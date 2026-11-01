// Host side of the engine: MKEngine and its implementation (the device code and its launchers: qwen38_mk.cu).

#include "mk_shared.cuh"
#include "mk_launch.h"

using namespace mk;

namespace {

// ------------------------------------------------------------------------------------------
// host side: loading / repacking
// ------------------------------------------------------------------------------------------

uint16_t f2h_bits(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t *>(&h);
}

// A persistent thread pool for parallel_rows: the loader calls it ~1000 times, too often to spawn
// hardware_concurrency() threads per call. One job at a time; a call while the pool is busy (another thread's job, or a
// call from inside a job) spawns threads of its own. Rows go out in chunks from a shared counter (every
// row's work is independent; nothing depends on which thread runs it). MK_LOAD_POOL=0: spawn per call.
class RowPool {
  public:
    static RowPool & get() {
        static RowPool p;
        return p;
    }
    bool run(int64_t n, const std::function<void(int64_t)> & f) {
        if (nth_ <= 0) return false;
        std::unique_lock<std::mutex> job(job_mu_, std::try_to_lock);
        if (!job.owns_lock()) return false;
        {
            std::lock_guard<std::mutex> l(mu_);
            fn_ = &f;
            n_ = n;
            ch_ = std::max<int64_t>(1, n / (8 * (int64_t) (nth_ + 1)));
            next_ = 0;
            active_ = nth_;
            err_ = nullptr;
            gen_++;
        }
        cv_.notify_all();
        work();
        std::exception_ptr err;
        {
            std::unique_lock<std::mutex> l(mu_);
            cvd_.wait(l, [&] { return active_ == 0; });
            fn_ = nullptr;
            std::swap(err, err_);
        }
        if (err) std::rethrow_exception(err); // (the first row that threw, on any thread)
        return true;
    }
    ~RowPool() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
            gen_++;
        }
        cv_.notify_all();
        for (auto & t : th_) t.join();
    }

  private:
    RowPool() {
        const char * e = getenv("MK_LOAD_POOL");
        nth_ = e && atoi(e) == 0 ? 0 : (int) std::max(1u, std::thread::hardware_concurrency()) - 1;
        try {
            for (int i = 0; i < nth_; i++) th_.emplace_back([this] { loop(); });
        } catch (...) { // (no more threads: the pool is the ones that started)
            nth_ = (int) th_.size();
        }
    }
    // a row that throws stops the job: no more rows are handed out, run() rethrows its exception
    void work() {
        try {
            for (;;) {
                const int64_t r0 = next_.fetch_add(ch_);
                if (r0 >= n_) return;
                const int64_t r1 = std::min(n_, r0 + ch_);
                for (int64_t r = r0; r < r1; r++) (*fn_)(r);
            }
        } catch (...) {
            next_.store(n_);
            std::lock_guard<std::mutex> l(mu_);
            if (!err_) err_ = std::current_exception();
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            work();
            {
                std::lock_guard<std::mutex> l(mu_);
                active_--;
            }
            cvd_.notify_all();
        }
    }
    int nth_ = 0;
    std::vector<std::thread> th_;
    std::mutex mu_, job_mu_;
    std::condition_variable cv_, cvd_;
    const std::function<void(int64_t)> * fn_ = nullptr;
    int64_t n_ = 0, ch_ = 1;
    std::atomic<int64_t> next_{0};
    int active_ = 0;
    std::exception_ptr err_; // (under mu_)
    uint64_t gen_ = 0;
    bool stop_ = false;
};

template <class F>
void parallel_rows(int64_t n, F f) {
    if (n <= 0) return;
    const std::function<void(int64_t)> fn = [&](int64_t r) { f(r); };
    if (RowPool::get().run(n, fn)) return;
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::exception_ptr err;
    std::vector<std::thread> th;
    try {
        for (int i = 0; i < nt; i++) {
            th.emplace_back([&, i] {
                try {
                    for (int64_t r = i; r < n && !stop.load(std::memory_order_relaxed); r += nt) f(r);
                } catch (...) {
                    stop = true;
                    std::lock_guard<std::mutex> l(mu);
                    if (!err) err = std::current_exception();
                }
            });
        }
    } catch (...) { // (a thread could not be created: the started ones are joined before the error goes up)
        stop = true;
        for (auto & t : th) t.join();
        throw;
    }
    for (auto & t : th) t.join();
    if (err) std::rethrow_exception(err);
}

// GGUF Q6_K block (210 B: ql[128] | qh[64] | int8 scales[16] | fp16 d) from 256 integers q in [-32, 31],
// one scale per 16 of them and the bits of d.
void put_q6k(uint8_t * blk, const int * q, const int8_t * sc, uint16_t dbits) {
    uint8_t *ql = blk, *qh = blk + 128;
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < 32; l++) {
            const int v0 = q[128 * n + l] + 32, v1 = q[128 * n + l + 32] + 32;
            const int v2 = q[128 * n + l + 64] + 32, v3 = q[128 * n + l + 96] + 32;
            ql[64 * n + l] = (uint8_t) ((v0 & 0xF) | ((v2 & 0xF) << 4));
            ql[64 * n + l + 32] = (uint8_t) ((v1 & 0xF) | ((v3 & 0xF) << 4));
            qh[32 * n + l] = (uint8_t) (((v0 >> 4) & 3) | (((v1 >> 4) & 3) << 2) | (((v2 >> 4) & 3) << 4) | (((v3 >> 4) & 3) << 6));
        }
    memcpy(blk + 192, sc, 16);
    memcpy(blk + 208, &dbits, 2);
}

// Q3_K or IQ3_S row -> Q6_K with the same values. Both store a small integer times d times an integer
// sub-block scale (Q3_K: d (sc - 32) q, q in [-4, 3], per 16; IQ3_S: d (1 + 2 s) q, |q| <= 15, per 32),
// so q = value / (d * scale) is exact and Q6_K holds it with the same d and scale. f: the row dequantized.
void reencode_q6k(uint32_t type, const uint8_t * src, const float * f, uint8_t * out, int64_t K) {
    for (int64_t b = 0; b < K / 256; b++) {
        const uint8_t * blk = src + b * 110;
        const float * y = f + b * 256;
        uint16_t dbits;
        int8_t sc[16];
        if (type == gguf::Q3_K) { // hmask[32] | qs[64] | scales[12] (6 bits each) | fp16 d
            memcpy(&dbits, blk + 108, 2);
            uint32_t aux[4];
            memcpy(aux, blk + 96, 12);
            const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
            aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
            aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
            aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
        } else { // IQ3_S: fp16 d | qs[64] | qh[8] | signs[32] | scales[4] (4 bits per 32)
            memcpy(&dbits, blk, 2);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) (1 + 2 * ((blk[106 + i / 4] >> (4 * ((i / 2) & 1))) & 0xf));
        }
        const float d = ggml_fp16_to_fp32(dbits);
        int q[256];
        for (int e = 0; e < 256; e++) {
            const float s = d * (float) sc[e / 16];
            q[e] = s == 0.0f ? 0 : (int) lrintf(y[e] / s);
        }
        put_q6k(out + b * 210, q, sc, dbits);
    }
}

// IQ4_NL row (18 B blocks: fp16 d | 16 bytes of 4-bit indices) -> Q8_0 (fp16 d | 32 int8) with the same
// d and q = kvalues_iq4nl[index]: the same values.
void reencode_iq4nl_q80(const uint8_t * src, uint8_t * out, int64_t K) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * s = src + b * 18;
        uint8_t * o = out + b * 34;
        memcpy(o, s, 2);
        for (int j = 0; j < 16; j++) {
            o[2 + j] = (uint8_t) kv[s[2 + j] & 0xf];
            o[2 + j + 16] = (uint8_t) kv[s[2 + j] >> 4];
        }
    }
}

// Q4_0 row (18 B blocks: fp16 d | 16 bytes, element j in the low nibble of byte j and element j + 16 in its high nibble,
// value d * (q - 8)) -> Q8_0 with the same d and q - 8 as int8: the same values (Q4_0 tensors outside the MTP block)
void reencode_q40_q80(const uint8_t * src, uint8_t * out, int64_t K) {
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * s = src + b * 18;
        uint8_t * o = out + b * 34;
        memcpy(o, s, 2);
        for (int j = 0; j < 16; j++) {
            o[2 + j] = (uint8_t) (int8_t) ((s[2 + j] & 0xf) - 8);
            o[2 + j + 16] = (uint8_t) (int8_t) ((s[2 + j] >> 4) - 8);
        }
    }
}

// GGUF IQ4_XS blocks (136 B: fp16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]) -> the layout rows_iq4x and the
// prefill GEMM read: pairs of blocks in 272 B (16-byte aligned, no padding): the two 8-byte GGUF headers, then
// each block's 128 index bytes in Q4_K order (see rows_iq4x). nblocks must be even.
void repack_iq4xs(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * 136, *qs = s + 8;
        uint8_t * pr = dst + (i >> 1) * 272;
        memcpy(pr + 8 * (i & 1), s, 8);
        uint8_t * o = pr + 16 + 128 * (i & 1);
        auto idx = [&](int v) {
            const int ib = v >> 5, j = v & 31;
            return j < 16 ? qs[16 * ib + j] & 0xf : qs[16 * ib + j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

// GGUF IQ4_NL (8 blocks of 18 B: fp16 d, u8 qs[16]) per 256 weights -> 144 B: the eight fp16 d, then the 128 index
// bytes in the order of repack_iq4xs (rows_iq4nl, prefill TIQ4NL)
void repack_iq4nl(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t * s = src + i * 144;
        uint8_t * o = dst + i * 144;
        for (int ib = 0; ib < 8; ib++) memcpy(o + 2 * ib, s + 18 * ib, 2);
        auto idx = [&](int v) {
            const uint8_t * qs = s + 18 * (v >> 5) + 2;
            const int j = v & 31;
            return j < 16 ? qs[j] & 0xf : qs[j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[16 + 32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

// Q3_K's 12 bytes of packed 6-bit scales -> sc - 32 (ggml's unpacking)
void q3k_scales(const uint8_t * packed, int8_t (&sc)[16]) {
    uint32_t aux[4];
    memcpy(aux, packed, 12);
    const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
    aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
    aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
    aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
    for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
}

// GGUF Q3_K row (110 B blocks: hmask[32] | qs[64] | scales[12] | fp16 d) -> the native layout of rows_q3k: values (96 B
// per block) and the Q6_K meta stream (18 B per block in pairs). y: the row as ggml dequantizes it, for the self-check
// (the native values, dequantized as ggml does, must reproduce it bit for bit). nb must be even.
bool repack_q3k(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *hm = blk, *qs = blk + 32;
        uint8_t u3[256];
        for (int e = 0; e < 256; e++) {
            const int n = e >> 7, j = (e >> 5) & 3, l = e & 31;
            u3[e] = (uint8_t) (((qs[32 * n + l] >> (2 * j)) & 3) | (((hm[l] >> (4 * n + j)) & 1) << 2));
        }
        uint8_t * o = q + (size_t) b * 96;
        memset(o, 0, 96);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int i = 0; i < 16; i++) {
                const int ea = 64 * p + 16 * hh + i, eb = ea + 32, m = i & 3, j = i >> 2;
                o[8 * w + m] |= (uint8_t) ((u3[ea] & 3) << (2 * j));
                o[8 * w + 4 + m] |= (uint8_t) ((u3[eb] & 3) << (2 * j));
                o[64 + 4 * w + m] |= (uint8_t) (((u3[ea] >> 2) << j) | ((u3[eb] >> 2) << (4 + j)));
            }
        }
        int8_t sc[16];
        q3k_scales(blk + 96, sc);
        uint16_t dbits;
        memcpy(&dbits, blk + 108, 2);
        uint8_t * mp = meta + (size_t) (b & ~1) * 18;
        memcpy(mp + 16 * (b & 1), sc, 16);
        memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
        // self-check: decode the native bytes as rows_q3k does, dequantize as ggml (dl = d (sc - 32), dl q)
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, j = i >> 2;
            const int lo = (o[8 * w + 4 * isb + m] >> (2 * j)) & 3, hb = (o[64 + 4 * w + m] >> (4 * isb + j)) & 1;
            const int8_t s = (int8_t) mp[16 * (b & 1) + 4 * p + 2 * isb + ((r >> 4) & 1)];
            const float v = (d * (float) s) * (float) (lo + 4 * hb - 4);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

// ggml's iq3s_grid, recovered with ggml's own IQ3_S dequantization (d = 1, scale 1, no signs: index -> 4 magnitudes)
std::vector<uint32_t> iq3s_grid_host() {
    std::vector<uint32_t> g(512);
    const auto * tr = ggml_get_type_traits(GGML_TYPE_IQ3_S);
    uint8_t blk[110] = {};
    const uint16_t one = 0x3c00;
    float y[256];
    for (int idx = 0; idx < 512; idx++) {
        memset(blk, 0, sizeof blk);
        memcpy(blk, &one, 2);
        blk[2] = (uint8_t) (idx & 0xff); // qs[0]: index 0 of sub-block 0
        blk[66] = (uint8_t) (idx >> 8);  // qh[0] bit 0
        tr->to_float(blk, y, 256);
        uint32_t v = 0;
        for (int m = 0; m < 4; m++) {
            const int gm = (int) y[m];
            if ((float) gm != y[m] || gm < 1 || gm > 15) throw std::runtime_error("iq3s_grid: unexpected IQ3_S dequantization");
            v |= (uint32_t) gm << (8 * m);
        }
        g[idx] = v;
    }
    return g;
}

// GGUF IQ3_S row (110 B blocks: fp16 d | qs[64] | qh[8] | signs[32] | scales[4]) -> the native layout of rows_iq3s:
// values (96 B per block) and meta (16 B per block). Self-check against y as repack_q3k.
bool repack_iq3s(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y, const std::vector<uint32_t> & grid) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        uint8_t * o = q + (size_t) b * 96;
        uint8_t * mt = meta + (size_t) b * 16;
        memset(o, 0, 96);
        memset(mt, 0, 16);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int isb = 0; isb < 2; isb++) {
                const int ib = 2 * p + isb; // the 32-weight sub-block of A (isb 0) or B (isb 1)
                for (int c = 0; c < 4; c++) {
                    const int k = 4 * hh + c; // grid index k of the sub-block (its weights 4k .. 4k + 3)
                    o[8 * w + 4 * isb + c] = qs[8 * ib + k];
                    mt[w] |= (uint8_t) (((qh[ib] >> k) & 1) << (4 * isb + c));
                }
                for (int i = 0; i < 16; i++) {
                    const int off = 16 * hh + i; // weight of the sub-block
                    const int sgn = (signs[4 * ib + off / 8] >> (off % 8)) & 1;
                    o[64 + 4 * w + (i & 3)] |= (uint8_t) (sgn << (4 * isb + (i >> 2)));
                }
            }
        }
        memcpy(mt + 8, scales, 4);
        memcpy(mt + 12, blk, 2);
        uint16_t dbits;
        memcpy(&dbits, blk, 2);
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, c = i >> 2;
            const int idx = o[8 * w + 4 * isb + c] | (((mt[w] >> (4 * isb + c)) & 1) << 8);
            const int g = (grid[idx] >> (8 * m)) & 0xff;
            const int sgn = (o[64 + 4 * w + m] >> (4 * isb + c)) & 1;
            const float db = d * (float) (1 + 2 * ((mt[8 + p] >> (4 * isb)) & 0xf));
            const float v = db * (float) g * (sgn ? -1.f : 1.f);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

// Host: work split for a chunk of T queries at n0 on `slots` resident CTAs.  The items of the last, partial
// wave (the highest q tiles of the last kv heads) are split into S tile ranges when that shortens the tail;
// S = 1 / nfull = all items: no split.  part_floats: capacity of the partials buffer.
inline void a2_plan(int T, int n0, int slots, size_t part_floats, int & nfull, int & S) {
    const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV, rem = slots > 0 ? N % slots : 0;
    nfull = N;
    S = 1;
    if (rem == 0) return;
    const int qt = (N - rem) % nqt; // smallest tail item
    const int ntile_min = (n0 + qt * A2_Q + std::min(A2_Q, T - qt * A2_Q) - 1) / A2_K + 1;
    const int smax = std::min(std::min(32, ntile_min / 2), (int) std::min<size_t>(1024, part_floats / ((size_t) A2_PART * rem)));
    double best = 1.0; // tail length in item times
    for (int s = 2; s <= smax; s++) {
        const double t = (double) ((rem * s + slots - 1) / slots) / s;
        if (t < best - 0.02) {
            best = t;
            S = s;
        }
    }
    if (S > 1) nfull = N - rem;
}

#include "wv_host.cuh"
#include "loader.cuh"

} // namespace

struct MKEngine::Impl {
    gguf::File file;
    LoadClock lc;                         // load phases (MK_LOADLOG=1 prints them)
    lclk::time_point t_load = lclk::now(); // (after the GGUF parse)
    ModelW hw{};
    Bufs hb{};
    std::vector<void *> allocs;
    void * embd_host = nullptr;
    int ctx;
    int nsm = 0;
    float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f; // sampling (temp 0: greedy)
    int smp_top_k = TOPK_MAX;
    unsigned long long smp_seed = 0;
    int ndraft;          // MTP drafts the next launch produces (at most; the gate may stop earlier)
    int nd_avail;        // drafts produced by the previous launch (verified by the next decode cycle)
    bool all_acc = true; // the last decode cycle accepted all of its drafts
    int last_acc = 0;    // drafts the last decode cycle accepted
    // draft-depth policy (MKEngine::set_policy). Default: up to n_draft drafts, the on-device gate on the running
    // product c_k of the MTP's own top-1 probabilities stops drafting after draft k >= 2 when c_k < 0.4 and drops
    // the 4th draft (the T = 5 verify row) when c_4 < 0.4; sampled drafts when sampling. Tuned on replayed logs
    // + live A/B. Without gates and sampled drafts: "depth=0,drop=0,stop=0,chain=0,h1=0".
    struct Policy {
        int depth = 1;       // 0: n_draft - 1 after a cycle with a rejection (at least 3), 1: always n_draft,
                             // 2: 1 + the accepted count of the last cycle (clamped to [dmin, n_draft])
        int dmin = 1;
        // generalised gate (any depth): stop after draft k >= 1 below 0.4, drop the deepest draft (k = n_draft - 1, the
        // T = n_draft + 1 verify row) below drop_last (at n_draft 4: the default gates above)
        float gate_drop[MAXT - 1] = {}, gate_stop[MAXT - 1] = {0.f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f};
        float drop_last = 0.4f;
        // context cap: from position dctx on, at most 4 drafts (T <= 5: at long context the wide pass's second
        // head-group pass over K / V costs more than drafts 5..7 gain)
        int dctx = INT_MAX;
        // g7 (default on): at depth > 5 (the wide tier T 6-8 costs the same per cycle, a draft there only its ~0.5 ms MTP
        // pass) the gates are stop after drafts 2-4 below 0.25, drop draft 5 below 0.3 (stay at T <= 5), stop after drafts
        // 5-6 below 0.3, never drop the deepest draft (from an offline policy search on open-policy logs; greedy output
        // is unchanged). Any explicit stop / drop key turns it off.
        int g7 = 1;
        int gate_chain = 1;
        int h1 = 1; // sampling: sampled drafts (Launch::spec_smp)
    } pol;
    // the policy's draft depth at position pos: n_draft, or at most 4 from Policy::dctx on
    int depth_top(int pos) const { return pos >= pol.dctx ? std::min(n_draft_max, 4) : n_draft_max; }
#if MK_SPEC_LOG
    FILE * slog = nullptr;               // MK_SPEC_LOG_FILE: one JSON line per decode cycle
    std::vector<float2> slog_q;          // dlog of the drafts the next cycle verifies
    void slog_drafts() {
        if (!slog) return;
        slog_q.resize((size_t) MAXT * (TOPK_MAX + 1));
        CK(cudaMemcpy(slog_q.data(), hb.dlog, slog_q.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    }
#endif
    std::vector<int> seq; // the sequence so far: prompt(s) and emitted tokens (position = index)
    // Images (MKEngine::MKImage): spans of positions whose input is an embedding row, with M-RoPE positions. llama.cpp's
    // layout for Qwen-VL (mtmd, IMROPE): the span's rope base r is the position after the text before it; raster row q
    // has (t, h, w) = (r, r + q / nx, r + q % nx), RoPE frequency i takes t, h, w for i % 3 = 0, 1, 2 (sections 11 / 11 /
    // 10 of the 32 pairs); the text after the span continues at r + max(nx, ny), so each span shifts the rope position of
    // everything after it down by nx * ny - max(nx, ny). Attention stays causal in sequence order (llama.cpp's 2D mask
    // over (t, h, w) orders a span's rows by raster position, which is their sequence order). Only hb.rope changes: its
    // row n holds the (cos, sin) of position n's rope angles, so the decode and prefill kernels index it by sequence
    // position (as for text). rope_base keeps the plain table (row p = position p): every rewritten row is a gather from it.
    struct Span {
        int pos, nx, ny;
        bool operator==(const Span & o) const { return pos == o.pos && nx == o.nx && ny == o.ny; }
        int n() const { return nx * ny; }
    };
    std::vector<Span> spans;      // the images in seq (ascending, disjoint)
    std::vector<Span> rope_spans; // the spans hb.rope's rows reflect
    std::vector<float2> rope_base;
    float * emb_host = nullptr;   // the prefill's image rows: pinned, mapped (cP.emb = its device alias)
    size_t emb_cap = 0;           // (floats)
    volatile int * err_host = nullptr; // cB.err's host side
    volatile Ctl * ctl_eg_host = nullptr; // cB.ctl_eg's host side
    bool bar_live = false;                // the cB.bar counts run on from the previous launch (zeroed once, and after bench launches)
    // ctl after a decode launch: the kernel's mapped mirror (a device-to-host copy without one)
    void ctl_fetch(Ctl & c) {
        if (ctl_eg_host) memcpy(&c, const_cast<const Ctl *>(ctl_eg_host), sizeof(Ctl));
        else CK(cudaMemcpy(&c, hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    }
    bool err_mtp_warned = false;
#if MK_TEST_NAN
    int test_nan_at = getenv("MK_TEST_NAN_AT") ? atoi(getenv("MK_TEST_NAN_AT")) : 0, test_nan_n = 0;
#endif
    void launch_failed() {
        const int bad = err_host[0], mtp = err_host[1];
        err_host[0] = err_host[1] = 0;
        if (mtp && !err_mtp_warned) {
            err_mtp_warned = true;
            fprintf(stderr, "[mk] warning: MTP catch-up out of range near position %d (drafts only, output unaffected)\n", hctl.n);
        }
        if (bad)
            throw MKCudaError("no finite logit in the output head (NaN / inf model state); the sequence is reset (a parallel "
                              "slot's when it begins its next prompt)");
    }
    // Rewrites hb.rope's rows from the first position where `spans` and `rope_spans` differ (none: no copy). Text rows are
    // rope_base rows shifted by the images before them, image rows per-frequency gathers; pageable host-to-device copies
    // on the legacy stream, ordered after every kernel that read the old rows.
    void rope_sync() {
        if (spans == rope_spans) return;
        size_t k = 0;
        while (k < spans.size() && k < rope_spans.size() && spans[k] == rope_spans[k]) k++;
        const int nr = hb.ctx; // the table's rows (KV_PAD past ctx)
        int from = nr, shift = 0;
        if (k < spans.size()) from = std::min(from, spans[k].pos);
        if (k < rope_spans.size()) from = std::min(from, rope_spans[k].pos);
        for (size_t j = 0; j < k; j++) shift += spans[j].n() - std::max(spans[j].nx, spans[j].ny);
        float2 * dst = const_cast<float2 *>(hb.rope);
        std::vector<float2> rows;
        size_t j = k;
        for (int p = from; p < nr;) {
            const int next = j < spans.size() ? spans[j].pos : nr;
            if (p < next) { // text rows p .. next - 1
                CK(cudaMemcpy(dst + (size_t) p * 32, rope_base.data() + (size_t) (p - shift) * 32, (size_t) (next - p) * 32 * sizeof(float2),
                              cudaMemcpyHostToDevice));
                p = next;
                continue;
            }
            const Span & s = spans[j++];
            const int r = p - shift, n = std::min(s.n(), nr - p);
            rows.resize((size_t) n * 32);
            for (int q = 0; q < n; q++) {
                const int pos3[3] = {r, r + q / s.nx, r + q % s.nx};
                for (int i = 0; i < 32; i++) rows[(size_t) q * 32 + i] = rope_base[(size_t) pos3[i % 3] * 32 + i];
            }
            CK(cudaMemcpy(dst + (size_t) p * 32, rows.data(), rows.size() * sizeof(float2), cudaMemcpyHostToDevice));
            shift += s.n() - std::max(s.nx, s.ny);
            p += s.n();
        }
        rope_spans = spans;
    }
    // the sequence goes back to `len` positions (reset, rewind, truncate): the images from there on are dropped; one cut
    // by `len` stays (its rows before len are processed; a prefill continuing inside it must supply it again)
    void spans_cut(int len) {
        while (!spans.empty() && spans.back().pos >= len) spans.pop_back();
        rope_sync();
    }
    // DeltaNet checkpoints: the state after position pos - 1 (recurrent state, conv ring, h ring, ctl), which lets a
    // caller rewind to pos (KV rows before it are never rewritten) and prefill from there. A pool of MK_CK_SLOTS
    // (default 8) checkpoints of 0.17 GB each in pinned host memory (MK_CK_DEVICE=1: one, in device memory); when it
    // is full, ck_assign evicts by coverage. Saving never waits on the host and adds no VRAM:
    // - Inside a prefill (pf_run, Snap): at position P of a chunk, each DeltaNet layer's scan is split at P and its
    //   first part writes the state at P into a staging slot in hp.xm (idle until the chunk's MTP part), next to the
    //   conv ring at P (pk_ck_conv); a device-to-host copy on ck_sv moves the staging slot into the checkpoint while
    //   the next layers run. The h ring at P goes straight into the mapped host buffer (pk_ck_hist). Stream 0 waits
    //   (on the device) only before reusing a staging slot and before the MTP part. The scan is split only at points
    //   of its 32-token grid, which changes no arithmetic: saving a checkpoint never changes what the prefill computes.
    //   A position off the grid (the exact ck_at of a chat turn) is saved by a side scan from the grid point before
    //   it (the state a prefill ending there computes).
    // - At the end of a prompt (ck_save_end): a device-to-device snapshot into the idle batched-prefill scratch
    //   on stream 0 (~0.4 ms), then the device-to-host copy on ck_sv, which runs under the first decode launch.
    // - Restore: the host-to-device copy runs on ck_st in CK_NG groups of DeltaNet layers, issued once the next
    //   prefill has uploaded its tokens (ck_issue_restore: nothing on stream 0 then queues behind it on the copy
    //   engine); stream 0 waits for a group's event only before the first kernel that reads that group's state
    //   (pf_run: dn_prep of the group's first layer). Same bytes, same order of use: bit-identical.
    // If the scratch is too small for the end snapshot (MK_PF_CAP below ~640), the end save copies straight from the
    // state and stream 0 waits for it before the state is next modified; below ~340 there are too few staging slots
    // too, and a checkpoint inside a prompt splits the prefill into two pf_runs around it.
    static constexpr size_t CK_SL = (size_t) NVH * 128 * 128, CK_CL = (size_t) CONV_RING * QKV; // floats per DeltaNet layer
    static constexpr size_t CK_NS = 48 * CK_SL, CK_NC = 48 * CK_CL, CK_NH = (size_t) H_RING * D;
    static constexpr int CK_NG = 5;
    static constexpr int ck_gb[CK_NG + 1] = {0, 1, 4, 12, 24, 48}; // restore group g: DeltaNet layers [ck_gb[g], ck_gb[g + 1])
    struct CkSlot {
        void * mem = nullptr;                                       // pinned host block (device memory: MK_CK_DEVICE)
        float *dstate = nullptr, *conv = nullptr, *hhist = nullptr; // in it
        float * hdev = nullptr;                                     // hhist as kernels address it
        Ctl ctl{};
        int pos = -1;                                               // -1: free
        cudaEvent_t ev = nullptr;                                   // the copies into this slot (ck_sv) are done
        bool pending = false;                                       // ... may still be running
        char * ring = nullptr;                                      // MK_EXACT + MK_KV_WIN: the q8 window rings at pos
        bool ring_ok = false;                                       // ring holds this checkpoint's rings
        bool pin = false;                                           // pinned (pin_set; meaningful while pos >= 0)
        unsigned pin_t = 0;                                         // ... by the request with this stamp
    };
    struct Snap { int pos, slot; };
    std::vector<CkSlot> ck;
    bool ck_dev = false;
    // MK_EXACT=1 (exact reuse): every result of a prefill row is independent of how the sequence was cut into prefill
    // calls and chunks, so a prefill continued from a checkpoint computes bit for bit what a prefill from 0 computes:
    // - the prefill attention runs unsplit (no a2_plan tail split: one online softmax per row over absolute tiles),
    // - the MK_KV_WIN ring / main split is taken per query tile (PBufs::wsrow), as decode takes it per row,
    // - checkpoints sit on the absolute 32-token scan grid (ck_exact is snapped like the others; callers restore only
    //   checkpoints of a prefill that started on the grid, which every restore and a prefill from 0 do) and are only
    //   saved inside the batched prefill (no end-of-prompt save: the prompt's last row runs through the megakernel),
    // - with MK_KV_WIN, a checkpoint also keeps the q8 window rings as of its position (a restore then needs no
    //   lossy refill from the low-bit main buffer); pinned host, ~NKV * kvrc * 272 B per ring side and slot.
    // The caller (mk_server --exact-reuse) never appends to decoded rows: it restores the last checkpoint at or
    // before the divergence and prefills from there (the reply's rows were written by the decode kernels).
    bool exact = false;
    bool exact_off_grid = false; // a batched prefill started off the grid (warned once)
    struct RingSide { int8_t * q; __half * s; size_t off; }; // device ring values / scales, offset in CkSlot::ring
    std::vector<RingSide> rsides[N_ATT_SLOTS];
    size_t ring_bytes_ck = 0;          // per checkpoint
    cudaEvent_t rs_ev[N_ATT_SLOTS] = {}; // pf_run: a chunk's whole-ring copies of slot s were issued (ck_sv)
    cudaEvent_t rs_ev_pf = nullptr, rs_ev_done = nullptr;
    bool rs_busy = false;              // ring copies on ck_sv may still read rings stream 0 will overwrite
    size_t ring_vb() const { return (size_t) NKV * hb.kvrc * HD; }
    size_t ring_sb() const { return (size_t) NKV * hb.kvrc * (HD / 32) * sizeof(__half); }
    double ck_ms = 0;             // last save / restore enqueue time (MK_MEMLOG)
    cudaStream_t ck_st = nullptr; // restores
    cudaStream_t ck_sv = nullptr; // saves
    cudaEvent_t ck_ev_src = nullptr, ck_ev_sv = nullptr, ck_ev_scr = nullptr, ck_ev_grp[CK_NG] = {};
    bool ck_scr_busy = false; // a save copy on ck_sv may still read the prefill scratch (ck_ev_scr)
    int ck_direct = -1;       // slot of an end save that reads the live state directly: stream 0 waits before modifying it
    int ck_rs_slot = -1;      // slot of a restore whose copies are not issued yet (ck_issue_restore)
    int ck_rs_src = -1;       // slot of the last restore (its copies may be running until its prefill waited for them)
    int ck_rs_next = CK_NG;   // first restore group stream 0 has not waited for (CK_NG: no restore pending)
    struct CkChunk { char * dev; char * scr; size_t off, n; };
    std::vector<CkChunk> ck_plan; // end save: state -> prefill scratch (stream 0) -> slot byte offset off (ck_sv)
    // staging slots of in-prefill saves in hp.xm: one DeltaNet layer's state + conv ring each
    static constexpr size_t STG_F = (CK_SL + CK_CL + 63) & ~(size_t) 63;
    static constexpr int STG_MAX = 16;
    int stg_n = 0, stg_next = 0, stg_last = -1;
    cudaEvent_t stg_ev[STG_MAX] = {};
    bool stg_busy[STG_MAX] = {};

    // Allocated at construction: cudaHostAlloc of 0.17 GB takes 0.1-0.5 s under WSL2, which would otherwise land
    // in the first prompt's time. A pinned slot that fails to allocate ends the pool there (the first: one device slot).
    void ck_alloc() {
        const char * cd = getenv("MK_CK_DEVICE");
        ck_dev = cd && cd[0] == '1';
        int nslots = 8;
        if (const char * e = getenv("MK_CK_SLOTS")) nslots = std::max(1, std::min(64, atoi(e)));
        const size_t nf = CK_NS + CK_NC + CK_NH;
        for (int i = 0; i < (ck_dev ? 1 : nslots); i++) {
            CkSlot s;
            if (!ck_dev && cudaHostAlloc(&s.mem, nf * 4, cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                s.mem = nullptr;
                if (i > 0) {
                    fprintf(stderr, "[mk] warning: no pinned host memory for DeltaNet checkpoint %d: keeping %d\n", i + 1, i);
                    break;
                }
                ck_dev = true;
            }
            if (ck_dev) s.mem = dalloc<float>(nf, false);
            s.dstate = static_cast<float *>(s.mem);
            s.conv = s.dstate + CK_NS;
            s.hhist = s.conv + CK_NC;
            if (ck_dev) s.hdev = s.hhist;
            else CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&s.hdev), s.hhist, 0));
            CK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
            ck.push_back(s);
        }
        pin_max = std::min(2, (int) ck.size() / 4);
        CK(cudaStreamCreateWithFlags(&ck_st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&ck_sv, cudaStreamNonBlocking));
        for (cudaEvent_t * e : {&ck_ev_src, &ck_ev_sv, &ck_ev_scr}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        for (auto & e : ck_ev_grp) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto & e : stg_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        // MK_EXACT (see `exact`): with MK_KV_WIN rings, each checkpoint also holds the rings as of its position
        if (const char * e = getenv("MK_EXACT")) exact = atoi(e) != 0;
        if (exact && hb.kvwin > 0) {
            if (ck_dev) throw std::runtime_error("MK_EXACT with MK_KV_WIN needs host checkpoints (MK_CK_DEVICE=0)");
            for (int s = 0; s < N_ATT_SLOTS; s++)
                for (int w = 0; w < 2; w++)
                    if (hb.kvrp[w][s]) {
                        rsides[s].push_back({hb.kvrp[w][s], hb.kvrs[w][s], ring_bytes_ck});
                        ring_bytes_ck += ring_vb() + ring_sb();
                    }
            for (auto & s : ck)
                if (cudaHostAlloc(reinterpret_cast<void **>(&s.ring), ring_bytes_ck, cudaHostAllocDefault) != cudaSuccess) {
                    cudaGetLastError();
                    s.ring = nullptr;
                    fprintf(stderr, "[mk] warning: no pinned host memory for a checkpoint's KV window rings: its restores refill them (not exact)\n");
                }
            for (int L = 0, d = 0; L < NL; L++) { // slot L / 4 (pf_run): restore group of the DeltaNet layer before it
                if (!hw.L[L].attn) {
                    d++;
                    continue;
                }
                int g = 0;
                while (g + 1 < CK_NG && ck_gb[g + 1] <= d - 1) g++;
                ring_grp[L / 4] = g;
            }
            ring_grp[N_ATT_SLOTS - 1] = CK_NG - 1; // MTP block: after the trunk
            for (auto & e : rs_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            for (cudaEvent_t * e : {&rs_ev_pf, &rs_ev_done}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
            fprintf(stderr, "[mk] MK_EXACT: checkpoints keep the KV window rings (%.1f MiB each, pinned host)\n", ring_bytes_ck / 1048576.0);
        }
        if (exact) fprintf(stderr, "[mk] MK_EXACT=1: prefill rows independent of the chunking (attention unsplit, per-tile window, grid checkpoints)\n");
        mem_mark(ck_dev ? "DeltaNet checkpoint" : "DeltaNet checkpoints (host)");
    }
    void ck_free() {
        if (ck_st) cudaStreamSynchronize(ck_st);
        if (ck_sv) cudaStreamSynchronize(ck_sv);
        for (auto & s : ck) {
            if (s.ev) cudaEventDestroy(s.ev);
            if (s.mem && !ck_dev) cudaFreeHost(s.mem);
            if (s.ring) cudaFreeHost(s.ring);
        }
        for (cudaEvent_t e : rs_ev)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : {rs_ev_pf, rs_ev_done})
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : {ck_ev_src, ck_ev_sv, ck_ev_scr})
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ck_ev_grp)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : stg_ev)
            if (e) cudaEventDestroy(e);
        if (ck_st) cudaStreamDestroy(ck_st);
        if (ck_sv) cudaStreamDestroy(ck_sv);
    }
    // end-save snapshot plan over the prefill scratch buffers (after pf_init; empty when they are too small) and
    // the staging slots in hp.xm
    void ck_make_plan() {
        ck_plan.clear();
        stg_n = (int) std::min<size_t>(STG_MAX, (size_t) hp.cap * D / STG_F);
        if (ck_dev) return; // (device checkpoint: the end save copies on stream 0)
        const size_t cap = hp.cap;
        const std::pair<void *, size_t> scr[] = {
            {hp.y0, cap * FF * 4}, {hp.y1, y1_floats * 4}, {hp.xh2, cap * FF * 2}, {hp.y2, cap * VD * 4}, {hp.x, cap * D * 4},
            {hp.xm, cap * D * 4},  {hp.dq, cap * QKV * 2}, {hp.xh, cap * 2 * D * 2}, {hp.qh, cap * NH * HD * 2}};
        const std::tuple<void *, size_t, size_t> reg[] = { // state, its byte offset in a slot, bytes
            {hb.dstate, 0, CK_NS * 4}, {hb.conv, CK_NS * 4, CK_NC * 4}, {hb.h_hist, (CK_NS + CK_NC) * 4, CK_NH * 4}};
        size_t si = 0, so = 0; // scratch buffer, offset in it
        for (const auto & r : reg) {
            size_t done = 0;
            while (done < std::get<2>(r)) {
                while (si < std::size(scr) && scr[si].second - so < 256) si++, so = 0;
                if (si == std::size(scr)) {
                    ck_plan.clear();
                    return;
                }
                const size_t n = std::min(std::get<2>(r) - done, (scr[si].second - so) & ~(size_t) 255);
                ck_plan.push_back({static_cast<char *>(std::get<0>(r)) + done, static_cast<char *>(scr[si].first) + so,
                                   std::get<1>(r) + done, n});
                done += n;
                so += n;
            }
        }
    }
    // stream 0: before overwriting the prefill scratch, wait for an end save's copy still reading it
    void ck_scr_guard() {
        if (!ck_scr_busy) return;
        CK(cudaStreamWaitEvent(0, ck_ev_scr, 0));
        ck_scr_busy = false;
    }
    // stream 0: before modifying the live state, wait for an end save that reads it directly (no snapshot)
    void ck_guard_state() {
        if (ck_direct < 0) return;
        CK(cudaStreamWaitEvent(0, ck[ck_direct].ev, 0));
        ck_direct = -1;
    }
    // MK_EXACT + MK_KV_WIN, pf_run chunk n .. n + T - 1 with checkpoints sp[0 .. sn) in slots ss: a checkpoint's rings as
    // of its position P = the rings before the chunk stores its K / V rows (positions P - kvrc .. n - 1: whole-ring
    // copies, issued at the chunk start) with the rows of positions n .. P - 1 laid over them after the slot's
    // pk_attn_prep (ring_snap_rows). All copies on ck_sv; stream 0 waits (rs_ev[s]) before slot s's prep overwrites
    // its rings, and (ring_snap_guard) before a later chunk or launch could overwrite rows a copy still reads.
    int ring_grp[N_ATT_SLOTS] = {}; // restore group after whose wait stream 0 first touches slot s's rings
    void ring_snap_begin(int sn, const int * ss) {
        CK(cudaEventRecord(rs_ev_pf, 0));
        CK(cudaStreamWaitEvent(ck_sv, rs_ev_pf, 0));
        for (int j = 0; j < sn; j++) {
            if (ck[ss[j]].pending) CK(cudaStreamWaitEvent(ck_sv, ck[ss[j]].ev, 0)); // (ck_sv is in order anyway)
            ck[ss[j]].ring_ok = ck[ss[j]].ring != nullptr;
        }
        const bool rs_in = ck_rs_next < CK_NG; // a restore's copies may still be writing the rings
        for (int s = 0; s < N_ATT_SLOTS; s++) {
            if (rs_in && !rsides[s].empty()) CK(cudaStreamWaitEvent(ck_sv, ck_ev_grp[ring_grp[s]], 0));
            for (int j = 0; j < sn; j++) {
                char * dst = ck[ss[j]].ring;
                if (!dst) continue;
                for (const RingSide & r : rsides[s]) {
                    CK(cudaMemcpyAsync(dst + r.off, r.q, ring_vb(), cudaMemcpyDeviceToHost, ck_sv));
                    CK(cudaMemcpyAsync(dst + r.off + ring_vb(), r.s, ring_sb(), cudaMemcpyDeviceToHost, ck_sv));
                }
            }
            CK(cudaEventRecord(rs_ev[s], ck_sv));
        }
        CK(cudaEventRecord(rs_ev_done, ck_sv));
        rs_busy = true;
    }
    void ring_snap_rows(int s, int n, int sn, const int * sp, const int * ss) {
        if (rsides[s].empty()) return;
        CK(cudaEventRecord(rs_ev_pf, 0));
        CK(cudaStreamWaitEvent(ck_sv, rs_ev_pf, 0));
        const size_t rc = hb.kvrc, vrow = HD, srow = (HD / 32) * sizeof(__half);
        for (int j = 0; j < sn; j++) {
            char * dst = ck[ss[j]].ring;
            if (!dst) continue;
            for (int p = n; p < sp[j];) {
                const int r0 = p % (int) rc, c = std::min(sp[j] - p, (int) rc - r0);
                for (const RingSide & r : rsides[s]) {
                    CK(cudaMemcpy2DAsync(dst + r.off + r0 * vrow, rc * vrow, r.q + r0 * vrow, rc * vrow, c * vrow, NKV,
                                         cudaMemcpyDeviceToHost, ck_sv));
                    CK(cudaMemcpy2DAsync(dst + r.off + ring_vb() + r0 * srow, rc * srow, reinterpret_cast<const char *>(r.s) + r0 * srow,
                                         rc * srow, c * srow, NKV, cudaMemcpyDeviceToHost, ck_sv));
                }
                p += c;
            }
        }
        CK(cudaEventRecord(rs_ev_done, ck_sv));
    }
    void ring_snap_guard() {
        if (!rs_busy) return;
        CK(cudaStreamWaitEvent(0, rs_ev_done, 0));
        rs_busy = false;
    }
    // Issues a due restore's host-to-device copies on ck_st (5 groups, an event after each). Deferred from ck_restore
    // to just after the next prefill's token upload: a host-to-device copy on stream 0 (ctl, token ids, draft
    // vocabulary rows) queued behind the 0.17 GB on the copy engine would stall the host for the whole restore.
    void ck_issue_restore() {
        if (ck_rs_slot < 0) return;
        const CkSlot & s = ck[ck_rs_slot];
        ck_rs_src = ck_rs_slot;
        ck_rs_slot = -1;
        // stream 0's earlier work on the state (and on the slot's h ring) is complete before ck_st overwrites the
        // state, and so are the slot's save copies and a direct save still reading the state (all normally long done)
        ring_snap_guard(); // (ring copies of an earlier save still reading the rings)
        CK(cudaEventRecord(ck_ev_src, 0));
        CK(cudaStreamWaitEvent(ck_st, ck_ev_src, 0));
        if (s.pending) CK(cudaStreamWaitEvent(ck_st, s.ev, 0));
        if (ck_direct >= 0) {
            CK(cudaStreamWaitEvent(ck_st, ck[ck_direct].ev, 0));
            ck_direct = -1;
        }
        const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        CK(cudaMemcpyAsync(hb.h_hist, s.hhist, CK_NH * 4, k, ck_st)); // group 0
        for (int g = 0; g < CK_NG; g++) {
            const size_t l0 = ck_gb[g], nl = ck_gb[g + 1] - ck_gb[g];
            CK(cudaMemcpyAsync(hb.conv + l0 * CK_CL, s.conv + l0 * CK_CL, nl * CK_CL * 4, k, ck_st));
            CK(cudaMemcpyAsync(hb.dstate + l0 * CK_SL, s.dstate + l0 * CK_SL, nl * CK_SL * 4, k, ck_st));
            if (s.ring_ok) // MK_EXACT: the KV window rings of the attention slots stream 0 reaches after this group
                for (int a = 0; a < N_ATT_SLOTS; a++)
                    if (ring_grp[a] == g)
                        for (const RingSide & r : rsides[a]) {
                            CK(cudaMemcpyAsync(r.q, s.ring + r.off, ring_vb(), cudaMemcpyHostToDevice, ck_st));
                            CK(cudaMemcpyAsync(r.s, s.ring + r.off + ring_vb(), ring_sb(), cudaMemcpyHostToDevice, ck_st));
                        }
            CK(cudaEventRecord(ck_ev_grp[g], ck_st));
        }
        ck_rs_next = 0;
    }
    // stream 0: wait for the restore groups up to the one holding DeltaNet layer dl (-1: all of them)
    void ck_wait_restore(int dl = -1) {
        ck_issue_restore();
        if (ck_rs_next >= CK_NG || (dl >= 0 && dl < ck_gb[ck_rs_next])) return;
        int g = CK_NG - 1;
        if (dl >= 0)
            while (ck_gb[g] > dl) g--;
        CK(cudaStreamWaitEvent(0, ck_ev_grp[g], 0));
        ck_rs_next = g + 1;
        if (ck_rs_next >= CK_NG) ck_rs_src = -1; // (stream 0 waited for all of it)
    }
    // checkpoint at the end of a prompt into slot si (prefill read ctl back after its last launch: hctl is the device's)
    void ck_save_end(int si) {
        ck_wait_restore();
        CkSlot & s = ck[si];
        const auto t0 = std::chrono::steady_clock::now();
        if (ck_dev) { // device checkpoint: stream-ordered device copies
            CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToDevice, 0));
        } else {
            ck_scr_guard(); // a previous end save may still read the scratch (normally long done)
            for (const auto & c : ck_plan) CK(cudaMemcpyAsync(c.scr, c.dev, c.n, cudaMemcpyDeviceToDevice, 0));
            CK(cudaEventRecord(ck_ev_sv, 0));
            CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
            char * base = static_cast<char *>(s.mem);
            if (!ck_plan.empty()) {
                for (const auto & c : ck_plan) CK(cudaMemcpyAsync(base + c.off, c.scr, c.n, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaEventRecord(ck_ev_scr, ck_sv));
                ck_scr_busy = true;
            } else {
                CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToHost, ck_sv));
                ck_direct = si;
            }
            CK(cudaEventRecord(s.ev, ck_sv));
            s.pending = true;
        }
        s.ctl = hctl;
        s.pos = hctl.dn_c;
        s.pin = false;
        s.ring_ok = false; // (no KV window rings: a restore refills them)
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog)
            fprintf(stderr, "[mk-mem] checkpoint save at %d enqueued in %.2f ms (%s)\n", s.pos, ck_ms,
                    ck_dev ? "device" : ck_plan.empty() ? "host, direct" : "host, via the prefill scratch");
    }
    void ck_restore(int si) {
        ck_wait_restore(); // (a previous restore not yet waited for: normally none)
        const auto t0 = std::chrono::steady_clock::now();
        ck_rs_slot = si; // issued by the next pf_run after its token upload, or by whatever touches the state first
        const CkSlot & s = ck[si];
        hctl = s.ctl;
        hctl.mtp_valid = std::min(hctl.mtp_valid, s.pos);
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        seq.resize(s.pos);
        for (auto & o : ck) // later checkpoints belong to the abandoned continuation
            if (o.pos > s.pos) o.pos = -1;
        if (s.ring_ok) kv_top = std::min(kv_top, s.pos); // MK_EXACT: the restore copies the rings as of s.pos
        else ring_rewind(s.pos);
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog) fprintf(stderr, "[mk-mem] checkpoint restore to %d enqueued in %.2f ms\n", s.pos, ck_ms);
    }
    // MK_KV_WIN: the sequence goes back to position P (checkpoint restore, truncate). The abandoned positions P .. kv_top - 1
    // (kv_top: every launch's rows, draft rows included) occupied the ring slots of the window rows P - kvrc .. kv_top - kvrc - 1,
    // so exactly those rows are refilled from the main buffer (they stay at its precision until they age out of the window;
    // a fresh prefill of the same text has them at q8). Positions >= P are rewritten before they are read.
    int kv_top = 0;
    void ring_rewind(int P) {
        if (hb.kvwin > 0) {
            const int p0 = std::max(0, P - hb.kvrc), p1 = std::min(P, kv_top - hb.kvrc);
            if (p1 > p0) {
                ring_snap_guard(); // (MK_EXACT: ring copies of a checkpoint save still reading those rows)
                mkl::k_ring_refill(p0, p1);
                CK(cudaGetLastError());
            }
            if (memlog) fprintf(stderr, "[mk-mem] KV window ring: rewind to %d (rows written to %d), refilled %d rows\n", P, kv_top, std::max(0, p1 - p0));
        }
        kv_top = std::min(kv_top, P);
    }
    int ck_find(int pos) const {
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos == pos && pos >= 0) return i;
        return -1;
    }
    // Pinned checkpoints (MKEngine::ck_pin): the caller's checkpoint at the end of a prompt's shared prefix (a client's
    // system prompt and tools), kept against ck_assign's thinning and carried by every park of the sequence, so a new
    // session of the same client resumes there however long the conversation grew. A request for position P pins the
    // checkpoint at the last position in (P - 32, P] (where ck_positions moves P, whatever the chunk start); at most
    // pin_max (min(2, slots / 4)) are pinned, more release the least recently requested ones. The flag goes with its
    // checkpoint (side requests hide it with the kept sequence, parks carry it, a rewind before it or a reset drops it
    // with the checkpoint), so a pin always covers a prefix of the live sequence.
    int pin_max = 0;
    unsigned pin_clock = 0, pin_stamp = 0; // the requests' stamps; the last ck_pin's
    std::vector<int> pin_req;              // pins for the next prefill to save (ck_pin; that prefill clears them)
    int pin_count() const {
        int n = 0;
        for (const auto & k : ck) n += k.pos >= 0 && k.pin;
        return n;
    }
    // the slot of the checkpoint a pin request for P covers (-1: none)
    int pin_slot(int P) const {
        int best = -1;
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos > P - 32 && ck[i].pos <= P && (best < 0 || ck[i].pos > ck[best].pos)) best = i;
        return best;
    }
    // pins slot si for the request stamped t; beyond pin_max the least recently requested other pin is released
    void pin_set(int si, unsigned t) {
        if (pin_max <= 0) return;
        ck[si].pin = true;
        ck[si].pin_t = t;
        while (pin_count() > pin_max) {
            int old = -1;
            for (int i = 0; i < (int) ck.size(); i++)
                if (i != si && ck[i].pos >= 0 && ck[i].pin &&
                    (old < 0 || ck[i].pin_t < ck[old].pin_t || (ck[i].pin_t == ck[old].pin_t && ck[i].pos < ck[old].pos)))
                    old = i;
            if (old < 0) break;
            ck[old].pin = false;
        }
    }
    // Slots for checkpoints at the new positions `want` (ascending): free ones, else evicting by coverage. While
    // there are more checkpoints (existing + new) than slots, the one dropped is the one (not `keep`; the slot of a
    // restore whose copies may still run only if nothing else is left) whose removal leaves the smallest gap relative
    // to its distance from the end E: (next - prev) / (E - prev). Old checkpoints thin out geometrically, recent ones
    // stay dense. Returns the slot of each new position (-1: dropped); evicted slots become free.
    // A side request (sw_on) takes free slots that hold none of the kept sequence's checkpoints first (one holding a
    // kept pinned checkpoint last); a slot that does gets its block saved to the host store before it is handed out
    // (sw_ck_take; no room: no checkpoint there). Pinned checkpoints and the new positions wpin marks (pin requests) are
    // never dropped.
    std::vector<int> ck_assign(const std::vector<int> & want, int keep, int E, const std::vector<char> & wpin = {}) {
        struct C { int pos, slot, wi; }; // slot >= 0: existing checkpoint; wi >= 0: new (index into want)
        std::vector<C> c;
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos >= 0) c.push_back({ck[i].pos, i, -1});
        for (int i = 0; i < (int) want.size(); i++) c.push_back({want[i], -1, i});
        std::stable_sort(c.begin(), c.end(), [](const C & a, const C & b) { return a.pos < b.pos; });
        while (c.size() > ck.size()) {
            int best = -1;
            double bs = 1e300;
            for (int pass = 0; pass < 2 && best < 0; pass++)
                for (int i = 0; i < (int) c.size(); i++) {
                    if (c[i].wi >= 0 ? c[i].pos == keep || (c[i].wi < (int) wpin.size() && wpin[c[i].wi]) : ck[c[i].slot].pin) continue;
                    if (!pass && c[i].slot >= 0 && (c[i].slot == ck_rs_src || c[i].slot == ck_rs_slot)) continue;
                    const double prev = i ? c[i - 1].pos : 0, next = i + 1 < (int) c.size() ? c[i + 1].pos : E;
                    const double sc = (next - prev) / std::max(1.0, E - prev);
                    if (sc < bs) bs = sc, best = i;
                }
            if (best < 0) break;
            c.erase(c.begin() + best);
        }
        std::vector<char> used(ck.size(), 0);
        for (const auto & x : c)
            if (x.slot >= 0) used[x.slot] = 1;
        for (int i = 0; i < (int) ck.size(); i++)
            if (!used[i]) ck[i].pos = -1;
        std::vector<int> ord(ck.size()); // free slots in this order
        std::iota(ord.begin(), ord.end(), 0);
        if (sw_on) {
            auto rank = [&](int i) { return !sw_ck_held(i) ? 0 : sw_ck[i].pin ? 2 : 1; };
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return rank(a) < rank(b); });
        }
        std::vector<int> out(want.size(), -1);
        int f = 0;
        for (const auto & x : c)
            if (x.wi >= 0) {
                while (f < (int) ck.size() && used[ord[f]]) f++;
                if (f == (int) ck.size()) break;
                used[ord[f]] = 1;
                out[x.wi] = ord[f];
            }
        if (sw_on)
            for (int & s : out)
                if (s >= 0 && !sw_ck_take(s)) s = -1;
        return out;
    }
    // staging slot for one DeltaNet layer's piece of an in-prefill checkpoint (stream 0 waits for its previous copy)
    float * stg_state(int s) { return hp.xm + (size_t) s * STG_F; }
    float * stg_conv(int s) { return stg_state(s) + CK_SL; }
    int stg_take() {
        const int s = stg_next;
        stg_next = (stg_next + 1) % stg_n;
        if (stg_busy[s]) {
            CK(cudaStreamWaitEvent(0, stg_ev[s], 0));
            stg_busy[s] = false;
        }
        return s;
    }
    // copies DeltaNet layer dl's staged pieces (staging slots sg[j] -> checkpoint slots ss[j]) on ck_sv
    void stg_flush(int dl, int sn, const int * sg, const int * ss) {
        CK(cudaEventRecord(ck_ev_sv, 0));
        CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
        const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        for (int j = 0; j < sn; j++) {
            CkSlot & s = ck[ss[j]];
            if (ss[j] == ck_rs_src) CK(cudaStreamWaitEvent(ck_sv, ck_ev_grp[CK_NG - 1], 0)); // (still being restored from)
            CK(cudaMemcpyAsync(s.dstate + (size_t) dl * CK_SL, stg_state(sg[j]), CK_SL * 4, k, ck_sv));
            CK(cudaMemcpyAsync(s.conv + (size_t) dl * CK_CL, stg_conv(sg[j]), CK_CL * 4, k, ck_sv));
            CK(cudaEventRecord(stg_ev[sg[j]], ck_sv));
            stg_busy[sg[j]] = true;
            stg_last = sg[j];
        }
    }
    // stream 0, before hp.xm is used again (a chunk's MTP part): every staged copy has finished (ck_sv is in order)
    void stg_drain() {
        if (stg_last < 0) return;
        CK(cudaStreamWaitEvent(0, stg_ev[stg_last], 0));
        for (bool & b : stg_busy) b = false;
        stg_last = -1;
    }
    // draft vocabulary: ids of head_d rows (base list + tokens added from the sequence)
    std::vector<int> dv_ids;
    std::vector<uint8_t> dv_in;
    int dv_base = 0;
    // dv_ids is what a fresh prefill of seq[0, dv_n) builds (the base list, then ONE dv_add of its text tokens), or -1:
    // not (step's additions). A rewind / truncate rebuilds it that way, and a prefill continuing at dv_n rebuilds it over
    // the whole new sequence, so a reused prefix drafts (and samples) exactly as the same prompt sent fresh.
    int dv_n = 0;
    int * dv_pin = nullptr; // dv_add's pinned staging: the ids (dv_base + DV_EXTRA), then [dv_base + DV_EXTRA] the row count

    // Adds the tokens not yet in the draft head's vocabulary (appends their ids).
    void dv_add(const int * toks, size_t n) {
        if (dv_frozen) return; // (parallel slots: the base list for every sequence)
        if (dv_in.empty()) return; // full-vocabulary draft head
        const int n0 = (int) dv_ids.size();
        for (size_t i = 0; i < n && (int) dv_ids.size() < dv_base + DV_EXTRA - 1; i++)
            if (!dv_in[toks[i]]) {
                dv_in[toks[i]] = 1;
                dv_ids.push_back(toks[i]);
            }
        if ((int) dv_ids.size() == n0) return;
        if (dv_ids.size() & 1) dv_ids.push_back(dv_ids.back()); // the draft lm_head takes rows in pairs
        const int k = (int) dv_ids.size() - n0;
        Mat & m = hw.head_d;
        if (dv_pin) {
            // asynchronous on the launch stream (ordered before the next launch). The staged ids [n0, n0 + k) are not
            // rewritten before a later synchronisation: appends go past them, and dv_truncate's synchronous row-count copy
            // waits for every copy queued before it (the next dv_add after it may restage the same ids)
            m.N = (int) dv_ids.size();
            memcpy(dv_pin + n0, dv_ids.data() + n0, k * 4);
            dv_pin[dv_base + DV_EXTRA] = m.N;
            CK(cudaMemcpyAsync(const_cast<int *>(hw.draft_ids) + n0, dv_pin + n0, k * 4, cudaMemcpyHostToDevice, 0));
            CK(mkl::set_cW_async(dv_pin + dv_base + DV_EXTRA, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N),
                                 cudaMemcpyHostToDevice, 0));
            return;
        }
        CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + n0, dv_ids.data() + n0, k * 4, cudaMemcpyHostToDevice));
        m.N = (int) dv_ids.size();
        CK(mkl::set_cW(&m.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    void dv_reset() {
        dv_truncate(dv_base);
        dv_n = 0;
    }
    // dv as a fresh prefill of seq[0, n) with image spans sp builds it (positions inside an image hold placeholder ids:
    // skipped, as prefill skips them)
    void dv_rebuild(int n, const std::vector<Span> & sp) {
        if (dv_frozen) return; // (parallel slots: the base list for every sequence)
        dv_reset();
        dv_n = n;
        if (dv_in.empty()) return;
        std::vector<int> text;
        text.reserve((size_t) n);
        size_t k = 0;
        for (int q = 0; q < n; q++) {
            while (k < sp.size() && sp[k].pos + sp[k].n() <= q) k++;
            if (k == sp.size() || q < sp[k].pos) text.push_back(seq[q]);
        }
        dv_add(text.data(), text.size());
    }
    // back to the first n ids (n >= dv_base, even: the size after a dv_add; a cancelled prefill's additions)
    void dv_truncate(size_t n) {
        if (dv_frozen) return; // (parallel slots: the base list for every sequence)
        if (dv_in.empty() || dv_ids.size() <= n) return;
        for (size_t i = n; i < dv_ids.size(); i++) dv_in[dv_ids[i]] = 0;
        for (size_t i = 0; i < n; i++) dv_in[dv_ids[i]] = 1; // a padding duplicate may have been cleared
        dv_ids.resize(n);
        hw.head_d.N = (int) n;
        CK(mkl::set_cW(&hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    Ctl hctl{};

    // Watchdog. Every wait for the device in the run-time paths goes through wait_device(), which records an
    // event and polls it with a deadline (MK_WATCHDOG: seconds per decode launch, default 60; prefill chunks and
    // checkpoint copies get at least 120 s; 0 = the runtime's plain blocking waits). The CUDA runtime's own waits
    // never time out: after a GPU / driver fault the WSL2 guest is not told about (nvlddmkm event 153 on the Windows
    // side, no error in the guest) the host thread would spin indefinitely with an idle GPU.
    // The watchdog turns that into a message and exit code 3. The first wd_spin_s of a wait spin (a decode launch
    // finishes in 20-35 ms, so no latency is added); after that it sleeps between polls.
    cudaEvent_t wd_ev = nullptr;
    double wd_limit = 60.0, wd_spin_s = 0.25;
    int wd_T = 0, wd_mode = -1;
    void wait_device(const char * what, double limit) {
        if (limit <= 0.0 || !wd_ev) {
            CK(cudaStreamSynchronize(0));
            return;
        }
        CK(cudaEventRecord(wd_ev, 0));
        wait_event(wd_ev, what, limit);
    }
    // the same deadline for an event recorded elsewhere (the checkpoint copy stream)
    void wait_event(cudaEvent_t ev, const char * what, double limit) {
        if (limit <= 0.0) {
            CK(cudaEventSynchronize(ev));
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const cudaError_t e = cudaEventQuery(ev);
            if (e == cudaSuccess) return;
            if (e != cudaErrorNotReady) mk_cuda_fail(e, (std::string("(the GPU work of ") + what + ")").c_str(), __FILE__, __LINE__);
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el > limit) {
                const cudaError_t le = cudaPeekAtLastError();
                fprintf(stderr,
                        "[mk] watchdog: the GPU did not finish %s within %g s (mode %d, T %d, position %d, committed %d, "
                        "launches %ld, last CUDA error: %s). The device or its driver is stuck: exiting with code 3.\n",
                        what, limit, wd_mode, wd_T, hctl.n, hctl.dn_c, launches_seen, cudaGetErrorString(le));
                fflush(stderr);
                _exit(3);
            }
#ifdef _WIN32
            // (sleep_for(100 us) is Sleep(1) there: until the next timer tick, 15.6 ms unless the timer resolution was raised)
            if (el > wd_spin_s) mk_win_sleep_us(el > 2.0 ? 2000 : 100);
#else
            if (el > wd_spin_s) std::this_thread::sleep_for(std::chrono::microseconds(el > 2.0 ? 2000 : 100));
#endif
        }
    }

    // MK_MEMLOG=1: device memory accounting. Each mark prints the bytes dalloc'd since the previous mark
    // and the drop of cudaMemGetInfo's free memory over the same span (the difference is allocation
    // granularity, module code, local memory or another process).
    const bool memlog = getenv("MK_MEMLOG") != nullptr;
    size_t alloc_bytes = 0, mark_alloc = 0, mark_free = 0;
    long launches_seen = 0, pf_runs_seen = 0;
    void mem_mark(const char * tag) {
        if (!memlog) return;
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (mark_free)
            fprintf(stderr, "[mk-mem] %-26s alloc %9.2f MiB   free drop %9.2f MiB   (free %9.1f MiB)\n", tag,
                    (alloc_bytes - mark_alloc) / 1048576.0, ((double) mark_free - (double) fr) / 1048576.0, fr / 1048576.0);
        else
            fprintf(stderr, "[mk-mem] %-26s free %.1f MiB of %.1f MiB\n", tag, fr / 1048576.0, tot / 1048576.0);
        mark_free = fr;
        mark_alloc = alloc_bytes;
    }

    size_t wbytes = 0;
    bool profiling = false;
    std::vector<double> ph_ns = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> ph_n = std::vector<long>(PH_COUNT, 0);
    double launch_ns = 0;
    long launches = 0;
    // min over launches of each barrier-to-barrier interval, keyed by (T, barrier index)
    std::vector<std::vector<double>> best_ns = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<std::vector<int>> best_tag = std::vector<std::vector<int>>(MAXT + 1);
    std::vector<std::vector<double>> best_fill = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<double> arr_first = std::vector<double>(PH_COUNT, 0.0), arr_med = std::vector<double>(PH_COUNT, 0.0),
                        arr_last = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> arr_n = std::vector<long>(PH_COUNT, 0);

    // Device buffers come from the default memory pool (stream-ordered allocator, never trimmed): it packs the
    // allocations, where cudaMalloc rounds each one up to whole 2 MiB pages (~0.5 GB over the ~900 weight tensors).
    // MK_CUDAMALLOC=1: plain cudaMalloc.
    bool pool_init = false, use_pool = true;
    int cuda_dev = 0;    // the device current at construction (a caller's cudaSetDevice)
    double dalloc_s = 0; // time in dalloc (load report)
    // MK_POISON=1 (test runs, read-before-write check): every device allocation is filled with 0xFF bytes (NaN in fp32 /
    // fp16, -1 in integers) instead of zeros, the weight buffers too (their padding), except DA_INIT0 buffers, whose zeros
    // are their initial value (arrival counters). An output that differs from a normal run's read memory nothing wrote.
    // MK_POISON_SKIP=i,j,..: allocations number i, j zero-filled instead (bisecting); MK_POISON_LOG=1 lists every one.
    static constexpr int DA_INIT0 = 2; // dalloc fill: zeros that are the buffer's initial value (kept under MK_POISON)
    bool poison = false;               // (read on the first dalloc)
    std::vector<long> poison_skip;
    long n_dalloc = 0;
    template <class T> T * dalloc(size_t n, int zero = 1) {
        void * p;
        if (!pool_init) {
            pool_init = true;
            poison = getenv("MK_POISON") && atoi(getenv("MK_POISON")) != 0;
            if (poison) {
                for (const char * s = getenv("MK_POISON_SKIP"); s && *s; s = strchr(s, ',') ? strchr(s, ',') + 1 : "")
                    poison_skip.push_back(atol(s));
                fprintf(stderr, "[mk] MK_POISON=1: device allocations filled with 0xFF bytes (test mode; %zu skipped)\n", poison_skip.size());
            }
            const char * cm = getenv("MK_CUDAMALLOC");
            int ok = 0;
            cudaDeviceGetAttribute(&ok, cudaDevAttrMemoryPoolsSupported, cuda_dev);
            use_pool = ok && !(cm && cm[0] == '1');
            if (use_pool) {
                cudaMemPool_t pool;
                CK(cudaDeviceGetDefaultMemPool(&pool, cuda_dev));
                unsigned long long thr = ~0ull; // keep freed memory in the pool (release() hands it back to the driver)
                CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
            }
        }
        const auto ta = lclk::now();
        if (use_pool) {
            CK(cudaMallocAsync(&p, n * sizeof(T) + 16, 0));
            CK(cudaStreamSynchronize(0));
        } else {
            CK(cudaMalloc(&p, n * sizeof(T) + 16));
        }
        allocs.push_back(p); // (before the fill: a buffer whose fill fails is freed with the others)
        alloc_bytes += n * sizeof(T) + 16;
        if (poison) {
            const bool skip = zero == DA_INIT0 || std::find(poison_skip.begin(), poison_skip.end(), n_dalloc) != poison_skip.end();
            CK(cudaMemset(p, skip ? 0 : 0xff, n * sizeof(T) + 16));
            CK(cudaStreamSynchronize(0)); // (before an upload on another stream writes the buffer)
            if (getenv("MK_POISON_LOG")) fprintf(stderr, "[mk-poison] #%ld %zu bytes%s\n", n_dalloc, n * sizeof(T), skip ? " zeroed" : "");
        } else if (zero) {
            CK(cudaMemset(p, 0, n * sizeof(T) + 16));
        }
        n_dalloc++;
        dalloc_s += lsec(ta);
        if (wrec_on) wrec.push_back({p, n * sizeof(T)});
        return (T *) p;
    }

    // A tensor of the file in a type the kernels read. Q3_K and IQ3_S are re-encoded as Q6_K and
    // IQ4_NL as Q8_0, exactly: every weight is a small integer times an fp16 scale in both (checked
    // bit for bit against ggml's dequantization of the source row). `to == T_F32` dequantizes
    // (ssm_alpha / ssm_beta, which are folded into fp32 rows).
    struct Src {
        uint32_t type = 0;
        std::vector<uint64_t> ne;
        const uint8_t * data = nullptr;
        uint64_t nbytes = 0;
        std::vector<uint8_t> own; // the re-encoded copy, when there is one
    };
    // a tensor's shape is [k, n] (any further dimensions 1): its bytes are exactly n rows of k (gguf::shape_is)
    static bool shape2(const std::vector<uint64_t> & ne, uint64_t k, uint64_t n) { return gguf::shape_is(ne, {k, n}); }
    int n_conv = 0;
    double conv_s = 0;
    // The MTP draft block's (blk.64 + nextn.eh_proj) matrices stored wider than Q4_K (Q5_K, Q6_K, Q8_0, Q4_0, IQ4_NL;
    // UD-Q4_K_XL: all of them, Q4_K_M: eh, v, ffn_down) are re-quantised to Q4_K at load (ggml_quantize_chunk, no
    // imatrix): the dp4a kernels read none of those types. Lossy, but only the drafts can change: they are verified by
    // the exact main model, so the output (greedy ids, the sampled distribution) is unchanged and only acceptance moves.
    // Each MTP pass streams ~0.1 GB less.
    bool req_q4 = false; // want(): re-quantise the current tensor to Q4_K (set while the MTP block loads)
    int n_req = 0;
    double req_s = 0;

    // MK_DRAFT_Q4 (default 1): the draft head's base rows (the draft vocabulary, --draft-vocab) re-quantised from lm_head's
    // Q6_K to a Q4_K copy at load (ggml_quantize_chunk, no imatrix; +94 MB of VRAM): each draft streams 0.69x the head's
    // bytes. Lossy, but only the drafts can change (they are verified by the exact main head), so the output (greedy ids,
    // the sampled distribution) is unchanged and only acceptance moves. MK_DRAFT_Q4=0: off. Policy key dq4=0|1 switches it.
    bool draft_q4 = true;
    bool draft_q4_env = false; // MK_DRAFT_Q4 given explicitly (only then is a missing draft vocabulary worth a note)
    int dq4_rows = 0; // rows of the copy (0: none)
    void set_draft_q4(int on) {
        if (!dq4_rows) throw std::runtime_error("policy dq4 needs the draft head's Q4_K copy (MK_DRAFT_Q4, on by default) and a draft vocabulary");
        hw.dq4_n = on ? dq4_rows : 0;
        CK(mkl::set_cW(&hw.dq4_n, sizeof(int), offsetof(ModelW, dq4_n)));
    }

    // MK_REENC3=1: Q3_K and IQ3_S re-encoded to Q6_K at load (instead of their native layouts; A/B tests)
    const bool reenc3 = getenv("MK_REENC3") && atoi(getenv("MK_REENC3")) != 0;
    int n_nat = 0; // Q3_K / IQ3_S tensors repacked to their native layouts
    double nat_s = 0;
    bool iq3s_grid_up = false;

    // The MTP block's source: the model file; a second GGUF (MKEngine's mtp argument: --mtp FILE, MK_MTP=FILE),
    // whose blk.<NL>.* tensors replace the model file's (its token_embd / output copies are not used: the MTP block shares
    // the model's embedding and lm_head); or none (a file without the MTP layer, or --no-mtp / MK_NO_MTP=1): plain
    // decoding at depth 0, no MTP weights, no MTP KV slot, no draft head.
    std::unique_ptr<gguf::File> mtpf;
    bool has_mtp = true;
    std::string mtp_desc = "model"; // (state-file key) "model", "none" or the --mtp file's identity
    int kv_slots = N_ATT_SLOTS;     // KV slots allocated (N_ATT_SLOTS - 1 without the MTP block: its slot is the last)
    static bool is_mtp_name(const std::string & n) {
        static const std::string p = "blk." + std::to_string(NL) + ".";
        return n.compare(0, p.size(), p) == 0;
    }
    const gguf::File & src_of(const std::string & name) const { return mtpf && is_mtp_name(name) ? *mtpf : file; }

    // The GGUF tensor `name`, its bytes read in (Prefetch, MK_LOAD_IO)
    std::unique_ptr<Prefetch> pfx, pfx_mtp;
    const gguf::Tensor & tget(const std::string & name) {
        const gguf::File & f = src_of(name);
        const auto & t = f.get(name);
        if (Prefetch * px = &f == &file ? pfx.get() : pfx_mtp.get()) px->need(t);
        return t;
    }
    // MK_MTP_Q4 re-quantisation of one tensor's rows to Q4_K (rs_to bytes per row into out)
    static void requant_q4k(const gguf::Tensor & t, uint8_t * out) {
        const int64_t K = (int64_t) t.ne[0];
        int64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
        const size_t rs_from = ggml_row_size((ggml_type) t.type, K), rs_to = ggml_row_size(GGML_TYPE_Q4_K, K);
        const auto * tr = ggml_get_type_traits((ggml_type) t.type);
        parallel_rows(rows, [&](int64_t r) {
            thread_local std::vector<float> f;
            f.resize((size_t) K);
            tr->to_float(t.data + (size_t) r * rs_from, f.data(), K);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), out + (size_t) r * rs_to, 0, 1, K, nullptr); // lossy by design (the MTP block only)
        });
    }
    static bool requant_type(uint32_t t) {
        return t == gguf::Q5_K || t == gguf::Q6_K || t == gguf::Q8_0 || t == gguf::Q4_0 || t == gguf::IQ4_NL;
    }
    // The MTP block's re-quantisation (~2.5 s of CPU on UD-Q4_K_XL) runs on a background thread from early in the load
    // (rq_start; MK_LOAD_RQBG=0: inline); want() takes its results. Same rows, same calls, same bytes.
    std::map<std::string, std::vector<uint8_t>> rq_res;
    double rq_s = 0; // the background thread's own time
    struct Joiner {
        std::thread t;
        ~Joiner() {
            if (t.joinable()) t.join();
        }
    } rq_th;
    bool rq_bg = false;
    void rq_start() {
        rq_bg = has_mtp && !(getenv("MK_LOAD_RQBG") && atoi(getenv("MK_LOAD_RQBG")) == 0);
        if (!rq_bg) return;
        std::vector<const gguf::Tensor *> ts;
        const gguf::File & mf = mtpf ? *mtpf : file;
        for (const char * r : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                               "ffn_down.weight", "nextn.eh_proj.weight"}) {
            const auto it = mf.tensors.find("blk." + std::to_string(NL) + "." + r);
            if (it != mf.tensors.end() && requant_type(it->second.type)) ts.push_back(&it->second);
        }
        rq_th.t = std::thread([this, ts] {
            const auto t0 = lclk::now();
            try {
                for (const gguf::Tensor * t : ts) {
                    int64_t rows = 1;
                    for (size_t d = 1; d < t->ne.size(); d++) rows *= (int64_t) t->ne[d];
                    std::vector<uint8_t> v(ggml_row_size(GGML_TYPE_Q4_K, (int64_t) t->ne[0]) * (size_t) rows);
                    requant_q4k(*t, v.data());
                    rq_res[t->name] = std::move(v);
                }
            } catch (...) { // (want() requantizes a tensor with no result here itself)
            }
            rq_s = lsec(t0);
        });
    }

    Src want(const std::string & name, int to = -1) {
        const auto & t = tget(name);
        Src s;
        s.type = t.type;
        s.ne = t.ne;
        s.data = t.data;
        s.nbytes = t.nbytes;
        const bool to_f32 = to == T_F32 && t.type != gguf::F32;
        const bool to_q4 = !to_f32 && req_q4 && requant_type(t.type);
        const bool to_q6 = !to_f32 && !to_q4 && reenc3 && (t.type == gguf::Q3_K || t.type == gguf::IQ3_S);
        // native IQ4_NL unless MK_IQ4NL_Q8; Q4_0 (outside the MTP block) always (the kernels do not read it)
        const bool to_q8 = !to_f32 && !to_q4 && ((t.type == gguf::IQ4_NL && getenv("MK_IQ4NL_Q8")) || t.type == gguf::Q4_0);
        if (!to_f32 && !to_q4 && !to_q6 && !to_q8) return s; // anything the kernels cannot read is refused by up_mat / need()
        const ggml_type from = (ggml_type) t.type;
        const ggml_type dst = to_f32 ? GGML_TYPE_F32 : to_q4 ? GGML_TYPE_Q4_K : to_q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
        const int64_t K = (int64_t) t.ne[0];
        int64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
        const size_t rs_from = ggml_row_size(from, K), rs_to = ggml_row_size(dst, K);
        const auto t0 = std::chrono::steady_clock::now();
        if (to_q4 && rq_bg) { // the background thread's result (rq_start)
            if (rq_th.t.joinable()) {
                auto _ = lc.scope("MTP Q4_K requant (wait for the background thread)");
                rq_th.t.join();
            }
            const auto it = rq_res.find(name);
            if (it != rq_res.end()) s.own = std::move(it->second);
            if (it != rq_res.end()) rq_res.erase(it);
        }
        if (to_q4 && s.own.empty()) {
            s.own.resize(rs_to * (size_t) rows);
            requant_q4k(t, s.own.data());
        } else if (!to_q4) {
        s.own.resize(rs_to * (size_t) rows);
        const auto * tr = ggml_get_type_traits(from);
        const auto * tr_dst = ggml_get_type_traits(dst);
        std::atomic<bool> exact{true};
        parallel_rows(rows, [&](int64_t r) {
            thread_local std::vector<float> f, g;
            f.resize((size_t) K);
            const uint8_t * src = t.data + (size_t) r * rs_from;
            uint8_t * out = s.own.data() + (size_t) r * rs_to;
            tr->to_float(src, f.data(), K);
            if (to_f32) {
                memcpy(out, f.data(), (size_t) K * 4);
                return;
            }
            if (to_q6) reencode_q6k(t.type, src, f.data(), out, K);
            else if (t.type == gguf::Q4_0) reencode_q40_q80(src, out, K);
            else reencode_iq4nl_q80(src, out, K);
            g.resize((size_t) K);
            tr_dst->to_float(out, g.data(), K);
            if (memcmp(f.data(), g.data(), (size_t) K * 4) != 0) exact = false;
        });
        if (!exact) throw std::runtime_error(name + ": re-encoding is not exact");
        }
        s.type = (uint32_t) dst;
        s.data = s.own.data();
        s.nbytes = s.own.size();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (to_q4) {
            n_req++;
            req_s += dt;
        } else {
            n_conv++;
            conv_s += dt;
        }
        return s;
    }

    // ---- weight upload ----
    // Every weight goes up through `upl` (loader.cuh Uploader: a worker thread, MK_LOAD_ASYNC=0 synchronous). Tensors the
    // kernels read as stored are copied straight from the GGUF mapping; repacked ones are built in one of two reused host
    // buffer sets (hset), so the next tensor is prepared while the previous one is copied. Reusing a set changes
    // no uploaded byte (the repacks write every byte they upload).
    struct HSet {
        HBuf q, m, o, om;
        uint64_t seq = 0; // last upload reading this set
    };
    HSet hs[2];
    int hs_i = 0;
    Uploader upl;
    HSet & hset() {
        HSet & S = hs[hs_i];
        hs_i ^= 1;
        auto _ = lc.scope("upload stall (host buffer set busy)");
        upl.wait(S.seq);
        return S;
    }
    void put(void * d, const void * s, size_t n, HSet * S = nullptr) {
        const uint64_t q = upl.put(d, s, n);
        if (S && q) S->seq = q;
        wbytes += n;
    }
    // device buffer of n bytes holding s (the weight allocations are recorded for MK_WDIGEST)
    template <class T> T * dput(const void * s, size_t n_elem, HSet * S = nullptr) {
        T * d = dalloc<T>(n_elem, false);
        put(d, s, n_elem * sizeof(T), S);
        return d;
    }

    const float * up_f32(const std::string & name, size_t expect) {
        const auto & t = tget(name);
        if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
        size_t n = t.nbytes / 4;
        if (expect && n != expect) throw std::runtime_error(name + " bad size");
        return dput<float>(t.data, n);
    }

    Mat up_mat(const std::string & name, int K, int N, int to = -1) {
        Src t = want(name, to < 0 ? (int) src_of(name).get(name).type : to);
        if (!shape2(t.ne, (uint64_t) K, (uint64_t) N))
            throw std::runtime_error(name + " has shape " + gguf::shape_str(t.ne) + ", expected " + gguf::shape_str({(uint64_t) K, (uint64_t) N}));
        Mat m{};
        m.K = K;
        m.N = N;
        m.type = (int) t.type;
        const int nb = K / 256;
        // a source owned by want() (re-encoded / re-quantised copy) must stay alive until its upload is done
        auto keep = [&](Src & s) {
            if (!s.own.empty()) own_keep.push_back(std::move(s.own));
        };
        if (t.type == gguf::Q4_K || t.type == gguf::Q5_K) {
            m.rb = (int) gguf::type_row_bytes(t.type, K);
            if (wv_lay) {
                HSet & S = hset();
                wv_place(m, t.data, t.nbytes, nullptr, 0, S);
            } else
            {
                m.w = dput<uint8_t>(t.data, t.nbytes);
                keep(t);
            }
        } else if (t.type == gguf::Q6_K) {
            // two streams (rows_q6k): 192 B of values per block, and per row a meta stream of 18 B per block in
            // pairs (sc even | sc odd | d even | d odd)
            if (nb & 1) throw std::runtime_error(name + ": Q6_K rows need an even number of 256-blocks");
            m.rb = nb * Q6_BLK;
            m.nd = nb * (Q6_META / 2); // halves
            HSet & S = hset();
            const size_t qn = (size_t) N * m.rb, mn = (size_t) N * nb * Q6_META;
            uint8_t * qd = S.q.get(qn);
            uint8_t * md = S.m.get(mn);
            const uint8_t * src = t.data;
            {
                auto _ = lc.scope("repack Q6_K two-stream");
                parallel_rows(N, [&](int64_t r) {
                    for (int b = 0; b < nb; b++) {
                        const uint8_t * blk = src + ((size_t) r * nb + b) * 210;
                        const uint8_t * ql = blk;
                        const uint8_t * qh = blk + 128;
                        const int8_t * sc = (const int8_t *) (blk + 192);
                        uint16_t dbits;
                        memcpy(&dbits, blk + 208, 2);
                        uint8_t vals[256];
                        for (int nn = 0; nn < 2; nn++)
                            for (int l = 0; l < 32; l++) {
                                const uint8_t * L = ql + 64 * nn;
                                const uint8_t * H = qh + 32 * nn;
                                vals[128 * nn + l + 0] = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4);
                                vals[128 * nn + l + 32] = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4);
                                vals[128 * nn + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                                vals[128 * nn + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                            }
                        uint8_t * dst = qd + (size_t) r * m.rb + b * Q6_BLK;
                        for (int p = 0; p < 4; p++)
                            for (int i = 0; i < 32; i++)
                                dst[32 * p + i] = (vals[64 * p + i] & 0xF) | ((vals[64 * p + 32 + i] & 0xF) << 4);
                        for (int p = 0; p < 4; p++)
                            for (int hh = 0; hh < 2; hh++) {
                                uint32_t hA = 0, hB = 0;
                                for (int i = 0; i < 16; i++) {
                                    const int sh = 8 * (i % 4) + 2 * (i / 4);
                                    hA |= (uint32_t) (vals[64 * p + 16 * hh + i] >> 4) << sh;
                                    hB |= (uint32_t) (vals[64 * p + 32 + 16 * hh + i] >> 4) << sh;
                                }
                                memcpy(dst + 128 + 8 * (2 * p + hh), &hA, 4);
                                memcpy(dst + 128 + 8 * (2 * p + hh) + 4, &hB, 4);
                            }
                        uint8_t * mp = md + ((size_t) r * nb + (b & ~1)) * Q6_META;
                        memcpy(mp + 16 * (b & 1), sc, 16);
                        memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
                    }
                });
            }
            // main layers: ffn_down's tile-major values and meta pairs, every Q6_K fragment-ordered with the meta inline
            // (one stream: no meta, m.d stays null)
            if (wv_lay) wv_place(m, qd, qn, md, mn, S);
            else
            {
                m.w = dput<uint8_t>(qd, qn, &S);
                m.d = reinterpret_cast<const __half *>(dput<uint8_t>(md, mn, &S));
            }
        } else if (t.type == gguf::IQ4_XS) {
            m.type = T_IQ4X;
            if (nb & 1) throw std::runtime_error(name + ": IQ4_XS rows need an even number of 256-blocks");
            m.rb = nb * 136;
            HSet & S = hset();
            const size_t qn = (size_t) N * m.rb;
            uint8_t * qd = S.q.get(qn);
            {
                auto _ = lc.scope("repack IQ4_XS");
                parallel_rows(N, [&](int64_t r) { repack_iq4xs(t.data + (size_t) r * nb * 136, qd + (size_t) r * m.rb, nb); });
            }
            if (wv_lay) wv_place(m, qd, qn, nullptr, 0, S);
            else
            m.w = dput<uint8_t>(qd, qn, &S);
        } else if (t.type == gguf::Q3_K || t.type == gguf::IQ3_S) {
            // native (rows_q3k / rows_iq3s, prefill TQ3K / TIQ3S): values 96 B per block + meta (Q3_K: the Q6_K meta
            // stream, 18 B; IQ3_S: 16 B), checked bit for bit against ggml's dequantization of every row
            const bool q3 = t.type == gguf::Q3_K;
            m.type = q3 ? T_Q3K : T_IQ3S;
            if (nb & 1) throw std::runtime_error(name + ": Q3_K / IQ3_S rows need an even number of 256-blocks");
            const int mb = q3 ? Q6_META : IQ3S_META;
            m.rb = nb * Q3_BLK;
            m.nd = nb * mb / 2; // halves
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<uint32_t> grid;
            if (!q3) {
                grid = iq3s_grid_host();
                if (!iq3s_grid_up) {
                    CK(mkl::set_g_iq3s_grid(grid.data(), grid.size() * 4));
                    iq3s_grid_up = true;
                }
            }
            HSet & S = hset();
            const size_t qn = (size_t) N * m.rb, mn = (size_t) N * nb * mb;
            uint8_t * qd = S.q.get(qn);
            uint8_t * md = S.m.get(mn);
            const auto * tr = ggml_get_type_traits((ggml_type) t.type);
            std::atomic<bool> exact{true};
            {
                auto _ = lc.scope("repack Q3_K / IQ3_S native (+ check)");
                parallel_rows(N, [&](int64_t r) {
                    thread_local std::vector<float> f;
                    f.resize((size_t) K);
                    const uint8_t * src = t.data + (size_t) r * nb * 110;
                    tr->to_float(src, f.data(), K);
                    uint8_t * qo = qd + (size_t) r * m.rb;
                    uint8_t * mo = md + (size_t) r * nb * mb;
                    if (!(q3 ? repack_q3k(src, qo, mo, nb, f.data()) : repack_iq3s(src, qo, mo, nb, f.data(), grid))) exact = false;
                });
            }
            if (!exact) throw std::runtime_error(name + ": native Q3_K / IQ3_S repack is not exact");
            if (wv_lay && wv_fd) wv_place(m, qd, qn, md, mn, S); // main-layer ffn_down: tile-major values and meta
            else
            {
                m.w = dput<uint8_t>(qd, qn, &S);
                m.d = reinterpret_cast<const __half *>(dput<uint8_t>(md, mn, &S));
            }
            n_nat++;
            nat_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else if (t.type == gguf::IQ4_NL) {
            m.type = T_IQ4NL;
            m.rb = nb * 144;
            HSet & S = hset();
            const size_t qn = (size_t) N * m.rb;
            uint8_t * qd = S.q.get(qn);
            {
                auto _ = lc.scope("repack IQ4_NL");
                parallel_rows(N, [&](int64_t r) { repack_iq4nl(t.data + (size_t) r * nb * 144, qd + (size_t) r * m.rb, nb); });
            }
            if (wv_lay) wv_place(m, qd, qn, nullptr, 0, S);
            else
            m.w = dput<uint8_t>(qd, qn, &S);
        } else if (t.type == gguf::Q8_0) {
            const int nb32 = K / 32;
            if (K % 256) throw std::runtime_error(name + ": Q8_0 rows need a multiple of 256 values");
            m.rb = K;
            m.nd = nb32;
            HSet & S = hset();
            const size_t qn = (size_t) N * K, mn = (size_t) N * nb32 * 2;
            uint8_t * qd = S.q.get(qn);
            uint8_t * md = S.m.get(mn); // fp16 scales, one per natural 32-block
            const uint8_t * src = t.data;
            {
                auto _ = lc.scope("repack Q8_0");
                parallel_rows(N, [&](int64_t r) {
                    const uint8_t * row = src + (size_t) r * nb32 * 34;
                    int8_t vals[256];
                    uint8_t * dst = qd + (size_t) r * K;
                    for (int j = 0; j < nb32; j++) memcpy(md + ((size_t) r * nb32 + j) * 2, row + 34 * j, 2);
                    for (int b = 0; b < K / 256; b++) { // (K % 256 == 0: the 32-unit permutation stays inside a 256-block)
                        for (int j = 0; j < 8; j++) memcpy(&vals[32 * j], row + 34 * (8 * b + j) + 2, 32);
                        for (int u = 8 * b; u < 8 * b + 8; u++) {
                            const int p = (u >> 1) & 3, hh = u & 1;
                            memcpy(dst + 32 * u, &vals[64 * p + 16 * hh], 16);
                            memcpy(dst + 32 * u + 16, &vals[64 * p + 32 + 16 * hh], 16);
                        }
                    }
                });
            }
            if (wv_lay) wv_place(m, qd, qn, md, mn, S);
            else
            {
                m.w = dput<uint8_t>(qd, qn, &S);
                m.d = (const __half *) dput<uint16_t>(md, mn / 2, &S);
            }
        } else {
            throw std::runtime_error(name + ": unsupported quant type " + std::to_string(t.type));
        }
        return m;
    }
    std::vector<std::vector<uint8_t>> own_keep; // want()'s copies, until the uploads are done (freed after the load)

    // the matrices of the main layers (set while they load) of the wv types are re-laid-out on the host before the upload
    // (wv_fd while a main layer's ffn_down loads: its Q6_K / Q3_K / IQ3_S too)
    bool wv_lay = false, wv_fd = false;
    int wv_n = 0;
    double wv_s = 0;
    size_t wv_relay(Mat & m, const uint8_t * w, const uint8_t * meta, size_t meta_n, uint8_t * out, uint8_t * out_meta) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t r = wv_relayout(m.type, m.N, m.K, m.rb, w, meta, meta_n, out, out_meta);
        m.fm = 1;
        wv_n++;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        wv_s += s;
        lc.add("wv re-layout", s);
        return r;
    }
    // MK_LOAD_GPURL (default 1): the wv layout made on the device (mkl::wv_dev_launch, on the upload worker: the
    // pre-wv bytes go to a staging buffer, a kernel writes them re-laid into the matrix's buffer); 0, or a layout / size the
    // device path does not take: wv_relayout on the host. Same bytes either way.
    const bool gpurl = !(getenv("MK_LOAD_GPURL") && atoi(getenv("MK_LOAD_GPURL")) == 0);
    std::map<int, WvProbe> wv_probes;
    uint8_t * stg_d = nullptr;   // device staging (cudaMalloc, freed after the weights)
    size_t stg_cap = 0;
    uint64_t relay_seq = 0;      // the last device re-layout queued
    std::vector<void *> wv_tabs; // the probes' gather tables (cudaMalloc)
    int wv_dev_n = 0;
    void wv_place(Mat & m, const uint8_t * w, size_t wn, const uint8_t * meta, size_t meta_n, HSet & S) {
        const bool q6f = m.type == T_Q6K;
        const size_t on = q6f ? wn + meta_n : wn, omn = q6f ? 0 : meta_n;
        const size_t moff = (wn + 255) & ~(size_t) 255;
        bool dev = gpurl && wv_dev_ok(m.type, m.N, m.K, m.rb, meta_n, wv_probes);
        if (dev && !stg_d) {
            stg_cap = (size_t) 192 << 20;
            if (cudaMalloc(&stg_d, stg_cap) != cudaSuccess) {
                cudaGetLastError();
                stg_d = nullptr;
                stg_cap = 0;
            }
        }
        dev = dev && moff + meta_n <= stg_cap;
        if (dev && wv_fm(m.type)) {
            WvProbe & p = wv_probes.at(m.type);
            if (!p.dtab) {
                void * d = nullptr;
                CK(cudaMalloc(&d, p.tab.size() * 4));
                CK(cudaMemcpy(d, p.tab.data(), p.tab.size() * 4, cudaMemcpyHostToDevice));
                wv_tabs.push_back(d);
                p.dtab = static_cast<const uint32_t *>(d);
            }
        }
        if (!dev) {
            uint8_t * o = S.o.get(on);
            uint8_t * om = meta_n ? S.om.get(meta_n) : nullptr;
            wv_relay(m, w, meta, meta_n, o, om);
            m.w = dput<uint8_t>(o, on, &S);
            if (omn) m.d = reinterpret_cast<const __half *>(dput<uint8_t>(om, omn, &S));
            return;
        }
        uint8_t * dw = dalloc<uint8_t>(on, false);
        uint8_t * dm = omn ? dalloc<uint8_t>(omn, false) : nullptr;
        wbytes += on + omn;
        const int type = m.type, N = m.N, K = m.K, rb = m.rb;
        uint8_t * sw = stg_d, * sm = stg_d + moff;
        const WvProbe * pp = wv_fm(m.type) ? &wv_probes.at(m.type) : nullptr; // (a map node: stable while others are added)
        const uint64_t q = upl.put2(sw, w, wn, sm, meta, meta_n, [=](cudaStream_t st) {
            return mkl::wv_dev_launch(type, N, K, rb, sw, sm, meta_n, dw, dm, st, pp ? pp->dtab : nullptr, pp ? (int) pp->tab.size() : 0,
                                      pp ? pp->G : 0, pp ? pp->seg : 0);
        });
        if (q) S.seq = relay_seq = q;
        m.w = dw;
        if (omn) m.d = reinterpret_cast<const __half *>(dm);
        m.fm = 1;
        wv_n++;
        wv_dev_n++;
    }
    // after the weights: the staging buffer and the gather tables go back (before the KV cache is allocated)
    void wv_dev_done() {
        if (!stg_d && wv_tabs.empty()) return;
        auto _ = lc.scope("wv device re-layout tail (wait)");
        upl.wait(relay_seq);
        if (stg_d) cudaFree(stg_d);
        for (void * t : wv_tabs) cudaFree(t);
        stg_d = nullptr;
        stg_cap = 0;
        wv_tabs.clear();
        for (auto & kv : wv_probes) kv.second.dtab = nullptr;
    }

    QBuf qbuf(int U) {
        QBuf q;
        q.U = U;
        q.xa = dalloc<int8_t>((size_t) RMAX * U * 16);
        q.xb = dalloc<int8_t>((size_t) RMAX * U * 16);
        q.meta = dalloc<uint2>((size_t) RMAX * U);
        q.ssq = dalloc<float>((size_t) RMAX * U);
        return q;
    }

    // MK_FAULT=cuda:<n> | sticky:<n> (tests of the error path only): the n-th prefill (1-based) fails half-way
    // through the batched layers, with a rejected CUDA call (non-sticky: an invalid memcpy direction; the state is
    // left half-updated) or a trapping kernel (sticky: the context is lost)
    int fault_kind = 0, fault_at = 0, fault_seen = 0;
    bool fault_now = false;
    void fault_inject() {
        fault_now = false;
        fprintf(stderr, "[mk] MK_FAULT: injecting a %s CUDA error\n", fault_kind == 1 ? "non-sticky" : "sticky");
        if (fault_kind == 1) CK(cudaMemcpy(hp.h, hp.h, 4, (cudaMemcpyKind) 99));
        else mkl::pk_fault();
    }

    int n_draft_max;
    int n_draft_cap; // the depth given at construction (set_n_draft: at most this)
    int kv_mode; // KV cache format: 0 fp16, 1 q8_0, -1 auto (q8_0 when fp16 does not fit)
    // A --mtp file must hold the MTP layer of this model: same architecture and shapes, its MTP block at layer index NL
    // (block_count NLA), the same vocabulary; every tensor of the block present (shapes: up_mat / up_f32 as they load)
    void check_mtp_file(const gguf::File & m, const std::string & mp) const {
        auto bad = [&](const std::string & what) {
            throw std::runtime_error("--mtp " + mp + ": " + what + " (an MTP layer file of this model is needed)");
        };
        auto gi = [](const gguf::File & f, const std::string & k) -> long long {
            const auto it = f.ints.find(k);
            return it == f.ints.end() ? -1 : (long long) it->second;
        };
        const auto arch = m.strs.find("general.architecture");
        if (arch == m.strs.end() || arch->second != "qwen35")
            bad("architecture " + (arch == m.strs.end() ? std::string("missing") : "'" + arch->second + "'") + ", expected qwen35");
        for (const char * k : {"embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv",
                               "attention.key_length", "attention.value_length"}) {
            const std::string key = std::string("qwen35.") + k;
            if (gi(m, key) != gi(file, key))
                bad(key + " " + std::to_string(gi(m, key)) + ", the model has " + std::to_string(gi(file, key)));
        }
        if (gi(m, "qwen35.block_count") != NLA) bad("block_count " + std::to_string(gi(m, "qwen35.block_count")) + ", expected " +
                                                   std::to_string(NLA) + " (the MTP block at layer " + std::to_string(NL) + ")");
        if (gi(m, "qwen35.nextn_predict_layers") == 0) bad("nextn_predict_layers 0");
        const auto vm = m.arrs.find("tokenizer.ggml.tokens"), vf = file.arrs.find("tokenizer.ggml.tokens");
        if (vm != m.arrs.end() && vf != file.arrs.end() && vm->second != vf->second)
            bad("vocabulary of " + std::to_string(vm->second) + " tokens, the model has " + std::to_string(vf->second));
        for (const char * k : {"tokenizer.ggml.model", "tokenizer.ggml.pre"}) {
            const auto a = m.strs.find(k), b = file.strs.find(k);
            if (a != m.strs.end() && b != file.strs.end() && a->second != b->second) bad(std::string(k) + " '" + a->second + "', the model has '" + b->second + "'");
        }
        for (const char * k : {"token_embd.weight", "output.weight"}) { // (copies the engine does not use: their shape only)
            const auto it = m.tensors.find(k);
            if (it != m.tensors.end() && !shape2(it->second.ne, D, VOCAB)) bad(std::string(k) + " is not [" + std::to_string(D) + ", " + std::to_string(VOCAB) + "]");
        }
        for (const char * r : {"attn_norm.weight", "post_attention_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                               "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                               "ffn_down.weight", "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                               "nextn.shared_head_norm.weight"}) {
            const std::string n = "blk." + std::to_string(NL) + "." + r;
            if (!m.tensors.count(n)) bad("missing tensor " + n);
        }
    }
    // the state-file identity of a file: real path, size, mtime, hash of its first and last 4 MiB
    static std::string file_ident(const std::string & p, const gguf::File & f);

    Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_, const std::string & mtp_)
        : file(path), ctx(ctx_), ndraft(ndraft_), nd_avail(ndraft_), n_draft_max(ndraft_), n_draft_cap(ndraft_), kv_mode(kv_mode_) {
        // a throw anywhere below: ~Impl does not run for a half-built engine, so this frees what it allocated (destroyed
        // after upl_halt: no queued upload still reads a buffer). A caller can then retry, e.g. with a smaller context.
        struct FreeOnThrow {
            Impl * p;
            int n0;
            ~FreeOnThrow() {
                if (std::uncaught_exceptions() <= n0) return;
#if MK_SPEC_LOG
                if (p->slog) fclose(p->slog);
#endif
                p->release();
            }
        } free_on_throw{this, std::uncaught_exceptions()};
        // a refused file (a throw anywhere below) unwinds this body before the members go: the upload worker stops first,
        // so no queued copy reads a buffer a destroyed member owned (own_keep, wv_probes: declared after upl)
        struct UplHalt {
            Uploader & u;
            ~UplHalt() { u.halt(); }
        } upl_halt{upl};
        model_path = path;
        if (file.strs["general.architecture"] != "qwen35") throw std::runtime_error("expected qwen35 architecture");
        // block_count NLA: the 64 layers + the MTP layer; NL: a conversion without the MTP layer
        if ((file.ints["qwen35.block_count"] != NLA && file.ints["qwen35.block_count"] != NL) || file.ints["qwen35.embedding_length"] != D)
            throw std::runtime_error("unexpected model dimensions (this kernel is specialised for the 27B)");
        if (ctx > ctx_limit())
            throw std::runtime_error("context of " + std::to_string(ctx) + " positions exceeds the model's context length (" +
                                     std::to_string(ctx_limit()) + ", qwen35.context_length)");
        {
            // MTP source: the mtp argument (CLI --mtp FILE / --no-mtp), else MK_NO_MTP=1 / MK_MTP=FILE, else the model file's
            std::string mtp = mtp_;
            if (mtp.empty()) {
                if (getenv("MK_NO_MTP") && atoi(getenv("MK_NO_MTP")) != 0) mtp = "none";
                else if (getenv("MK_MTP") && getenv("MK_MTP")[0]) mtp = getenv("MK_MTP");
            }
            const bool file_mtp = file.tensors.count("blk." + std::to_string(NL) + ".nextn.eh_proj.weight") != 0;
            if (mtp == "none") {
                has_mtp = false;
                fprintf(stderr, "[mk] MTP layer off (--no-mtp / MK_NO_MTP=1): plain decoding, depth 0\n");
            } else if (!mtp.empty()) {
                mtpf.reset(new gguf::File(mtp));
                check_mtp_file(*mtpf, mtp);
                mtp_desc = file_ident(mtp, *mtpf);
                fprintf(stderr, "[mk] MTP layer from %s%s\n", mtp.c_str(), file_mtp ? " (instead of the model file's own)" : "");
            } else if (!file_mtp) {
                has_mtp = false;
                fprintf(stderr, "[mk] the model file has no MTP layer (blk.%d): plain decoding, depth 0 (--mtp FILE loads one from another GGUF)\n", NL);
            }
            if (!has_mtp) {
                mtp_desc = "none";
                kv_slots = N_ATT_SLOTS - 1;
                if (ndraft > 0) fprintf(stderr, "[mk] n_draft %d -> 0 (no MTP layer)\n", ndraft);
                ndraft = nd_avail = n_draft_max = n_draft_cap = 0;
            }
        }
        // read-ahead of the tensor data from here on (loader.cuh Prefetch; MK_LOAD_IO threads, 0: off), 3 GiB ahead at most
        // (the model file without an MTP block it does not supply; a --mtp file: its MTP block only, a few threads)
        if (const int nio = getenv("MK_LOAD_IO") ? atoi(getenv("MK_LOAD_IO")) : 8; nio > 0) {
            if (has_mtp && !mtpf) pfx.reset(new Prefetch(file, nio, (size_t) 3 << 30));
            else pfx.reset(new Prefetch(file, nio, (size_t) 3 << 30, [](const std::string & n) { return !is_mtp_name(n); }));
            if (mtpf) pfx_mtp.reset(new Prefetch(*mtpf, std::max(1, nio / 4), (size_t) 3 << 30, is_mtp_name));
        }
        cudaDeviceProp prop;
        {
            auto _ = lc.scope("CUDA context");
            CK(cudaGetDevice(&cuda_dev));
            CK(cudaGetDeviceProperties(&prop, cuda_dev));
            CK(cudaFree(0));
        }
        { // the persistent kernel runs one block per SM (mk_min_grid: launches of up to MAXT rows when n_draft > 4)
            const int rows = ndraft + 1 > MAXT_PF ? MAXT : MAXT_PF, need = mk_min_grid(rows);
            if (prop.multiProcessorCount < need)
                throw std::runtime_error(std::string(prop.name) + " has " + std::to_string(prop.multiProcessorCount) +
                                         " SMs; the megakernel runs one block per SM and needs at least " + std::to_string(need) +
                                         " (launches of " + std::to_string(rows) + " rows at n_draft " + std::to_string(ndraft) +
                                         "; n_draft <= 4: " + std::to_string(mk_min_grid(MAXT_PF)) + ")");
        }
#ifdef __CUDA_ARCH_LIST__
        { // (without machine code for the GPU the driver JIT-compiles the megakernel from PTX, or finds no kernel image)
            bool ok = false;
            std::string have;
            for (int a : {__CUDA_ARCH_LIST__}) {
                ok |= a / 100 == prop.major && a / 10 % 10 <= prop.minor;
                have += (have.empty() ? "sm_" : ", sm_") + std::to_string(a / 10);
            }
            if (!ok)
                throw std::runtime_error(std::string(prop.name) + " is sm_" + std::to_string(10 * prop.major + prop.minor) +
                                         "; this build has GPU code for " + have +
                                         " only: rebuild for it (make ARCH=..., build.ps1 -Arch ...; docs/BUILDING.md)");
        }
#endif
        upl.start(!(getenv("MK_LOAD_ASYNC") && atoi(getenv("MK_LOAD_ASYNC")) == 0));
        // token_embd's mapped pinned table (0.7 GB: ~0.8 s under WSL2) allocated by a helper thread while the weights load
        // (MK_LOAD_PINBG=0: at its turn)
        const bool embd_dev = getenv("MK_EMBD_DEVICE") && getenv("MK_EMBD_DEVICE")[0] == '1';
        Joiner embd_th;
        cudaError_t embd_err = cudaSuccess;
        double embd_pin_s = 0;
        const size_t embd_nb = file.get("token_embd.weight", {(uint64_t) D, (uint64_t) VOCAB}).nbytes; // (rows are read by token id)
        auto embd_alloc = [&] {
            const auto t0 = lclk::now();
            embd_err = cudaHostAlloc(&embd_host, embd_nb, cudaHostAllocMapped | cudaHostAllocWriteCombined);
            embd_pin_s = lsec(t0);
        };
        if (!embd_dev && !(getenv("MK_LOAD_PINBG") && atoi(getenv("MK_LOAD_PINBG")) == 0)) embd_th.t = std::thread(embd_alloc);
        nsm = std::min(prop.multiProcessorCount, MK_MAX_GRID); // (blocks of the persistent kernel)
        if (nsm < prop.multiProcessorCount)
            fprintf(stderr, "[mk] %d SMs: the persistent kernel runs %d blocks (MK_MAX_GRID)\n", prop.multiProcessorCount, nsm);
        CK(cudaEventCreateWithFlags(&wd_ev, cudaEventDisableTiming));
        if (const char * w = getenv("MK_WATCHDOG")) wd_limit = atof(w);
        if (const char * w = getenv("MK_WATCHDOG_SPIN")) wd_spin_s = atof(w);
        upl.wd_s = wd_limit > 0.0 ? std::max(wd_limit, 120.0) : 0.0; // (the weight upload: as the prefill chunks)
        if (const char * f = getenv("MK_FAULT")) {
            const std::string s = f;
            const size_t c = s.find(':');
            fault_kind = s.compare(0, c, "cuda") == 0 ? 1 : s.compare(0, c, "sticky") == 0 ? 2 : 0;
            fault_at = c == std::string::npos ? 1 : atoi(s.c_str() + c + 1);
            if (!fault_kind || fault_at < 1) throw std::runtime_error("MK_FAULT: cuda:<n> or sticky:<n>");
        }
        if (memlog) {
            CK(cudaFree(0));
            mem_mark("context");
        }

        auto blk = [](int i, const char * s) { return "blk." + std::to_string(i) + "." + s; };
        // The kernels take the weight type as a template argument and dispatch on each matrix's
        // type at run time (gemv_pairs_any, resid_any, ffn_gu), for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS (plus what want()
        // re-encodes into those). Check every role so that an unsupported type is refused at load.
        auto tname = [](int t) -> std::string {
            switch (t) {
                case gguf::F32: return "F32";
                case gguf::F16: return "F16";
                case gguf::Q4_0: return "Q4_0";
                case gguf::Q8_0: return "Q8_0";
                case gguf::Q3_K: return "Q3_K";
                case gguf::Q4_K: return "Q4_K";
                case gguf::Q5_K: return "Q5_K";
                case gguf::Q6_K: return "Q6_K";
                case gguf::IQ4_NL: return "IQ4_NL";
                case gguf::IQ3_S: return "IQ3_S";
                case gguf::IQ4_XS: return "IQ4_XS";
                default: return "type " + std::to_string(t);
            }
        };
        auto need = [&](const std::string & name, int got, std::initializer_list<int> ok) {
            for (int t : ok)
                if (got == t) return;
            std::string want;
            for (int t : ok) want += (want.empty() ? "" : " or ") + tname(t);
            throw std::runtime_error(name + " is " + tname(got) + ", the kernel needs " + want +
                                     " (see README: the GGUF)");
        };
        auto needq = [&](const std::string & name, int got) {
            need(name, got, {T_Q4K, T_Q5K, T_Q6K, T_Q8_0, T_IQ4X, T_IQ4NL, T_Q3K, T_IQ3S});
        };
        if (const char * e = getenv("MK_MTP_Q4")) {
            if (strcmp(e, "1"))
                throw std::runtime_error(std::string("MK_MTP_Q4=") + e + " was removed: the MTP block is always re-quantised to Q4_K");
        }
        if (const char * e = getenv("MK_DRAFT_Q4")) {
            if (strcmp(e, "0") && strcmp(e, "1")) throw std::runtime_error("MK_DRAFT_Q4: 0 or 1");
            draft_q4 = e[0] == '1';
            draft_q4_env = true;
        }
        rq_start();
        wrec_on = true; // (MK_WDIGEST: the weight allocations from here to token_embd, without the draft ids)
        for (int i = 0; i < (has_mtp ? NLA : NL); i++) { // (no MTP block: hw.L[NL] stays empty)
            LayerW & L = hw.L[i];
            const bool mtp = i == NL;
            req_q4 = mtp;
            wv_lay = !mtp;
            L.attn = mtp || ((i + 1) % 4 == 0);
            L.attn_norm = up_f32(blk(i, "attn_norm.weight"), D);
            L.post_norm = up_f32(blk(i, "post_attention_norm.weight"), D);
            if (L.attn) {
                L.q = up_mat(blk(i, "attn_q.weight"), D, QG);
                L.k = up_mat(blk(i, "attn_k.weight"), D, KVD);
                L.v = up_mat(blk(i, "attn_v.weight"), D, KVD);
                L.o = up_mat(blk(i, "attn_output.weight"), VD, D);
                needq(blk(i, "attn_q.weight"), L.q.type);
                needq(blk(i, "attn_k.weight"), L.k.type);
                needq(blk(i, "attn_v.weight"), L.v.type);
                needq(blk(i, "attn_output.weight"), L.o.type);
                L.qn = up_f32(blk(i, "attn_q_norm.weight"), HD);
                L.kn = up_f32(blk(i, "attn_k_norm.weight"), HD);
            } else {
                L.qkv = up_mat(blk(i, "attn_qkv.weight"), D, QKV);
                L.gate = up_mat(blk(i, "attn_gate.weight"), D, VD);
                L.sout = up_mat(blk(i, "ssm_out.weight"), VD, D);
                needq(blk(i, "attn_qkv.weight"), L.qkv.type);
                needq(blk(i, "attn_gate.weight"), L.gate.type);
                needq(blk(i, "ssm_out.weight"), L.sout.type);
                L.conv = up_f32(blk(i, "ssm_conv1d.weight"), 4 * QKV);
                L.dt = up_f32(blk(i, "ssm_dt.bias"), NVH);
                L.A = up_f32(blk(i, "ssm_a"), NVH);
                L.snorm = up_f32(blk(i, "ssm_norm.weight"), HK);
                const auto & raw_a = tget(blk(i, "ssm_alpha.weight"));
                const auto & raw_b = tget(blk(i, "ssm_beta.weight"));
                if (raw_a.type == gguf::Q8_0 && raw_b.type == gguf::Q8_0 && shape2(raw_a.ne, D, NVH) && shape2(raw_b.ne, D, NVH) &&
                    !getenv("MK_AB_F32")) {
                    // Q8_0 alpha / beta (UD) kept as int8 rows + fp16 block scales; the kernels form
                    // (q * d) * attn_norm on the fly, the same fp32 values as the folded rows below (MK_AB_F32=1)
                    HSet & S = hset();
                    const size_t nq = (size_t) 2 * NVH * D, nd = (size_t) 2 * NVH * (D / 32);
                    int8_t * q = reinterpret_cast<int8_t *>(S.q.get(nq));
                    uint16_t * ds = reinterpret_cast<uint16_t *>(S.m.get(nd * 2));
                    for (int w = 0; w < 2; w++) {
                        const uint8_t * src = (w ? raw_b : raw_a).data;
                        for (size_t blk_ = 0; blk_ < (size_t) NVH * (D / 32); blk_++) {
                            memcpy(&ds[(size_t) w * NVH * (D / 32) + blk_], src + blk_ * 34, 2);
                            memcpy(&q[((size_t) w * NVH * (D / 32) + blk_) * 32], src + blk_ * 34 + 2, 32);
                        }
                    }
                    L.abq = dput<int8_t>(q, nq, &S);
                    L.abd = reinterpret_cast<const __half *>(dput<uint16_t>(ds, nd, &S));
                    L.ab = nullptr;
                } else {
                // alpha/beta projections in fp32 with the attn_norm weight folded in
                const Src ta = want(blk(i, "ssm_alpha.weight"), T_F32);
                const Src tb = want(blk(i, "ssm_beta.weight"), T_F32);
                const auto & tn = tget(blk(i, "attn_norm.weight"));
                // read below as raw floats, so anything but F32 would be reinterpreted silently
                need(blk(i, "ssm_alpha.weight"), (int) ta.type, {T_F32});
                need(blk(i, "ssm_beta.weight"), (int) tb.type, {T_F32});
                if (!shape2(ta.ne, D, NVH) || !shape2(tb.ne, D, NVH))
                    throw std::runtime_error(blk(i, "ssm_alpha/beta.weight") + " bad shape");
                HSet & S = hset();
                const size_t nab = (size_t) 2 * NVH * D;
                float * ab = reinterpret_cast<float *>(S.q.get(nab * 4));
                const float * fa = (const float *) ta.data;
                const float * fb = (const float *) tb.data;
                const float * fn = (const float *) tn.data;
                for (int h = 0; h < NVH; h++)
                    for (int e = 0; e < D; e++) {
                        ab[(size_t) h * D + e] = fa[(size_t) h * D + e] * fn[e];
                        ab[(size_t) (NVH + h) * D + e] = fb[(size_t) h * D + e] * fn[e];
                    }
                L.ab = dput<float>(ab, nab, &S);
                L.abq = nullptr;
                L.abd = nullptr;
                }
            }
            L.fg = up_mat(blk(i, "ffn_gate.weight"), D, FF);
            L.fu = up_mat(blk(i, "ffn_up.weight"), D, FF);
            wv_fd = true;
            L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D);
            wv_fd = false;
            if (!mtp && !L.fd.fm) throw std::runtime_error(blk(i, "ffn_down.weight") + ": the wide pass needs every ffn_down in the wv layout");
            needq(blk(i, "ffn_gate.weight"), L.fg.type);
            needq(blk(i, "ffn_up.weight"), L.fu.type);
            needq(blk(i, "ffn_down.weight"), L.fd.type);
        }
        mem_mark("weights: 64 layers + MTP");
        wv_lay = false;
        fprintf(stderr, "[mk] wv: %d main-pass matrices in the wv layout (%d on the device, the others re-laid-out on the host in %.1f s; "
                        "Q6_K everywhere (FM), ffn_down every type; elsewhere Q3_K / IQ3_S stay on dp4a), MAXT %d\n", wv_n, wv_dev_n, wv_s, MAXT);
        if (has_mtp) {
            req_q4 = true;
            hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D);
            req_q4 = false;
            needq(blk(NL, "nextn.eh_proj.weight"), hw.eh.type);
            hw.enorm = up_f32(blk(NL, "nextn.enorm.weight"), D);
            hw.hnorm = up_f32(blk(NL, "nextn.hnorm.weight"), D);
            hw.shnorm = up_f32(blk(NL, "nextn.shared_head_norm.weight"), D);
        }
        // Every matrix outside the wv layout must be of a type this build's dp4a dispatch keeps: main layers a hybrid type
        // (Q3_K / IQ3_S: wv.cuh gemv_pairs_dp), the MTP block (solo and batched passes: gemv_pairs_any,
        // resid_any, rows16_any) after its re-quantisation also Q4_K / IQ4_XS. The loader above guarantees it for every file
        // it accepts; anything else is refused here rather than mis-read.
        {
            auto dp4a_need = [&](const std::string & name, const Mat & m, bool mtp) {
                const int t = m.type;
                if (m.fm || t == T_Q3K || t == T_IQ3S || (mtp && (t == T_Q4K || t == T_IQ4X))) return;
                throw std::runtime_error(name + " is " + tname(t) + (mtp ? " in the MTP block" : " outside the wv layout") +
                                         ": this build's dp4a kernels do not read it");
            };
            for (int i = 0; i < (has_mtp ? NLA : NL); i++) {
                const LayerW & L = hw.L[i];
                auto chk = [&](const char * r, const Mat & m) { dp4a_need(blk(i, r), m, i == NL); };
                if (L.attn) {
                    chk("attn_q.weight", L.q);
                    chk("attn_k.weight", L.k);
                    chk("attn_v.weight", L.v);
                    chk("attn_output.weight", L.o);
                } else {
                    chk("attn_qkv.weight", L.qkv);
                    chk("attn_gate.weight", L.gate);
                    chk("ssm_out.weight", L.sout);
                }
                chk("ffn_gate.weight", L.fg);
                chk("ffn_up.weight", L.fu);
                chk("ffn_down.weight", L.fd);
            }
            if (has_mtp) dp4a_need(blk(NL, "nextn.eh_proj.weight"), hw.eh, true);
        }
        hw.onorm = up_f32("output_norm.weight", D);
        hw.head = up_mat("output.weight", D, VOCAB);
        // Q8_0: some fine-tunes (~0.3 GB more than Q6_K); lm_head reads either type
        if (hw.head.type != T_Q6K && hw.head.type != T_Q8_0)
            throw std::runtime_error("output.weight is " + tname(hw.head.type) + ", the kernel needs Q6_K or Q8_0 (see README: the GGUF)");
        mem_mark("weights: eh, norms, lm_head");
        hw.head_d = hw.head;
        hw.draft_ids = nullptr;
        wrec_on = false;
        if (!draft_vocab.empty() && has_mtp) { // (no MTP block: no draft head)
            FILE * f = ggml_fopen(draft_vocab.c_str(), "rb"); // (a UTF-8 path on Windows too)
            if (!f) throw std::runtime_error("cannot open draft vocab " + draft_vocab);
            std::vector<int> ids;
            int id;
            while (fread(&id, 4, 1, f) == 1) {
                if (id < 0 || id >= VOCAB) throw std::runtime_error("bad id in draft vocab");
                ids.push_back(id);
            }
            fclose(f);
            if (ids.size() & 1) ids.push_back(ids.back());
            // room for DV_EXTRA more rows: tokens of the current sequence that are not in the list
            // are added on the fly (identifiers of pasted code, see dv_add)
            // the draft head reads lm_head rows through the id list (lm_head), so no copy of the rows is kept
            const int cap = (int) ids.size() + DV_EXTRA;
            int * did = dalloc<int>(cap, false);
            CK(cudaMemcpy(did, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            Mat m = hw.head;
            m.N = (int) ids.size();
            hw.head_d = m;
            hw.draft_ids = did;
            dv_ids = ids;
            dv_base = (int) ids.size();
            if (cudaHostAlloc(reinterpret_cast<void **>(&dv_pin), ((size_t) cap + 1) * 4, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                dv_pin = nullptr; // (synchronous copies)
            }
            dv_in.assign(VOCAB, 0);
            for (int t : ids) dv_in[t] = 1;
            fprintf(stderr, "[mk] MTP draft head restricted to %d tokens (%s)\n", m.N, draft_vocab.c_str());
            if (draft_q4) { // MK_DRAFT_Q4: the rows' Q4_K copy (row r = lm_head row ids[r], from the file's Q6_K or Q8_0)
                auto _ = lc.scope("draft head Q4_K copy (MK_DRAFT_Q4)");
                const auto t0 = lclk::now();
                const auto & t = tget("output.weight");
                const size_t rs_from = ggml_row_size((ggml_type) t.type, D), rs_to = ggml_row_size(GGML_TYPE_Q4_K, D);
                if ((t.type != gguf::Q6_K && t.type != gguf::Q8_0) || rs_to != (size_t) DQ4_RB) throw std::runtime_error("MK_DRAFT_Q4: unexpected lm_head");
                std::vector<uint8_t> q(rs_to * ids.size());
                const auto * tr = ggml_get_type_traits((ggml_type) t.type);
                parallel_rows((int64_t) ids.size(), [&](int64_t r) {
                    thread_local std::vector<float> f;
                    f.resize((size_t) D);
                    tr->to_float(t.data + (size_t) ids[r] * rs_from, f.data(), D);
                    ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), q.data() + (size_t) r * rs_to, 0, 1, D, nullptr); // lossy by design
                });
                hw.dq4 = dput<uint8_t>(q.data(), q.size());
                own_keep.push_back(std::move(q)); // (until its upload is done)
                hw.dq4_n = dq4_rows = (int) ids.size();
                fprintf(stderr, "[mk] draft head: %d rows re-quantised to Q4_K in %.2f s (%.1f MiB; drafts only, MK_DRAFT_Q4=0: off)\n",
                        dq4_rows, lsec(t0), (double) rs_to * dq4_rows / 1048576.0);
            }
        } else if (draft_q4 && draft_q4_env) {
            fprintf(stderr, "[mk] MK_DRAFT_Q4 ignored: it needs a draft vocabulary (--draft-vocab)\n");
        }
        mem_mark("draft head copy");
        wrec_on = true;
        {
            const auto & t = tget("token_embd.weight");
            // the embedding lookups dequantise each type (embed_prologue, the MTP M0 phases, pk_embed*, pk_mtp_in)
            if (t.type != gguf::Q4_K && t.type != gguf::Q6_K && t.type != gguf::Q8_0)
                throw std::runtime_error("token_embd.weight is " + tname((int) t.type) + ", the engine reads Q4_K, Q6_K or Q8_0 (see README: the GGUF)");
            if (!shape2(t.ne, D, VOCAB)) throw std::runtime_error("token_embd.weight bad shape");
            hw.embd_t = (int) t.type; // (gguf::Q4_K / Q6_K / Q8_0 == T_Q4K / T_Q6K / T_Q8_0)
            // mapped pinned host memory: a cycle reads only a few 2.8 KB rows over PCIe, saves 0.7 GB of VRAM
            auto _ = lc.scope("token_embd (pinned table)");
            if (!embd_dev) {
                if (embd_th.t.joinable()) embd_th.t.join();
                else embd_alloc();
            }
            if (!embd_dev && embd_err == cudaSuccess) {
                // (write-combined: several threads, 64 MiB pieces)
                const size_t PC = (size_t) 64 << 20, np = (t.nbytes + PC - 1) / PC;
                parallel_rows((int64_t) np, [&](int64_t i) {
                    const size_t o = (size_t) i * PC;
                    memcpy(static_cast<uint8_t *>(embd_host) + o, t.data + o, std::min(PC, (size_t) t.nbytes - o));
                });
                void * d = nullptr;
                CK(cudaHostGetDevicePointer(&d, embd_host, 0));
                hw.embd = (const uint8_t *) d;
                fprintf(stderr, "[mk] token_embd in mapped host memory (%.2f GB, pinned in %.2f s)\n", t.nbytes / 1e9, embd_pin_s);
            } else {
                cudaGetLastError();
                embd_host = nullptr;
                hw.embd = dput<uint8_t>(t.data, t.nbytes);
                wbytes -= t.nbytes; // (not counted as weights)
            }
        }
        wrec_on = false;
        mem_mark("token_embd");
        wv_dev_done();
        CK(mkl::set_cW(&hw, sizeof(ModelW)));
        mem_mark("cW symbol (module load)");
        if (n_conv)
            fprintf(stderr, "[mk] %d tensors re-encoded in %.1f s (exact: Q3_K / IQ3_S -> Q6_K with MK_REENC3, IQ4_NL -> Q8_0 with MK_IQ4NL_Q8; ssm_alpha / beta -> F32)\n",
                    n_conv, conv_s);
        if (n_nat)
            fprintf(stderr, "[mk] %d Q3_K / IQ3_S tensors repacked to their native layouts in %.1f s (exact, checked per row)\n", n_nat, nat_s);
        if (n_req)
            fprintf(stderr, "[mk] MTP block: %d matrices re-quantised to Q4_K in %.1f s (drafts only)\n", n_req, req_s);

        hb.x = dalloc<float>(RMAX * D);
        hb.xm = dalloc<float>(RMAX * D);
        hb.qn = qbuf(U_D);
        hb.qo = qbuf(U_VD);
        hb.qa = qbuf(U_FF);
        hb.qe = qbuf(U_EH);
        hb.z = dalloc<float>(RMAX * VD);
        hb.ab = dalloc<float>(AB_SPLIT * RMAX * 2 * NVH);
        hb.qbuf = dalloc<float>(RMAX * QG);
        hb.kbuf = dalloc<float>(RMAX * KVD);
        hb.vbuf = dalloc<float>(RMAX * KVD);
        hb.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
        hb.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
        mem_mark("decode act bufs, conv, kvgb");
        hb.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
        mem_mark("GDN recurrent state");
        // stockfix switches (bit-exact): K1 MK_K1, K2 MK_K2 (both default on)
        hb.k1 = !(getenv("MK_K1") && atoi(getenv("MK_K1")) == 0);
        hb.k2 = !(getenv("MK_K2") && atoi(getenv("MK_K2")) == 0);
        fprintf(stderr, "[mk] stockfix: K1 ab dealing %d, K2 combine %d (tb %d)\n", hb.k1, hb.k2, MK_K2_TB);
        // stock2 switches: K6 MK_K6 (bits: 2 dealing across blocks, exact; 4 8-row halves of the Q8_0
        // k / v tiles, exact), K8 MK_K8 (bits, prefill, exact: 1 pk_ab on a second stream, 2 pk_norm_quant reads x once,
        // 4 pk_dn_prep MK_K8_NT tokens per block); compile-time K7 (state layout), K6_LD (Q8_0 tile loads)
        hb.k6 = getenv("MK_K6") ? atoi(getenv("MK_K6")) : MK_K6_DEF;
        if (hb.k6 & 1)
            throw std::runtime_error("MK_K6=" + std::to_string(hb.k6) + ": bit 1 (the K-split of the Q8_0 k / v tiles) was removed; bits 2 and 4 remain (default 4)");
        k8 = getenv("MK_K8") ? atoi(getenv("MK_K8")) : MK_K8_DEF;
        if (const char * e = getenv("MK_K8_NT")) k8_nt = atoi(e) == 8 ? 8 : 4;
        fprintf(stderr, "[mk] stock2: K6 %d, K8 %d (nt %d)\n", hb.k6, k8, k8_nt);
        fprintf(stderr, "[mk] K5: split dn_proj -> dn_core barrier, delta-net heads in dn_proj arrival order\n");
        fprintf(stderr, "[mk] C1: ffn_gu -> ffn_down per-chunk waits instead of a grid barrier\n");
        hb.ctx = ctx + KV_PAD; // rows per kv head (KV_PAD: the MTP draft rows written past the last admitted position)
        {
            const size_t rows = (size_t) kv_slots * NKV * hb.ctx; // (no MTP block: no slot N_ATT_SLOTS - 1)
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            // fp16 needs 2 x rows x HD x 2 bytes; leave ~1.5 GB for the prefill buffers, checkpoint and runtime.
            // A non-empty MK_KV picks the quantized cache in auto mode too.
            const char * kv_spec = getenv("MK_KV");
            const bool want_kv = kv_spec && kv_spec[0];
            const bool q8 = kv_mode == 1 || (kv_mode < 0 && (want_kv || rows * HD * 4 + (size_t) 1536 * 1024 * 1024 > free_b));
            hb.kvq8 = q8;
            hb.dattn = !(getenv("MK_DATTN") && atoi(getenv("MK_DATTN")) == 0);
            // MK_MTP_WIN=N: the MTP block's decode attention sees the last N positions only (its prefill attention and
            // the KV layout are unchanged). Drafts are verified, so the output is exact; acceptance can change.
            hb.mtp_win = getenv("MK_MTP_WIN") ? atoi(getenv("MK_MTP_WIN")) : 0;
            if (hb.mtp_win > 0) fprintf(stderr, "[mk] MTP decode attention window %d positions (MK_MTP_WIN)\n", hb.mtp_win);
            // the catch-up rows before the last 5 run K / V only (same drafts; not in MK_KV_EMU builds); env MK_MTP_KVC=0: off
            hb.mtp_kvc = !(getenv("MK_MTP_KVC") && atoi(getenv("MK_MTP_KVC")) == 0);
            // MK_KV (q8 cache): per attention slot and K / V, q8_0 (default), q4 (kv_store_q4) or q5 / q6 / q7
            // (kv_store_qn), e.g. "V*=6r,K16=4r": comma-separated <K|V|*><slot>=<4..8>[r], slot 0..15 = attention layer
            // 4 slot + 3, 16 = MTP block, * or empty = all slots; r: stored Hadamard-rotated (K: the queries get the same
            // rotation; V: the head's attention output is rotated back)
            const size_t srows = (size_t) NKV * hb.ctx; // rows of one slot
            for (int s = 0; s < N_ATT_SLOTS; s++) {
                hb.kvb[0][s] = hb.kvb[1][s] = 8;
                hb.kvr[s] = hb.kvrv[s] = 0;
            }
            if (want_kv) {
                if (!q8) throw std::runtime_error("MK_KV needs the q8 KV cache (--kv q8 or auto, not f16)");
                const std::string str = kv_spec;
                size_t i = 0;
                while (i < str.size()) {
                    size_t j = str.find(',', i);
                    if (j == std::string::npos) j = str.size();
                    const std::string tok = str.substr(i, j - i);
                    i = j + 1;
                    if (tok.empty()) continue;
                    const size_t e = tok.find('=');
                    if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV: bad item " + tok);
                    const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
                    if (md.empty() || md.size() > 2 || md[0] < '4' || md[0] > '8' || (md.size() == 2 && md[1] != 'r'))
                        throw std::runtime_error("MK_KV: expected 4..8 or 4r..8r in " + tok);
                    const bool rot = md.size() == 2;
                    const bool all = sl.empty() || sl == "*";
                    const int slot = all ? -1 : std::stoi(sl);
                    if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV: bad slot in " + tok);
                    for (int w = 0; w < 2; w++)
                        if (tok[0] == '*' || tok[0] == "KV"[w])
                            for (int s = 0; s < N_ATT_SLOTS; s++)
                                if (all || s == slot) {
                                    hb.kvb[w][s] = md[0] - '0';
                                    (w == 0 ? hb.kvr : hb.kvrv)[s] = rot;
                                }
                }
            }
            for (int s = kv_slots; s < N_ATT_SLOTS; s++) { // (not allocated)
                hb.kvb[0][s] = hb.kvb[1][s] = 8;
                hb.kvr[s] = hb.kvrv[s] = 0;
            }
            size_t vbytes[2] = {0, 0};
            int n4 = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < kv_slots; s++) {
                    vbytes[w] += srows * HD * hb.kvb[w][s] / 8;
                    n4 += hb.kvb[w][s] != 8 || (w == 0 ? hb.kvr[s] : hb.kvrv[s]);
                }
            hb.kc = q8 ? (void *) dalloc<int8_t>(vbytes[0]) : (void *) dalloc<__half>(rows * HD);
            hb.vc = q8 ? (void *) dalloc<int8_t>(vbytes[1]) : (void *) dalloc<__half>(rows * HD);
            hb.ksc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            hb.vsc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            size_t off[2] = {0, 0};
            for (int s = 0; s < N_ATT_SLOTS; s++)
                for (int w = 0; w < 2; w++) { // slots packed in order: all-q8 gives the same addresses as [slot][head][ctx][HD]
                    if (s >= kv_slots) {
                        hb.kvp[w][s] = nullptr;
                        hb.kvs[w][s] = nullptr;
                        continue;
                    }
                    hb.kvp[w][s] = static_cast<int8_t *>(w ? hb.vc : hb.kc) + off[w];
                    hb.kvs[w][s] = q8 ? (w ? hb.vsc : hb.ksc) + s * srows * (HD / 32) : nullptr;
                    off[w] += srows * HD * hb.kvb[w][s] / 8;
                }
            // MK_KV_WIN=W: every low-bit side keeps its last >= W positions as q8 in a ring (kv_ws / kv_src; the ring holds
            // (ceil(W / ATT_BUCKET) + 1) ATT_BUCKET rows per kv head so a bucket-aligned window always fits)
            hb.kvwin = 0;
            hb.kvrc = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    hb.kvrp[w][s] = nullptr;
                    hb.kvrs[w][s] = nullptr;
                }
            size_t ring_bytes = 0;
            if (want_kv && getenv("MK_KV_WIN") && atoi(getenv("MK_KV_WIN")) > 0) {
                hb.kvwin = atoi(getenv("MK_KV_WIN"));
                // + ATP slack rows: rows written past the attention partition of the launch that reads the window (the MTP
                // block's draft rows and catch-up row, up to n_draft past it, the wide pass at -d 7 included) then land in the
                // ring slots of positions before the window, not of its oldest rows (multiple of ATP: a tile never wraps)
                const int kvwr = ((hb.kvwin + ATT_BUCKET - 1) / ATT_BUCKET + 1) * ATT_BUCKET; // the most rows a window spans
                static_assert(MAXT + MAXT <= ATP, "ring slack");
                hb.kvrc = kvwr + ATP;
                // MK_KV_WIN_ALL=1 (test knob): rings for the q8 sides too. Their ring rows are the same q8 rows as the main
                // buffer (and k_ring_refill re-stores a q8 row exactly), so output must stay byte-identical to no window:
                // exercises the ring write / read / refill paths with an exact expectation.
                const bool win_all = getenv("MK_KV_WIN_ALL") && atoi(getenv("MK_KV_WIN_ALL")) != 0;
                int nr = 0;
                for (int w = 0; w < 2; w++)
                    for (int s = 0; s < kv_slots; s++)
                        if (hb.kvb[w][s] != 8 || win_all) {
                            hb.kvrp[w][s] = dalloc<int8_t>((size_t) NKV * hb.kvrc * HD);
                            hb.kvrs[w][s] = dalloc<__half>((size_t) NKV * hb.kvrc * (HD / 32));
                            ring_bytes += (size_t) NKV * hb.kvrc * (HD + HD / 16);
                            nr++;
                        }
                fprintf(stderr, "[mk] KV recent window: the last %d..%d positions of %d low-bit sides stay q8 (MK_KV_WIN=%d; rings of %d rows per head, %.1f MiB)\n",
                        kvwr - ATT_BUCKET + 1, kvwr, nr, hb.kvwin, hb.kvrc, ring_bytes / 1048576.0);
            }
            const double gb = q8 ? (vbytes[0] + vbytes[1] + 2.0 * rows * (HD / 32) * 2 + ring_bytes) / 1e9 : rows * HD * 4.0 / 1e9;
            fprintf(stderr, "[mk] KV cache %s, %.2f GB for %d positions\n", q8 ? (n4 ? "q8_0 + q4 (MK_KV)" : "q8_0") : "fp16", gb, ctx);
            if (n4) {
                std::string k, v, kr, vr;
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    k += (char) ('0' + hb.kvb[0][s]);
                    v += (char) ('0' + hb.kvb[1][s]);
                    kr += hb.kvr[s] ? 'r' : '.';
                    vr += hb.kvrv[s] ? 'r' : '.';
                }
                fprintf(stderr, "[mk] MK_KV bits per slot (layers 3, 7, .., 63, MTP): K %s  V %s; rotated: K %s  V %s\n", k.c_str(), v.c_str(), kr.c_str(), vr.c_str());
            }
        }
        mem_mark("KV cache");
        hb.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
        hb.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
        hb.h_hist = dalloc<float>((size_t) H_RING * D);
        mem_mark("attn partials, h ring");
        {
            std::vector<float2> rt((size_t) hb.ctx * 32); // (rows past ctx: the MTP draft rows, KV_PAD)
            for (int p = 0; p < hb.ctx; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            float2 * d = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(d, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
            hb.rope = d;
            rope_base = std::move(rt); // (images: rope_sync gathers rewritten rows from it)
        }
        mem_mark("RoPE table");
        hb.hm = dalloc<float>(D);
        hb.blk_best = dalloc<float2>((size_t) nsm * MAXT * TOPK_MAX);
        hb.blk_lse = dalloc<float2>((size_t) nsm);
        hb.dlog = dalloc<float2>((size_t) MAXT * (TOPK_MAX + 1));
        hb.dq = dalloc<float2>((size_t) MAXT * TOPK_MAX);
        hb.cand = dalloc<float2>((size_t) RMAX * TOPK_MAX);
        hb.logits = nullptr; // 5 MB used by scoring launches only (score_logits): logits_alloc()
        hb.pcnt = nullptr;   // 0.5 MB for the presence / frequency penalty, allocated by the first set_penalty
        hb.bar = dalloc<unsigned>(C1_WORDS); // count word | go word | C1's per-chunk counters, each on its own line
        hb.cnt = dalloc<unsigned>(4);
        hb.ctl = dalloc<Ctl>(1);
        hb.wvpart = dalloc<float>((size_t) WV_S * RMAX * D);
        hb.wvcnt = dalloc<unsigned>(2 * (D / 32), DA_INIT0); // (starts at 0; the last arrival of each use resets it)
        profiling = getenv("MK_PROFILE") != nullptr;
        hb.prof = profiling ? dalloc<unsigned long long>(4096 + 1024 * 128) : nullptr;
        mem_mark("logits, sampling, ctl");
        if (const char * spec = getenv("MK_KV_EMU_SPEC")) kv_emu_parse(spec);
        {
            int * eh = nullptr;
            CK(cudaHostAlloc(reinterpret_cast<void **>(&eh), 2 * sizeof(int), cudaHostAllocMapped));
            eh[0] = eh[1] = 0;
            err_host = eh;
            CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&hb.err), eh, 0));
        }
        {
            Ctl * ce = nullptr;
            CK(cudaHostAlloc(reinterpret_cast<void **>(&ce), sizeof(Ctl), cudaHostAllocMapped));
            memset(ce, 0, sizeof(Ctl));
            ctl_eg_host = ce;
            CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&hb.ctl_eg), ce, 0));
        }
        CK(mkl::set_cBs(&hb, sizeof(Bufs)));

        CK(cudaFuncSetAttribute(mkl::k_mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
        int nb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mkl::k_mk_cycle, NT, SM_TOTAL));
        if (nb < 1) throw std::runtime_error("megakernel does not fit on an SM");
        // The driver reserves local memory for every thread slot of the device (1,536 per SM x 82 SMs), sized by
        // the larger of the stack-size limit (default 1 KB per thread = 123 MiB) and the launched kernels' own
        // (static) frames. No kernel here needs a dynamic stack, so the limit only adds a floor: drop it.
        {
            const char * sl = getenv("MK_STACK_LIMIT");
            CK(cudaDeviceSetLimit(cudaLimitStackSize, sl ? (size_t) atol(sl) : 0));
        }
        if (memlog) {
            mem_mark("mk_cycle attributes");
            cudaFuncAttributes fa{};
            CK(cudaFuncGetAttributes(&fa, mkl::k_mk_cycle));
            size_t stk = 0, heap = 0, fifo = 0;
            CK(cudaDeviceGetLimit(&stk, cudaLimitStackSize));
            CK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
            CK(cudaDeviceGetLimit(&fifo, cudaLimitPrintfFifoSize));
            fprintf(stderr, "[mk-mem] mk_cycle: %d regs, local %zu B/thread, const %zu B, static smem %zu B; limits: stack %zu B/thread, "
                            "malloc heap %zu B, printf fifo %zu B; %d SMs x %d threads/SM\n",
                    fa.numRegs, (size_t) fa.localSizeBytes, (size_t) fa.constSizeBytes, (size_t) fa.sharedSizeBytes, stk, heap, fifo, nsm,
                    prop.maxThreadsPerMultiProcessor);
        }
        // the DeltaNet checkpoint and the batched-prefill scratch are allocated here, not inside the first prompt's time
        {
            auto _ = lc.scope("checkpoint pool (pinned host)");
            ck_alloc();
        }
        {
            auto _ = lc.scope("prefill scratch");
            pf_init();
        }
        ck_make_plan();
        if (exact && stg_n < 2)
            fprintf(stderr, "[mk] warning: MK_EXACT with MK_PF_CAP %d: no staging slots for checkpoints inside a prefill, so "
                            "none are saved (a reuse prefills from position 0); MK_PF_CAP >= 384 keeps them\n", hp.cap);
        {
            auto _ = lc.scope("upload tail (wait)");
            upl.finish();
        }
        own_keep.clear();
        own_keep.shrink_to_fit();
        for (HSet & S : hs) S = HSet{};
        const double pf_wait = pfx ? pfx->wait_s : 0.0;
        const size_t pf_bytes = pfx ? pfx->bytes() : 0;
        pfx.reset();
        load_report(pf_wait, pf_bytes);
        if (getenv("MK_WDIGEST")) weights_digest();
        fprintf(stderr, "[mk] weights %.2f GB on device, %d SMs, ctx %d, n_draft %d\n", wbytes / 1e9, nsm, ctx, ndraft);
    }

    // one line of load phases (MK_LOADLOG=1: every phase on its own line)
    void load_report(double pf_wait, size_t pf_bytes) {
        if (rq_th.t.joinable()) rq_th.t.join(); // (rq_s is the thread's until it ends)
        const double tot = lsec(t_load);
        double known = pf_wait;
        double rep = 0;
        std::string rest;
        for (const auto & p : lc.ph) {
            known += p.second;
            if (p.first.compare(0, 6, "repack") == 0) rep += p.second;
        }
        auto g = [&](const char * k) { return lc.get(k); };
        fprintf(stderr, "[mk] load %.2f s: context %.2f, read-ahead wait %.2f (%.1f GB read), repack %.2f, wv re-layout %.2f, "
                        "MTP requant wait %.2f (background %.2f), token_embd %.2f, ckpt pool %.2f, prefill scratch %.2f, upload tail %.2f, "
                        "other %.2f (device allocs %.2f of it), upload stalls %.2f | uploads %.2f GB, %.2f s in copies (%s)\n",
                tot, g("CUDA context"), pf_wait, pf_bytes / 1e9, rep, g("wv re-layout"), g("MTP Q4_K requant (wait for the background thread)"),
                rq_s, g("token_embd (pinned table)"), g("checkpoint pool (pinned host)"), g("prefill scratch"), g("upload tail (wait)"),
                tot - known, dalloc_s, g("upload stall (host buffer set busy)"), upl.bytes / 1e9, upl.busy_s, getenv("MK_LOAD_ASYNC") && atoi(getenv("MK_LOAD_ASYNC")) == 0 ? "synchronous" : "worker thread");
        if (getenv("MK_LOADLOG"))
            for (const auto & p : lc.ph) fprintf(stderr, "[mk-load] %-52s %7.3f s\n", p.first.c_str(), p.second);
    }

    // MK_WDIGEST=1 (load gates): a 64-bit hash over every buffer allocated while the weights load (dalloc, in allocation
    // order: size + device bytes) and the mapped token_embd table. Slow (reads ~17 GB back); test runs only.
    bool wrec_on = false;
    std::vector<std::pair<const void *, size_t>> wrec;
    void weights_digest() {
        CK(cudaDeviceSynchronize());
        const auto t0 = lclk::now();
        const size_t PC = (size_t) 64 << 20, NB = 32;
        struct Pc { const uint8_t * p; size_t n; bool host; };
        std::vector<Pc> pcs;
        size_t tot = 0;
        for (const auto & r : wrec) {
            for (size_t o = 0; o < r.second; o += PC) pcs.push_back({static_cast<const uint8_t *>(r.first) + o, std::min(PC, r.second - o), false});
            tot += r.second;
        }
        if (embd_host) {
            const size_t n = file.get("token_embd.weight").nbytes;
            for (size_t o = 0; o < n; o += PC) pcs.push_back({static_cast<const uint8_t *>(embd_host) + o, std::min(PC, n - o), true});
            tot += n;
        }
        std::vector<unsigned long long> ph(pcs.size());
        std::unique_ptr<uint8_t[]> buf(new uint8_t[NB * PC]);
        for (size_t b = 0; b < pcs.size(); b += NB) {
            const size_t k = std::min(NB, pcs.size() - b);
            for (size_t j = 0; j < k; j++)
                if (!pcs[b + j].host) CK(cudaMemcpy(buf.get() + j * PC, pcs[b + j].p, pcs[b + j].n, cudaMemcpyDeviceToHost));
            parallel_rows((int64_t) k, [&](int64_t j) {
                const Pc & q = pcs[b + j];
                if (q.host) memcpy(buf.get() + j * PC, q.p, q.n); // (write-combined memory: copy first)
                ph[b + j] = wd_hash(buf.get() + j * PC, q.n);
            });
        }
        unsigned long long h = 0xcbf29ce484222325ULL;
        for (const auto & r : wrec) h = wd_hash(reinterpret_cast<const uint8_t *>(&r.second), sizeof(size_t), h);
        h = wd_hash(reinterpret_cast<const uint8_t *>(ph.data()), ph.size() * 8, h);
        fprintf(stderr, "[mk] weights digest %016llx (%zu buffers + token_embd, %.2f GB, %.1f s)\n", h, wrec.size(), tot / 1e9, lsec(t0));
    }

#if MK_SPEC_LOG
    // one JSON line per decode cycle: start position n, T, accepted a, the verified drafts d, the verify
    // rows' targets tg, per draft its MTP [lse, best logit, (logit, id) x TOPK_MAX], and when sampling the
    // main rows' candidates p [(logit, id) x TOPK_MAX]
    void slog_cycle(int n0, int T, const int * drafts, const Ctl & c) {
        if (!slog) return;
        const bool smp = smp_temp > 0.f;
        std::vector<float2> cand;
        if (smp) {
            cand.resize((size_t) T * TOPK_MAX);
            CK(cudaMemcpy(cand.data(), hb.cand, cand.size() * sizeof(float2), cudaMemcpyDeviceToHost));
        }
        auto id_of = [](float y) { int v; memcpy(&v, &y, 4); return v; };
        fprintf(slog, "{\"n\":%d,\"T\":%d,\"a\":%d,\"nd\":%d,\"d\":[", n0, T, c.n_acc, nd_avail);
        for (int i = 1; i < T; i++) fprintf(slog, "%s%d", i > 1 ? "," : "", drafts[i]);
        fprintf(slog, "],\"tg\":[");
        for (int i = 0; i < T; i++) fprintf(slog, "%s%d", i ? "," : "", c.tgt[i]);
        fprintf(slog, "],\"o\":[");
        for (int i = 0; i < c.n_out; i++) fprintf(slog, "%s%d", i ? "," : "", c.out[i]);
        fprintf(slog, "],\"q\":[");
        for (int i = 0; i + 1 < T; i++) {
            const float2 * q = slog_q.data() + (size_t) i * (TOPK_MAX + 1);
            fprintf(slog, "%s[%.7g,%.7g", i ? "," : "", q[TOPK_MAX].x, q[TOPK_MAX].y);
            for (int j = 0; j < TOPK_MAX; j++) fprintf(slog, ",%.7g,%d", q[j].x, id_of(q[j].y));
            fprintf(slog, "]");
        }
        fprintf(slog, "]");
        if (smp) {
            fprintf(slog, ",\"p\":[");
            for (int i = 0; i < T; i++) {
                fprintf(slog, "%s[", i ? "," : "");
                for (int j = 0; j < TOPK_MAX; j++)
                    fprintf(slog, "%s%.7g,%d", j ? "," : "", cand[(size_t) i * TOPK_MAX + j].x, id_of(cand[(size_t) i * TOPK_MAX + j].y));
                fprintf(slog, "]");
            }
            fprintf(slog, "]");
        }
        fprintf(slog, "}\n");
        fflush(slog);
        slog_drafts(); // the drafts this launch produced, verified by the next cycle
    }
    // after a prefill: the whole sequence (its last id is the first output token) and where the fed part began
    void slog_prompt(int n0) {
        if (!slog) return;
        fprintf(slog, "{\"prompt\":%zu,\"n0\":%d,\"ids\":[", seq.size(), n0);
        for (size_t i = 0; i < seq.size(); i++) fprintf(slog, "%s%d", i ? "," : "", seq[i]);
        fprintf(slog, "]}\n");
        fflush(slog);
    }
#endif

    ~Impl() {
#if MK_SPEC_LOG
        if (slog) fclose(slog);
#endif
        try { // (CK throws; a destructor must not)
            print_profile();
            mem_mark("at exit");
            ck_wait_restore();
        } catch (const std::exception & e) {
            fprintf(stderr, "[mk] teardown: %s\n", e.what());
        }
        release();
    }
    // the device and pinned resources: ~Impl, and a constructor that throws part way (~Impl does not run then). Every
    // handle starts null or empty, so a half-built engine frees what it got; no CUDA call when nothing was made.
    void release() noexcept {
        if (!allocs.empty()) cudaDeviceSynchronize(); // (a non-blocking stream's work may still use them)
        ck_free(); // (checkpoint copies may still read or write device buffers)
        for (auto & l : lanes) { // (a live lane's resources are in the members above, the engine's own ones in its Lane)
            ck_slots_free(l.ck);
            if (l.ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(l.ctl_eg_host));
        }
        for (auto & p : parked)
            for (char * c : p.second.blob.chunks) cudaFreeHost(c);
        for (char * c : sw_blob.chunks) cudaFreeHost(c);
        for (char * c : hs_free) cudaFreeHost(c);
        if (pf_s2) { // (MK_K8 bit 1)
            cudaStreamSynchronize(pf_s2);
            cudaStreamDestroy(pf_s2);
        }
        for (cudaEvent_t e : {pf_e0, pf_e1, pf_cev[0], pf_cev[1], wd_ev})
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : pf_gev) cudaEventDestroy(e);
        for (void * p : allocs) use_pool ? cudaFreeAsync(p, 0) : cudaFree(p);
        if (!allocs.empty()) pool_trim(); // (a caller's retry finds the VRAM free)
        // a constructor that threw while the weights loaded: wv_dev_done did not give back the staging buffer (192 MiB) and
        // the gather tables (cudaMalloc, outside allocs)
        if (stg_d) cudaFree(stg_d);
        for (void * t : wv_tabs) cudaFree(t);
        stg_d = nullptr;
        wv_tabs.clear();
        if (embd_host) cudaFreeHost(embd_host);
        if (emb_host) cudaFreeHost(emb_host);
        if (err_host) cudaFreeHost(const_cast<int *>(err_host));
        if (ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(ctl_eg_host));
        if (dv_pin) cudaFreeHost(dv_pin);
    }

    // The default pool keeps what is freed into it, and what a failed allocation grew it by before failing: back to the
    // device (after the stream-ordered frees). No CUDA call without the pool.
    void pool_trim() noexcept {
        if (!use_pool) return;
        cudaStreamSynchronize(0);
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, cuda_dev) == cudaSuccess) cudaMemPoolTrimTo(pool, 0);
    }

    // the most positions a sequence may have: the model's context length (2^20 without the key)
    int ctx_limit() const {
        const auto cl = file.ints.find("qwen35.context_length");
        return (int) std::min<uint64_t>(cl != file.ints.end() && cl->second > 0 ? cl->second : 1u << 20, 1u << 20);
    }

    // emb_host holding at least `floats` (grown: pinned, mapped; cP.emb follows it)
    void emb_reserve(size_t floats) {
        if (floats <= emb_cap) return;
        pf_init();
        CK(cudaDeviceSynchronize()); // (no kernel still reads the old rows)
        if (emb_host) CK(cudaFreeHost(emb_host));
        emb_host = nullptr;
        emb_cap = 0;
        CK(cudaHostAlloc(reinterpret_cast<void **>(&emb_host), floats * sizeof(float), cudaHostAllocMapped));
        emb_cap = floats;
        float * dev = nullptr;
        CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), emb_host, 0));
        hp.emb = dev;
        CK(mkl::set_cP(&hp, sizeof(PBufs)));
    }

    void launch(int mode, int T, int last, int score = 0) {
        sw_need(hctl.n + T + ndraft); // (side request: the kept sequence's rows there go to the host store first)
        kv_top = std::max(kv_top, hctl.n + T + ndraft); // rows hctl.n .. + T - 1, the MTP's draft rows after them (ring_rewind)
        Launch L{mode, T, last, ndraft, score, smp_temp, smp_top_p, smp_min_p, smp_top_k, smp_seed};
        for (int k = 0; k < MAXT - 1; k++) {
            L.gate_drop[k] = pol.gate_drop[k];
            L.gate_stop[k] = pol.gate_stop[k];
        }
        const int top = depth_top(hctl.n); // drop_last: the deepest draft of this position's policy depth
        if (pol.g7 && top > 5) {
            for (int k = 1; k <= 3; k++) L.gate_stop[k] = 0.25f;
            L.gate_stop[4] = L.gate_stop[5] = 0.3f;
            L.gate_drop[4] = 0.3f;
        } else if (top >= 1) L.gate_drop[top - 1] = std::max(L.gate_drop[top - 1], pol.drop_last);
        L.gate_chain = pol.gate_chain;
        L.spec_smp = smp_temp > 0.f && pol.h1;
        // chain=2 with sampled drafts: the chain holds the drafted token's own q' probability, so a drop gate on it would
        // pick which drafts get verified by their value (biased output: tests/spec_rng_sim.py); stop gates only
        if (L.gate_chain == 2 && L.spec_smp)
            for (int k = 0; k < MAXT - 1; k++) L.gate_drop[k] = 0.f;
        if (!score && (pen_pres != 0.f || pen_freq != 0.f)) {
            L.pres = pen_pres;
            L.freq = pen_freq;
            pcnt_dirty = true;
        }
        void * args[] = {&L};
        ck_wait_restore(); // (cheap no-ops unless a checkpoint copy is pending on ck_st)
        ck_guard_state();
        ring_snap_guard();
#if MK_TEST_NAN
        if (mode == MODE_DECODE && test_nan_at > 0 && ++test_nan_n == test_nan_at) {
            __half * ks = hb.kvs[0][0] ? hb.kvs[0][0] : hb.ksc;
            if (ks) CK(cudaMemset(ks, 0xff, (size_t) NKV * hb.ctx * (HD / 32) * sizeof(__half)));
            fprintf(stderr, "[mk] MK_TEST_NAN: NaN K scales before decode launch %d\n", test_nan_n);
        }
#endif
        if (!bar_live) { // (the counts run on from the previous launch's BAR_BASE / C1_BASE: zeroed once, and after bench launches)
            CK(cudaMemsetAsync(hb.bar, 0, C1_WORDS * 4));
            bar_live = true;
        }
        if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
        CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        wd_T = T;
        wd_mode = mode;
        wait_device(mode == MODE_DECODE ? "a decode launch" : "a prefill launch", wd_limit);
        if (err_host[0] | err_host[1]) launch_failed();
        launches_seen++;
        if (memlog && launches_seen == 1) mem_mark("first mk_cycle launch");
        if (profiling && mode == MODE_DECODE) prof_collect(T);

    }

    // MK_PROFILE: one decode launch's barrier stamps (T: its rows; 0 = a batched launch)
    void prof_collect(int T) {
        std::vector<unsigned long long> pr(4096 + 1024 * 128);
        CK(cudaMemcpy(pr.data(), hb.prof, pr.size() * 8, cudaMemcpyDeviceToHost));
        unsigned long long prev = pr[0] >> 8;
        auto & bn = best_ns[T];
        auto & bt = best_tag[T];
        auto & bf = best_fill[T];
        for (int k = 1; k < 4096 && pr[k]; k++) {
            const unsigned long long t = pr[k] >> 8;
            const int tag = (int) (pr[k] & 0xff);
            const double dt = (double) (t - prev);
            ph_ns[tag] += dt;
            ph_n[tag]++;
            launch_ns += dt;
            prev = t;
            const unsigned long long mark = pr[2048 + k] & ((1ull << 56) - 1);
            const double fill = mark ? (double) ((mark - (t - (unsigned long long) dt)) & ((1ull << 56) - 1)) : 0.0;
            if ((int) bn.size() < k) { bn.resize(k, 1e30); bt.resize(k, -1); bf.resize(k, 0.0); }
            if (k < 1024) {
                std::vector<double> arr;
                const unsigned long long t0 = t - (unsigned long long) dt;
                for (int b = 0; b < nsm; b++) {
                    const unsigned long long a = pr[4096 + k * 128 + b];
                    if (a) arr.push_back((double) (((a & ((1ull << 56) - 1)) - t0) & ((1ull << 56) - 1)));
                }
                if (arr.size() == (size_t) nsm) {
                    std::sort(arr.begin(), arr.end());
                    arr_first[tag] += arr.front();
                    arr_med[tag] += arr[arr.size() / 2];
                    arr_last[tag] += arr.back();
                    arr_n[tag]++;
                }
            }
            if (bt[k - 1] != tag) { bt[k - 1] = tag; bn[k - 1] = dt; bf[k - 1] = fill; } // sequence changed
            else if (dt < bn[k - 1]) { bn[k - 1] = dt; bf[k - 1] = fill; }
        }
        launches++;
    }

    void print_profile() {
        if (!profiling || !launches) return;
        static const char * names[PH_COUNT] = {"embed", "attn_proj", "attn_core", "attn_comb", "resid_o", "dn_proj",
                                               "dn_core", "resid_sout", "ffn_gu", "ffn_down", "lm_head", "mtp_m0",
                                               "mtp_eh", "mtp_attn_proj", "mtp_attn_core", "mtp_attn_comb",
                                               "mtp_resid_o", "mtp_ffn_gu", "mtp_ffn_down", "mtp_lm_head",
                                               "mtp_kv_m0", "mtp_kv_eh", "mtp_kv_proj"};
        fprintf(stderr, "[mk-profile] %ld decode launches, mean %.3f ms in barriers-to-barrier phases\n", launches,
                launch_ns / launches / 1e6);
        for (int i = 0; i < PH_COUNT; i++) {
            if (!ph_n[i]) continue;
            fprintf(stderr, "  %-16s total/launch %8.3f ms   per call %8.2f us   calls/launch %6.1f\n", names[i],
                    ph_ns[i] / launches / 1e6, ph_ns[i] / ph_n[i] / 1e3, (double) ph_n[i] / launches);
        }
        fprintf(stderr, "[mk-profile] block arrival at barrier (us after phase start; mean over launches): first / median / last\n");
        for (int i = 0; i < PH_COUNT; i++)
            if (arr_n[i])
                fprintf(stderr, "  %-16s %8.2f %8.2f %8.2f\n", names[i], arr_first[i] / arr_n[i] / 1e3, arr_med[i] / arr_n[i] / 1e3,
                        arr_last[i] / arr_n[i] / 1e3);
        for (int T = 0; T <= MAXT; T++) { // (T = 0: batched launches)
            if (best_ns[T].empty()) continue;
            std::vector<double> tot(PH_COUNT, 0.0), fil(PH_COUNT, 0.0);
            std::vector<int> cnt(PH_COUNT, 0);
            double all = 0;
            for (size_t k = 0; k < best_ns[T].size(); k++) {
                if (best_tag[T][k] < 0) continue;
                tot[best_tag[T][k]] += best_ns[T][k];
                fil[best_tag[T][k]] += best_fill[T][k];
                cnt[best_tag[T][k]]++;
                all += best_ns[T][k];
            }
            fprintf(stderr, "[mk-profile] T=%d min-composed launch %.3f ms (interference-free estimate)\n", T, all / 1e6);
            for (int i = 0; i < PH_COUNT; i++)
                if (cnt[i])
                    fprintf(stderr, "  %-16s %8.3f ms  %8.2f us/call (fill %5.2f us)  x%d\n", names[i], tot[i] / 1e6,
                            tot[i] / cnt[i] / 1e3, fil[i] / cnt[i] / 1e3, cnt[i]);
            if (getenv("MK_PROFILE_SEQ")) // every barrier position: its phase and min time (per-layer phase costs)
                for (size_t k = 0; k < best_ns[T].size(); k++)
                    if (best_tag[T][k] >= 0)
                        fprintf(stderr, "[mk-seq] T=%d k=%zu %s %.2f fill %.2f\n", T, k, names[best_tag[T][k]],
                                best_ns[T][k] / 1e3, best_fill[T][k] / 1e3);
        }
    }

    // grid barrier cost. MK_GSB=N: launches of N barriers with no work (and of 0 barriers: the launch cost) for
    // every gsync form (gs_arrive_wait) at nsm and nsm / 2 blocks, round-robin over the configurations so clock drift hits
    // them alike; per barrier = (median t(N) - median t(0)) / N. MK_GSB_REPS (default 15) rounds; 1.5 s GPU warm-up first.
    void gsync_bench_host(int N) {
        bar_live = false; // (these launches zero the barrier words themselves and leave no BAR_BASE)
        const int reps = getenv("MK_GSB_REPS") ? std::max(3, atoi(getenv("MK_GSB_REPS"))) : 15;
        mkl::st_warm(nsm * 4, (unsigned long long) 1500 * 1000000ull);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const int nbs[2] = {nsm, nsm / 2};
        std::vector<float> t[2][4][2]; // [blocks][form][N or 0] ms
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        for (int r = 0; r < reps; r++)
            for (int bi = 0; bi < 2; bi++)
                for (int f = 0; f < 4; f++)
                    for (int z = 0; z < 2; z++) {
                        Launch L{MODE_BENCH, 1, 3, z ? 0 : N, f};
                        void * args[] = {&L};
                        CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                        CK(cudaEventRecord(a));
                        CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nbs[bi]), dim3(NT), args, SM_TOTAL, 0));
                        CK(cudaEventRecord(b));
                        CK(cudaEventSynchronize(b));
                        float ms = 0.f;
                        CK(cudaEventElapsedTime(&ms, a, b));
                        t[bi][f][z].push_back(ms);
                    }
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
        auto med = [](std::vector<float> v) {
            std::sort(v.begin(), v.end());
            return (double) v[v.size() / 2];
        };
        static const char * fn[4] = {"0 upstream (red.release, acquire polls)", "1 flag broadcast, relaxed polls",
                                     "2 flag broadcast, acquire polls", "3 upstream arrival, relaxed polls"};
        for (int bi = 0; bi < 2; bi++)
            for (int f = 0; f < 4; f++) {
                const double mn = med(t[bi][f][0]), m0 = med(t[bi][f][1]);
                const double best = *std::min_element(t[bi][f][0].begin(), t[bi][f][0].end());
                const double b0 = *std::min_element(t[bi][f][1].begin(), t[bi][f][1].end());
                fprintf(stderr, "[mk-gsb] blocks %2d form %s: %.3f us per barrier (median; min %.3f), N %d, launch %.1f us, %d reps\n",
                        nbs[bi], fn[f], (mn - m0) * 1e3 / N, (best - b0) * 1e3 / N, N, m0 * 1e3, reps);
            }
        fprintf(stderr, "[mk-gsb] the engine's gsync is form 0\n");
    }

    void bench() {
        bar_live = false; // (these launches zero the barrier words themselves and leave no BAR_BASE)
        if (const char * g = getenv("MK_GSB")) {
            gsync_bench_host(std::max(1, atoi(g)));
            return;
        }
        const char * names[3] = {"ffn_gu", "ffn_down", "dn_proj"};
        if (profiling) {
            // i-cache test: 4 back-to-back iterations of one phase inside one launch
            for (int which = 0; which < 3; which++) {
                std::vector<double> best(5, 1e30);
                for (int rep = 0; rep < 60; rep++) {
                    Launch L{MODE_BENCH, 4, which, which == 2 ? 0 : 1, 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                    CK(cudaMemsetAsync(hb.prof, 0, 4096 * 8));
                    CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    std::vector<unsigned long long> pr(8);
                    CK(cudaMemcpy(pr.data(), hb.prof, 8 * 8, cudaMemcpyDeviceToHost));
                    for (int k = 1; k <= 4; k++) best[k] = std::min(best[k], (double) ((pr[k] >> 8) - (pr[k - 1] >> 8)));
                }
                fprintf(stderr, "[mk-bench] %-8s T=4 back-to-back iterations: %.1f / %.1f / %.1f / %.1f us\n", names[which],
                        best[1] / 1e3, best[2] / 1e3, best[3] / 1e3, best[4] / 1e3);
            }
            return;
        }
        for (int which = 0; which < 3; which++)
            for (int T : {1, 4}) {
                float best = 1e30f;
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int rep = 0; rep < 400; rep++) {
                    cudaEventRecord(a);
                    Launch L{MODE_BENCH, T, which, which == 2 ? 0 : (rep % 16), 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                    CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    cudaEventRecord(b);
                    CK(cudaEventSynchronize(b));
                    float ms;
                    cudaEventElapsedTime(&ms, a, b);
                    best = std::min(best, ms);
                }
                fprintf(stderr, "[mk-bench] %-8s T=%d: %.1f us/launch (1 layer + barrier + launch overhead)\n", names[which], T, best * 1000);
            }
        err_host[0] = err_host[1] = 0; // (bench launches run on synthetic state)
    }

    // ---------------- batched prefill (prefill.cuh) ----------------
    PBufs hp{};
    bool pf_ready = false;
    // the host enqueues prefill chunk k + 1 before it waits for chunk k (pf_cev); MK_PF_SYNC=1: the token upload of every
    // chunk is synchronous and the host waits for each chunk's trunk before its MTP part (the GPU idles for the host's wake-up
    // and launches at both points). MK_PF_GAPS=1: report the GPU's idle time at those points per prefill run.
    bool pf_sync = false, pf_gaps = false;
    // MKEngine::cancel: pf_run reads pf_cancel (set from any thread) at each chunk boundary and stops there (pf_cancelled)
    std::atomic<int> pf_cancel{0};
    bool pf_cancelled = false;
    int pf_stop = 0; // (the position it stopped at)
    // MKEngine::prefill_progress (read from any thread): the batched part of the running prefill() covers positions
    // pf_from .. pf_to - 1; pf_run moves pf_pos past each chunk once the GPU finished it. Host counters only.
    std::atomic<int> pf_from{0}, pf_to{0}, pf_pos{0};
    cudaEvent_t pf_cev[2] = {};
    std::vector<cudaEvent_t> pf_gev; // MK_PF_GAPS: timing events, 4 per chunk (start, trunk end, MTP part start, end)

    void pf_init() {
        if (pf_ready) return;
        // MK_PF_CAP: tokens per prefill chunk (multiple of 128); scratch is ~0.33 MB per token
        int cap = 2048;
        if (const char * e = getenv("MK_PF_CAP")) cap = std::max(128, std::min(2048, atoi(e) / 128 * 128));
        hp.cap = cap;
        mem_mark("before prefill scratch");
        hp.x = dalloc<float>((size_t) cap * D);
        hp.xm = dalloc<float>((size_t) cap * D);
        hp.rs = dalloc<float>(cap);
        hp.xh = dalloc<__half>((size_t) cap * 2 * D); // GEMM inputs: K = D, VD (attention / delta-net out) or 2 D (MTP eh)
        hp.xh2 = dalloc<__half>((size_t) cap * FF);
        hp.y0 = dalloc<float>((size_t) cap * FF);
        // y1: [cap][FF] for the two-GEMM gate / up path (up's fp32 output) and as a2_plan's tail-split partials (non-exact
        // mode: its capacity bounds the split count, i.e. the numerics, so it stays FF there). In MK_EXACT mode with every
        // gate / up pair fused (same-type or V2 mixed), y1's other users need VD floats per token (DeltaNet z; attention
        // k / v, 2 KVD; EXA partials, whose split is numerically free): 45 KB less per token. MK_PF_Y1=1: always FF.
        bool all_fused = true;
        for (int L = 0; L < NL; L++) all_fused &= glu_fused(hw.L[L].fg, hw.L[L].fu) || (glu_mix_on && glu_mix(hw.L[L].fg, hw.L[L].fu, 0));
        const bool y1full = getenv("MK_PF_Y1") && atoi(getenv("MK_PF_Y1")) == 1;
        y1_floats = (size_t) cap * (exact && all_fused && !y1full ? VD : FF);
        hp.y1 = dalloc<float>(y1_floats);
        hp.y2 = dalloc<float>((size_t) cap * VD);
        hp.dq = dalloc<__half>((size_t) cap * QKV);
        hp.gb = dalloc<float>((size_t) cap * 2 * NVH);
        hp.ab = dalloc<float>((size_t) 8 * cap * 2 * NVH);
        hp.h = dalloc<float>((size_t) (cap + 1) * D);
        hp.qh = dalloc<__half>((size_t) cap * NH * HD);
        hp.tok = dalloc<int>(cap);
        hp.erow = dalloc<int>(cap);
        hp.emb = nullptr; // (emb_reserve)
        hp.amax = dalloc<int>(cap);
        hp.wsrow = exact ? 1 : 0;
        mem_mark("prefill scratch");
        CK(mkl::set_cP(&hp, sizeof(PBufs)));
        mmh_attr<pf::TQ4K, false>();
        mmh_attr<pf::TQ5K, false>();
        mmh_attr<pf::TQ6K, false>();
        mmh_attr<pf::TQ80, false>();
        mmh_attr<pf::TIQ4, false>();
        mmh_attr<pf::TIQ4NL, false>();
        mmh_attr<pf::TQ3K, false>();
        mmh_attr<pf::TIQ3S, false>();
        mmh_attr<pf::TQ4K, true>();
        mmh_attr<pf::TQ5K, true>();
        mmh_attr<pf::TIQ4, true>();
        mmh_attr<pf::TQ4K, false, 1>();
        mmh_attr<pf::TQ5K, false, 1>();
        mmh_attr<pf::TQ80, false, 1>();
        mmh_attr<pf::TIQ4, false, 1>();
        mmh_attr<pf::TIQ4NL, false, 1>();
        mmh_attr<pf::TQ4K, true, 1>();
        mmh_attr<pf::TQ5K, true, 1>();
        mmh_attr<pf::TIQ4, true, 1>();
        mmh_attr<pf::TQ6K, false, 1>();
        mmh_attr<pf::TQ3K, false, 1>();
        mmh_attr<pf::TIQ3S, false, 1>();
        glu_mix_attr();
        if (const char * e = getenv("MK_PF_BN")) { // force one GEMM token tile (tuning / tests); 0 = by prompt size
            pf_bn_force = atoi(e);
            if (pf_bn_force != 0 && pf_bn_force != 32 && pf_bn_force != 64 && pf_bn_force != 128)
                throw std::runtime_error("MK_PF_BN: 0, 32, 64 or 128");
        }
        mkl::pk_attn_attr();
        int dev = 0, nsm = 0, occ = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
        mkl::pk_attn2_occupancy(&occ);
        a2_slots = nsm * occ;
        // MK_ATTN=1: pk_attn; MK_ATTN_SPLIT=0: no tail split; MK_ATTN_PV16=0: fp32-accumulated P V
        // (pk_attn2 with both off is bit-identical to pk_attn)
        if (const char * e = getenv("MK_ATTN")) attn_old = atoi(e) == 1;
        if (const char * e = getenv("MK_ATTN_SPLIT")) attn_split = atoi(e) != 0;
        if (const char * e = getenv("MK_ATTN_PV16")) attn_pv16 = atoi(e) != 0;
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < N_ATT_SLOTS; s++)
                if (attn_old && hb.kvb[w][s] != 8) throw std::runtime_error("MK_ATTN=1 (the original prefill attention) reads q8_0 only: no MK_KV");
        // MK_ATTN_S8 (q8 cache, with PV16): 1 = int8 Q K^T, Q scales per 32-dim block (pk_attn8); 2 = per row; 0 = off
        if (const char * e = getenv("MK_ATTN_S8")) attn_s8 = atoi(e);
        if (attn_s8) {
            int occ8 = 0;
            mkl::pk_attn8_occupancy(&occ8);
            if (occ8 != occ) attn_s8 = 0; // the split plan assumes pk_attn2's residency
        }
        // MK_EXACT: pk_attn8's EXA form (absolute key segments of MK_EXACT_SEG positions, default 4096); other prefill
        // attention kernels run unsplit (one online softmax per row over absolute tiles)
        if (exact) {
            attn_split = false;
            if (const char * e = getenv("MK_EXACT_SEG")) ex_segt = std::max(1, atoi(e) / A2_K);
            if (const char * e = getenv("MK_EXACT_SPLIT")) ex_split = atoi(e) != 0;
            mkl::pk_attn8_exa_attr();
        }
        if (k8 & 1) { // MK_K8 bit 1: pk_ab's stream (non-blocking: ordered only by pf_e0 / pf_e1)
            CK(cudaStreamCreateWithFlags(&pf_s2, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&pf_e0, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&pf_e1, cudaEventDisableTiming));
        }
        if (const char * e = getenv("MK_PF_SYNC")) pf_sync = atoi(e) != 0;
        if (const char * e = getenv("MK_PF_GAPS")) pf_gaps = atoi(e) != 0;
        for (cudaEvent_t & e : pf_cev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        mem_mark("prefill kernel attributes");
        pf_ready = true;
    }

    // causal attention of the chunk's T queries (positions n ..) over the cache of attention slot `slot` -> y2
    int a2_slots = 0;
    size_t y1_floats = 0; // hp.y1 in floats (pf_init: cap x FF, or cap x VD in MK_EXACT mode with every gate / up fused)
    // the slot's main buffer holds K q6 / V q5 (the D / D-prime presets) or K q5 / V q5 (preset F, K rotated or not):
    // pk_attn8's compile-time-format instantiations
    bool kv_k6v5(int slot) const { return hb.kvb[0][slot] == 6 && hb.kvb[1][slot] == 5; }
    bool kv_k5v5(int slot) const { return hb.kvb[0][slot] == 5 && hb.kvb[1][slot] == 5; }
    bool attn_old = false, attn_split = true, attn_pv16 = true;
    int attn_s8 = 1;
    // MK_EXACT: key tiles per absolute attention segment (pk_attn8 EXA). 4096 positions (long-context prefills, fresh or
    // after a restore, are faster than with 8192 or 16384)
    int ex_segt = 128;
    // EXA partials, two buffers: y1 (k, v of the layer, dead after pk_attn_prep) as for a2_plan, then xh (the attention
    // input, dead after the q / k / v GEMMs until pk_attn_out). A slot with a KV window ring runs its chunk in pieces
    // (pf_run), and a piece's partials in y1 would overwrite the k / v of the next pieces' rows: there xh2 (the ffn_down
    // input, idle until the FFN, half y1's size), then xh.
    std::pair<float *, float *> ex_part(bool pieces) {
        return {pieces ? reinterpret_cast<float *>(hp.xh2) : hp.y1, reinterpret_cast<float *>(hp.xh)};
    }
    std::pair<size_t, size_t> ex_cap(bool pieces) const { // (partial slots)
        return {(pieces ? (size_t) hp.cap * FF / 2 : y1_floats) / A2_PART, (size_t) hp.cap * D / A2_PART};
    }
    // MK_EXACT work split (pk_attn8 EXA): numerically free (every split gives the same rows), so this only balances.
    // The last N - nfull items (blockIdx order) are split into a head unit (segments 0 .. h - 1, folded in registers ->
    // partial slot 0) and single-segment units (segment h + u - 1 -> slot u), launched interleaved (every head, then
    // every item's segment h, ..) so that the CTA scheduler fills the tail evenly; S = 1 + (most segments of a split item)
    // - h units and partial slots per split item (ex_part). The plan minimises a list-scheduling estimate (a unit
    // takes its key tiles + a per-CTA overhead; the CTAs start in blockIdx order on a2_slots resident slots) over
    // nsplit in {the last partial wave, + one full wave} x h; memoised per (T, n0, capacity). The first nA split items'
    // slots go to the first partial buffer, the others to the second. MK_EXACT_SPLIT=0: whole items only.
    struct ExPlan { int nfull, S, h, nA; };
    std::map<std::tuple<int, int, size_t, size_t>, ExPlan> ex_memo;
    bool ex_split = true;
    ExPlan ex_plan(int T, int n0, std::pair<size_t, size_t> cap) {
        const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV, P = std::max(1, a2_slots), segt = ex_segt;
        if (!ex_split) return {N, 1, 0, 0};
        const auto key = std::make_tuple(T, n0, cap.first, cap.second);
        const auto f = ex_memo.find(key);
        if (f != ex_memo.end()) return f->second;
        if (ex_memo.size() > 4096) ex_memo.clear();
        // costs in key tiles (~3.2 us of one CTA): c0 per CTA (Q fragments, pipeline fill); cf per fold of a segment into R
        // (R.o read + write, 192 KB at a resident CTA's share of DRAM bandwidth); cw per partial slot written; cc per slot
        // read by pk_attn8_combx (the whole GPU's bandwidth, after the attention kernel)
        const double c0 = 2.0, cf = 4.0, cw = 3.0, cc = 0.04;
        std::vector<int> nt(N); // key tiles of each item
        for (int i = 0; i < N; i++) {
            const int t0 = i % nqt * A2_Q;
            nt[i] = (n0 + t0 + std::min(A2_Q, T - t0) - 1) / A2_K + 1;
        }
        std::vector<double> fr; // the slots' busy-until times (min-heap)
        auto sim = [&](int nfull, int S, int h) {
            fr.assign(P, 0.0);
            auto run = [&](double d) {
                std::pop_heap(fr.begin(), fr.end(), std::greater<double>());
                fr.back() += d;
                std::push_heap(fr.begin(), fr.end(), std::greater<double>());
            };
            for (int i = 0; i < nfull; i++) run(nt[i] + c0 + cf * ((nt[i] - 1) / segt));
            const int ns = N - nfull;
            for (int u = 0; u < (ns > 0 ? S : 0); u++)
                for (int j = 0; j < ns; j++) {
                    const int t = nt[nfull + j], k0 = u == 0 ? 0 : h + u - 1, k1 = u == 0 ? h : k0 + 1;
                    if (k0 * segt >= t) {
                        run(0.1); // (idle unit)
                        continue;
                    }
                    const int e = std::min(t, k1 * segt);
                    run(e - k0 * segt + c0 + cw + (u == 0 ? cf * ((e - 1) / segt) : 0.0));
                }
            // + pk_attn8_combx: one more launch, reads S partial slots per split item
            return *std::max_element(fr.begin(), fr.end()) + (ns > 0 ? c0 + cc * ns * S : 0.0);
        };
        ExPlan best{N, 1, 0, 0};
        double bt = sim(N, 1, 0);
        const int rem = N % P;
        for (const int ns : {rem, rem + P}) {
            if (ns <= 0 || ns > N) continue;
            int mx = 0; // the most segments of a split item
            for (int j = N - ns; j < N; j++) mx = std::max(mx, (nt[j] + segt - 1) / segt);
            for (int h = 1; h < mx; h++) {
                const int S = 1 + mx - h;
                const size_t na = cap.first / S; // (split items whose slots fit the first buffer)
                if ((size_t) ns > na + cap.second / S) continue;
                const double t = sim(N - ns, S, h);
                if (t < bt) bt = t, best = {N - ns, S, h, (int) std::min<size_t>(na, ns)};
            }
        }
        ex_memo[key] = best;
        return best;
    }
    // (r0: the queries are chunk rows r0 .. r0 + T - 1 at positions n ..: MK_EXACT with KV window rings, see pf_run)
    void attn(int T, int n, int slot, int r0 = 0) {
        if (attn_old) {
            if (r0) throw std::runtime_error("prefill attention: MK_ATTN=1 (pk_attn) has no row offset (MK_EXACT with MK_KV_WIN)");
            if (hb.kvq8) mkl::pk_attn<true>(T, n, slot);
            else mkl::pk_attn<false>(T, n, slot);
            return;
        }
        if (exact && hb.kvq8 && attn_pv16 && attn_s8 == 1) { // MK_EXACT: pk_attn8's EXA form
            const int N = (T + A2_Q - 1) / A2_Q * NKV;
            const bool pcs = hb.kvrp[0][slot] || hb.kvrp[1][slot]; // (the chunk runs in pieces: pf_run)
            const ExPlan p = ex_plan(T, n, ex_cap(pcs));
            const auto pt = ex_part(pcs);
            const int grid = p.nfull + (N - p.nfull) * p.S;
            if (kv_k6v5(slot))
                mkl::pk_attn8<false, true, true, 6, 5>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
            else if (kv_k5v5(slot))
                mkl::pk_attn8<false, true, true, 5, 5>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
            else if (hb.kvb[0][slot] != 8)
                mkl::pk_attn8<false, true, true>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
            else mkl::pk_attn8<false, false, true>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
            if (N > p.nfull) mkl::pk_attn8_combx(N - p.nfull, T, n, p.nfull, p.S, ex_segt, p.h, r0, pt.first, pt.second, p.nA);
            return;
        }
        // y1 (k, v of the layer) is free between pk_attn_prep and the FFN: partials of split items
        const int N = (T + A2_Q - 1) / A2_Q * NKV;
        int nfull = N, S = 1;
        if (attn_split) a2_plan(T, n, a2_slots, y1_floats, nfull, S);
        if (r0 && N > nfull) // (r0 != 0 only with MK_EXACT, which runs unsplit)
            throw std::runtime_error("prefill attention: the split merge (pk_attn2_comb) has no row offset");
        const int grid = nfull + (N - nfull) * S;
        if (hb.kvq8) {
            // pk_attn8 reads K in any MK_KV format (low-bit tiles are expanded to int8 in its staging) and any V format
            // (a8_cv_v converts). Rotated K too: pk_attn_prep rotates Q (fp32 fwht256) before its fp16 store to qh, so the
            // int8 Q fragments are rotated as well; pk_attn_out un-rotates a rotated V's output for every kernel. MK_ATTN_S8=0
            // keeps pk_attn2 (fp16 Q) for comparison.
            const bool s8 = attn_pv16;
            const bool kl = hb.kvb[0][slot] != 8; // low-bit K: the expanding instantiation (q8 K: the plain kernel)
            if (s8 && attn_s8 == 1) {
                if (kv_k6v5(slot)) mkl::pk_attn8<false, true, false, 6, 5>(grid, T, n, slot, nfull, S, hp.y1, r0);
                else if (kv_k5v5(slot)) mkl::pk_attn8<false, true, false, 5, 5>(grid, T, n, slot, nfull, S, hp.y1, r0);
                else if (kl) mkl::pk_attn8<false, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
                else mkl::pk_attn8<false, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
            } else if (s8 && attn_s8 == 2) {
                if (kl) mkl::pk_attn8<true, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
                else mkl::pk_attn8<true, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
            }
            else if (attn_pv16) mkl::pk_attn2<true, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else mkl::pk_attn2<true, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
        } else {
            if (attn_pv16) mkl::pk_attn2<false, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else mkl::pk_attn2<false, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
        }
        if (N > nfull) mkl::pk_attn2_comb(N - nfull, T, nfull, S, hp.y1);
    }

    // GEMM token tile (pf::mmh NB): every NB gives bit-identical results, so this is a pure speed choice.
    // At 128 a short prompt pads to 128 columns and the CTA is tensor-pipe bound on the padding (42 tokens: 2.1 us
    // per 64-K stage, 86 of 128 columns wasted); 32 / 64 cut the MMA work and add CTAs.
    // The choice is a small cost model over waves of CTAs (nrt row tiles x token tiles) on the nsm SMs, with the
    // per-stage time of a lone CTA at NB = 32 / 64 / 128 (0.80 / 0.95 / 1.44, relative; the model's Q4_K .. Q8_0
    // shapes, microbenchmark). NB 64 / 128 run one CTA per SM (shared memory), NB 32 two (a
    // pair takes 1.77x as long as one CTA). It picks 32 for <= 32 tokens; up to 64 tokens 64, or 32 when the GEMM is
    // narrow (40 row tiles x 2 token tiles still fit one wave); up to 128 tokens 128, or 64 when narrow; above that
    // mostly 128. Chunks of more than 384 tokens always use 128.
    int pf_bn_force = 0; // MK_PF_BN
    int pf_nb(int ntok, int nrt) const {
        if (pf_bn_force) return pf_bn_force;
        if (ntok > 384) return 128;
        const int n32 = nrt * ((ntok + 31) / 32), n64 = nrt * ((ntok + 63) / 64), n128 = nrt * ((ntok + 127) / 128);
        const double t32 = 0.80 * (n32 <= nsm ? 1.0 : 1.77 * ((n32 + 2 * nsm - 1) / (2 * nsm)));
        const double t64 = 0.95 * ((n64 + nsm - 1) / nsm), t128 = 1.44 * ((n128 + nsm - 1) / nsm);
        return t32 <= t64 && t32 <= t128 ? 32 : t64 <= t128 ? 64 : 128;
    }
    template <int TYPE, bool GLU, int LAY = 0> void mmh_attr() { mkl::pf_mmh_attr<TYPE, GLU, LAY>(); }
    // nrt row tiles (CTAs along N) x token tiles of pf_nb tokens; LAY 1: the matrix is in the wv layout
    template <int TYPE, bool GLU, int LAY = 0>
    void mmh_launch(int nrt, const pf::WMat & W, const __half * X, int ntok, const pf::Out & O, const uint8_t * W2 = nullptr) {
        mkl::pf_mmh<TYPE, GLU, LAY>(pf_nb(ntok, nrt), nrt, &W, X, ntok, &O, W2);
    }

    // pf::mmh's shapes: K in pairs of BK-wide stages (the pipelined loop steps two stages; fp16 sums span two), N in whole
    // tiles of `rows` (BM; the fused gate / up: BM / 2 of each). Another shape would skip rows or columns silently.
    static void pf_shape_check(const Mat & m, int rows) {
        if (m.K % (2 * pf::BK) != 0 || m.N % rows != 0)
            throw std::runtime_error("prefill GEMM: unsupported matrix shape K " + std::to_string(m.K) + " x N " + std::to_string(m.N) +
                                     " (needs K % " + std::to_string(2 * pf::BK) + " == 0 and N % " + std::to_string(rows) + " == 0)");
    }
    static_assert(D % (2 * pf::BK) == 0 && 2 * D % (2 * pf::BK) == 0 && FF % (2 * pf::BK) == 0 && VD % (2 * pf::BK) == 0, "prefill GEMM: K");
    static_assert(D % pf::BM == 0 && FF % pf::BM == 0 && VD % pf::BM == 0 && QG % pf::BM == 0 && KVD % pf::BM == 0 && QKV % pf::BM == 0 &&
                      VOCAB % pf::BM == 0,
                  "prefill GEMM: N");
#if MK_PF_INFCHK
    // MK_PF_INFCHK builds: the (layer, site) the counters charge from the next launch on (stream-ordered; < 0: unchanged)
    void chk_set(int layer, int site) {
        if (layer >= 0) CK(mkl::set_chk_layer_async(&layer, sizeof(int), 0, cudaMemcpyHostToDevice, 0));
        if (site >= 0) CK(mkl::set_chk_site_async(&site, sizeof(int), 0, cudaMemcpyHostToDevice, 0));
    }
    void chk_mat(const Mat & m) { // a GEMM's site and layer, from its matrix
        for (int l = 0; l < NLA; l++) {
            const LayerW & W = hw.L[l];
            const Mat * ms[10] = {&W.q, &W.k, &W.v, &W.o, &W.qkv, &W.gate, &W.sout, &W.fg, &W.fu, &W.fd}; // (pf::CHK_Q ..)
            for (int i = 0; i < 10; i++)
                if (m.w && ms[i]->w == m.w) return chk_set(l, i);
        }
        if (m.w == hw.eh.w) chk_set(NL, pf::CHK_EH);
        else chk_set(pf::CHK_LAYERS - 1, pf::CHK_HEAD);
    }
    // prints the counts of the prefill that just finished (device synchronised) and clears them
    void chk_report(int npos) {
        static const char * const nm[pf::CHK_SITES] = {"attn_q", "attn_k", "attn_v", "attn_o", "dn_qkv", "dn_gate", "dn_out", "ffn_gate",
                                                      "ffn_up", "ffn_down", "mtp_eh", "head", "in norm_quant", "in dn_out",
                                                      "in mtp_in", "in attn_gate", "in silu*up"};
        constexpr int NL_ = pf::CHK_LAYERS, NS = pf::CHK_SITES;
        std::vector<unsigned long long> c((size_t) NL_ * NS * 2);
        CK(mkl::get_chk_cnt(c.data(), c.size() * sizeof(unsigned long long)));
        unsigned long long t[2][2] = {{0, 0}, {0, 0}}; // [GEMM | input][kind]
        for (int l = 0; l < NL_; l++)
            for (int s = 0; s < NS; s++)
                for (int k = 0; k < 2; k++) t[s >= pf::CHK_IN_NORM][k] += c[((size_t) l * NS + s) * 2 + k];
        fprintf(stderr, "[mk] MK_PF_INFCHK prefill of %d positions: GEMM fp16 partial sums inf/NaN %llu, non-finite GEMM outputs %llu;"
                        " fp16 GEMM inputs clamped (|v| > 65504) %llu, NaN (stored as -65504) %llu\n", npos, t[0][0], t[0][1], t[1][0], t[1][1]);
        for (int s = 0; s < NS; s++) {
            unsigned long long a[2] = {0, 0};
            std::string where;
            for (int l = 0; l < NL_; l++) {
                const unsigned long long x = c[((size_t) l * NS + s) * 2], y = c[((size_t) l * NS + s) * 2 + 1];
                a[0] += x;
                a[1] += y;
                if ((x || y) && where.size() < 120) where += " L" + std::to_string(l) + ":" + std::to_string(x) + "/" + std::to_string(y);
            }
            if (a[0] || a[1])
                fprintf(stderr, "[mk]   %-14s %s %llu, %s %llu  (layer:counts%s)\n", nm[s], s < pf::CHK_IN_NORM ? "partials" : "clamped", a[0],
                        s < pf::CHK_IN_NORM ? "outputs" : "NaN", a[1], where.c_str());
        }
        std::fill(c.begin(), c.end(), 0ull);
        CK(mkl::set_chk_cnt(c.data(), c.size() * sizeof(unsigned long long)));
    }
#endif
    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0)
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr) {
        pf_shape_check(m, pf::BM);
#if MK_PF_INFCHK
        chk_mat(m);
#endif
        const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
        const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
        const pf::Out O{y, ldy, mode, nullptr};
        const int nrt = m.N / pf::BM;
        if (m.fm) {
            switch (m.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q5K: mmh_launch<pf::TQ5K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q8_0: mmh_launch<pf::TQ80, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ4NL: mmh_launch<pf::TIQ4NL, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q6K: mmh_launch<pf::TQ6K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q3K: mmh_launch<pf::TQ3K, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ3S: mmh_launch<pf::TIQ3S, false, 1>(nrt, W, X, ntok, O); return;
                default: throw std::runtime_error("gemm: unsupported weight type in the wv layout");
            }
        }
        switch (m.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, false>(nrt, W, X, ntok, O); break;
            case T_Q5K: mmh_launch<pf::TQ5K, false>(nrt, W, X, ntok, O); break;
            case T_Q6K: mmh_launch<pf::TQ6K, false>(nrt, W, X, ntok, O); break;
            case T_Q8_0: mmh_launch<pf::TQ80, false>(nrt, W, X, ntok, O); break;
            case T_IQ4X: mmh_launch<pf::TIQ4, false>(nrt, W, X, ntok, O); break;
            case T_IQ4NL: mmh_launch<pf::TIQ4NL, false>(nrt, W, X, ntok, O); break;
            case T_Q3K: mmh_launch<pf::TQ3K, false>(nrt, W, X, ntok, O); break;
            case T_IQ3S: mmh_launch<pf::TIQ3S, false>(nrt, W, X, ntok, O); break;
            default: throw std::runtime_error("gemm: unsupported weight type");
        }
    }

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh: one fused GEMM when gate and
    // up share a type with in-row scales, else two GEMMs into y0 / y1 and pk_glu
    static bool glu_fused(const Mat & g, const Mat & u) {
        if (g.fm != u.fm) return false;
        return g.type == u.type && g.rb == u.rb && (g.type == T_Q4K || g.type == T_Q5K || g.type == T_IQ4X);
    }
    void gemm_glu(const Mat & g, const Mat & u, int ntok) {
        pf_shape_check(g, pf::BM / 2);
        pf_shape_check(u, pf::BM / 2);
#if MK_PF_INFCHK
        chk_mat(g); // (the fused launches; two GEMMs + pk_glu set their own)
#endif
        const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const int nrt = g.N / (pf::BM / 2);
        if (glu_fused(g, u) && g.fm) {
            switch (g.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_Q5K: mmh_launch<pf::TQ5K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                default: break;
            }
        }
        if (glu_fused(g, u)) {
            switch (g.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_Q5K: mmh_launch<pf::TQ5K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                default: break;
            }
        }
        if (glu_mix_on && glu_mix(g, u, ntok)) return;
        gemm(g, ntok, hp.y0, g.N, pf::OUT_STORE);
        gemm(u, ntok, hp.y1, u.N, pf::OUT_STORE);
        mkl::pk_glu(ntok, hp.y0, hp.y1, hp.xh2, g.N);
    }
    // V2: a gate / up pair of different types (or layouts) as one mixed-type fused launch, pf::mmh<TG, true, NB, LG, TU, LU>
    // (warps 0-3 gate rows, 4-7 up rows, each half its own type): the same products, sums and silu * up as the two GEMMs +
    // pk_glu, so bit-identical, without the fp32 y0 / y1 round trip (y1 can then shrink, pf_init). MK_PF_GLU_MIX=0: off.
    // The pairs of UD-Q4_K_XL (gate / up; main layers are in the wv layout except Q3_K); any other pair takes the two-GEMM
    // path.
    bool glu_mix_on = !(getenv("MK_PF_GLU_MIX") && atoi(getenv("MK_PF_GLU_MIX")) == 0);
    template <int TG, int LG, int TU, int LU> void mmh_attr_mix() { mkl::pf_mmh_mix_attr<TG, LG, TU, LU>(); }
    template <int TG, int LG, int TU, int LU> void mmh_launch_mix(int nrt, const pf::WMat & Wg, const pf::WMat & Wu, int ntok, const pf::Out & O) {
        mkl::pf_mmh_mix<TG, LG, TU, LU>(pf_nb(ntok, nrt), nrt, &Wg, &Wu, hp.xh, ntok, &O);
    }
    void glu_mix_attr() {
#define MK_GLU_ATTR(TG, LG, TU, LU) mmh_attr_mix<TG, LG, TU, LU>();
        MK_GLU_MIX_PAIRS(MK_GLU_ATTR)
#undef MK_GLU_ATTR
    }
    // ntok > 0: launch it if the pair has an instantiation; ntok = 0: only whether it has one
    bool glu_mix(const Mat & g, const Mat & u, int ntok) {
        if (glu_fused(g, u) || g.K != u.K || g.N != u.N) return false;
        const pf::WMat Wg{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd}, Wu{u.w, u.d, u.type, u.K, u.N, u.rb, u.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const int nrt = g.N / (pf::BM / 2), gl = g.fm, ul = u.fm;
#define MK_GLU_GO(TG, LG, TU, LU)                                                                                          \
    if (g.type == TG && gl == LG && u.type == TU && ul == LU) {                                                            \
        if (ntok > 0) mmh_launch_mix<TG, LG, TU, LU>(nrt, Wg, Wu, ntok, O);                                                \
        return true;                                                                                                       \
    }
        MK_GLU_MIX_PAIRS(MK_GLU_GO)
#undef MK_GLU_GO
        return false;
    }

    // MK_PF_PROFILE=1: per-category GPU time of the batched prefill (events around each launch)
    struct PfEv { const char * cat; cudaEvent_t a, b; };
    std::vector<PfEv> pf_ev;
    bool pf_prof = getenv("MK_PF_PROFILE") != nullptr;
    // K8 (MK_K8 bits, read at init): 1 pk_ab on pf_s2 (after pf_e0 = the layer's pk_norm_quant; pk_dn_prep waits for
    // pf_e1 = pk_ab), 2 pk_norm_quant1, 4 pk_dn_prep_t
    int k8 = MK_K8_DEF;
    cudaStream_t pf_s2 = nullptr;
    cudaEvent_t pf_e0 = nullptr, pf_e1 = nullptr;
    void pf_norm_quant(const float * src, const float * w, int T) {
        if (k8 & 2) mkl::pk_norm_quant1(T, src, w);
        else mkl::pk_norm_quant(T, src, w);
    }
    void pf_dn_prep(int T, int n, int dl, const LayerW & W) {
        if (k8 & 4) {
            if (k8_nt == 8) mkl::pk_dn_prep_t<8>(T, n, dl, W.conv, W.dt, W.A);
            else mkl::pk_dn_prep_t<4>(T, n, dl, W.conv, W.dt, W.A);
        } else {
            mkl::pk_dn_prep(T, n, dl, W.conv, W.dt, W.A);
        }
    }
    int k8_nt = MK_K8_NT; // (MK_K8_NT env: 4 or 8)
    template <class F> void pf_time(const char * cat, F f) {
        if (!pf_prof) { f(); return; }
        // MK_PF_PROFILE=2: idempotent kernels are timed alone (5 repeats, min)
        static const char * idem[] = {"attn_prep", "attn_out", "dn_out", "norm_quant", "dn_ab", "attn", "embed", "gemm_ffn_gu", "gemm_ffn_gum", "gemm_ffn_gu2", "gemm_dn_in", "gemm_attn_qkv"};
        if (getenv("MK_PF_PROFILE")[0] == '2' && std::any_of(std::begin(idem), std::end(idem), [&](const char * c) { return !strcmp(c, cat); })) {
            float best = 1e30f;
            for (int r = 0; r < 5; r++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                f();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
            pf_iso[cat] += best;
            return;
        }
        PfEv e{cat, nullptr, nullptr};
        CK(cudaEventCreate(&e.a));
        CK(cudaEventCreate(&e.b));
        CK(cudaEventRecord(e.a));
        f();
        CK(cudaEventRecord(e.b));
        pf_ev.push_back(e);
    }
    std::map<std::string, double> pf_iso;
    void pf_report() {
        if (!pf_prof || pf_ev.empty()) return;
        CK(cudaDeviceSynchronize());
        // per category: sum, and median-per-launch x launches (robust to time-slicing by other GPU users)
        std::vector<std::string> cats;
        std::vector<std::vector<float>> ts;
        for (auto & e : pf_ev) {
            float ms;
            CK(cudaEventElapsedTime(&ms, e.a, e.b));
            auto it = std::find(cats.begin(), cats.end(), std::string(e.cat));
            if (it == cats.end()) { cats.push_back(e.cat); ts.push_back({}); it = cats.end() - 1; }
            ts[it - cats.begin()].push_back(ms);
            cudaEventDestroy(e.a);
            cudaEventDestroy(e.b);
        }
        pf_ev.clear();
        double all = 0, all_med = 0;
        std::vector<std::tuple<double, double, std::string, size_t>> rows;
        for (size_t i = 0; i < cats.size(); i++) {
            auto v = ts[i];
            double sum = 0;
            for (float x : v) sum += x;
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2] * v.size();
            rows.emplace_back(med, sum, cats[i], v.size());
            all += sum;
            all_med += med;
        }
        for (auto & kv : pf_iso) fprintf(stderr, "  [isolated] %-12s %8.1f ms\n", kv.first.c_str(), kv.second);
        pf_iso.clear();
        std::sort(rows.rbegin(), rows.rend());
        fprintf(stderr, "[mk-prefill-profile] %.1f ms in kernels (median-based %.1f ms)\n", all, all_med);
        for (auto & r : rows)
            fprintf(stderr, "  %-14s median-based %8.1f ms %5.1f%%   sum %8.1f ms   x%zu\n", std::get<2>(r).c_str(), std::get<0>(r),
                    100 * std::get<0>(r) / all_med, std::get<1>(r), std::get<3>(r));
    }

    // Runs positions n0 .. n0 + toks.size() - 1 through the trunk (and the MTP block's K/V).
    // score: instead of MTP, greedy argmax of every position -> pred, and / or the logits of the
    // positions >= first -> sink.
    // snaps: checkpoints to save on the way (positions n0 < P <= n0 + toks.size(), ascending, slots from ck_assign;
    // at most stg_n per chunk: ck_positions); complete when pf_run returns.
    // erow (images, aligned with toks): each token's row in emb_host, -1 for a token_embd row; chunks with an image row
    // run pk_embed_img / pk_mtp_in<true>, the others the text kernels.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0, const std::vector<Snap> * snaps = nullptr, const int * erow = nullptr) {
        pf_init();
        const int kv_top0 = kv_top;
        sw_need(n0 + (int) toks.size()); // (side request)
        kv_top = std::max(kv_top, n0 + (int) toks.size()); // (ring_rewind)
        pf_cancelled = false;
        ck_scr_guard(); // the scratch may still be the source of an end save's copy
        ck_guard_state();
        const int N = (int) toks.size(), cap = hp.cap;
        // chunk c's GPU work is enqueued before the host waits for chunk c - 1 (its pf_cev event), so the GPU does not
        // idle while the host wakes up and launches; the scoring paths (pred / sink) read results back per chunk
        const bool ahead = !pf_sync && !pred && !sink;
        const int nch = (N + cap - 1) / cap;
        if (pf_gaps) // 4 timing events per chunk: start, trunk end, MTP part start, end
            while (pf_gev.size() < (size_t) 4 * nch) {
                cudaEvent_t e;
                CK(cudaEventCreate(&e));
                pf_gev.push_back(e);
            }
        auto gap_mark = [&](int c, int k) {
            if (pf_gaps) CK(cudaEventRecord(pf_gev[4 * c + k], 0));
        };
        for (int c0 = 0; c0 < N; c0 += cap) {
            // MKEngine::cancel: stop at this chunk boundary (a host flag, no device sync; the chunks already enqueued finish
            // in the tail wait below). The caller (MKEngine::prefill) discards what they computed.
            if (c0 > 0 && !pred && !sink && pf_cancel.load(std::memory_order_relaxed)) {
                pf_cancelled = true;
                pf_stop = n0 + c0;
                kv_top = std::max(kv_top0, n0 + c0); // (rows of the chunks never enqueued were not written)
                break;
            }
#if MK_KV_EMU
            kv_demote_to(n0 + c0 - hb.kvemu_win); // recent-window emulation: rows older than the window before this chunk
#endif
            const int T = std::min(cap, N - c0), n = n0 + c0, ci = c0 / cap;
            int sn = 0, sp[STG_MAX], ss[STG_MAX]; // this chunk's checkpoints (n < P <= n + T) and their slots
            if (snaps && !pred && !sink)
                for (const Snap & q : *snaps)
                    if (q.pos > n && q.pos <= n + T && sn < stg_n) sp[sn] = q.pos, ss[sn++] = q.slot;
            const bool img = erow && std::any_of(erow + c0, erow + c0 + T, [](int r) { return r >= 0; });
            if (ahead) { // stream-ordered behind the previous chunk (pageable source: staged at the call, no device wait)
                CK(cudaMemcpyAsync(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice, 0));
                if (img) CK(cudaMemcpyAsync(hp.erow, erow + c0, T * sizeof(int), cudaMemcpyHostToDevice, 0));
            } else {
                CK(cudaMemcpy(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice));
                if (img) CK(cudaMemcpy(hp.erow, erow + c0, T * sizeof(int), cudaMemcpyHostToDevice));
            }
            ck_issue_restore(); // a due checkpoint restore streams in from here on (waited for per layer group below)
            ring_snap_guard();  // (MK_EXACT: a previous chunk's ring row copies, before this chunk's preps)
            const bool rsnap = sn > 0 && ring_bytes_ck > 0; // MK_EXACT + MK_KV_WIN: the checkpoints keep the rings
            for (int j = 0; j < sn; j++) ck[ss[j]].ring_ok = false;
            if (rsnap) ring_snap_begin(sn, ss);
            gap_mark(ci, 0); // (after the token upload: MK_PF_SYNC's idle is before it)
            pf_time("embed", [&] { // (token_embd's type)
                if (hw.embd_t == T_Q6K) {
                    if (img) mkl::pk_embed_img<T_Q6K>(T);
                    else mkl::pk_embed<T_Q6K>(T);
                } else if (hw.embd_t == T_Q8_0) {
                    if (img) mkl::pk_embed_img<T_Q8_0>(T);
                    else mkl::pk_embed<T_Q8_0>(T);
                } else {
                    if (img) mkl::pk_embed_img<T_Q4K>(T);
                    else mkl::pk_embed<T_Q4K>(T);
                }
            });
            int dl = 0;
            for (int L = 0; L < NL; L++) {
                const LayerW & W = hw.L[L];
#if MK_PF_INFCHK
                chk_set(L, -1); // (this layer's input stores; its GEMMs set their own site)
#endif
                if (fault_now && L == NL / 2) fault_inject();
                pf_time("norm_quant", [&] { pf_norm_quant(hp.x, W.attn_norm, T); });
                if (W.attn) {
                    pf_time("gemm_attn_qkv", [&] { gemm(W.q, T, hp.y0, QG, pf::OUT_STORE); });
                    pf_time("gemm_attn_qkv", [&] { gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE); gemm(W.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE); });
                    const int sl = L / 4;
                    if (rsnap) CK(cudaStreamWaitEvent(0, rs_ev[sl], 0)); // (the rings' whole copies are done)
                    if (exact && (hb.kvrp[0][sl] || hb.kvrp[1][sl])) {
                        // MK_EXACT with a KV window ring: the chunk in pieces ending at multiples of ATT_BUCKET (as decode
                        // launches end): a piece's rows read the window of their bucket (PBufs::wsrow), which the ring
                        // (kvrc rows) still holds after the piece's own rows are stored; a whole chunk's rows would have
                        // overwritten the oldest window rows of its first queries
                        for (int a = 0; a < T;) {
                            const int b = std::min(T, ((n + a) / ATT_BUCKET + 1) * ATT_BUCKET - n);
                            pf_time("attn_prep", [&] { mkl::pk_attn_prep(b - a, n, sl, W.qn, W.kn, 1, a); });
                            pf_time("attn", [&] { attn(b - a, n + a, sl, a); });
                            a = b;
                        }
                    } else {
                        pf_time("attn_prep", [&] { mkl::pk_attn_prep(T, n, sl, W.qn, W.kn, 1); });
                        pf_time("attn", [&] { attn(T, n, sl); });
                    }
                    if (rsnap) ring_snap_rows(sl, n, sn, sp, ss);
                    pf_time("attn_out", [&] { mkl::pk_attn_out(T, L / 4); });
                    pf_time("gemm_attn_o", [&] { gemm(W.o, T, hp.x, D, pf::OUT_ADD); });
                } else {
                    // MK_K8 bit 1: pk_ab (reads x and rs, writes ab: only pk_dn_prep reads it) on pf_s2 next to the GEMMs, which
                    // read xh; pk_dn_prep waits for it. (MK_PF_PROFILE: in order on stream 0, so it is timed alone.)
                    const bool k8a = (k8 & 1) && pf_s2 && !pf_prof;
                    if (k8a) {
                        CK(cudaEventRecord(pf_e0, 0));
                        CK(cudaStreamWaitEvent(pf_s2, pf_e0, 0));
                        mkl::pk_ab(W.ab, W.abq, W.abd, W.attn_norm, T, pf_s2);
                        CK(cudaEventRecord(pf_e1, pf_s2));
                    }
                    pf_time("gemm_dn_in", [&] { gemm(W.qkv, T, hp.y0, QKV, pf::OUT_STORE); gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE); });
                    if (k8a) CK(cudaStreamWaitEvent(0, pf_e1, 0));
                    else pf_time("dn_ab", [&] { mkl::pk_ab(W.ab, W.abq, W.abd, W.attn_norm, T); });
                    ck_wait_restore(dl); // this layer's conv ring / state may still be arriving from the checkpoint
                    int sg[STG_MAX], ta = -1; // staging slots of this layer's checkpoint pieces (+ a side scan's input)
                    for (int j = 0; j < sn; j++) sg[j] = stg_take();
                    for (int j = 0; j < sn; j++)
                        if ((sp[j] - n) % 32 && sp[j] - n > 32) ta = stg_take();
                    pf_time("dn_prep", [&] {
                        pf_dn_prep(T, n, dl, W);
                        for (int j = 0; j < sn; j++) mkl::pk_ck_conv(n, sp[j], dl, stg_conv(sg[j]));
                        mkl::pk_conv_ring(T, n, dl);
                    });
                    pf_time("dn_scan", [&] {
                        // The scan is split only at points of its 32-token grid, which changes no arithmetic: the state
                        // at an on-grid checkpoint goes to its staging slot on the way. An off-grid one (the exact
                        // ck_at) is computed on the side: a scan of the rows from the grid point before it, from the
                        // state there, into its staging slot (= what a prefill ending at it computes); the main scan
                        // then rewrites those rows' y2.
                        float * live = hb.dstate + (size_t) dl * CK_SL;
                        const float * in = live;
                        int t = 0;
                        for (int j = 0; j < sn; j++) {
                            const int te = sp[j] - n, tg = te / 32 * 32;
                            if (te == tg) {
                                mkl::pk_dn_scan(tg - t, t, in, stg_state(sg[j]));
                                in = stg_state(sg[j]);
                                t = tg;
                                continue;
                            }
                            if (tg > t) {
                                mkl::pk_dn_scan(tg - t, t, in, stg_state(ta));
                                in = stg_state(ta);
                                t = tg;
                            }
                            mkl::pk_dn_scan(te - tg, tg, in, stg_state(sg[j]));
                        }
                        mkl::pk_dn_scan(T - t, t, in, live);
                    });
                    if (sn) stg_flush(dl, sn, sg, ss);
                    pf_time("dn_out", [&] { mkl::pk_dn_out(T, W.snorm); });
                    pf_time("gemm_dn_out", [&] { gemm(W.sout, T, hp.x, D, pf::OUT_ADD); });
                    dl++;
                }
                pf_time("norm_quant", [&] { pf_norm_quant(hp.x, W.post_norm, T); });
                // (profile: mixed-type fused gate / up pairs are reported as gemm_ffn_gum, pairs that run as two GEMMs +
                // pk_glu as gemm_ffn_gu2)
                pf_time(glu_fused(W.fg, W.fu) ? "gemm_ffn_gu" : glu_mix_on && glu_mix(W.fg, W.fu, 0) ? "gemm_ffn_gum" : "gemm_ffn_gu2",
                        [&] { gemm_glu(W.fg, W.fu, T); });
                pf_time("gemm_ffn_down", [&] { gemm(W.fd, T, hp.x, D, pf::OUT_ADD, 0, hp.xh2); });
            }
            wd_T = T;
            wd_mode = MODE_PREFILL;
            gap_mark(ci, 1);
            if (!ahead) wait_device("a prefill chunk", std::max(wd_limit, 120.0));
            gap_mark(ci, 2);
            if (pred || sink) {
#if MK_PF_INFCHK
                chk_set(pf::CHK_LAYERS - 1, -1);
#endif
                pf_norm_quant(hp.x, hw.onorm, T);
                std::vector<float> rows;
                const int tile = std::min(128, cap * FF / VOCAB); // logits rows that fit in y0
                for (int s = 0; s < T; s += tile) {
                    const int nt = std::min(tile, T - s);
                    const int lo = std::max(s, first - n); // rows from lo on are at positions >= first
                    if (!pred && lo >= s + nt) continue;
                    gemm(hw.head, nt, hp.y0, VOCAB, pf::OUT_STORE, s);
                    if (pred) mkl::pk_argmax(nt, s);
                    if (sink && lo < s + nt) {
                        rows.resize((size_t) (s + nt - lo) * VOCAB);
                        CK(cudaMemcpy(rows.data(), hp.y0 + (size_t) (lo - s) * VOCAB, rows.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        (*sink)(n + lo, s + nt - lo, rows.data());
                    }
                }
                if (pred) {
                    std::vector<int> am(T);
                    CK(cudaMemcpy(am.data(), hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                    pred->insert(pred->end(), am.begin(), am.end());
                }
                continue;
            }
            if (c0 == 0) { // hp.h row 0: the hidden state before the chunk, read only by pk_mtp_in (after every restore wait)
                if (n0 == 0) CK(cudaMemsetAsync(hp.h, 0, D * sizeof(float), 0));
                else CK(cudaMemcpyAsync(hp.h, hb.h_hist + (size_t) ((n0 - 1) % H_RING) * D, D * sizeof(float), cudaMemcpyDeviceToDevice, 0));
            }
            stg_drain(); // (the MTP part writes hp.xm)
            for (int j = 0; j < sn; j++) { // h ring at each checkpoint: positions before the chunk, from the ring now
                if (ck[ss[j]].pending) CK(cudaStreamWaitEvent(0, ck[ss[j]].ev, 0)); // (an earlier save into the slot)
                if (sp[j] - H_RING < n) mkl::pk_ck_hist(n - (sp[j] - H_RING), sp[j] - H_RING, n, 1, ck[ss[j]].hdev);
            }
            pf_time("mtp", [&] {
                mkl::pk_final_h(T, n);
                for (int j = 0; j < sn; j++) { // ... and positions of the chunk, from its h rows
                    const int lo = std::max(n, sp[j] - H_RING);
                    mkl::pk_ck_hist(sp[j] - lo, lo, n, 0, ck[ss[j]].hdev);
                }
                if (!has_mtp) return; // (no MTP block: no MTP K / V rows)
#if MK_PF_INFCHK
                chk_set(NL, -1);
#endif
                if (hw.embd_t == T_Q6K) {
                    if (img) mkl::pk_mtp_in<true, T_Q6K>(T);
                    else mkl::pk_mtp_in<false, T_Q6K>(T);
                } else if (hw.embd_t == T_Q8_0) {
                    if (img) mkl::pk_mtp_in<true, T_Q8_0>(T);
                    else mkl::pk_mtp_in<false, T_Q8_0>(T);
                } else {
                    if (img) mkl::pk_mtp_in<true, T_Q4K>(T);
                    else mkl::pk_mtp_in<false, T_Q4K>(T);
                }
                gemm(hw.eh, T, hp.xm, D, pf::OUT_STORE);
                const LayerW & M = hw.L[NL];
                pf_norm_quant(hp.xm, M.attn_norm, T);
                gemm(M.k, T, hp.y1, 2 * KVD, pf::OUT_STORE);
                gemm(M.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE);
                if (rsnap) CK(cudaStreamWaitEvent(0, rs_ev[N_ATT_SLOTS - 1], 0));
                mkl::pk_attn_prep(T, n, 16, M.qn, M.kn, 0);
                if (rsnap) ring_snap_rows(N_ATT_SLOTS - 1, n, sn, sp, ss);
            });
            CK(cudaMemcpyAsync(hp.h, hp.h + (size_t) T * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
            gap_mark(ci, 3);
            if (ahead) { // (the watchdog still covers every chunk, one behind)
                CK(cudaEventRecord(pf_cev[ci & 1], 0));
                if (ci > 0) wait_event(pf_cev[(ci - 1) & 1], "a prefill chunk", std::max(wd_limit, 120.0));
            }
            pf_pos.store(n, std::memory_order_relaxed); // (the chunks before this one are done: waited for above)
        }
        if (snaps && !pred && !sink)
            for (const Snap & q : *snaps) { // (its last piece was copied before the last chunk's MTP part)
                CK(cudaEventRecord(ck[q.slot].ev, ck_sv));
                ck[q.slot].pending = true;
            }
        CK(cudaGetLastError());
        wait_device("the prefill tail", std::max(wd_limit, 120.0));
        if (!pf_cancelled) pf_pos.store(n0 + N, std::memory_order_relaxed);
        if (memlog && pf_runs_seen++ == 0) mem_mark("first prefill run");
        if (pf_gaps && !pred && !sink && !pf_cancelled) { // GPU idle at the host's points: before a chunk's first kernel, before its MTP part
            double g0 = 0, g1 = 0, wall = 0;
            float ms;
            for (int c = 0; c < nch; c++) {
                if (c > 0) {
                    CK(cudaEventElapsedTime(&ms, pf_gev[4 * (c - 1) + 3], pf_gev[4 * c]));
                    g0 += ms;
                }
                CK(cudaEventElapsedTime(&ms, pf_gev[4 * c + 1], pf_gev[4 * c + 2]));
                g1 += ms;
            }
            CK(cudaEventElapsedTime(&ms, pf_gev[0], pf_gev[4 * (nch - 1) + 3]));
            wall = ms;
            fprintf(stderr, "[mk] prefill gaps (%s): %d chunks, %.1f ms on the GPU; idle before chunks %.2f ms, before MTP parts %.2f ms\n",
                    ahead ? "ahead" : "MK_PF_SYNC", nch, wall, g0, g1);
        }
#if MK_PF_INFCHK
        chk_report(N);
#endif
        pf_report();
    }

    // Checkpoint positions for a batched part n0 .. hend - 1 (pf_run chunks of hp.cap from n0): the requested P with
    // n0 < P <= hend, `exact` as it is (saved by a side scan), every other one moved down onto the DeltaNet scan's
    // 32-token grid from its chunk's start; ascending, without duplicates or existing checkpoints, at most stg_n per
    // chunk (one less with `exact`, whose side scan may need a staging slot; dropping the earliest).
    // pins: pin requests (pin_req) to save too, never dropped; *vpin marks their positions in the result.
    std::vector<int> ck_positions(const std::vector<int> & cks, int exact, int n0, int hend, const std::vector<int> & pins = {},
                                  std::vector<char> * vpin = nullptr) {
        const int cap = hp.cap;
        auto chunk = [&](int P) { return (P - n0 - 1) / cap; }; // the chunk P belongs to (n < P <= n + T)
        auto at = [&](int P) { // where P is saved (-1: outside this part or on an existing checkpoint)
            if (P <= n0 || P > hend) return -1;
            if (P != exact) {
                const int cs = n0 + chunk(P) * cap;
                P = cs + (P - cs) / 32 * 32;
                if (P <= n0) return -1;
            }
            return ck_find(P) < 0 ? P : -1;
        };
        std::vector<int> v, pv;
        for (int P : cks)
            if ((P = at(P)) >= 0) v.push_back(P);
        for (int P : pins)
            if ((P = at(P)) >= 0) v.push_back(P), pv.push_back(P);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        auto kept = [&](int P) { return P == exact || std::find(pv.begin(), pv.end(), P) != pv.end(); };
        for (size_t i = 0; i < v.size();) { // per chunk: at most stg_n staging slots
            size_t j = i;
            while (j < v.size() && chunk(v[j]) == chunk(v[i])) j++;
            const bool ex = std::find(v.begin() + i, v.begin() + j, exact) != v.begin() + j;
            int extra = (int) (j - i) + (ex ? 1 : 0) - stg_n;
            for (int pass = 0; pass < 2; pass++) // (a pin only when nothing else is left)
                for (size_t k = i; k < j && extra > 0;)
                    if (v[k] != exact && (pass || !kept(v[k]))) v.erase(v.begin() + k), j--, extra--;
                    else k++;
            i = j;
        }
        if (vpin) {
            vpin->assign(v.size(), 0);
            for (size_t k = 0; k < v.size(); k++) (*vpin)[k] = std::find(pv.begin(), pv.end(), v[k]) != pv.end();
        }
        return v;
    }

    // cB.logits ([MAXT][VOCAB] fp32, 5 MB): only scoring launches (Launch::score == 2) write it
    void logits_alloc() {
        if (hb.logits) return;
        hb.logits = dalloc<float>((size_t) MAXT * VOCAB);
        CK(mkl::set_cBs(&hb.logits, sizeof(hb.logits), live_view * sizeof(Bufs) + offsetof(Bufs, logits)));
        mem_mark("scoring logits");
    }

    // presence / frequency penalty (Launch::pres / freq): the response's token counts (cB.pcnt), zeroed by the next
    // prefill once a penalised launch has counted into them
    float pen_pres = 0.f, pen_freq = 0.f;
    bool pcnt_dirty = false;
    // MK_KV_EMU_SPEC / set_kv_emu (MK_KV_EMU builds): comma-separated <K|V|*><slot>=<bits>[flags] and sink=N. slot
    // 0..15 (attention layer 4 slot + 3), 16 (MTP) or * / empty (all). Flags (kv_store_emu): r Hadamard rotation,
    // s random signs + rotation, a q4_0-style scale, l Lloyd-Max codebook with rms scale (bits 2..6), n one scale per
    // 256-value row (else per 32). "8" without flags = plain q8. sink=N: positions < N stay q8.
    void kv_emu_parse(const std::string & s) {
#if MK_KV_EMU
        for (int w = 0; w < 2; w++)
            for (int q = 0; q < N_ATT_SLOTS; q++) hb.kvemu[w][q] = 0;
        hb.kvemu_sink = 0;
        hb.kvemu_win = 0;
        demoted = 0;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            const std::string tok = s.substr(i, j - i);
            i = j + 1;
            if (tok.empty()) continue;
            if (tok.compare(0, 5, "sink=") == 0) {
                hb.kvemu_sink = std::stoi(tok.substr(5));
                continue;
            }
            if (tok.compare(0, 4, "win=") == 0) {
                hb.kvemu_win = std::stoi(tok.substr(4));
                continue;
            }
            const size_t e = tok.find('=');
            if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV_EMU_SPEC: bad item " + tok);
            const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
            int bits = 0, mode = 0;
            size_t k = 0;
            while (k < md.size() && isdigit((unsigned char) md[k])) bits = 10 * bits + (md[k++] - '0');
            for (; k < md.size(); k++) {
                if (md[k] == 'r') mode |= 16;
                else if (md[k] == 'a') mode |= 32;
                else if (md[k] == 'l') mode |= 64;
                else if (md[k] == 'n') mode |= 128;
                else if (md[k] == 's') mode |= 256 | 16;
                else if (md[k] == 't') mode |= 2048 | 256 | 16;
                else if (md[k] == 'T') mode |= 4096 | 256 | 16;
                else throw std::runtime_error("MK_KV_EMU_SPEC: bad mode " + md);
            }
            if (bits < 2 || bits > 8) throw std::runtime_error("MK_KV_EMU_SPEC: bits must be 2..8 in " + tok);
            if ((mode & 64) && (bits > 6 || (mode & 32))) throw std::runtime_error("MK_KV_EMU_SPEC: l needs 2..6 bits and no a: " + tok);
            if ((mode & (2048 | 4096)) && (mode & (32 | 64 | 128))) throw std::runtime_error("MK_KV_EMU_SPEC: t / T take no a, l, n: " + tok);
            mode |= bits;
            const bool all = sl.empty() || sl == "*";
            const int slot = all ? -1 : std::stoi(sl);
            if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV_EMU_SPEC: bad slot in " + tok);
            for (int w = 0; w < 2; w++)
                if (tok[0] == '*' || tok[0] == "KV"[w])
                    for (int q = 0; q < kv_slots; q++) // (no MTP block: its slot stays 0, not allocated)
                        if (all || q == slot) hb.kvemu[w][q] = mode == 8 ? 0 : mode;
        }
        fprintf(stderr, "[mk] MK_KV_EMU_SPEC %s: K", s.c_str());
        for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[0][q]);
        fprintf(stderr, " | V");
        for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[1][q]);
        fprintf(stderr, " | sink %d win %d\n", hb.kvemu_sink, hb.kvemu_win);
        if (!hb.kvq8) throw std::runtime_error("MK_KV_EMU_SPEC needs the q8 KV cache (--kv q8)");
        for (int w = 0; w < 2; w++)
            for (int q = 0; q < N_ATT_SLOTS; q++)
                if (hb.kvb[w][q] != 8 || hb.kvr[q] || hb.kvrv[q]) throw std::runtime_error("MK_KV_EMU_SPEC needs an all-q8 cache (no MK_KV)");
#else
        (void) s;
        throw std::runtime_error("MK_KV_EMU_SPEC needs a build with -DMK_KV_EMU=1");
#endif
    }
    void set_kv_emu(const std::string & spec) { // between sequences (no kernel in flight): new spec, cB re-uploaded
        kv_emu_parse(spec);
#if MK_KV_EMU
        CK(cudaDeviceSynchronize());
        CK(mkl::set_cBs(&hb, sizeof(Bufs), live_view * sizeof(Bufs)));
#endif
    }
    int demoted = 0; // recent-window emulation: positions < demoted have been demoted to their slot's mode
    void kv_demote_to(int upto) { // demote everything older than `upto` (positions demoted .. upto - 1)
#if MK_KV_EMU
        if (hb.kvemu_win <= 0 || upto <= demoted) return;
        CK(cudaDeviceSynchronize());
        mkl::k_kv_demote(demoted, upto);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        demoted = upto;
#else
        (void) upto;
#endif
    }
    void set_penalty(float pres, float freq) {
        if (!std::isfinite(pres) || !std::isfinite(freq)) throw std::runtime_error("penalties must be finite");
        if ((pres != 0.f || freq != 0.f) && !hb.pcnt) {
            // (hb.pcnt only after the kernels' copy: a failed copy leaves both unset, and the next call retries)
            decltype(hb.pcnt) pc = dalloc<unsigned short>(VOCAB, DA_INIT0); // (the first response's counts: these zeros)
            CK(mkl::set_cBs(&pc, sizeof(pc), live_view * sizeof(Bufs) + offsetof(Bufs, pcnt)));
            hb.pcnt = pc;
            mem_mark("penalty counts");
        }
        pen_pres = pres;
        pen_freq = freq;
    }
    void pcnt_clear() {
        if (!pcnt_dirty) return;
        CK(cudaMemsetAsync(hb.pcnt, 0, (size_t) VOCAB * sizeof(unsigned short)));
        pcnt_dirty = false;
    }

    // ---- post-prefill state files (MKEngine::prefill_state) ----
    // A file holds what a batched prefill of positions 0 .. hend - 1 leaves for the megakernel launch of the prompt's last
    // token: every attention slot's K / V rows < hend (values + scales, or fp16), the MK_KV_WIN rings, the DeltaNet states,
    // conv rings and the h ring; nothing else of the sequence state is read before it is rewritten (the restore runs no
    // checkpoint, and prefill() then sets ctl / seq / the draft vocabulary as after the batched part). Named by a hash of
    // the key text (stored in the file and compared on load): the binary's bytes, the model file, ctx, the KV format, the
    // prefill chunk cap and every MK_* variable not known to be decode-only or diagnostic.
    std::string st_dir, model_path;
    int st_how = 0;
    double st_s = 0;
    static unsigned long long file_hash(const std::string & path) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return 0;
        std::vector<uint8_t> b((size_t) 8 << 20);
        unsigned long long h = 0xcbf29ce484222325ULL;
        for (ssize_t r; (r = read(fd, b.data(), b.size())) > 0;) h = wd_hash(b.data(), (size_t) r, h);
        close(fd);
        return h;
    }
    // the object file holding this code (the CLI binary, or libqwen38mk.so under the server), from /proc/self/maps
    static unsigned long long self_hash() {
        static unsigned long long h = 0;
        if (h) return h;
#ifdef _WIN32
        char mp[PATH_MAX];
        std::string path = mk_win_module_path(reinterpret_cast<const void *>(&self_hash), mp, sizeof mp) ? mp : "(module)";
#else
        const uintptr_t a = reinterpret_cast<uintptr_t>(&self_hash);
        std::string path = "/proc/self/exe";
        if (FILE * m = fopen("/proc/self/maps", "r")) {
            char line[4096];
            while (fgets(line, sizeof line, m)) {
                unsigned long lo, hi;
                if (sscanf(line, "%lx-%lx", &lo, &hi) != 2 || a < lo || a >= hi) continue;
                const char * p = strchr(line, '/');
                if (p) {
                    path = p;
                    while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
                }
                break;
            }
            fclose(m);
        }
#endif
        h = file_hash(path);
        if (!h) throw std::runtime_error("state files: cannot read " + path + " to identify the build");
        return h;
    }
    static std::string hex64(unsigned long long v) {
        char b[24];
        snprintf(b, sizeof b, "%016llx", v);
        return b;
    }
    void st_name(const std::vector<int> & head, std::string & path, std::string & key) {
        std::ostringstream k;
        k << "mkstate 1 bin " << hex64(self_hash());
        struct stat st{};
        char rp[PATH_MAX];
        if (!realpath(model_path.c_str(), rp)) snprintf(rp, sizeof rp, "%s", model_path.c_str());
        stat(rp, &st);
        const size_t edge = std::min<size_t>(file.size, (size_t) 4 << 20);
        unsigned long long mh = wd_hash(file.base, edge);
        mh = wd_hash(file.base + file.size - edge, edge, mh);
#ifdef _WIN32
        k << " model " << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtime << ".0"
#else
        k << " model " << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtim.tv_sec << "." << (long long) st.st_mtim.tv_nsec
#endif
          << " " << hex64(mh);
        k << " mtp " << mtp_desc; // (the MTP block's weights decide its KV slot; none: no slot)
        k << " ctx " << ctx << " kvq8 " << hb.kvq8 << " kvb";
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < N_ATT_SLOTS; s++) k << " " << hb.kvb[w][s] << (w ? hb.kvrv[s] : hb.kvr[s]);
        k << " win " << hb.kvwin << " " << hb.kvrc << " exact " << exact << " cap " << hp.cap << " slots " << N_ATT_SLOTS << " nkv " << NKV
          << " hring " << H_RING << " conv " << CONV_RING;
        // environment: every MK_* setting except those that only change decode, logging, timing or the load path
        static const char * const skip[] = {"MK_STATE", "MK_POLICY", "MK_PROFILE", "MK_SPEC_LOG", "MK_MEMLOG", "MK_LOADLOG", "MK_LOAD_",
                                            "MK_WDIGEST", "MK_WATCHDOG", "MK_CK_SLOTS", "MK_BENCH", "MK_STACK_LIMIT",
                                            "MK_K1=", "MK_K2=", "MK_GSB", // (K1 / K2 / gsync bench: decode only)
                                            "MK_K6="};                    // (K6: decode only; MK_K8 = prefill: kept)
        std::vector<std::string> env;
        for (char ** e = environ; *e; e++) {
            if (strncmp(*e, "MK_", 3)) continue;
            bool sk = false;
            for (const char * s : skip) sk = sk || !strncmp(*e, s, strlen(s));
            if (!sk) env.push_back(*e);
        }
        std::sort(env.begin(), env.end());
        for (const auto & e : env) k << " env " << e;
        k << " tokens " << head.size() << " " << hex64(wd_hash(reinterpret_cast<const uint8_t *>(head.data()), head.size() * sizeof(int)));
        key = k.str();
        path = st_dir + "/" + hex64(wd_hash(reinterpret_cast<const uint8_t *>(key.data()), key.size())) + ".mks";
    }
    // the device regions of the state after a batched prefill of 0 .. hend - 1 (pointer, bytes), in file order
    std::vector<std::pair<char *, size_t>> st_regions(int hend) {
        std::vector<std::pair<char *, size_t>> r;
        const size_t cx = (size_t) hb.ctx, n = (size_t) hend; // (row stride per kv head)
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++)
                for (int g = 0; g < NKV; g++) {
                    if (hb.kvq8) {
                        const size_t rb = (size_t) HD * hb.kvb[w][s] / 8;
                        r.push_back({reinterpret_cast<char *>(hb.kvp[w][s]) + (size_t) g * cx * rb, n * rb});
                        r.push_back({reinterpret_cast<char *>(hb.kvs[w][s] + (size_t) g * cx * (HD / 32)), n * (HD / 32) * sizeof(__half)});
                    } else {
                        __half * b = static_cast<__half *>(w ? hb.vc : hb.kc) + ((size_t) s * NKV + g) * cx * HD;
                        r.push_back({reinterpret_cast<char *>(b), n * HD * sizeof(__half)});
                    }
                }
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++)
                if (hb.kvrp[w][s]) {
                    r.push_back({reinterpret_cast<char *>(hb.kvrp[w][s]), (size_t) NKV * hb.kvrc * HD});
                    r.push_back({reinterpret_cast<char *>(hb.kvrs[w][s]), (size_t) NKV * hb.kvrc * (HD / 32) * sizeof(__half)});
                }
        r.push_back({reinterpret_cast<char *>(hb.dstate), CK_NS * 4});
        r.push_back({reinterpret_cast<char *>(hb.conv), CK_NC * 4});
        r.push_back({reinterpret_cast<char *>(hb.h_hist), CK_NH * 4});
        return r;
    }
    // file: magic "MKSTATE1" | u64 key bytes | key | u64 hend | u64 regions | u64 bytes of each region | the regions' bytes
    void st_save(const std::string & path, const std::string & key, int hend) {
        static const char magic[9] = "MKSTATE1";
        const auto t0 = lclk::now();
        CK(cudaDeviceSynchronize());
        mkdir(st_dir.c_str(), 0755);
        const auto R = st_regions(hend);
        const std::string tmp = path + ".tmp" + std::to_string(getpid());
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[mk] state file: cannot write %s (not saved)\n", tmp.c_str());
            return;
        }
        bool ok = fwrite(magic, 8, 1, f) == 1;
        auto w64 = [&](uint64_t v) { ok = ok && fwrite(&v, 8, 1, f) == 1; };
        w64(key.size());
        ok = ok && fwrite(key.data(), 1, key.size(), f) == key.size();
        w64((uint64_t) hend);
        w64(R.size());
        size_t tot = 0;
        for (const auto & r : R) w64(r.second), tot += r.second;
        const size_t PC = (size_t) 64 << 20;
        std::unique_ptr<uint8_t[]> buf(new uint8_t[PC]);
        for (const auto & r : R)
            for (size_t o = 0; ok && o < r.second; o += PC) {
                const size_t n = std::min(PC, r.second - o);
                CK(cudaMemcpy(buf.get(), r.first + o, n, cudaMemcpyDeviceToHost));
                ok = fwrite(buf.get(), 1, n, f) == n;
            }
        ok = (fclose(f) == 0) && ok;
#ifdef _WIN32
        if (ok) remove(path.c_str()); // (rename does not replace an existing file on Windows)
#endif
        if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
            unlink(tmp.c_str());
            fprintf(stderr, "[mk] state file: writing %s failed (not saved)\n", path.c_str());
            return;
        }
        st_s = lsec(t0);
        fprintf(stderr, "[mk] state file saved: %s (%.2f GB, %d positions) in %.2f s\n", path.c_str(), tot / 1e9, hend, st_s);
    }
    bool st_load(const std::string & path, const std::string & key, int hend) {
        static const char magic[9] = "MKSTATE1";
        const auto t0 = lclk::now();
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        fstat(fd, &st);
        const size_t fsz = (size_t) st.st_size;
        void * mp = fsz ? mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        close(fd);
        if (mp == MAP_FAILED) return false;
        const uint8_t * b = static_cast<const uint8_t *>(mp);
        size_t pos = 0;
        auto r64 = [&](uint64_t & v) {
            if (pos + 8 > fsz) return false;
            memcpy(&v, b + pos, 8);
            pos += 8;
            return true;
        };
        const auto R = st_regions(hend);
        uint64_t kn = 0, h = 0, nr = 0, v = 0;
        bool ok = fsz >= 8 && !memcmp(b, magic, 8);
        pos = 8;
        ok = ok && r64(kn) && pos + kn <= fsz && key.size() == kn && !memcmp(b + pos, key.data(), kn);
        pos += kn;
        ok = ok && r64(h) && h == (uint64_t) hend && r64(nr) && nr == R.size();
        size_t tot = 0;
        for (size_t i = 0; ok && i < R.size(); i++) ok = r64(v) && v == R[i].second, tot += R[i].second;
        ok = ok && pos + tot == fsz;
        if (!ok) {
            munmap(mp, fsz);
            fprintf(stderr, "[mk] state file %s does not match (rewriting it)\n", path.c_str());
            return false;
        }
        CK(cudaDeviceSynchronize()); // (no copy of an earlier sequence still reading the state)
        { // read the file in with several threads (cold: one thread is bound by its page faults)
            const size_t PC = (size_t) 4 << 20, np = (fsz + PC - 1) / PC;
            std::atomic<size_t> next{0};
            std::atomic<unsigned> sink{0};
            std::vector<std::thread> th;
            for (int t = 0; t < 8; t++)
                th.emplace_back([&] {
                    unsigned s = 0;
                    for (size_t i; (i = next++) < np;)
                        for (size_t o = i * PC; o < std::min(fsz, (i + 1) * PC); o += 4096) s += b[o];
                    sink += s;
                });
            for (auto & t : th) t.join();
        }
        sw_need(hend); // (side request)
        for (const auto & r : R) {
            CK(cudaMemcpy(r.first, b + pos, r.second, cudaMemcpyHostToDevice));
            pos += r.second;
        }
        munmap(mp, fsz);
        kv_top = std::max(kv_top, hend); // (as pf_run of 0 .. hend - 1)
        // MK_STATE_WARM_MS (default 1000; 0: off): GPU warm-up before the decode (st_warm)
        const double wms = getenv("MK_STATE_WARM_MS") ? atof(getenv("MK_STATE_WARM_MS")) : 1000.0;
        if (wms > 0) {
            mkl::st_warm(nsm * 4, (unsigned long long) (wms * 1e6));
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
        }
        st_s = lsec(t0);
        fprintf(stderr, "[mk] state file restored: %s (%.2f GB, %d positions) in %.2f s (incl. %.1f s GPU warm-up, MK_STATE_WARM_MS)\n",
                path.c_str(), tot / 1e9, hend, st_s, wms / 1000.0);
        return true;
    }
    // ---- conversation state in host memory (MKEngine::side_begin / park) ----
    // Host store: pinned host chunks of hs_chunk bytes (MK_STORE_CHUNK_MIB, default 32), allocated on first use and kept in
    // a free list (cudaHostAlloc runs at ~1 GB/s under WSL2, a first touch of pageable memory about as slow: a chunk is
    // paid for once), at most hs_budget bytes (MKEngine::store_budget; 0: no store). A Blob lays items (bytes of device or
    // host memory) end to end over its chunks; blob_get copies each item back to where it came from.
    size_t hs_chunk = (size_t) (getenv("MK_STORE_CHUNK_MIB") ? std::max(1, atoi(getenv("MK_STORE_CHUNK_MIB"))) : 32) << 20;
    size_t hs_budget = 0, hs_alloc = 0;
    std::vector<char *> hs_free;
    double hs_ms = 0;      // the last side / park operation: wall time
    size_t hs_bytes = 0;   // ... bytes it copied
    struct Blob {
        struct Item {
            char * at; // device memory (async copies on stream 0), or host memory (memcpy)
            size_t n;
            bool host;
        };
        std::vector<char *> chunks;
        std::vector<Item> items;
        size_t used = 0;
    };
    char * hs_take() { // a chunk: from the free list, else a new one within the budget (nullptr: none)
        if (!hs_free.empty()) {
            char * c = hs_free.back();
            hs_free.pop_back();
            return c;
        }
        return hs_new();
    }
    char * hs_new() {
        if (hs_alloc + hs_chunk > hs_budget) return nullptr;
        char * c = nullptr;
        if (cudaHostAlloc(reinterpret_cast<void **>(&c), hs_chunk, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        hs_alloc += hs_chunk;
        return c;
    }
    void hs_release(Blob & b) {
        for (char * c : b.chunks) hs_free.push_back(c);
        b = Blob();
        hs_trim();
    }
    void hs_trim() { // (free chunks above the budget: after a budget cut)
        while (hs_alloc > hs_budget && !hs_free.empty()) {
            cudaFreeHost(hs_free.back());
            hs_free.pop_back();
            hs_alloc -= hs_chunk;
        }
    }
    // appends n bytes at src (device memory: a device-to-host copy on stream 0, which the caller waits for; host memory:
    // memcpy now) to b; false: the store is full (b then holds a partial item: the caller releases it)
    bool blob_put(Blob & b, const void * src, size_t n, bool host) {
        const char * s = static_cast<const char *>(src);
        b.items.push_back({const_cast<char *>(s), n, host});
        for (size_t done = 0; done < n;) {
            const size_t ci = b.used / hs_chunk, off = b.used % hs_chunk;
            if (ci == b.chunks.size()) {
                char * c = hs_take();
                if (!c) return false;
                b.chunks.push_back(c);
            }
            const size_t k = std::min(n - done, hs_chunk - off);
            if (host) memcpy(b.chunks[ci] + off, s + done, k);
            else CK(cudaMemcpyAsync(b.chunks[ci] + off, s + done, k, cudaMemcpyDeviceToHost, 0));
            done += k;
            b.used += k;
        }
        return true;
    }
    // copies every item of b back (device items first, as host-to-device copies on stream 0 that the caller waits for, so
    // that the host items' memcpy runs meanwhile)
    void blob_get(const Blob & b) {
        for (int pass = 0; pass < 2; pass++) {
            size_t pos = 0;
            for (const auto & it : b.items) {
                if (it.host == (pass == 0)) {
                    pos += it.n;
                    continue;
                }
                for (size_t done = 0; done < it.n;) {
                    const size_t ci = pos / hs_chunk, off = pos % hs_chunk, k = std::min(it.n - done, hs_chunk - off);
                    if (it.host) memcpy(it.at + done, b.chunks[ci] + off, k);
                    else CK(cudaMemcpyAsync(it.at + done, b.chunks[ci] + off, k, cudaMemcpyHostToDevice, 0));
                    done += k;
                    pos += k;
                }
            }
        }
    }
    // the sequence's KV rows a .. b - 1 of every slot (values + scales, or fp16), K and V, each kv head
    bool kv_put(Blob & bl, int a, int b) {
        if (b <= a) return true;
        const size_t cx = (size_t) hb.ctx, n = (size_t) (b - a);
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++)
                for (int g = 0; g < NKV; g++) {
                    if (hb.kvq8) {
                        const size_t rb = (size_t) HD * hb.kvb[w][s] / 8;
                        if (!blob_put(bl, reinterpret_cast<char *>(hb.kvp[w][s]) + ((size_t) g * cx + a) * rb, n * rb, false) ||
                            !blob_put(bl, hb.kvs[w][s] + ((size_t) g * cx + a) * (HD / 32), n * (HD / 32) * sizeof(__half), false))
                            return false;
                    } else {
                        __half * r = static_cast<__half *>(w ? hb.vc : hb.kc) + (((size_t) s * NKV + g) * cx + a) * HD;
                        if (!blob_put(bl, r, n * HD * sizeof(__half), false)) return false;
                    }
                }
        return true;
    }
    // ring slots a .. b - 1 (of kvrc) of every MK_KV_WIN ring side, each kv head
    bool ring_put(Blob & bl, int a, int b) {
        if (hb.kvwin <= 0 || b <= a) return true;
        const size_t rc = hb.kvrc, n = (size_t) (b - a);
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++)
                if (hb.kvrp[w][s])
                    for (int g = 0; g < NKV; g++)
                        if (!blob_put(bl, hb.kvrp[w][s] + ((size_t) g * rc + a) * HD, n * HD, false) ||
                            !blob_put(bl, hb.kvrs[w][s] + ((size_t) g * rc + a) * (HD / 32), n * (HD / 32) * sizeof(__half), false))
                            return false;
        return true;
    }
    // the recurrent state: DeltaNet states, conv rings, h ring (what a checkpoint holds; kvgb is not needed: the next
    // prefill refeeds every position from dn_c on)
    bool live_put(Blob & bl) {
        return blob_put(bl, hb.dstate, CK_NS * 4, false) && blob_put(bl, hb.conv, CK_NC * 4, false) &&
               blob_put(bl, hb.h_hist, CK_NH * 4, false);
    }
    size_t kv_token_bytes() const {
        size_t t = 0;
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++) t += NKV * (hb.kvq8 ? (size_t) HD * hb.kvb[w][s] / 8 + (HD / 32) * sizeof(__half) : HD * sizeof(__half));
        return t;
    }
    size_t ring_bytes() const {
        size_t t = 0;
        if (hb.kvwin > 0)
            for (int s = 0; s < kv_slots; s++)
                for (int w = 0; w < 2; w++)
                    if (hb.kvrp[w][s]) t += ring_vb() + ring_sb();
        return t;
    }
    // the host side of a sequence: everything a continuation reads besides the device state above
    struct Book {
        std::vector<int> seq, dv;
        Ctl ctl{};
        std::vector<Span> spans;
        int kv_top = 0, nd_avail = 0, ndraft = 0, last_acc = 0, demoted = 0, dv_n = 0;
        bool all_acc = true;
    };
    Book book_get() const {
        Book k;
        k.seq = seq;
        k.dv = dv_ids;
        k.dv_n = dv_n;
        k.ctl = hctl;
        k.spans = spans;
        k.kv_top = kv_top;
        k.nd_avail = nd_avail;
        k.ndraft = ndraft;
        k.last_acc = last_acc;
        k.demoted = demoted;
        k.all_acc = all_acc;
        return k;
    }
    void book_set(const Book & k) {
        seq = k.seq;
        dv_set(k.dv);
        dv_n = k.dv_n;
        hctl = k.ctl;
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        spans = k.spans;
        rope_sync();
        kv_top = k.kv_top;
        nd_avail = k.nd_avail;
        ndraft = k.ndraft;
        last_acc = k.last_acc;
        demoted = k.demoted;
        all_acc = k.all_acc;
    }
    // the draft vocabulary as `ids` (a dv_ids saved by book_get: the base list, then the sequence's additions in order)
    void dv_set(const std::vector<int> & ids) {
        if (dv_in.empty()) return;
        dv_truncate(dv_base);
        for (size_t i = dv_base; i < ids.size(); i++) {
            dv_in[ids[i]] = 1;
            dv_ids.push_back(ids[i]);
        }
        const int k = (int) dv_ids.size() - dv_base;
        if (k > 0) CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + dv_base, dv_ids.data() + dv_base, k * 4, cudaMemcpyHostToDevice));
        hw.head_d.N = (int) dv_ids.size();
        CK(mkl::set_cW(&hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    // no copy of a checkpoint save or restore in flight, nothing on the device still reading or writing the state
    void hs_quiesce() {
        ck_wait_restore();
        ring_snap_guard();
        ck_guard_state();
        CK(cudaDeviceSynchronize());
        for (auto & k : ck) k.pending = false;
        ck_scr_busy = false;
        ck_rs_src = -1;
    }
    // Side request (MKEngine::side_begin): the kept sequence's rows main_top = sw_top and its window ring; sw_hi rows and
    // sw_rhi ring slots are in sw_blob. Every launch / prefill that writes rows < X calls sw_need(X) first (rings_all:
    // every ring slot, for side_unpark). The checkpoint pool serves the side request: the kept sequence's checkpoints are
    // hidden (sw_ck: position, ctl, rings flag), and a slot holding one has its block (and rings) copied into sw_blob
    // before the side request writes it (sw_ck_take); side_end copies them back with the rows.
    static constexpr int SW_BLK = 1024; // rows per copy-out step
    bool sw_on = false, sw_broken = false;
    int sw_top = 0, sw_hi = 0, sw_rhi = 0;
    Blob sw_blob;
    Book sw_book;
    struct SwCk {
        int pos;
        Ctl ctl;
        bool ring_ok, saved;
        bool pin;
        unsigned pin_t;
    };
    std::vector<SwCk> sw_ck;
    double sw_ms_out = 0;
    void sw_need(int X, bool rings_all = false) {
        if (!sw_on || sw_broken) return;
        const int up = (X + MAXT + SW_BLK - 1) / SW_BLK * SW_BLK;
        const int want = std::max(sw_hi, std::min(sw_top, up)),
                  rwant = hb.kvwin > 0 ? std::max(sw_rhi, rings_all ? hb.kvrc : std::min(hb.kvrc, up)) : 0;
        if (want <= sw_hi && rwant <= sw_rhi) return;
        const auto t0 = std::chrono::steady_clock::now();
        CK(cudaDeviceSynchronize()); // (nothing still writing those rows; the copies then run before the next kernel)
        const size_t u0 = sw_blob.used;
        const bool ok = kv_put(sw_blob, sw_hi, want) && ring_put(sw_blob, sw_rhi, rwant);
        wait_device("side request copies", std::max(wd_limit, 120.0));
        sw_hi = want;
        sw_rhi = rwant;
        sw_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            sw_broken = true;
            fprintf(stderr, "[mk] side request: the host store is full at row %d: the kept sequence is dropped (a full prefill when it comes back)\n", X);
        } else if (memlog) {
            fprintf(stderr, "[mk-mem] side request: kept rows < %d, ring slots < %d (+%.1f MB)\n", sw_hi, sw_rhi, (sw_blob.used - u0) / 1e6);
        }
    }
    // slot i holds a checkpoint of the kept sequence that is not in sw_blob yet
    bool sw_ck_held(int i) const { return sw_on && !sw_broken && i < (int) sw_ck.size() && sw_ck[i].pos >= 0 && !sw_ck[i].saved; }
    // before the side request writes slot i: the kept checkpoint there goes to sw_blob (false: no room, nothing changed)
    bool sw_ck_take(int i) {
        if (!sw_ck_held(i)) return true;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t items = sw_blob.items.size(), used = sw_blob.used;
        const bool ok = blob_put(sw_blob, ck[i].mem, ck_block_bytes(), !ck_dev) &&
                        (!sw_ck[i].ring_ok || blob_put(sw_blob, ck[i].ring, ring_bytes_ck, true));
        if (ck_dev) wait_device("side request copies", std::max(wd_limit, 120.0)); // (device block: copies on stream 0)
        sw_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { // (the partial item goes; its chunks stay with the blob)
            sw_blob.items.resize(items);
            sw_blob.used = used;
            return false;
        }
        sw_ck[i].saved = true;
        return true;
    }
    // bytes the store can still hand out (free chunks, then new ones within the budget)
    size_t hs_room() const { return hs_free.size() * hs_chunk + (hs_budget > hs_alloc ? (hs_budget - hs_alloc) / hs_chunk * hs_chunk : 0); }
    // Parked conversations (MKEngine::park): a snapshot of everything above, plus up to max_cks of the newest checkpoints
    // and the pinned ones
    struct Parked {
        Blob blob;
        Book book;
        struct Ck {
            int slot, pos;
            Ctl ctl;
            bool ring_ok, pin;
            unsigned pin_t;
        };
        std::vector<Ck> cks;
    };
    std::map<int, Parked> parked;
    int park_next = 1;
    size_t hs_used() const { // bytes of the store in use (parked entries, the side request's copies)
        size_t n = sw_blob.chunks.size();
        for (const auto & p : parked) n += p.second.blob.chunks.size();
        return n * hs_chunk;
    }
    int seq_top() const { return std::max({kv_top, (int) seq.size(), hctl.dn_c}); }
    std::vector<int> park_cks(int max_cks) const { // slots of the newest checkpoints, newest first, then the pinned ones
        std::vector<int> v;
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos >= 0) v.push_back(i);
        std::sort(v.begin(), v.end(), [&](int a, int b) { return ck[a].pos > ck[b].pos; });
        const size_t n = std::min(v.size(), (size_t) std::max(0, max_cks));
        std::vector<int> out(v.begin(), v.begin() + n);
        for (size_t k = n; k < v.size() && max_cks > 0; k++)
            if (ck[v[k]].pin) out.push_back(v[k]);
        return out;
    }
    size_t ck_block_bytes() const { return (CK_NS + CK_NC + CK_NH) * 4; }
    double hs_park_ms = 0;    // the last side_end_park: its park of the side request's conversation
    size_t hs_park_bytes = 0;
    // park(): a snapshot of the sequence (device copies quiesced by the caller) as a new entry: its id, -1 no room
    int park_entry(int max_cks) {
        const auto t0 = std::chrono::steady_clock::now();
        Parked e;
        struct Rel { // (the entry's chunks go back unless it is kept, also on an exception)
            Impl & I;
            Blob & b;
            bool keep = false;
            ~Rel() {
                if (keep) return;
                if (cudaStreamSynchronize(0) != cudaSuccess) cudaGetLastError(); // (no copy into its chunks still in flight)
                I.hs_release(b);
            }
        } rel{*this, e.blob};
        e.book = book_get();
        // device state first (copies on stream 0), then the checkpoint blocks (pinned host: memcpy while those run)
        bool ok = kv_put(e.blob, 0, seq_top()) && ring_put(e.blob, 0, hb.kvwin > 0 ? hb.kvrc : 0) && live_put(e.blob);
        for (int si : park_cks(max_cks)) {
            if (!ok) break;
            const CkSlot & s = ck[si];
            ok = blob_put(e.blob, s.mem, ck_block_bytes(), !ck_dev) && (!s.ring_ok || blob_put(e.blob, s.ring, ring_bytes_ck, true));
            e.cks.push_back({si, s.pos, s.ctl, s.ring_ok, s.pin, s.pin_t});
        }
        wait_device("park copies", std::max(wd_limit, 120.0));
        if (!ok) return -1;
        const int id = park_next++;
        hs_bytes = e.blob.used;
        hs_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        rel.keep = true;
        parked.emplace(id, std::move(e));
        return id;
    }
    // side_end: the kept sequence comes back (rows, rings, recurrent state, its checkpoints' blocks, the bookkeeping)
    void side_restore() {
        blob_get(sw_blob);
        wait_device("side request copies", std::max(wd_limit, 120.0));
        book_set(sw_book);
        for (size_t k = 0; k < ck.size(); k++) {
            ck[k].pos = sw_ck[k].pos;
            ck[k].ctl = sw_ck[k].ctl;
            ck[k].ring_ok = sw_ck[k].ring_ok;
            ck[k].pin = sw_ck[k].pin;
            ck[k].pin_t = sw_ck[k].pin_t;
            ck[k].pending = false;
        }
        sw_on = false;
        hs_bytes = sw_blob.used;
        hs_release(sw_blob);
    }

    // ---- lanes (MKEngine::slots_*): sequences of their own, decoded together by batched launches ----
    // A lane holds a whole conversation: device state of its own (KV cache, recurrent state, rings, ctl), its host
    // bookkeeping, a small checkpoint pool. While a lane is live (lane_in) its state sits in the Impl's own members and
    // cView selects its full view, so every single-sequence path (prefill, rewind, truncate, a solo decode step) runs on it
    // unchanged; lane_out swaps it back. Batched launches (lanes_step) read every lane through its views at once. The
    // row buffers (x, qn, ..), the barrier words and the lm_head scratch are the engine's own, shared by all lanes.
    struct Lane {
        int ctx = 0;
        Bufs hb{};
        std::vector<int> seq;
        Ctl hctl{};
        int kv_top = 0, nd_avail = 0, ndraft = 0, last_acc = 0, demoted = 0, dv_n = 0;
        bool all_acc = true;
        float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f;
        int smp_top_k = TOPK_MAX;
        unsigned long long smp_seed = 0;
        float pen_pres = 0.f, pen_freq = 0.f; // (always 0: a lane has no penalty counts, hb.pcnt null)
        bool pcnt_dirty = false;
        std::vector<Span> spans, rope_spans;
        volatile Ctl * ctl_eg_host = nullptr;
        std::vector<CkSlot> ck;
        std::vector<CkChunk> ck_plan;
        std::vector<RingSide> rsides[N_ATT_SLOTS]; // MK_EXACT + MK_KV_WIN: its window rings as its checkpoints copy them
        std::vector<int> prompt; // slot_begin's prompt (a cancelled chunk queues the rest again from it)
        std::vector<int> pend;   // prompt tokens slot_begin queued, not prefilled yet
        int first = -1;          // the first token of the finished prefill: emitted by the next slots_step
        bool dead = false;       // a slot call on it failed (MKCudaError): state undefined until slot_begin resets it
        std::string dead_why;    // (that error, in the refusals)
    };
    std::vector<Lane> lanes;
    int live = -1;          // the lane whose state the Impl's members hold (-1: the engine's own sequence)
    int live_view = 0;      // cView: view_full(live), or 0
    bool dv_frozen = false; // lanes open: the draft vocabulary stays the base list (shared by every sequence)
    bool par_wide = false;  // batched passes up to RMAX rows (MK_PAR_WIDE build, the model's matrices and the GPU allow them)
    // slot_prefill's chunks before the prompt's last: every row in the batched prefill (no megakernel row, no output
    // token, no checkpoint at the chunk's end), so in MK_EXACT mode the rows are those of the whole prompt's prefill
    bool pf_mid = false;
    void lane_swap(Lane & l) {
        std::swap(ctx, l.ctx);
        std::swap(hb, l.hb);
        std::swap(seq, l.seq);
        std::swap(hctl, l.hctl);
        std::swap(kv_top, l.kv_top);
        std::swap(nd_avail, l.nd_avail);
        std::swap(ndraft, l.ndraft);
        std::swap(last_acc, l.last_acc);
        std::swap(demoted, l.demoted);
        std::swap(dv_n, l.dv_n);
        std::swap(all_acc, l.all_acc);
        std::swap(smp_temp, l.smp_temp);
        std::swap(smp_top_p, l.smp_top_p);
        std::swap(smp_min_p, l.smp_min_p);
        std::swap(smp_top_k, l.smp_top_k);
        std::swap(smp_seed, l.smp_seed);
        std::swap(pen_pres, l.pen_pres); // (the own sequence's penalties and their counts never reach a lane)
        std::swap(pen_freq, l.pen_freq);
        std::swap(pcnt_dirty, l.pcnt_dirty);
        std::swap(spans, l.spans);
        std::swap(rope_spans, l.rope_spans);
        std::swap(ctl_eg_host, l.ctl_eg_host);
        std::swap(ck, l.ck);
        std::swap(ck_plan, l.ck_plan);
        std::swap(rsides, l.rsides);
    }
    void set_cview(int v) {
        if (v == live_view) return;
        CK(mkl::set_cView(&v, sizeof(int)));
        live_view = v;
    }
    void lane_in(int k) {
        if (k < 0 || k >= (int) lanes.size()) throw std::runtime_error("no slot " + std::to_string(k));
        if (live == k) return;
        lane_out();
        hs_quiesce();
        lane_swap(lanes[k]);
        live = k;
        set_cview(view_full(k));
    }
    void lane_out() {
        if (live >= 0) {
            hs_quiesce();
            lane_swap(lanes[live]);
            live = -1;
        }
        set_cview(0); // (also after a failed copy left the device on a lane's view)
    }
    // ~LaneOut's lane_out: never throws. A failed device call marks the lane dead and is retried once; failing again it
    // is taken as sticky (as in mk_cuda_fail: exit code 4), since unsynchronised or on a stale view the next sequence's
    // work would run on the wrong state
    void lane_out_nothrow() noexcept {
        const int k = live;
        for (int i = 0;; i++) {
            std::string why = "unknown exception";
            try {
                lane_out();
                return;
            } catch (const std::exception & e) {
                why = e.what();
            } catch (...) {
            }
            lane_dead(k, why);
            if (i > 0) {
                fprintf(stderr, "[mk] the CUDA error repeats while a slot is put away (sticky): exiting with code 4\n");
                fflush(stdout);
                fflush(stderr);
                _exit(4);
            }
        }
    }
    // a slot call failed on lane k (MKCudaError, also the NaN check's): nothing runs on it until slot_begin resets it
    void lane_dead(int k, const std::string & why) {
        if (k < 0 || k >= (int) lanes.size()) return;
        Lane & l = lanes[k];
        l.dead = true;
        l.dead_why = why;
        l.pend.clear();
        l.first = -1;
    }
    // lane k for a call that continues its prompt or reply: refused when it does not exist or is dead
    Lane & lane_ok(int k) {
        if (k < 0 || k >= (int) lanes.size()) throw std::runtime_error("no slot " + std::to_string(k));
        if (lanes[k].dead)
            throw std::runtime_error("slot " + std::to_string(k) + " failed (" + lanes[k].dead_why +
                                     "): begin a new prompt");
        return lanes[k];
    }
    // a slot call's body f on lanes ks, then the engine's own sequence back in; an MKCudaError marks every lane in ks
    // dead on its way to the caller (MKCancelled does not: the prefill put the state back itself)
    template <class F> void lanes_call(const std::vector<int> & ks, F f) {
        try {
            f();
            lane_out(); // (here rather than in ~LaneOut: its failure is this call's error)
        } catch (const MKCudaError & e) {
            for (int k : ks) lane_dead(k, e.what());
            throw;
        }
    }
    // frees ck_new_slots' slots (events, pinned blocks, ring copies; nothing may still use them)
    void ck_slots_free(std::vector<CkSlot> & v) noexcept {
        for (CkSlot & s : v) {
            if (s.ev) cudaEventDestroy(s.ev);
            if (s.mem) cudaFreeHost(s.mem);
            if (s.ring) cudaFreeHost(s.ring);
        }
        v.clear();
    }
    // slots_open failing part way (allocs had na entries, alloc_bytes ab): its lanes and every device / pinned
    // allocation it made go, lanes stay empty, the engine's own sequence is live (never throws: results ignored, the last
    // error cleared)
    void slots_undo(size_t na, size_t ab) noexcept {
        lane_out_nothrow();
        cudaDeviceSynchronize();
        for (Lane & l : lanes) {
            ck_slots_free(l.ck);
            if (l.ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(l.ctl_eg_host));
        }
        lanes.clear();
        for (size_t i = na; i < allocs.size(); i++) use_pool ? cudaFreeAsync(allocs[i], 0) : cudaFree(allocs[i]);
        allocs.resize(na);
        alloc_bytes = ab;
        // trimmed: what the lanes held, and what a failed allocation left reserved (several GB after a large one failed
        // part way, outside the VRAM), go back to the device: an open reusing that remainder at once can fail with out of
        // memory, which the context then keeps reporting.
        pool_trim();
        cudaGetLastError();
    }
    // n pinned checkpoint slots (as ck_alloc's: the streams and events are the engine's)
    std::vector<CkSlot> ck_new_slots(int n) {
        std::vector<CkSlot> v;
        const size_t nf = CK_NS + CK_NC + CK_NH;
        for (int i = 0; i < n; i++) {
            CkSlot s;
            if (cudaHostAlloc(&s.mem, nf * 4, cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                break;
            }
            s.dstate = static_cast<float *>(s.mem);
            s.conv = s.dstate + CK_NS;
            s.hhist = s.conv + CK_NC;
            try {
                CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&s.hdev), s.hhist, 0));
                CK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
            } catch (...) { // (the slots made so far go with it)
                v.push_back(s);
                ck_slots_free(v);
                throw;
            }
            if (ring_bytes_ck > 0 && cudaHostAlloc(reinterpret_cast<void **>(&s.ring), ring_bytes_ck, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                s.ring = nullptr; // (as ck_alloc's: restores of this checkpoint refill the rings, not exact)
                fprintf(stderr, "[mk] warning: no pinned host memory for a slot checkpoint's KV window rings: its restores refill them (not exact)\n");
            }
            v.push_back(s);
        }
        return v;
    }
    // device bytes of one lane's KV cache of lctx positions in the engine's format (MK_KV bits per slot and side, fp16 scales,
    // the MK_KV_WIN q8 rings of the sides that have one)
    size_t lane_kv_bytes(int lctx) const {
        const size_t srows = (size_t) NKV * (lctx + KV_PAD);
        size_t b = 0;
        for (int w = 0; w < 2; w++)
            for (int sl = 0; sl < kv_slots; sl++) {
                b += srows * ((size_t) HD * hb.kvb[w][sl] / 8 + (HD / 32) * sizeof(__half));
                if (hb.kvrp[w][sl]) b += (size_t) NKV * hb.kvrc * (HD + (HD / 32) * sizeof(__half));
            }
        return b;
    }
    // device bytes of one lane of ctx positions (KV cache; state, rings, partials, ctl)
    size_t lane_bytes(int lctx) const {
        return lane_kv_bytes(lctx) + sizeof(float) * ((size_t) 48 * CONV_RING * QKV + (size_t) 48 * KV_RING * KVGB +
               (size_t) 48 * NVH * 128 * 128 + (size_t) NKV * MAX_CH * AR * 256 + (size_t) NKV * MAX_CH * AR * 2 +
               (size_t) H_RING * D + D);
    }
    // MKEngine::slots_open: n lanes of lctx positions each (0: the most that fit, leaving MK_SLOT_RESERVE_MIB, default
    // 1536, for the prefill and the runtime); returns lctx
    int slots_open(int n, int lctx) {
        if (!lanes.empty()) throw std::runtime_error("slots are open already");
        if (n < 1 || n > NSEQ) throw std::runtime_error("slots: 1 .. " + std::to_string(NSEQ));
        // (2+ lanes run batched passes of MAXT rows, two lanes always: embed_prologue skips the rows past a smaller grid)
        if (n >= 2 && nsm < mk_min_grid(MAXT))
            throw std::runtime_error("parallel slots: " + std::to_string(n) + " slots need a GPU with at least " +
                                     std::to_string(mk_min_grid(MAXT)) + " SMs (batched passes of " + std::to_string(MAXT) +
                                     " rows); this one has " + std::to_string(nsm) + " (1 slot runs)");
        // a slot's positions: at most the model's context length, also for "the most that fit"; checked before anything is
        // allocated (100,000,000 positions would otherwise build a 26 GB host RoPE table first)
        const int lmax = ctx_limit();
        if (lctx > lmax)
            throw std::runtime_error("parallel slots: " + std::to_string(lctx) + " positions per slot exceed the model's context length (" +
                                     std::to_string(lmax) + ", qwen35.context_length)");
        lane_out();
        // (any MK_KV / MK_KV_WIN preset: every lane gets the engine's per-slot formats and window rings of its own; every
        // device path reads them through the lane's views, the host's per-sequence ring state is in the Lane)
        if (!hb.kvq8) throw std::runtime_error("parallel slots need the q8 KV cache (--kv q8; MK_KV / MK_KV_WIN presets are fine)");
#if MK_KV_EMU
        for (int w = 0; w < 2; w++)
            for (int sl = 0; sl < N_ATT_SLOTS; sl++)
                if (hb.kvemu[w][sl]) throw std::runtime_error("parallel slots: no MK_KV_EMU_SPEC (emulated formats)");
#endif
        for (int L = 0; L < NL; L++)
            if (!hw.L[L].attn && !(hw.L[L].qkv.fm && hw.L[L].gate.fm))
                throw std::runtime_error("parallel slots: this model's delta-net projections are not all in the wv layout (types the batched launch does not serve)");
        if (!spans.empty()) throw std::runtime_error("parallel slots: the engine's own sequence holds images (reset it first)");
        // a failure from here on: slots_undo frees what this call allocated, lanes stay empty (a call while slots are
        // open throws above, before any change, so lanes it did not build are never undone)
        struct Undo {
            Impl * p;
            size_t na, ab;
            bool done;
            ~Undo() {
                if (!done) p->slots_undo(na, ab);
            }
        } undo{this, allocs.size(), alloc_bytes, false};
        par_wide = false;
#if MK_PAR_WIDE
        {
            // the RMAX-row pass (wv.cuh "batched passes wider than MAXT rows") runs every main matrix but ffn_gate / ffn_up in
            // the wv layout, and one item per warp in every weight phase
            std::string why;
            for (int L = 0; L < NL && why.empty(); L++) {
                const LayerW & W = hw.L[L];
                if (W.attn && !(W.q.fm && W.k.fm && W.v.fm && W.o.fm)) why = "an attention q / k / v / out matrix of a type outside the wv layout";
                if (!W.attn && !W.sout.fm) why = "a delta-net ssm_out matrix of a type outside the wv layout";
                if (!W.fd.fm) why = "an ffn_down matrix outside the wv layout";
            }
            const int G = nsm / WV_S;
            if (why.empty() && ((D / 16 + G - 1) / std::max(G, 1) > NW || (FF / 16 / WV_S + G - 1) / std::max(G, 1) > NW ||
                                (QKV / 16 + VD / 16 + nsm - 1) / nsm > NW || QG / 16 + 4 * (KVD / 16) > nsm * NW))
                why = "a GPU with fewer SMs than one item per warp needs";
            if (getenv("MK_PAR_RMAX") && atoi(getenv("MK_PAR_RMAX")) <= MAXT) why = "MK_PAR_RMAX";
            par_wide = why.empty();
            fprintf(stderr, "[mk] parallel slots: batched passes of %s\n", par_wide ? (std::to_string(RMAX) + " rows").c_str()
                                                                               : (std::to_string(MAXT) + " rows (" + why + ")").c_str());
        }
#endif
        CK(cudaDeviceSynchronize());
        if (lctx <= 0) {
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            const size_t reserve = (size_t) (getenv("MK_SLOT_RESERVE_MIB") ? atol(getenv("MK_SLOT_RESERVE_MIB")) : 1536) << 20;
            const size_t avail = free_b > reserve ? (free_b - reserve) / n : 0;
            int lo = 0, hi = lmax;
            while (lo < hi) {
                const int mid = (lo + hi + 1) / 2;
                if (lane_bytes(mid) <= avail) lo = mid;
                else hi = mid - 1;
            }
            lctx = lo / ATT_BUCKET * ATT_BUCKET;
            if (lctx < 2 * ATT_BUCKET) throw std::runtime_error("parallel slots: not enough free device memory");
        }
        // the slots' device memory against the device's free memory (+ what the allocation pool holds unused), before any of
        // it is allocated: past it a driver with the CUDA sysmem fallback (Windows and WSL2, on by default) does not fail but
        // puts the rest in system memory, charged to Windows commit, and the KV cache is then read over PCIe (4 x 200000
        // positions beside a ctx 8192 engine on a 24 GB GPU: ~26 GB of it, Windows commit down to 0 free, 2026-11-01).
        // MK_SLOT_SPILL=1: no check (the driver decides).
        if (!(getenv("MK_SLOT_SPILL") && atoi(getenv("MK_SLOT_SPILL")) != 0)) {
            const size_t need = (size_t) n * lane_bytes(lctx) +
                                (lctx + KV_PAD > hb.ctx ? (size_t) (lctx + KV_PAD) * 32 * sizeof(float2) : 0); // (+ the RoPE rows)
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            if (use_pool) {
                cudaMemPool_t pool;
                unsigned long long res = 0, used = 0;
                if (cudaDeviceGetMemPool(&pool, cuda_dev) == cudaSuccess &&
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res) == cudaSuccess &&
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess && res > used)
                    free_b += res - used;
                cudaGetLastError();
            }
            if (need > free_b) {
                char msg[384];
                snprintf(msg, sizeof msg,
                         "parallel slots: %d x %d positions need %.2f GB of device memory, %.2f GB is free (fewer or shorter slots, or "
                         "0 positions: the most that fit; MK_SLOT_SPILL=1 lets a driver's system-memory fallback take the rest)",
                         n, lctx, need / 1e9, free_b / 1e9);
                throw std::runtime_error(msg);
            }
        }
        // RoPE rows for the lanes (plain positions; the engine's own table when it is long enough)
        float2 * rope = const_cast<float2 *>(hb.rope);
        if (lctx + KV_PAD > hb.ctx) {
            std::vector<float2> rt((size_t) (lctx + KV_PAD) * 32);
            for (int p = 0; p < lctx + KV_PAD; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            rope = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(rope, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
        }
        const int nck = getenv("MK_SLOT_CK") ? std::max(1, atoi(getenv("MK_SLOT_CK"))) : 2;
        lanes.resize(n);
        for (int k = 0; k < n; k++) {
            Lane & l = lanes[k];
            l.ctx = lctx;
            l.hb = hb;
            Bufs & b = l.hb;
            b.ctx = lctx + KV_PAD;
            // the KV cache in the engine's layout (the constructor's): per slot and side its MK_KV bits (kvb, kvr / kvrv rotation
            // as hb's), slots packed in order (all-q8: [slot][kv head][ctx][HD]), fp16 scales [slot][kv head][ctx][HD / 32];
            // a q8 window ring of its own for every side the engine keeps one for (MK_KV_WIN, kvwin / kvrc as hb's)
            const size_t srows = (size_t) NKV * b.ctx;
            size_t vbytes[2] = {0, 0};
            for (int w = 0; w < 2; w++)
                for (int sl = 0; sl < kv_slots; sl++) vbytes[w] += srows * HD * hb.kvb[w][sl] / 8;
            b.kc = dalloc<int8_t>(vbytes[0]);
            b.vc = dalloc<int8_t>(vbytes[1]);
            b.ksc = dalloc<__half>((size_t) kv_slots * srows * (HD / 32));
            b.vsc = dalloc<__half>((size_t) kv_slots * srows * (HD / 32));
            size_t off[2] = {0, 0};
            for (int sl = 0; sl < N_ATT_SLOTS; sl++)
                for (int w = 0; w < 2; w++) {
                    b.kvrp[w][sl] = nullptr;
                    b.kvrs[w][sl] = nullptr;
                    if (sl >= kv_slots) {
                        b.kvp[w][sl] = nullptr;
                        b.kvs[w][sl] = nullptr;
                        continue;
                    }
                    b.kvp[w][sl] = static_cast<int8_t *>(w ? b.vc : b.kc) + off[w];
                    b.kvs[w][sl] = (w ? b.vsc : b.ksc) + (size_t) sl * srows * (HD / 32);
                    off[w] += srows * HD * hb.kvb[w][sl] / 8;
                    if (hb.kvrp[w][sl]) {
                        b.kvrp[w][sl] = dalloc<int8_t>((size_t) NKV * hb.kvrc * HD);
                        b.kvrs[w][sl] = dalloc<__half>((size_t) NKV * hb.kvrc * (HD / 32));
                    }
                }
            for (int sl = 0; sl < N_ATT_SLOTS; sl++) { // (ck_alloc's ring sides, offsets and order: the lane's rings)
                l.rsides[sl].clear();
                for (const RingSide & r : rsides[sl])
                    for (int w = 0; w < 2; w++)
                        if (r.q == hb.kvrp[w][sl]) l.rsides[sl].push_back({b.kvrp[w][sl], b.kvrs[w][sl], r.off});
            }
            b.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
            b.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
            b.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
            b.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
            b.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
            b.h_hist = dalloc<float>((size_t) H_RING * D);
            b.hm = dalloc<float>(D);
            b.dlog = dalloc<float2>((size_t) MAXT * (TOPK_MAX + 1));
            b.dq = dalloc<float2>((size_t) MAXT * TOPK_MAX);
            b.ctl = dalloc<Ctl>(1);
            b.pcnt = nullptr;
            b.logits = nullptr;
            b.rope = rope;
            {
                Ctl * ce = nullptr;
                CK(cudaHostAlloc(reinterpret_cast<void **>(&ce), sizeof(Ctl), cudaHostAllocMapped));
                memset(ce, 0, sizeof(Ctl));
                l.ctl_eg_host = ce;
                CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&b.ctl_eg), ce, 0));
            }
            l.ck = ck_new_slots(nck);
            CK(mkl::set_cBs(&b, sizeof(Bufs), view_full(k) * sizeof(Bufs)));
            CK(mkl::set_cBs(&b, sizeof(Bufs), view_main(k) * sizeof(Bufs)));
            {
                const Bufs vm = lane_main_view(l, k); // (batched MTP: lane k's rows at row k)
                CK(mkl::set_cBs(&vm, sizeof(Bufs), view_mtp(k) * sizeof(Bufs)));
            }
            lane_in(k);
            ck_make_plan();
            reset();
            seq.clear();
            lane_out();
#if MK_TEST_SLOTS_FAIL
            if (getenv("MK_TEST_SLOTS_FAIL_AT") && atoi(getenv("MK_TEST_SLOTS_FAIL_AT")) == k) {
                fprintf(stderr, "[mk] MK_TEST_SLOTS_FAIL: slots_open fails after lane %d\n", k);
                throw MKCudaError("MK_TEST_SLOTS_FAIL: slots_open failed after lane " + std::to_string(k));
            }
#endif
        }
        dv_reset();
        dv_frozen = true;
        undo.done = true;
        bool lowbit = false;
        for (int w = 0; w < 2; w++)
            for (int sl = 0; sl < kv_slots; sl++) lowbit |= hb.kvb[w][sl] != 8 || (w ? hb.kvrv[sl] : hb.kvr[sl]);
        int ck_rings = 0;
        for (const CkSlot & c : lanes[0].ck) ck_rings += c.ring != nullptr;
        fprintf(stderr, "[mk] %d parallel slots of %d positions (%s KV%s: %.2f GB, %.2f GB each in all, %zu checkpoints each",
                n, lctx, lowbit ? "MK_KV" : "q8", hb.kvwin > 0 ? " + MK_KV_WIN rings" : "", lane_kv_bytes(lctx) / 1e9,
                lane_bytes(lctx) / 1e9, lanes[0].ck.size());
        if (ring_bytes_ck > 0) fprintf(stderr, ", %d with the window rings, %.1f MiB pinned host each", ck_rings, ring_bytes_ck / 1048576.0);
        fprintf(stderr, ")\n");
        return lctx;
    }
    // the main view of lane k for a batched launch: its rows start at row r0 of the pass
    Bufs lane_main_view(const Lane & l, int r0) const {
        Bufs v = l.hb;
        v.x += (size_t) r0 * D;
        v.xm += (size_t) r0 * D;
        auto qoff = [&](QBuf & q) {
            q.xa += (size_t) r0 * q.U * 16;
            q.xb += (size_t) r0 * q.U * 16;
            q.meta += (size_t) r0 * q.U;
            q.ssq += (size_t) r0 * q.U;
        };
        qoff(v.qn);
        qoff(v.qo);
        qoff(v.qa);
        qoff(v.qe);
        v.z += (size_t) r0 * VD;
        v.ab += (size_t) r0 * 2 * NVH;
        v.qbuf += (size_t) r0 * QG;
        v.kbuf += (size_t) r0 * KVD;
        v.vbuf += (size_t) r0 * KVD;
        v.cand += (size_t) r0 * TOPK_MAX;
        return v;
    }
    // one batched decode launch of lanes ks (>= 2, none live, every one past its prefill): returns each lane's tokens
    // (empty: its context is full). Rows: lane i verifies T_i <= Tcap = MAXT / nl tokens (its drafts beyond are dropped,
    // as att_cap_T drops them) and drafts up to Tcap - 1 for the next launch.
    std::vector<std::vector<int>> lanes_step(const std::vector<int> & ks) {
        lane_out();
        const int nl = (int) ks.size();
        std::vector<std::vector<int>> out(nl);
        // rows per lane: the lanes' slots (lane_slot: T <= 5, or 8 rows) share RMAX rows (par_wide) or MAXT
        const int rmax = par_wide ? RMAX : MAXT;
        int tcap = rmax / nl >= MAXT ? MAXT : std::min(5, rmax / nl);
        // two lanes: both in one MAXT-row pass (3 drafts each) are faster than a wide pass with deeper drafts (the
        // wide passes are faster at 3 and 4 lanes)
        if (nl == 2) tcap = MAXT / 2;
        Launch L{};
        L.mode = MODE_DECODE;
        L.nl = 0;
        int rows = 0, top = 0;
        std::vector<int> li(nl, -1), nd(nl, 0), Ti(nl, 0);
        for (int i = 0; i < nl; i++) {
            Lane & l = lanes[ks[i]];
            const Ctl & c = l.hctl;
            if (c.n + MAXT + 2 >= l.ctx) continue; // (context full: no tokens)
            int T = att_cap_T(c.n, std::min(tcap, 1 + l.nd_avail));
            if (T > 5 && att_cap_T(c.n, MAXT) < MAXT) T = 5; // (an 8-row slot lies in one attention partition, as step_live's)
            int d = std::min(depth_top(c.n), tcap - 1);
            if (pol.depth == 0) d = (!l.all_acc && d > 3) ? d - 1 : d;
            else if (pol.depth == 2) d = std::min(d, std::max(std::min(pol.dmin, d), 1 + l.last_acc));
            Launch::Lane & q = L.ln[L.nl];
            q.lane = ks[i];
            q.T = T;
            q.r0 = rows;
            q.nd = d;
            q.top_k = l.smp_top_k;
            q.spec_smp = l.smp_temp > 0.f && pol.h1;
            q.temp = l.smp_temp;
            q.top_p = l.smp_top_p;
            q.min_p = l.smp_min_p;
            q.seed = l.smp_seed;
            li[i] = L.nl++;
            nd[i] = d;
            Ti[i] = T;
            rows += lane_slot(T);
            top = std::max(top, d);
        }
        if (L.nl == 0) return out;
        if (L.nl == 1) { // (the others are full: a solo step)
            int k = -1;
            for (int i = 0; i < nl; i++)
                if (li[i] >= 0) k = i;
            lane_in(ks[k]);
            std::vector<int> o;
            step_live(o);
            out[k] = o;
            lane_out();
            return out;
        }
        L.T = rows <= 5 ? rows : rows <= MAXT ? MAXT : RMAX;
        L.mtpb = hb.mtp_win <= 0; // (MK_MTP_WIN: the window start follows the pass's first row)
        L.ndraft = top;
        for (int k = 0; k < MAXT - 1; k++) {
            L.gate_drop[k] = pol.gate_drop[k];
            L.gate_stop[k] = pol.gate_stop[k];
        }
        if (top >= 1) L.gate_drop[top - 1] = std::max(L.gate_drop[top - 1], pol.drop_last);
        L.gate_chain = pol.gate_chain;
        L.spec_smp = 0;
        for (int i = 0; i < L.nl; i++) L.spec_smp |= L.ln[i].spec_smp;
        if (L.gate_chain == 2 && L.spec_smp)
            for (int k = 0; k < MAXT - 1; k++) L.gate_drop[k] = 0.f;
        for (int i = 0; i < nl; i++) {
            if (li[i] < 0) continue;
            Lane & l = lanes[ks[i]];
            l.kv_top = std::max(l.kv_top, l.hctl.n + Ti[i] + nd[i]);
            const Bufs v = lane_main_view(l, L.ln[li[i]].r0);
            CK(mkl::set_cBs_async(&v, sizeof(Bufs), view_main(ks[i]) * sizeof(Bufs), cudaMemcpyHostToDevice, 0));
            if (L.mtpb) { // (the lane's MTP view: its buffers as they are now, rows at row ks[i])
                const Bufs vm = lane_main_view(l, ks[i]);
                CK(mkl::set_cBs_async(&vm, sizeof(Bufs), view_mtp(ks[i]) * sizeof(Bufs), cudaMemcpyHostToDevice, 0));
            }
        }
        void * args[] = {&L};
        ck_wait_restore();
        ck_guard_state();
        ring_snap_guard();
        if (!bar_live) {
            CK(cudaMemsetAsync(hb.bar, 0, C1_WORDS * 4));
            bar_live = true;
        }
        if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
        CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        wd_T = L.T;
        wd_mode = MODE_DECODE;
        wait_device("a batched decode launch", wd_limit);
        if (err_host[0] | err_host[1]) launch_failed();
        launches_seen++;
        if (profiling) prof_collect(0);
        for (int i = 0; i < nl; i++) {
            if (li[i] < 0) continue;
            Lane & l = lanes[ks[i]];
            Ctl & c = l.hctl;
            memcpy(&c, const_cast<const Ctl *>(l.ctl_eg_host), sizeof(Ctl));
            l.nd_avail = nd[i] > 0 ? std::min(nd[i], c.nd_prod) : 0;
            l.ndraft = nd[i];
            out[i].assign(c.out, c.out + c.n_out);
            l.seq.insert(l.seq.end(), c.out, c.out + c.n_out);
            l.dv_n = -1;
            l.all_acc = c.n_acc == Ti[i] - 1;
            l.last_acc = c.n_acc;
        }
        return out;
    }
    // one solo decode step of the live sequence (MKEngine::step's body): appends the emitted tokens to out
    int step_live(std::vector<int> & out);

    void reset() {
        ck_rs_slot = -1;   // (a restore not issued yet is moot)
        ck_wait_restore(); // (an issued one still writes the state)
        ck_guard_state();
        kv_top = 0; // (MK_KV_WIN: every position is rewritten before it is read; the ring needs nothing)
        CK(cudaMemset(hb.dstate, 0, sizeof(float) * (size_t) 48 * NVH * 128 * 128));
        CK(cudaMemset(hb.h_hist, 0, sizeof(float) * (size_t) H_RING * D));
        memset(&hctl, 0, sizeof(hctl));
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        CK(cudaMemset(hb.cnt, 0, 16));
    }
};

MKEngine::MKEngine(const std::string & path, int ctx, int n_draft, const std::string & draft_vocab, int kv_mode,
                   const std::string & mtp)
    : p_(nullptr), n_draft_(n_draft) {
    if (n_draft < 0 || n_draft > MAXT - 1) throw std::runtime_error("n_draft must be in [0, " + std::to_string(MAXT - 1) + "]");
    if (mkl::abi_sig() != mk_abi_sig())
        throw std::runtime_error("engine build: the device object (qwen38_mk.cu) was compiled with other settings than the host files: rebuild both");
    p_ = new Impl(path, ctx, n_draft, draft_vocab, kv_mode, mtp);
    struct DeleteOnThrow { // (~MKEngine does not run for a constructor that throws: a bad MK_POLICY, a CUDA error)
        Impl *& p;
        int n0;
        ~DeleteOnThrow() {
            if (std::uncaught_exceptions() <= n0) return;
            delete p;
            p = nullptr;
        }
    } delete_on_throw{p_, std::uncaught_exceptions()};
    n_draft = n_draft_ = p_->n_draft_cap; // (0 without an MTP block)
    // the wide pass (T >= 6) runs the pipelined q8-cache decode attention only (attn_core<T > 5>: attn_core_w48q, or
    // attn_core_w48m for the low-bit MK_KV sides)
    if (n_draft > 4 && !p_->hb.kvq8) throw std::runtime_error("n_draft > 4 needs the q8 KV cache (--kv q8)");
    reset();
    if (const char * pol = getenv("MK_POLICY")) set_policy(pol);
#if MK_SPEC_LOG
    if (const char * f = getenv("MK_SPEC_LOG_FILE")) p_->slog = fopen(f, "a");
#endif
}

// "key=value,...": depth=0|1|2 (see Impl::Policy), dmin=N; on-device early stop (Launch::gate_*): drop=F,
// stop=F (every draft), dropK=F / stopK=F (draft K = 1..7), chain=0|1; h1=0|1 (sampled drafts when
// sampling); dq4=0|1 (MK_DRAFT_Q4=1 only: the draft head reads lm_head / its Q4_K copy; kept until changed);
// droplast=F (the deepest draft), dctx=N (from position N on at most 4 drafts),
// chain=2 (sampled drafts: the chain multiplies each drafted token's probability under q' instead of the top-1's);
// g7=0|1 (the depth > 5 gates, Impl::Policy); blk=0 only (blk=1, block verification of sampled drafts, is refused);
// "" = defaults
void MKEngine::set_policy(const std::string & spec) {
    Impl::Policy pol;
    float dk[MAXT - 1], sk[MAXT - 1]; // per-draft overrides
    for (int j = 0; j < MAXT - 1; j++) dk[j] = sk[j] = -1.f;
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string kv = spec.substr(i, j - i);
        i = j + 1;
        if (kv.empty()) continue;
        const size_t e = kv.find('=');
        if (e == std::string::npos) throw std::runtime_error("policy: expected key=value, got " + kv);
        const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
        if (k == "drop" || k == "stop" || (k.size() == 5 && (k.compare(0, 4, "drop") == 0 || k.compare(0, 4, "stop") == 0)))
            pol.g7 = 0; // explicit gates replace the depth > 5 defaults
        if (k == "depth") pol.depth = std::stoi(v);
        else if (k == "dmin") pol.dmin = std::stoi(v);
        else if (k == "drop") for (float & x : pol.gate_drop) x = std::stof(v);
        else if (k == "stop") for (float & x : pol.gate_stop) x = std::stof(v);
        else if (k.size() == 5 && (k.compare(0, 4, "drop") == 0 || k.compare(0, 4, "stop") == 0) && k[4] >= '1' &&
                 k[4] < '1' + MAXT - 1) // dropK / stopK: draft K alone (e.g. drop4: the T = 5 verify row)
            (k[0] == 'd' ? dk : sk)[k[4] - '1'] = std::stof(v);
        else if (k == "droplast") pol.drop_last = std::stof(v), pol.g7 = 0; // the deepest draft (k = n_draft - 1)
        else if (k == "g7") pol.g7 = std::stoi(v);
        else if (k == "dctx") pol.dctx = std::stoi(v); // from position dctx on: at most 4 drafts
        else if (k == "chain") pol.gate_chain = std::stoi(v);
        else if (k == "h1") pol.h1 = std::stoi(v);
        else if (k == "blk") {
            if (std::stoi(v)) throw std::runtime_error("policy blk=" + v + " was removed (block verification of sampled drafts; blk=0 is the only value)");
        }
        else if (k == "mtpq") throw std::runtime_error("policy mtpq was removed: the MTP block is always re-quantised to Q4_K");
        else if (k == "dq4") p_->set_draft_q4(std::stoi(v)); // MK_DRAFT_Q4=1 only; from the next launch
        else throw std::runtime_error("policy: unknown key " + k);
    }
    for (int j = 0; j < MAXT - 1; j++) {
        if (dk[j] >= 0.f) pol.gate_drop[j] = dk[j];
        if (sk[j] >= 0.f) pol.gate_stop[j] = sk[j];
    }
    p_->pol = pol;
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() { p_->bench(); }

void MKEngine::set_penalty(float presence, float frequency) { p_->set_penalty(presence, frequency); }

void MKEngine::set_kv_emu(const std::string & spec) { p_->set_kv_emu(spec); }

void MKEngine::reset() {
    p_->pcnt_clear();
    p_->reset();
    p_->demoted = 0;
    p_->seq.clear();
    p_->spans_cut(0);
    for (auto & k : p_->ck) k.pos = -1;
    p_->dv_reset();
    stats = MKStats();
}

void MKEngine::cancel(bool on) { p_->pf_cancel.store(on ? 1 : 0); }

void MKEngine::prefill_progress(int * done, int * total) const {
    const int a = p_->pf_from.load(), b = p_->pf_to.load(), q = p_->pf_pos.load();
    *total = std::max(b - a, 0);
    *done = std::min(std::max(q - a, 0), *total);
}

int MKEngine::position() const { return p_->hctl.n; }

void MKEngine::set_n_draft(int n) {
    if (!p_->has_mtp && n > 0) { // (no MTP block: depth 0, a notice once)
        static bool told = false;
        if (!told) fprintf(stderr, "[mk] n_draft %d -> 0 (no MTP layer)\n", n);
        told = true;
        n = 0;
    }
    if (n < 0 || n > p_->n_draft_cap)
        throw std::runtime_error("set_n_draft: 0 .. " + std::to_string(p_->n_draft_cap) + " (the engine's construction depth)");
    p_->n_draft_max = p_->ndraft = p_->nd_avail = n;
    n_draft_ = n;
}

// token ids outside the vocabulary would index host arrays and embedding rows: refused before anything changes
static void check_ids(const std::vector<int> & toks, size_t pos0) {
    for (size_t i = 0; i < toks.size(); i++)
        if (toks[i] < 0 || toks[i] >= VOCAB)
            throw std::runtime_error("token id " + std::to_string(toks[i]) + " at position " + std::to_string(pos0 + i) +
                                     " out of range (vocabulary " + std::to_string(VOCAB) + ")");
}

int MKEngine::prefill_state(const std::vector<int> & tokens, const std::string & dir, int * how) {
    if (dir.empty()) throw std::runtime_error("prefill_state: no directory");
    struct Clear {
        Impl * p;
        ~Clear() { p->st_dir.clear(); }
    } clear{p_};
    p_->st_dir = dir;
    p_->st_how = 0;
    const int r = prefill(tokens);
    if (how) *how = p_->st_how;
    return r;
}

int MKEngine::prefill(const std::vector<int> & tokens, int ck_at) {
    return prefill(tokens, ck_at >= 0 ? std::vector<int>{ck_at} : std::vector<int>{}, ck_at);
}

int MKEngine::prefill(const std::vector<int> & tokens, const std::vector<int> & cks_in, int ck_exact) {
    return prefill(tokens, cks_in, ck_exact, {});
}

int MKEngine::prefill(const std::vector<int> & tokens, const std::vector<int> & cks_in, int ck_exact,
                      const std::vector<MKImage> & images_in) {
    struct PinClear { // (ck_pin's requests are for this call only, whatever happens to it)
        Impl * p;
        ~PinClear() { p->pin_req.clear(); }
    } pin_clear{p_};
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    check_ids(tokens, p_->seq.size());
    std::vector<int> cks = cks_in;
    if (p_->exact && ck_exact >= 0) { // MK_EXACT: on the scan grid like the others (a side scan's state is not on it)
        cks.push_back(ck_exact);
        ck_exact = -1;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    std::vector<int> & seq = p_->seq;
    if ((int) (seq.size() + tokens.size()) + 8 > p_->ctx) throw std::runtime_error("prompt exceeds context");
    // Everything from the committed delta-net position on is fed: on a continued sequence that is
    // the last emitted token (not yet processed) and any accepted drafts not yet folded into the
    // recurrent state, followed by the new tokens.  Their KV / MTP rows are simply rewritten.
    const int n0 = c.dn_c;
    const int s0 = (int) seq.size(), end = s0 + (int) tokens.size(); // this prompt's positions: s0 .. end - 1
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    const bool rows_only = p_->pf_mid && batched; // (returns -1: no output token)
    // Images, checked before anything changes. The spans after this prompt: the earlier ones before n0, then these.
    // An earlier span cut by n0 (a rewind to a checkpoint inside it) must come again (same position and grid): its rows
    // from n0 on are fed from it.
    using Span = Impl::Span;
    std::vector<Span> nsp;
    const Span * cut = nullptr;
    for (const Span & s : p_->spans)
        if (s.pos + s.n() <= n0) nsp.push_back(s);
        else if (s.pos < n0) cut = &s;
    std::vector<MKImage> images = images_in;
    std::sort(images.begin(), images.end(), [](const MKImage & a, const MKImage & b) { return a.pos < b.pos; });
    std::vector<int> erow; // the batched part's rows in emb_host (positions n0 .. end - 2), -1: token_embd
    if (!images.empty() || cut) {
        if (!batched || end - n0 < 2 || !p_->st_dir.empty())
            throw std::runtime_error("images need the batched prefill (not MK_PREFILL=mk, not prefill_state)");
        bool cut_ok = false;
        int prev = nsp.empty() ? 0 : nsp.back().pos + nsp.back().n();
        size_t rows = 0;
        for (const MKImage & m : images) {
            if (m.nx <= 0 || m.ny <= 0 || !m.embd) throw std::runtime_error("image: empty grid or no embedding rows");
            if (m.pos < prev) throw std::runtime_error("images overlap, or one starts inside an earlier one");
            if (m.pos + (long long) m.nx * m.ny > end - 1)
                throw std::runtime_error("an image must end before the prompt's last token");
            const int n = m.nx * m.ny; // (no overflow: checked against the prompt's end)
            if (m.pos < s0) { // continues the span cut by n0
                if (!cut || cut->pos != m.pos || cut->nx != m.nx || cut->ny != m.ny)
                    throw std::runtime_error("image at " + std::to_string(m.pos) + " starts before this prompt (" + std::to_string(s0) +
                                             ") and is not the image the sequence was rewound into");
                cut_ok = true;
            }
            nsp.push_back({m.pos, m.nx, m.ny});
            rows += (size_t) (m.pos + n - std::max(m.pos, n0));
            prev = m.pos + n;
        }
        if (cut && !cut_ok)
            throw std::runtime_error("the sequence was rewound to " + std::to_string(n0) + ", inside the image at " +
                                     std::to_string(cut->pos) + ": pass that image again");
        p_->emb_reserve(rows * D);
        erow.assign((size_t) (end - 1 - n0), -1);
        size_t r = 0;
        for (const MKImage & m : images) {
            const int a = std::max(m.pos, n0), n = m.pos + m.nx * m.ny - a;
            memcpy(p_->emb_host + r * D, m.embd + (size_t) (a - m.pos) * D, (size_t) n * D * sizeof(float));
            for (int q = 0; q < n; q++) erow[(size_t) (a - n0 + q)] = (int) (r + q);
            r += n;
        }
    }
    // MKEngine::cancel: a batched part stopped at a chunk boundary leaves a partial prefill in the KV rows, the recurrent
    // state and its checkpoint slots, so nothing of this call may count as processed. The checkpoints this call saved or
    // evicted are dropped, the last earlier one is restored (KV rows before it were never rewritten; without one: a
    // reset), the draft vocabulary and the image spans go back to their state before the call.
    struct ClearCancel {
        Impl * p;
        ~ClearCancel() { p->pf_cancel.store(0); }
    } clear_cancel{p_};
    const std::vector<Span> spans0 = p_->spans;
    const size_t dv0 = p_->dv_ids.size();
    const int dvn0 = p_->dv_n;
    std::vector<int> ck0;
    for (const auto & k : p_->ck) ck0.push_back(k.pos);
    auto stop_if_cancelled = [&] {
        if (!p_->pf_cancelled) return;
        p_->pf_cancelled = false;
        int best = -1;
        for (size_t k = 0; k < p_->ck.size(); k++) {
            Impl::CkSlot & s = p_->ck[k];
            if (s.pos != ck0[k]) s.pos = -1;
            else if (s.pos >= 0 && (best < 0 || s.pos > p_->ck[best].pos)) best = (int) k;
        }
        p_->spans = spans0;
        const int back = best >= 0 ? p_->ck[best].pos : 0;
        if (best >= 0) {
            p_->ck_restore(best);
            p_->spans_cut(back);
            if (back < s0 || dvn0 == s0) { // (as a fresh prefill of seq[0, back) has it)
                p_->dv_rebuild(back, p_->spans);
            } else {
                p_->dv_truncate(dv0);
                p_->dv_n = dvn0;
            }
        } else {
            reset();
        }
        throw MKCancelled("prefill of positions " + std::to_string(n0) + ".." + std::to_string(end - 1) + " cancelled at " +
                          std::to_string(p_->pf_stop) + "; the sequence is back at position " + std::to_string(back));
    };
    p_->pcnt_clear(); // a new response: the penalties count its generated tokens only
    if (p_->fault_at && ++p_->fault_seen == p_->fault_at) p_->fault_now = true;
    seq.insert(seq.end(), tokens.begin(), tokens.end());
    const bool dv_fresh = p_->dv_n == s0; // (after a reset, rewind or truncate: as a fresh prefill of seq[0, end) builds it)
    if (dv_fresh) {
        p_->dv_rebuild(end, nsp);
    } else if (images.empty()) {
        p_->dv_add(tokens.data(), tokens.size());
    } else { // (the draft vocabulary takes the prompt's text tokens, not the image rows' placeholder ids)
        std::vector<int> text;
        size_t k = 0;
        for (int q = s0; q < end; q++) {
            while (k < images.size() && images[k].pos + images[k].nx * images[k].ny <= q) k++;
            if (k == images.size() || q < images[k].pos) text.push_back(seq[q]);
        }
        p_->dv_add(text.data(), text.size());
    }
    if (!dv_fresh) p_->dv_n = -1;
    p_->spans = nsp;
    p_->rope_sync();
    const int * er = erow.empty() ? nullptr : erow.data();
    const std::vector<int> feed(seq.begin() + n0, seq.end());
    c.mtp_valid = std::min(c.mtp_valid, n0);
    size_t i = 0;
    bool end_save = !p_->exact; // a checkpoint at the end of the prompt, unless ck_exact is inside the batched part
    { // prefill_progress: this call's batched part (an empty range while it is set: total 0)
        const int to = rows_only ? n0 + (int) feed.size() : batched && feed.size() >= 2 ? n0 + (int) feed.size() - 1 : n0;
        p_->pf_to.store(n0);
        p_->pf_from.store(n0);
        p_->pf_pos.store(n0);
        p_->pf_to.store(to);
    }
    if (batched && (feed.size() >= 2 || rows_only)) {
        if (p_->exact && n0 % 32 != 0 && !p_->exact_off_grid) {
            p_->exact_off_grid = true;
            fprintf(stderr, "[mk] warning: MK_EXACT: a prefill from position %d, off the 32-token scan grid: its rows depend on "
                            "where it starts (not exact)\n", n0);
        }
        // all but the last token in batched kernels; the last one goes through the megakernel,
        // which produces the first output token and the MTP drafts (rows_only: every token in batched kernels)
        const std::vector<int> head(feed.begin(), feed.end() - (rows_only ? 0 : 1));
        const int hend = n0 + (int) head.size(); // the batched part: positions n0 .. hend - 1
        end_save = !p_->exact && !(ck_exact > n0 && ck_exact < hend); // (MK_EXACT: no megakernel state in checkpoints)
        // post-prefill state file (prefill_state): the batched part's result restored from the file, or computed and saved
        const bool st = !p_->st_dir.empty() && n0 == 0 && cks.empty() && ck_exact < 0;
        std::string st_path, st_key;
        if (st) p_->st_name(head, st_path, st_key);
        if (st && p_->st_load(st_path, st_key, hend)) {
            p_->st_how = 1;
        } else {
        if (p_->stg_n < 2) { // too few staging slots (MK_PF_CAP below ~340): two pf_runs around ck_exact, other positions ignored
            if (ck_exact > n0 && ck_exact < hend) { // (never with MK_EXACT: its ck_exact joined cks, which this path ignores)
                p_->pf_run(std::vector<int>(head.begin(), head.begin() + (ck_exact - n0)), n0, nullptr, nullptr, 0, nullptr, er);
                stop_if_cancelled();
                c.dn_c = ck_exact;
                c.mtp_valid = c.dn_c;
                CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
                const int si = p_->ck_assign({ck_exact}, ck_exact, (int) seq.size())[0];
                if (si >= 0) p_->ck_save_end(si);
                p_->pf_run(std::vector<int>(head.begin() + (ck_exact - n0), head.end()), ck_exact, nullptr, nullptr, 0, nullptr,
                           er ? er + (ck_exact - n0) : nullptr);
                stop_if_cancelled();
            } else {
                p_->pf_run(head, n0, nullptr, nullptr, 0, nullptr, er);
                stop_if_cancelled();
            }
        } else {
            std::vector<int> pins; // (pin requests no checkpoint covers yet: this prefill saves them)
            for (int P : p_->pin_req)
                if (p_->pin_slot(P) < 0) pins.push_back(P);
            std::vector<char> wpin;
            const std::vector<int> want = p_->ck_positions(cks, ck_exact, n0, hend, pins, &wpin);
            const std::vector<int> slot = p_->ck_assign(want, ck_exact, (int) seq.size(), wpin);
            std::vector<Impl::Snap> snaps;
            for (size_t k = 0; k < want.size(); k++)
                if (slot[k] >= 0) snaps.push_back({want[k], slot[k]});
            const auto t0 = std::chrono::steady_clock::now();
            p_->pf_run(head, n0, nullptr, nullptr, 0, &snaps, er);
            stop_if_cancelled();
            for (size_t k = 0; k < want.size(); k++) { // (the state after position want[k] - 1)
                if (slot[k] < 0) continue;
                Impl::CkSlot & s = p_->ck[slot[k]];
                s.ctl = c;
                s.ctl.dn_c = s.ctl.mtp_valid = want[k];
                s.pos = want[k];
                s.pin = false;
                if (wpin[k]) p_->pin_set(slot[k], p_->pin_stamp);
            }
            if (p_->memlog && !snaps.empty()) {
                fprintf(stderr, "[mk-mem] prefill %d..%d with %zu checkpoints (", n0, hend, snaps.size());
                for (const auto & q : snaps) fprintf(stderr, " %d%s", q.pos, p_->ck[q.slot].pin ? " (pinned)" : "");
                fprintf(stderr, " ) in %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        if (st) {
            p_->st_save(st_path, st_key, hend);
            p_->st_how = 2;
        }
        }
        i = head.size();
        c.dn_c = hend;
        c.mtp_valid = c.dn_c;
    }
    if (rows_only) { // (the next chunk continues from here: its prefill reads the control block back)
        c.n = c.dn_c;
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        CK(cudaGetLastError());
        return -1;
    }
    while (i < feed.size()) {
        const int T = att_cap_T(c.dn_c, (int) std::min<size_t>(MAXT_PF, feed.size() - i));
        const bool last = i + T == feed.size();
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->ndraft = p_->depth_top(c.n);
        p_->launch(MODE_PREFILL, T, last ? 1 : 0);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        i += T;
    }
    p_->nd_avail = p_->ndraft > 0 ? std::min(p_->ndraft, c.nd_prod) : 0;
    p_->all_acc = true;
    p_->last_acc = p_->nd_avail;
    CK(cudaGetLastError());
#if MK_SPEC_LOG
    p_->slog_drafts();
#endif
    if (end_save && !p_->pf_mid && p_->ck_find(c.dn_c) < 0) {
        const int si = p_->ck_assign({c.dn_c}, c.dn_c, c.dn_c)[0];
        if (si >= 0) p_->ck_save_end(si);
    }
    seq.push_back(c.out[0]);
#if MK_SPEC_LOG
    p_->slog_prompt(n0);
#endif
    return c.out[0];
}

void MKEngine::set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    if (temp > 0.f && (top_k <= 0 || top_k > TOPK_MAX))
        throw std::runtime_error("sampling needs 1 <= top_k <= " + std::to_string(TOPK_MAX));
    p_->smp_temp = temp;
    p_->smp_top_p = top_p;
    p_->smp_top_k = top_k;
    p_->smp_min_p = min_p;
    p_->smp_seed = seed;
}

const std::vector<int> & MKEngine::sequence() const { return p_->seq; }
int MKEngine::committed() const { return p_->hctl.dn_c; }
std::vector<int> MKEngine::checkpoints() const {
    std::vector<int> v;
    for (const auto & k : p_->ck)
        if (k.pos >= 0) v.push_back(k.pos);
    std::sort(v.begin(), v.end());
    return v;
}
int MKEngine::checkpoint_pos() const {
    const std::vector<int> v = checkpoints();
    return v.empty() ? -1 : v.back();
}
bool MKEngine::rewind_to(int pos) {
    const int si = p_->ck_find(pos);
    if (si < 0 || pos > (int) p_->seq.size()) return false;
    p_->ck_restore(si);
    p_->spans_cut(pos);
    p_->dv_rebuild(pos, p_->spans); // (the abandoned continuation's tokens leave the draft vocabulary)
    return true;
}
bool MKEngine::rewind_to_checkpoint() { return rewind_to(checkpoint_pos()); }

int MKEngine::ck_pin(const std::vector<int> & positions) {
    Impl & I = *p_;
    I.pin_req.clear();
    if (I.pin_max <= 0) return 0;
    const unsigned t = I.pin_stamp = ++I.pin_clock;
    int n = 0;
    for (size_t k = 0; k < positions.size() && (int) k < I.pin_max; k++) {
        const int si = I.pin_slot(positions[k]);
        if (si >= 0) {
            I.pin_set(si, t);
            n++;
        } else if (positions[k] > 0) {
            I.pin_req.push_back(positions[k]);
        }
    }
    return n;
}

std::vector<int> MKEngine::ck_pinned() const {
    std::vector<int> v;
    for (const auto & k : p_->ck)
        if (k.pos >= 0 && k.pin) v.push_back(k.pos);
    std::sort(v.begin(), v.end());
    return v;
}

void MKEngine::truncate(int len) {
    std::vector<int> & seq = p_->seq;
    if (len < p_->hctl.dn_c || len > (int) seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    p_->ring_rewind(len); // MK_KV_WIN: the dropped positions' ring slots held window rows
    seq.resize(len);
    p_->spans_cut(len);
    p_->dv_rebuild(len, p_->spans);
}

// ---- conversation state in host memory: side requests, parked conversations; Impl "host store" ----
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void MKEngine::store_budget(size_t bytes) {
    p_->hs_budget = bytes;
    p_->hs_trim();
}

void MKEngine::store_reserve(size_t bytes) {
    Impl & I = *p_;
    while (I.hs_free.size() * I.hs_chunk < bytes) {
        char * c = I.hs_new();
        if (!c) break;
        I.hs_free.push_back(c);
    }
}

MKEngine::StoreInfo MKEngine::store_info() const {
    StoreInfo s;
    s.budget = p_->hs_budget;
    s.allocated = p_->hs_alloc;
    s.used = p_->hs_used();
    s.parked = (int) p_->parked.size();
    s.last_ms = p_->hs_ms;
    s.last_bytes = p_->hs_bytes;
    s.side_ms = p_->sw_ms_out;
    s.side_park_ms = p_->hs_park_ms;
    s.side_park_bytes = p_->hs_park_bytes;
    return s;
}

void MKEngine::kv_sizes(size_t * per_token, size_t * rings, size_t * state, size_t * checkpoint) const {
    *per_token = p_->kv_token_bytes();
    *rings = p_->ring_bytes();
    *state = p_->ck_block_bytes();
    *checkpoint = p_->ck_block_bytes() + p_->ring_bytes_ck;
}

bool MKEngine::side_begin() {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("side_begin: a side request is already running");
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    I.sw_book = I.book_get();
    I.sw_ck.clear();
    for (const auto & k : I.ck) I.sw_ck.push_back({k.pos, k.ctl, k.ring_ok, false, k.pin, k.pin_t});
    I.sw_top = I.seq_top();
    I.sw_hi = I.sw_rhi = 0;
    I.sw_broken = false;
    I.sw_on = true; // (from here a CUDA error drops the kept sequence: the C API calls side_drop)
    const bool ok = I.live_put(I.sw_blob);
    I.wait_device("side request copies", std::max(I.wd_limit, 120.0));
    if (!ok) { // (the store is full: nothing changed)
        I.sw_on = false;
        I.hs_release(I.sw_blob);
        return false;
    }
    for (auto & k : I.ck) k.pos = -1;
    reset();
    I.sw_ms_out = ms_since(t0);
    return true;
}

int MKEngine::side_end() {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_end: no side request is running");
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    if (I.sw_broken) {
        side_drop();
        return 0;
    }
    I.side_restore();
    I.hs_ms = ms_since(t0);
    return 1;
}

int MKEngine::side_end_park(int max_checkpoints, int * id) {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_end_park: no side request is running");
    *id = 0;
    I.hs_park_ms = 0;
    I.hs_park_bytes = 0;
    auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    if (I.sw_broken) { // (the kept sequence is lost: the side request's conversation stays, with its checkpoints)
        I.sw_on = false;
        I.hs_release(I.sw_blob);
        return 0;
    }
    if (!I.seq.empty()) {
        const int pid = I.park_entry(max_checkpoints); // (before side_restore writes the kept sequence's rows and blocks back)
        if (pid > 0) {
            *id = pid;
            I.hs_park_bytes = I.hs_bytes;
            I.hs_park_ms = ms_since(t0);
        }
        t0 = std::chrono::steady_clock::now();
    }
    I.side_restore();
    I.hs_ms = ms_since(t0);
    return 1;
}

int MKEngine::side_unpark(int id) {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_unpark: no side request is running");
    if (!I.seq.empty()) throw std::runtime_error("side_unpark: the side request has a sequence already");
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return -1;
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    Impl::Parked & e = it->second;
    const int top = std::max({e.book.kv_top, (int) e.book.seq.size(), e.book.ctl.dn_c});
    if (!I.sw_broken) { // room for what the kept sequence moves out first (else nothing changes)
        const int rows = std::min(I.sw_top, (top + MAXT + Impl::SW_BLK - 1) / Impl::SW_BLK * Impl::SW_BLK) - I.sw_hi;
        size_t need = I.kv_token_bytes() * (size_t) std::max(0, rows) + 2 * I.hs_chunk; // (+ the last chunks' slack)
        if (I.hb.kvwin > 0) need += I.ring_bytes() / I.hb.kvrc * (size_t) (I.hb.kvrc - I.sw_rhi);
        for (const auto & c : e.cks)
            if (I.sw_ck_held(c.slot)) need += I.ck_block_bytes() + (I.sw_ck[c.slot].ring_ok ? I.ring_bytes_ck : 0);
        if (need > I.hs_room()) return 0;
    }
    I.sw_need(top, true); // (the entry's rows, every ring slot)
    for (const auto & c : e.cks)
        if (!I.sw_ck_take(c.slot)) I.sw_ck[c.slot].pos = -1; // (no room after all: the kept sequence loses that checkpoint)
    I.blob_get(e.blob);
    I.wait_device("unpark copies", std::max(I.wd_limit, 120.0));
    I.book_set(e.book);
    for (const auto & c : e.cks) {
        Impl::CkSlot & s = I.ck[c.slot];
        s.pos = c.pos;
        s.ctl = c.ctl;
        s.ring_ok = c.ring_ok;
        s.pin = c.pin;
        s.pin_t = c.pin_t;
        s.pending = false;
    }
    I.hs_bytes = e.blob.used;
    I.hs_ms = ms_since(t0);
    I.hs_release(e.blob);
    I.parked.erase(it);
    return 1;
}

void MKEngine::side_drop() {
    Impl & I = *p_;
    if (!I.sw_on) return;
    I.sw_on = false;
    if (cudaStreamSynchronize(0) != cudaSuccess) cudaGetLastError(); // (a failed call's copies into sw_blob land first)
    I.hs_release(I.sw_blob);
    reset();
}

bool MKEngine::side_active() const { return p_->sw_on; }

int MKEngine::park(int max_checkpoints) {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("park: a side request is running");
    if (I.seq.empty()) return 0;
    I.hs_quiesce();
    return I.park_entry(max_checkpoints);
}

bool MKEngine::unpark(int id) {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("unpark: a side request is running");
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return false;
    const auto t0 = std::chrono::steady_clock::now();
    reset(); // (the live sequence and its checkpoints go)
    I.hs_quiesce();
    Impl::Parked & e = it->second;
    I.blob_get(e.blob);
    I.wait_device("unpark copies", std::max(I.wd_limit, 120.0));
    I.book_set(e.book);
    for (const auto & c : e.cks) {
        Impl::CkSlot & s = I.ck[c.slot];
        s.pos = c.pos;
        s.ctl = c.ctl;
        s.ring_ok = c.ring_ok;
        s.pin = c.pin;
        s.pin_t = c.pin_t;
        s.pending = false;
    }
    I.hs_bytes = e.blob.used;
    I.hs_ms = ms_since(t0);
    I.hs_release(e.blob);
    I.parked.erase(it);
    return true;
}

bool MKEngine::park_drop(int id) {
    Impl & I = *p_;
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return false;
    I.hs_release(it->second.blob);
    I.parked.erase(it);
    return true;
}

size_t MKEngine::park_need(int max_checkpoints) const {
    const Impl & I = *p_;
    if (I.seq.empty()) return 0;
    size_t n = I.kv_token_bytes() * (size_t) I.seq_top() + I.ring_bytes() + I.ck_block_bytes();
    for (int si : I.park_cks(max_checkpoints)) n += I.ck_block_bytes() + (I.ck[si].ring_ok ? I.ring_bytes_ck : 0);
    return (n + I.hs_chunk - 1) / I.hs_chunk * I.hs_chunk;
}

size_t MKEngine::park_bytes(int id) const {
    auto it = p_->parked.find(id);
    return it == p_->parked.end() ? 0 : it->second.blob.chunks.size() * p_->hs_chunk;
}

std::vector<int> MKEngine::score(const std::vector<int> & tokens) {
    check_ids(tokens, 0);
    reset();
    std::vector<int> pred;
    const char * mode = getenv("MK_PREFILL");
    if (!(mode && std::string(mode) == "mk")) {
        p_->pf_run(tokens, 0, &pred);
        reset();
        return pred;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    size_t i = 0;
    while (i < tokens.size()) {
        const int T = att_cap_T((int) i, (int) std::min<size_t>(MAXT_PF, tokens.size() - i));
        c.n = (int) i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 1);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        for (int k = 0; k < T; k++) pred.push_back(c.tgt[k]);
        i += T;
    }
    return pred;
}

int MKEngine::n_vocab() { return VOCAB; }
int MKEngine::n_embd() { return D; }
int MKEngine::warm_launch(void * stream, const unsigned * stop, double ms, int blocks, int threads, int spin) {
    if (blocks < 1 || threads < 1 || threads > 1024 || !(ms > 0.0)) return (int) cudaErrorInvalidValue;
    mkl::kw_warm(blocks, threads, stop, (unsigned long long) (ms * 1e6), spin, (cudaStream_t) stream);
    return (int) cudaGetLastError();
}

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink) {
    score_logits(tokens, mk_from, first, sink, {});
}

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink,
                            const std::vector<MKImage> & images) {
    const int N = (int) tokens.size();
    if (N + 8 > p_->ctx) throw std::runtime_error("sequence exceeds context");
    check_ids(tokens, 0);
    reset();
    p_->logits_alloc();
    mk_from = std::min(std::max(mk_from, 0), N);
    std::vector<int> erow;
    if (!images.empty()) { // (the batched part's rows, as prefill stages them)
        size_t rows = 0;
        int prev = 0;
        std::vector<MKImage> im = images;
        std::sort(im.begin(), im.end(), [](const MKImage & a, const MKImage & b) { return a.pos < b.pos; });
        for (const MKImage & m : im) {
            if (m.nx <= 0 || m.ny <= 0 || !m.embd || m.pos < prev || m.pos + m.nx * m.ny > mk_from)
                throw std::runtime_error("score_logits: images must be disjoint, inside the batched part (mk_from)");
            prev = m.pos + m.nx * m.ny;
            rows += (size_t) m.nx * m.ny;
            p_->spans.push_back({m.pos, m.nx, m.ny});
        }
        p_->rope_sync();
        p_->emb_reserve(rows * D);
        erow.assign((size_t) mk_from, -1);
        size_t r = 0;
        for (const MKImage & m : im) {
            const int n = m.nx * m.ny;
            memcpy(p_->emb_host + r * D, m.embd, (size_t) n * D * sizeof(float));
            for (int q = 0; q < n; q++) erow[(size_t) m.pos + q] = (int) (r + q);
            r += n;
        }
    }
    if (mk_from > 0)
        p_->pf_run(std::vector<int>(tokens.begin(), tokens.begin() + mk_from), 0, nullptr, first < mk_from ? &sink : nullptr, first,
                   nullptr, erow.empty() ? nullptr : erow.data());
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    c.dn_c = mk_from;
    std::vector<float> rows((size_t) MAXT * VOCAB);
    for (int i = mk_from; i < N;) {
        const int T = att_cap_T(i, std::min(MAXT_PF, N - i));
#if MK_KV_EMU
        p_->kv_demote_to(i - p_->hb.kvemu_win); // recent-window emulation
#endif
        c.n = i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 2);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        const int s = std::max(0, first - i);
        if (s < T) {
            CK(cudaMemcpy(rows.data(), p_->hb.logits + (size_t) s * VOCAB, (size_t) (T - s) * VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
            sink(i + s, T - s, rows.data());
        }
        i += T;
    }
    reset();
}

int MKEngine::step(std::vector<int> & out) {
    const int T = p_->step_live(out);
    const Ctl & c = p_->hctl;
    stats.cycles++;
    stats.drafted += T - 1;
    stats.accepted += c.n_acc;
    stats.last_T = T;
    stats.last_nd = p_->nd_avail;
    stats.last_acc = c.n_acc;
    return c.n_out;
}

// ---- parallel slots (open-jet's slot API over lanes, see Impl::Lane) ----
// Every slot call leaves the engine's own sequence live again, so the single-sequence API is unaffected between them.
// An MKCudaError in one marks the lanes it ran on dead (Impl::lanes_call): refused until slot_begin starts them afresh.
#define MK_LANE_OUT()                                       \
    struct LaneOut {                                        \
        Impl * p;                                           \
        ~LaneOut() { p->lane_out_nothrow(); }               \
    } lane_out_guard{p_}

int MKEngine::slots_open(int n, int ctx, int /*seg_tokens*/) { return p_->slots_open(n, ctx); }

int MKEngine::slots_count() const { return (int) p_->lanes.size(); }

int MKEngine::slot_begin(int slot, const std::vector<int> & toks, float temp, float top_p, int top_k, float min_p,
                         unsigned long long seed, int /*share_len*/, bool /*pin*/) {
    Impl & I = *p_;
    MK_LANE_OUT();
    if (toks.empty()) throw std::runtime_error("empty prompt");
    check_ids(toks, 0);
    int reused = 0;
    I.lanes_call({slot}, [&] {
        I.lane_in(slot);
        set_sampling(temp, top_p, top_k, min_p, seed);
        Impl::Lane & L = I.lanes[slot];
        L.first = -1;
        L.pend.clear();
        if ((int) toks.size() + 8 > I.ctx) throw std::runtime_error("prompt exceeds the slot's context");
        if (L.dead) { // (a failed call left its state undefined: from scratch, nothing reused)
            reset();
            L.dead = false;
            L.dead_why.clear();
        }
        size_t c = 0;
        while (c < I.seq.size() && c < toks.size() && I.seq[c] == toks[c]) c++;
        if (c < I.seq.size() || c == toks.size()) {
            // back to a position <= c from which at least the prompt's last token is fed: a truncate (emitted tokens past
            // the committed position), else the newest checkpoint there, else from scratch
            const int target = (int) std::min(c, toks.size() - 1);
            int best = -1;
            for (const auto & k : I.ck)
                if (k.pos >= 0 && k.pos <= target && k.pos > best) best = k.pos;
            if (target >= I.hctl.dn_c) truncate(target);
            else if (best >= 0 && rewind_to(best)) {
            } else reset();
        }
        reused = (int) I.seq.size();
        L.prompt = toks;
        L.pend.assign(toks.begin() + reused, toks.end());
    });
    return reused;
}

int MKEngine::slot_prefill(int slot, int budget) {
    Impl & I = *p_;
    MK_LANE_OUT();
    Impl::Lane & L = I.lane_ok(slot);
    int left = 0;
    I.lanes_call({slot}, [&] {
        I.lane_in(slot);
        if (budget <= 0) { // (abandoned: the sequence keeps what was prefilled)
            L.pend.clear();
            return;
        }
        if (L.pend.empty()) return;
        size_t n = std::min<size_t>(L.pend.size(), (size_t) budget);
        // MK_EXACT: a chunk before the last ends on the 32-token scan grid (the next grid point if the budget reaches none)
        if (I.exact && n < L.pend.size()) {
            const size_t s = I.seq.size(), e = (s + n) / 32 * 32;
            n = e > s ? e - s : std::min(L.pend.size(), (s / 32 + 1) * 32 - s);
        }
        const std::vector<int> chunk(L.pend.begin(), L.pend.begin() + n);
        L.pend.erase(L.pend.begin(), L.pend.begin() + n);
        try {
            if (L.pend.empty()) { // the prompt's last chunk: its first token, a checkpoint before the prompt's last token
                const int end = (int) I.seq.size() + (int) chunk.size();
                L.first = prefill(chunk, end - 1);
            } else { // a chunk before the last: batched rows only (with MK_PREFILL=mk its last row samples a token: dropped)
                struct Mid {
                    Impl * p;
                    ~Mid() { p->pf_mid = false; }
                } mid{p_};
                I.pf_mid = true;
                if (prefill(chunk) >= 0) truncate((int) I.seq.size() - 1);
            }
        } catch (const MKCancelled &) {
            // the sequence went back to a checkpoint before this call (or was reset): a prefix of the prompt, whose
            // rest is pending again (the next slot_prefill continues there)
            const size_t s = I.seq.size();
            if (s <= L.prompt.size() && std::equal(I.seq.begin(), I.seq.end(), L.prompt.begin()))
                L.pend.assign(L.prompt.begin() + s, L.prompt.end());
            else
                L.pend.clear(); // (not a prefix: nothing queued, the next slot_begin matches again)
            throw;
        }
        left = (int) L.pend.size();
    });
    return left;
}

std::vector<std::vector<int>> MKEngine::slots_step(const std::vector<int> & slots) {
    Impl & I = *p_;
    MK_LANE_OUT();
    for (int s : slots) I.lane_ok(s); // (every slot before any of them changes)
    std::vector<std::vector<int>> out(slots.size());
    std::vector<int> ks, at, firsts;
    for (size_t i = 0; i < slots.size(); i++) {
        Impl::Lane & L = I.lanes[slots[i]];
        if (L.first >= 0) { // (its prefill's token first: this call only emits it)
            out[i].push_back(L.first);
            firsts.push_back(slots[i]);
            continue;
        }
        ks.push_back(slots[i]);
        at.push_back((int) i);
    }
    I.lanes_call(ks, [&] {
        if (ks.size() == 1) {
            I.lane_in(ks[0]);
            std::vector<int> o;
            const Ctl & c = I.hctl;
            if (c.n + MAXT + 2 < I.ctx) step(o); // (else full: no tokens)
            out[at[0]] = o;
        } else if (ks.size() >= 2) {
            const auto r = I.lanes_step(ks); // (a failure there: every lane in ks, the batched launch wrote them all)
            for (size_t j = 0; j < ks.size(); j++) out[at[j]] = r[j];
        }
    });
    for (int s : firsts) I.lanes[s].first = -1; // (taken only now: a throw above leaves the token for the next call)
    return out;
}

// the next chunk of slot pslot's prompt, then the step (separate launches; the chunk first: a failure in it leaves the
// decoding slots unstepped, so no emitted token is lost with the call's error)
std::vector<std::vector<int>> MKEngine::slots_step_mixed(int pslot, int budget, const std::vector<int> & slots, int & remain) {
    p_->lane_ok(pslot); // (a dead slot fails the call before anything changes)
    for (int s : slots) p_->lane_ok(s);
    remain = slot_prefill(pslot, budget);
    return slots_step(slots);
}

std::vector<int> MKEngine::slot_sequence(int slot) {
    MK_LANE_OUT();
    p_->lane_in(slot);
    return p_->seq;
}

std::vector<float> MKEngine::slot_h(int slot) {
    MK_LANE_OUT();
    p_->lane_in(slot);
    return last_hidden();
}

// MKEngine::step's body on the live sequence; returns the launch's T
int MKEngine::Impl::step_live(std::vector<int> & out) {
    Ctl & c = hctl;
    if (c.n + MAXT + 2 >= ctx) throw std::runtime_error("context full"); // (rows past ctx: KV_PAD)
    // verify the previous launch's drafts, minus any that would put this launch's rows across an
    // attention-partition boundary (once per ATT_BUCKET positions; keeps greedy output depth-independent)
    // T >= 6 runs as one MAXT-row pass (padding rows): the whole n + 1 .. n + MAXT must lie in one partition, else at most 5
    int T = att_cap_T(c.n, 1 + nd_avail);
    if (T > 5 && att_cap_T(c.n, MAXT) < MAXT) T = 5;
    // Draft depth (an upper bound: the on-device gate may stop drafting earlier), by Policy::depth (default 1: the
    // full depth). Depth 0: after a cycle that rejected a draft, draft one token less (but at least 3), since a deeper
    // draft is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    const int top = depth_top(c.n); // n_draft (at most 4 from Policy::dctx on)
    int nd = top;
    if (pol.depth == 0) nd = (!all_acc && top > 3) ? top - 1 : top;
    else if (pol.depth == 2) nd = std::min(top, std::max(std::min(pol.dmin, top), 1 + last_acc));
    ndraft = nd;
#if MK_SPEC_LOG
    const int n0 = c.n;
    int drafts[MAXT];
    for (int i = 1; i < T; i++) drafts[i] = c.tok[i];
#endif
    launch(MODE_DECODE, T, 0);
    ctl_fetch(c);
    nd_avail = nd > 0 ? std::min(nd, c.nd_prod) : 0;
#if MK_SPEC_LOG
    slog_cycle(n0, T, drafts, c);
#endif
    for (int i = 0; i < c.n_out; i++) out.push_back(c.out[i]);
    seq.insert(seq.end(), c.out, c.out + c.n_out);
    dv_add(c.out, c.n_out);
    dv_n = -1;
    all_acc = c.n_acc == T - 1;
    last_acc = c.n_acc;
    return T;
}

std::vector<float> MKEngine::last_hidden() {
    std::vector<float> h(D);
    const int pos = p_->hctl.dn_c - 1;
    p_->ck_wait_restore();
    CK(cudaMemcpy(h.data(), p_->hb.h_hist + (size_t) (pos % H_RING) * D, D * 4, cudaMemcpyDeviceToHost));
    return h;
}

bool MKEngine::exact() const { return p_->exact; }

std::vector<int> MKEngine::draft_vocab() {
    const Impl & I = *p_;
    if (I.dv_in.empty()) return {};
    int n = 0;
    CK(mkl::get_cW(&n, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    std::vector<int> v((size_t) std::max(n, 0));
    if (n > 0) CK(cudaMemcpy(v.data(), I.hw.draft_ids, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    if (v != I.dv_ids || I.hw.head_d.N != n) throw std::runtime_error("draft vocabulary: the device's rows differ from the host's list");
    return v;
}

bool MKEngine::has_mtp() const { return p_->has_mtp; }

std::string MKEngine::Impl::file_ident(const std::string & p, const gguf::File & f) {
    char rp[PATH_MAX];
    if (!realpath(p.c_str(), rp)) snprintf(rp, sizeof rp, "%s", p.c_str());
    struct stat st{};
    stat(rp, &st);
    const size_t edge = std::min<size_t>(f.size, (size_t) 4 << 20);
    unsigned long long mh = wd_hash(f.base, edge);
    mh = wd_hash(f.base + f.size - edge, edge, mh);
    std::ostringstream k;
#ifdef _WIN32
    k << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtime << ".0";
#else
    k << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtim.tv_sec << "." << (long long) st.st_mtim.tv_nsec;
#endif
    k << " " << hex64(mh);
    return k.str();
}

// state_digest (test helper for the exact-reuse probes; slow, host-side hashing): 64-bit words
//   [0] 0x4d4b444701, [1] n = committed position (dn_c), [2] mtp_valid, [3] kvrc (0: no MK_KV_WIN ring),
//   [4] ring_lo = max(0, n - kvrc), [5] N_ATT_SLOTS, [6] NKV, [7] H_RING; then
//   per (slot, w = K / V, kv head g, position p < n): its row in the main buffer (values + scales);
//   per (slot, w, g, position ring_lo <= p < n): its q8 ring row (0: the side has no ring);
//   48 DeltaNet states, 48 conv rings, H_RING h ring rows (ring index order).
static unsigned long long dg_hash(const void * p, size_t nb, unsigned long long h = 0xcbf29ce484222325ULL) {
    const unsigned char * b = static_cast<const unsigned char *>(p);
    size_t i = 0;
    for (; i + 8 <= nb; i += 8) {
        unsigned long long w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x100000001b3ULL;
        h ^= h >> 31;
    }
    for (; i < nb; i++) h = (h ^ b[i]) * 0x100000001b3ULL;
    return h;
}
std::vector<unsigned long long> MKEngine::state_digest() {
    Impl & I = *p_;
    I.ck_wait_restore();
    I.ring_snap_guard();
    CK(cudaDeviceSynchronize());
    Ctl c;
    CK(cudaMemcpy(&c, I.hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    typedef unsigned long long u64;
    const int n = c.dn_c, rc = I.hb.kvwin > 0 ? I.hb.kvrc : 0, rlo = rc ? std::max(0, n - rc) : n;
    std::vector<u64> out = {0x4d4b444701ULL, (u64) n, (u64) c.mtp_valid, (u64) rc, (u64) rlo, (u64) I.kv_slots, (u64) NKV, (u64) H_RING};
    const size_t ctx = (size_t) I.hb.ctx, sbq = (HD / 32) * sizeof(__half); // (row stride per kv head)
    std::vector<unsigned char> vb, sb;
    auto d2h = [&](std::vector<unsigned char> & dst, const void * src, size_t nb) {
        dst.resize(std::max<size_t>(1, nb));
        if (nb) CK(cudaMemcpy(dst.data(), src, nb, cudaMemcpyDeviceToHost));
    };
    for (int s = 0; s < I.kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                size_t rb, srb = 0;
                if (I.hb.kvq8) {
                    rb = (size_t) HD * I.hb.kvb[w][s] / 8;
                    d2h(vb, I.hb.kvp[w][s] + (size_t) g * ctx * rb, n * rb);
                    srb = sbq;
                    d2h(sb, I.hb.kvs[w][s] + (size_t) g * ctx * (HD / 32), n * srb);
                } else {
                    rb = (size_t) HD * sizeof(__half);
                    d2h(vb, static_cast<const __half *>(w ? I.hb.vc : I.hb.kc) + ((size_t) s * NKV + g) * ctx * HD, n * rb);
                }
                for (int p = 0; p < n; p++) out.push_back(dg_hash(sb.data() + p * srb, srb, dg_hash(vb.data() + p * rb, rb)));
            }
    for (int s = 0; s < I.kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                if (!rc || !I.hb.kvrp[w][s]) {
                    out.insert(out.end(), n - rlo, 0ULL);
                    continue;
                }
                d2h(vb, I.hb.kvrp[w][s] + (size_t) g * rc * HD, (size_t) rc * HD);
                d2h(sb, I.hb.kvrs[w][s] + (size_t) g * rc * (HD / 32), (size_t) rc * sbq);
                for (int p = rlo; p < n; p++) {
                    const size_t r = p % rc;
                    out.push_back(dg_hash(sb.data() + r * sbq, sbq, dg_hash(vb.data() + r * HD, HD)));
                }
            }
    for (int l = 0; l < 48; l++) {
        d2h(vb, I.hb.dstate + l * Impl::CK_SL, Impl::CK_SL * 4);
        out.push_back(dg_hash(vb.data(), Impl::CK_SL * 4));
    }
    for (int l = 0; l < 48; l++) {
        d2h(vb, I.hb.conv + l * Impl::CK_CL, Impl::CK_CL * 4);
        out.push_back(dg_hash(vb.data(), Impl::CK_CL * 4));
    }
    for (int r = 0; r < H_RING; r++) {
        d2h(vb, I.hb.h_hist + (size_t) r * D, (size_t) D * 4);
        out.push_back(dg_hash(vb.data(), (size_t) D * 4));
    }
    return out;
}
