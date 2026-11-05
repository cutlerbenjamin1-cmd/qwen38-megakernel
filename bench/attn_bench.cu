// Correctness + throughput of the prefill attention kernels (prefill_attn.cuh) on a random KV cache in the engine's layouts
// (one attention slot, view 0): pk_attn8 (the reference, as the engine runs it with MK_PFX bit 64) against pk_attn9, bit for
// bit, for every format the engine instantiates.
//   nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas=-dlcm=cg bench/attn_bench.cu -o bench/attn_bench
//   ./attn_bench <fmt> <ctx> <reps> <T:n0> ...     e.g. ./attn_bench F 262144 5 1024:39000 1024:100000 1024:257000
// fmt: q8 (all q8, no ring), Vl (K q8, V q5 at run time), D (K q6, V q5, 2048-position q8 ring), F (K q5, V q5, ring),
// any<k><v> (run-time KB / VB, ring), e.g. any45. Per config: ms per launch (min, median of reps, the variants alternating), and the number
// of output floats that differ from the reference.
// env: AB_VARS (substring of "01": 0 pk_attn8, 1 pk_attn9; default both), EXA=1 (MK_EXACT form, unsplit), EXA=2 (split, + pk_attn8_combx),
// QROW=1 (QROW form, q8 / low-bit generic), QSCALE (Q multiplier, default 1), SPECIAL=1 (Inf / NaN / zero rows in Q and the
// K scales), SEED, WARM_MS (default 4000: clocks), STOP_AFTER_OLD=1 / NCU_VAR=<v> NCU_N=<launches> (only that variant, for ncu).
#define MK_SHARED_NS
#include "../mk_shared.cuh"
#include <random>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

namespace {
__constant__ Bufs cBs[NVIEW];
__constant__ int cView;
__shared__ int s_view;
#define cB cBs[s_view]
#define MK_VIEW_INIT()                          \
    do {                                        \
        if (threadIdx.x == 0) s_view = cView;   \
        __syncthreads();                        \
    } while (0)
__constant__ PBufs cP;

// ---- copies of the engine's device helpers (qwen38_mk.cu, prefill_gemm.cuh)
__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
namespace pf {
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void hmma16(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma16z(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2 *>(&u); }
} // namespace pf
__device__ __forceinline__ int kv_ws(int L_ctx) {
    if (cB.kvwin <= 0) return 0;
    const int ws = (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET - 1 - (cB.kvwin + ATT_BUCKET - 1) / ATT_BUCKET;
    return ws > 0 ? ws * ATT_BUCKET : 0;
}
__device__ __forceinline__ int kv_wsm(int ws) { return cB.kvrc > 0 ? ws % cB.kvrc : 0; }
struct KvFmt { int kb, vb, kr, vr; };
__device__ __forceinline__ KvFmt kv_fmt(int slot) {
    return {cB.kvb[0][slot], cB.kvb[1][slot], cB.kvrp[0][slot] != nullptr, cB.kvrp[1][slot] != nullptr};
}
__device__ __forceinline__ int kv_ring_tile(int ring, int ws, int base, int tp) {
    return ring ? max(0, (ws - base + tp - 1) / tp) : 0x7fffffff;
}
__device__ __forceinline__ void kv_tile_src(int w, int slot, int g, int p0, int ws, int wsm, int bits, bool ring, const int8_t *& q, const __half *& s) {
    if (ring) {
        int r0 = wsm + (p0 - ws);
        if (r0 >= cB.kvrc) r0 -= cB.kvrc;
        const size_t rr = (size_t) g * cB.kvrc + r0;
        q = cB.kvrp[w][slot] + rr * HD;
        s = cB.kvrs[w][slot] + rr * (HD / 32);
    } else {
        const size_t row = (size_t) g * cB.ctx + p0;
        q = cB.kvp[w][slot] + row * (HD * bits / 8);
        s = cB.kvs[w][slot] + row * (HD / 32);
    }
}
__device__ __forceinline__ int kv_src(int w, int slot, int g, int p0, int ws, int wsm, const int8_t *& q, const __half *& s) {
    const bool ring = cB.kvrp[w][slot] != nullptr && p0 >= ws;
    const int bits = cB.kvb[w][slot];
    kv_tile_src(w, slot, g, p0, ws, wsm, bits, ring, q, s);
    return ring ? 8 : bits;
}
__device__ __forceinline__ unsigned kv_hi(const uint8_t * p, int hb) {
    unsigned x = p[0];
    if (hb > 1) x |= (unsigned) p[1] << 8;
    if (hb > 2) x |= (unsigned) p[2] << 16;
    return x;
}
__device__ __forceinline__ unsigned kv_hi4(unsigned x, int k0, int hb) {
    const unsigned m = (1u << hb) - 1u;
    x >>= hb * k0;
    return ((x & m) | (((x >> hb) & m) << 8) | (((x >> (2 * hb)) & m) << 16) | (((x >> (3 * hb)) & m) << 24)) << 4;
}
template <int HB>
__device__ __forceinline__ unsigned kv_spread4(unsigned x) {
    if constexpr (HB == 1) return ((x & 0xfu) * 0x02040810u) & 0x10101010u;
    if constexpr (HB == 2) {
        x = ((x & 0xffu) * 0x10010u) & 0x00f000f0u;
        return (x * 0x41u) & 0x30303030u;
    }
    if constexpr (HB == 3) {
        x &= 0xfffu;
        x = (x | (x << 10)) & 0x003f003fu;
        x = (x | (x << 5)) & 0x07070707u;
        return x << 4;
    }
    return 0u;
}
template <int BITS, bool SIGNED = true>
__device__ __forceinline__ void kv_rows_expand(uint8_t * st, int np, int r0, int rs, int ld) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    unsigned wd[3], hi[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        wd[k] = 0u;
        hi[k] = 0u;
        if (r < np) {
            const uint8_t * row = st + r * ld;
            wd[k] = *reinterpret_cast<const unsigned *>(row + 4 * lane);
            if constexpr (HB == 1) hi[k] = row[128 + lane];
            if constexpr (HB == 2) hi[k] = *reinterpret_cast<const unsigned short *>(row + 128 + 2 * lane);
            if constexpr (HB == 3) hi[k] = (unsigned) row[128 + 3 * lane] | ((unsigned) row[129 + 3 * lane] << 8) | ((unsigned) row[130 + 3 * lane] << 16);
        }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        if (r < np) {
            unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu;
            if constexpr (HB > 0) {
                lo |= kv_spread4<HB>(hi[k]);
                hv |= kv_spread4<HB>(hi[k] >> (4 * HB));
            }
            if constexpr (SIGNED) {
                constexpr unsigned bias = (1u << (BITS - 1)) * 0x01010101u;
                lo = ((lo | 0x80808080u) - bias) ^ 0x80808080u;
                hv = ((hv | 0x80808080u) - bias) ^ 0x80808080u;
            }
            *reinterpret_cast<uint2 *>(st + r * ld + 8 * lane) = make_uint2(lo, hv);
        }
    }
}
template <bool SIGNED = true>
__device__ __forceinline__ void kv_tile_expand(uint8_t * st, int bits, int np, int r0, int rs, int ld) {
    switch (bits) {
    case 4:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<4, SIGNED>(st, np, r, rs, ld);
        break;
    case 5:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<5, SIGNED>(st, np, r, rs, ld);
        break;
    case 6:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<6, SIGNED>(st, np, r, rs, ld);
        break;
    case 7:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<7, SIGNED>(st, np, r, rs, ld);
        break;
    default:
        break;
    }
}

#include "../prefill_attn.cuh"

// ---- host
inline void a2_plan(int T, int n0, int slots, size_t part_floats, int & nfull, int & S) { // prefill_host.cu
    const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV, rem = slots > 0 ? N % slots : 0;
    nfull = N;
    S = 1;
    if (rem == 0) return;
    const int qt = (N - rem) % nqt;
    const int ntile_min = (n0 + qt * A2_Q + std::min(A2_Q, T - qt * A2_Q) - 1) / A2_K + 1;
    const int smax = std::min(std::min(32, ntile_min / 2), (int) std::min<size_t>(1024, part_floats / ((size_t) A2_PART * rem)));
    double best = 1.0;
    for (int s = 2; s <= smax; s++) {
        const double t = (double) ((rem * s + slots - 1) / slots) / s;
        if (t < best - 0.02) {
            best = t;
            S = s;
        }
    }
    if (S > 1) nfull = N - rem;
}

enum Form { F_Q8, F_D, F_F, F_ANY };
struct Plan {
    dim3 grid;
    int T, n0, nfull, S, segt, exh, exa;
    float *part, *part2;
};
#define L_(K, ...) K<__VA_ARGS__><<<p.grid, A2_NT, A8_SMEM>>>(p.T, p.n0, 0, p.nfull, p.S, p.part, 0, p.segt, p.exh, p.part2, p.exa)
// one launch of variant v of form f (qrow: QROW; exa: EXA)
template <int V> void launch1(Form f, bool klow, bool qrow, bool exa, const Plan & p);
#define LAUNCH_BODY(K)                                                                           \
    if (exa) {                                                                                   \
        if (f == F_D) L_(K, false, true, true, 6, 5);                                             \
        else if (f == F_F) L_(K, false, true, true, 5, 5);                                        \
        else if (klow) L_(K, false, true, true);                                                 \
        else if (f == F_Q8) Q8EXA;                                                               \
        else L_(K, false, false, true);                                                          \
    } else if (qrow) {                                                                           \
        if (klow) L_(K, true, true);                                                             \
        else L_(K, true, false);                                                                 \
    } else {                                                                                     \
        if (f == F_D) L_(K, false, true, false, 6, 5);                                            \
        else if (f == F_F) L_(K, false, true, false, 5, 5);                                       \
        else if (klow) L_(K, false, true);                                                       \
        else if (f == F_Q8) Q8PLN;                                                               \
        else L_(K, false, false);                                                                \
    }
template <> void launch1<0>(Form f, bool klow, bool qrow, bool exa, const Plan & p) {
#define Q8EXA L_(pk_attn8, false, false, true)
#define Q8PLN L_(pk_attn8, false, false)
    LAUNCH_BODY(pk_attn8)
#undef Q8EXA
#undef Q8PLN
}
#ifdef AB_NEW
template <> void launch1<1>(Form f, bool klow, bool qrow, bool exa, const Plan & p) {
#define Q8EXA L_(pk_attn9, false, false, true, 0, 8)
#define Q8PLN L_(pk_attn9, false, false, false, 0, 8)
    LAUNCH_BODY(pk_attn9)
#undef Q8EXA
#undef Q8PLN
}
#endif
#define ATTR1(K, ...) CK(cudaFuncSetAttribute(K<__VA_ARGS__>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM))
#define ATTR_ALL(K)                  \
    ATTR1(K, false, true, true, 6, 5); \
    ATTR1(K, false, true, true, 5, 5); \
    ATTR1(K, false, true, true);       \
    ATTR1(K, false, false, true);      \
    ATTR1(K, true, true);              \
    ATTR1(K, true, false);             \
    ATTR1(K, false, true, false, 6, 5); \
    ATTR1(K, false, true, false, 5, 5); \
    ATTR1(K, false, true);             \
    ATTR1(K, false, false)

constexpr int NROWS_RAND = 8192;
// one value row -> the engine's storage: bits 8: int8 (HD bytes), else kv_store_qn's layout (HD * bits / 8 bytes); fp16 scale per 32 values
void store_row(const float * v, int bits, uint8_t * row, __half * sc) {
    const int maxq = (1 << (bits - 1)) - 1;
    uint8_t n[HD];
    for (int b = 0; b < HD / 32; b++) {
        float am = 1e-6f;
        for (int k = 0; k < 32; k++) am = std::max(am, std::fabs(v[32 * b + k]));
        const float d = am / maxq;
        sc[b] = __float2half(d);
        for (int k = 0; k < 32; k++) {
            int q = (int) std::lrint(v[32 * b + k] / d);
            q = std::max(-maxq - 1, std::min(maxq, q));
            n[32 * b + k] = bits == 8 ? (uint8_t) (int8_t) q : (uint8_t) (q + (1 << (bits - 1)));
        }
    }
    if (bits == 8) {
        memcpy(row, n, HD);
        return;
    }
    const int hb = bits - 4;
    memset(row, 0, HD * bits / 8);
    for (int w = 0; w < HD / 8; w++) {
        unsigned word = 0, hi = 0;
        for (int j = 0; j < 8; j++) {
            const unsigned x = n[8 * w + j];
            word |= (x & 15u) << (j < 4 ? 8 * j : 8 * (j - 4) + 4);
            if (hb) hi |= (x >> 4) << (hb * j);
        }
        memcpy(row + 4 * w, &word, 4);
        if (hb) memcpy(row + 128 + w * hb, &hi, hb);
    }
}
struct Side {
    int8_t * q;
    __half * s;
};
// [g][rows][..]: every row a copy of one of NROWS_RAND random rows (the kernels' timing does not depend on the values)
Side make_side(std::mt19937 & rng, int bits, size_t rows_per_head) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t rb = HD * bits / 8, total = rows_per_head * NKV;
    std::vector<uint8_t> rq(NROWS_RAND * rb);
    std::vector<__half> rs(NROWS_RAND * (HD / 32));
    for (int r = 0; r < NROWS_RAND; r++) {
        float v[HD];
        const float gain = std::exp(0.3f * nd(rng));
        for (int i = 0; i < HD; i++) v[i] = nd(rng) * gain;
        store_row(v, bits, rq.data() + r * rb, rs.data() + r * (HD / 32));
    }
    std::vector<uint8_t> hq(total * rb);
    std::vector<__half> hs(total * (HD / 32));
    for (size_t r = 0; r < total; r++) {
        const uint32_t x = (uint32_t) (r * 2654435761u) >> 19; // < 8192
        memcpy(hq.data() + r * rb, rq.data() + x * rb, rb);
        memcpy(hs.data() + r * (HD / 32), rs.data() + x * (HD / 32), HD / 32 * 2);
    }
    Side o;
    void * d;
    CK(cudaMalloc(&d, total * rb));
    CK(cudaMemcpy(d, hq.data(), total * rb, cudaMemcpyHostToDevice));
    o.q = static_cast<int8_t *>(d);
    CK(cudaMalloc(&d, total * (HD / 32) * 2));
    CK(cudaMemcpy(d, hs.data(), total * (HD / 32) * 2, cudaMemcpyHostToDevice));
    o.s = static_cast<__half *>(d);
    return o;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: attn_bench <q8|Vl|D|F|any<k><v>> <ctx> <reps> <T:n0> ...\n");
        return 1;
    }
    const std::string fs = argv[1];
    Form f = F_Q8;
    int kb = 8, vb = 8, win = 0;
    if (fs == "D") f = F_D, kb = 6, vb = 5, win = 2048;
    else if (fs == "F") f = F_F, kb = 5, vb = 5, win = 2048;
    else if (fs == "Vl") f = F_ANY, kb = 8, vb = 5, win = 0;
    else if (fs.rfind("any", 0) == 0 && fs.size() == 5) f = F_ANY, kb = fs[3] - '0', vb = fs[4] - '0', win = 2048;
    else if (fs != "q8") { fprintf(stderr, "bad format %s\n", fs.c_str()); return 1; }
    const int ctx = atoi(argv[2]), reps = atoi(argv[3]);
    const bool klow = kb != 8;
    const int exa_mode = getenv("EXA") ? atoi(getenv("EXA")) : 0;
    const bool exa = exa_mode > 0, qrow = getenv("QROW") && atoi(getenv("QROW")) != 0;
    const float qscale = getenv("QSCALE") ? (float) atof(getenv("QSCALE")) : 1.f;
    const bool special = getenv("SPECIAL") && atoi(getenv("SPECIAL")) != 0;
    const int warm_ms = getenv("WARM_MS") ? atoi(getenv("WARM_MS")) : 4000;
    const int ncu_n = getenv("NCU_N") ? atoi(getenv("NCU_N")) : 0;
    const std::string vsel = getenv("AB_VARS") ? getenv("AB_VARS") :
#ifdef AB_NEW
        "01";
#else
        "0";
#endif
    int maxT = 0;
    std::vector<std::pair<int, int>> cfg;
    for (int ai = 4; ai < argc; ai++) {
        int T, n0;
        if (sscanf(argv[ai], "%d:%d", &T, &n0) != 2 || n0 + T > ctx) { fprintf(stderr, "bad config %s\n", argv[ai]); return 1; }
        cfg.push_back({T, n0});
        maxT = std::max(maxT, T);
    }
    ATTR_ALL(pk_attn8);
#ifdef AB_NEW
    ATTR_ALL(pk_attn9);
    ATTR1(pk_attn9, false, false, false, 0, 8);
    ATTR1(pk_attn9, false, false, true, 0, 8);
#endif
    int nsm = 0, occ = 0;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pk_attn8<false, false>, A2_NT, A8_SMEM));
    const int slots = nsm * occ;
    printf("fmt %s (K %d V %d win %d) ctx %d SMs %d CTAs/SM %d smem %d exa %d qrow %d qscale %g special %d\n", fs.c_str(), kb, vb, win, ctx, nsm, occ, A8_SMEM, exa_mode, qrow, qscale, special);
    fflush(stdout);

    std::mt19937 rng(getenv("SEED") ? atoi(getenv("SEED")) : 1234);
    Bufs b;
    memset(static_cast<void *>(&b), 0, sizeof b);
    b.kvq8 = 1;
    b.ctx = ctx;
    const Side sk = make_side(rng, kb, ctx), sv = make_side(rng, vb, ctx);
    b.kvp[0][0] = sk.q, b.kvs[0][0] = sk.s, b.kvb[0][0] = kb;
    b.kvp[1][0] = sv.q, b.kvs[1][0] = sv.s, b.kvb[1][0] = vb;
    b.kc = sk.q, b.vc = sv.q, b.ksc = sk.s, b.vsc = sv.s;
    if (win) {
        const int Wb = (win + ATT_BUCKET - 1) / ATT_BUCKET;
        b.kvwin = win;
        b.kvrc = (Wb + 1) * ATT_BUCKET + ATP;
        if (kb != 8) {
            const Side r = make_side(rng, 8, b.kvrc);
            b.kvrp[0][0] = r.q, b.kvrs[0][0] = r.s;
        }
        if (vb != 8) {
            const Side r = make_side(rng, 8, b.kvrc);
            b.kvrp[1][0] = r.q, b.kvrs[1][0] = r.s;
        }
    }
    if (special) { // non-finite K scales in a few rows of main buffer (all tiles' rows repeat, so they are spread everywhere)
        std::vector<__half> s(ctx * NKV * (HD / 32));
        CK(cudaMemcpy(s.data(), sk.s, s.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < s.size(); r += 104729) s[r] = __float2half(r % 3 == 0 ? INFINITY : r % 3 == 1 ? NAN : 0.f);
        CK(cudaMemcpy(sk.s, s.data(), s.size() * 2, cudaMemcpyHostToDevice));
    }
    __half * qh;
    float * y2;
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<__half> hq((size_t) maxT * NH * HD);
        for (auto & x : hq) x = __float2half(nd(rng) * 3.f / 16.f * qscale);
        if (special) {
            for (size_t i = 0; i + HD < hq.size(); i += 523 * HD + 17) hq[i] = __float2half(i % 2 ? INFINITY : NAN);
            for (size_t r = 0; r + 1 < (size_t) maxT * NH; r += 97) memset(&hq[r * HD], 0, HD * 2);
        }
        CK(cudaMalloc(&qh, hq.size() * 2));
        CK(cudaMemcpy(qh, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y2, (size_t) maxT * VD * 4));
    }
    const size_t part_floats = (size_t) std::max(maxT * 17408, 64 * 64 * A2_PART);
    float * part;
    CK(cudaMalloc(&part, part_floats * 4));
    PBufs p;
    memset(static_cast<void *>(&p), 0, sizeof p);
    p.qh = qh, p.y2 = y2, p.cap = maxT, p.wsrow = exa ? 1 : 0;
    const int view0 = 0;
    CK(cudaMemcpyToSymbol(cBs, &b, sizeof b));
    CK(cudaMemcpyToSymbol(cView, &view0, sizeof view0));
    CK(cudaMemcpyToSymbol(cP, &p, sizeof p));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto run = [&](int v, const Plan & pl, int T, int n0) {
        if (v == 0) launch1<0>(f, klow, qrow, exa, pl);
#ifdef AB_NEW
        else launch1<1>(f, klow, qrow, exa, pl);
#endif
        const int N = (T + A2_Q - 1) / A2_Q * NKV;
        if (N > pl.nfull) {
            if (exa) pk_attn8_combx<<<N - pl.nfull, 256>>>(T, n0, pl.nfull, pl.S, pl.segt, pl.exh, 0, pl.part, pl.part2, pl.exa);
            else pk_attn2_comb<<<N - pl.nfull, 256>>>(T, pl.nfull, pl.S, pl.part);
        }
    };
    for (auto [T, n0] : cfg) {
        const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV;
        Plan pl{};
        pl.T = T, pl.n0 = n0, pl.part = part, pl.part2 = nullptr, pl.segt = 0, pl.exh = 0, pl.exa = 0;
        int nfull = N, S = 1;
        if (exa) {
            pl.segt = 4096 / A2_K;
            if (exa_mode == 2) {
                const int rem = N % slots > 0 ? N % slots : std::min(N, slots), ntm = (n0 + T - 1) / A2_K + 1, mx = (ntm + pl.segt - 1) / pl.segt, h = std::max(1, mx / 2);
                S = 1 + mx - h, nfull = N - rem;
                if (mx < 2 || (size_t) rem * S * A2_PART > part_floats) S = 1, nfull = N; // no split
                else pl.exh = h, pl.exa = rem;
            }
        } else {
            a2_plan(T, n0, slots, part_floats, nfull, S);
        }
        pl.nfull = nfull, pl.S = S;
        pl.grid = dim3(nfull + (N - nfull) * S);
        printf("T %d n0 %d: items %d, grid %d nfull %d S %d\n", T, n0, N, pl.grid.x, nfull, S);
        fflush(stdout);
        std::vector<float> ref((size_t) T * VD), out((size_t) T * VD);
        std::vector<int> vs;
        for (char c : vsel) vs.push_back(c - '0');
        if (ncu_n > 0) { // profiler run: N launches of the one variant
            for (int i = 0; i < ncu_n; i++) {
                run(vs[0], pl, T, n0);
                CK(cudaDeviceSynchronize());
            }
            continue;
        }
        bool have_ref = false;
        for (int v : vs) { // outputs (non-final launches may start at cold clocks: not timed)
            CK(cudaMemset(y2, 0xff, (size_t) T * VD * 4));
            run(v, pl, T, n0);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            if (!have_ref) {
                CK(cudaMemcpy(ref.data(), y2, (size_t) T * VD * 4, cudaMemcpyDeviceToHost));
                have_ref = true;
                size_t nnan = 0;
                double am = 0;
                for (float x : ref) nnan += std::isnan(x), am = std::max(am, (double) std::fabs(x));
                printf("  ref v%d: NaN %zu max|y| %.3g\n", v, nnan, am);
            } else {
                CK(cudaMemcpy(out.data(), y2, (size_t) T * VD * 4, cudaMemcpyDeviceToHost));
                size_t nbits = 0, first = (size_t) -1, nnan = 0;
                double md = 0;
                for (size_t i = 0; i < out.size(); i++) {
                    if (memcmp(&out[i], &ref[i], 4)) {
                        if (!nbits) first = i;
                        nbits++;
                        md = std::max(md, (double) std::fabs(out[i] - ref[i]));
                    }
                    nnan += std::isnan(out[i]);
                }
                printf("  v%d vs ref: differing %zu / %zu (first %zd, max|d| %.3g), NaN %zu%s\n", v, nbits, out.size(), (ssize_t) first, md, nnan, nbits ? "  <<< NOT EXACT" : "");
            }
            fflush(stdout);
        }
        if (reps <= 0) continue;
        const auto t00 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t00).count() < warm_ms) { // clocks
            run(vs[0], pl, T, n0);
            CK(cudaDeviceSynchronize());
        }
        std::vector<std::vector<float>> ms(vs.size());
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < vs.size(); i++) {
                CK(cudaEventRecord(e0));
                run(vs[i], pl, T, n0);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float t;
                CK(cudaEventElapsedTime(&t, e0, e1));
                ms[i].push_back(t);
            }
        double b0 = 0;
        for (size_t i = 0; i < vs.size(); i++) {
            std::sort(ms[i].begin(), ms[i].end());
            const double mn = ms[i][0], md = ms[i][ms[i].size() / 2];
            if (i == 0) b0 = md;
            printf("  v%d: min %9.3f med %9.3f ms  (%.2f us/position, %+.2f%%)\n", vs[i], mn, md, md * 1e3 / (n0 + T / 2), 100 * (md / b0 - 1));
        }
        fflush(stdout);
    }
    printf("DONE\n");
    return 0;
}
