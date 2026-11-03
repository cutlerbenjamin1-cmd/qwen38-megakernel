// wv: int8 tensor-core verify GEMM (mma.m16n8k32, tokens on N) for the main-pass GEMV phases.
// Included by qwen38_mk.cu inside its anonymous namespace, after the dp4a phases. Counterpart of the bench kernel
// bench/wvgemm_bench.cu (k_wv<TYPE, T, X16, PF = 0>, -DWV_TILEMAJ=2), which checks
// T = 1..8 bit-identical per row and errors vs fp64 at fp32-rounding level.
//
// A warp computes a 16-row tile (thread (g, tq): rows g, g + 8) x 8 tokens (one C column per token) per MMA over a K
// range, sequentially in K: every output is the same sequence of integer 32-sums and float FMAs at any T, so the path
// is T-invariant, but not bit-identical to the dp4a GEMV (per-32 integer sums); it must therefore serve every T.
//
// K permutation (the same for A and B): MMA k-slot 4 tq + i <-> element 8 tq + i, k-slot 16 + 4 tq + i <-> element
// 8 tq + 4 + i of the 32-element block: each thread supplies 8 consecutive elements of a weight row (one 8-byte piece of
// the fragment-ordered unit) and of a token row (one 8-byte LDS of the activation units).
// Integer -> float without I2F: the accumulator input is MAGIC = 0x4B400000 (12582912.0f), the MMA returns MAGIC + S
// (|S| < 2^22), fmaf(dx, that, -12582912 dx) = dx * S rounded once.
// Per-16 scales of the q16 activations (ffn_down, X16): "split" MMAs. Threads tq 0-1 hold chunk 0 (elements 0-15),
// tq 2-3 chunk 1; B column g carries token (g & 3) of the group and only the k-slots of chunk g >> 2, so C column n =
// the 16-element dot of token n & 3 over chunk n >> 2; the two chunk sums of a token meet in one shuffle after the K loop.
//
// Weight layout (loader, wv_relayout): every main-pass matrix of a wv type is stored as units (16-row tile, super-block)
// of 16 x BS contiguous bytes. Q4_K / Q5_K / IQ4_XS fragment-ordered ("FM"): the 16 rows' headers paired per g (row g
// | row g + 8; 16 B each for Q4_K / Q5_K, 8 B for IQ4_XS), [Q5_K: qh as 32 lanes x {row g 8 B | row g + 8 8 B}], then the
// 4 qs groups likewise (512 B each); every lane load is one 16-byte LDG and every warp load reads contiguous bytes.
// IQ4_NL / Q8_0: tile-major whole blocks (unit = the 16 rows' blocks of that super-block; Q8_0's scale stream likewise).
#pragma once

constexpr int WV_MAGIC = 0x4B400000;
constexpr float WV_FMAGIC = 12582912.0f;

__device__ __forceinline__ void mma16832(int (&d)[4], const int (&a)[4], int b0, int b1, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// float(byte k of w) + 2^23 (k compile-time after unrolling)
__device__ __forceinline__ float mag23(unsigned w, int k) { return __int_as_float(__byte_perm(w, 0x4b000000u, 0x7540u | k)); }

// the weight types the wv path serves (the others stay on dp4a at every T: per tensor T-invariant)
__host__ __device__ constexpr bool wv_type(int t) { return t == T_Q4K || t == T_Q5K || t == T_IQ4X || t == T_IQ4NL || t == T_Q8_0; }
// per-16 weight scales (the ffn_down offset types): split MMAs only
__host__ __device__ constexpr bool wv_s16(int t) { return t == T_Q6K || t == T_Q3K; }

// unit (tile, super-block sb) of an FM matrix, BS bytes per row and super-block
__device__ __forceinline__ const uint8_t * wv_unit(const Mat & m, int tile, int sb, int BS) {
    return m.w + ((size_t) tile * (m.K >> 8) + sb) * 16 * BS;
}
// tile-major whole blocks: block sb of row r (0..15) of the tile, and the meta stream (MB bytes per super-block)
__device__ __forceinline__ const uint8_t * wv_tblk(const Mat & m, int tile, int r, int sb, int BS) {
    return m.w + (((size_t) tile * (m.K >> 8) + sb) * 16 + r) * BS;
}
__device__ __forceinline__ const uint8_t * wv_tmeta(const Mat & m, int tile, int r, int sb, int MB) {
    return reinterpret_cast<const uint8_t *>(m.d) + (((size_t) tile * (m.K >> 8) + sb) * 16 + r) * MB;
}

// Per weight type: Raw = one super-block of one row as this thread loads it; Sc = its decoded scales; frag(k) = the two
// int8x4 words of elements 8 tq .. 8 tq + 7 of sub-block k; wsc(k) = the float scale of sub-block k and, for MIN == 1, the
// min coefficient (weight = fA q - gA). MIN: 0 none, 1 per-row min (Q4_K / Q5_K: one FFMA per output).
template <int TYPE> struct WV;

template <> struct WV<T_Q4K> { // 144 B: d, dmin | scales[12] | qs[128]
    static constexpr int MIN = 1, BS = 144;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { unsigned s0, s1, m0, m1; float d, nd, dm, ndm; };
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&r.h.x));
        const unsigned y = r.h.y, z = r.h.z, w = r.h.w;
        s.s0 = y & 0x3f3f3f3fu;
        s.m0 = z & 0x3f3f3f3fu;
        s.s1 = (w & 0x0f0f0f0fu) | ((y >> 2) & 0x30303030u);
        s.m1 = ((w >> 4) & 0x0f0f0f0fu) | ((z >> 2) & 0x30303030u);
        s.d = dm.x; s.nd = -8388608.0f * dm.x; s.dm = dm.y; s.ndm = -8388608.0f * dm.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        fA = fmaf(s.d, mag23(k < 4 ? s.s0 : s.s1, k & 3), s.nd);
        gA = fmaf(s.dm, mag23(k < 4 ? s.m0 : s.m1, k & 3), s.ndm);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = (r.q[k >> 1].x >> sh) & M4;
        w1 = (r.q[k >> 1].y >> sh) & M4;
    }
};

template <> struct WV<T_Q5K> { // 176 B: d, dmin | scales[12] | qh[32] | qs[128]
    static constexpr int MIN = 1, BS = 176;
    struct Raw { int4 h; int2 qh; int2 q[4]; };
    using Sc = WV<T_Q4K>::Sc;
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        WV<T_Q4K>::Raw t;
        t.h = r.h;
        WV<T_Q4K>::scales(t, s);
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) { WV<T_Q4K>::wsc(s, k, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | ((((unsigned) r.qh.x >> k) & 0x01010101u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | ((((unsigned) r.qh.y >> k) & 0x01010101u) << 4);
    }
};

template <> struct WV<T_IQ4X> { // per super-block: 8-B GGUF header (fp16 d | u16 scales_h | u8 scales_l[4]) | qs[128]
    static constexpr int MIN = 0, BS = 136;
    struct Raw { int2 h; int2 q[4]; };
    struct Sc { unsigned sl, sh; float d, nd; };
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.d = __half2float(__ushort_as_half((unsigned short) (r.h.x & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
        s.sh = (unsigned) r.h.x >> 16;
        s.sl = (unsigned) r.h.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        const int ls = (int) (((s.sl >> (4 * k)) & 0xfu) | (((s.sh >> (2 * k)) & 3u) << 4)); // k < 8: nibble k of sl
        fA = fmaf(s.d, __int_as_float(WV_MAGIC - 32 + ls), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_IQ4NL> { // repacked 144 B: d[8] (fp16, one per 32) | qs[128] in Q4_K order; tile-major blocks
    static constexpr int MIN = 0, BS = 144;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { int4 h; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, 144);
        r.h = ldw4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = ldw2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.h = r.h; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        const int w = k == 0 || k == 1 ? s.h.x : k == 2 || k == 3 ? s.h.y : k == 4 || k == 5 ? s.h.z : s.h.w;
        fA = __half2float(__ushort_as_half((unsigned short) ((unsigned) w >> (16 * (k & 1)))));
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_Q8_0> { // repacked: per unit 32 B (A 16 | B 16); fp16 d per natural 32-block in Mat::d; tile-major
    static constexpr int MIN = 0, BS = 256;
    struct Raw { int4 d; int2 q[8]; };
    struct Sc { int4 d; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * rp = wv_tblk(m, tile, row, sb, 256);
        r.d = ldw4(wv_tmeta(m, tile, row, sb, 16));
        // q[k] = the 8 bytes at 32 (2 kk + (tq >> 1)) + 16 kb + 8 (tq & 1), k = 2 kk + kb. The lane pair tq = 2 h,
        // 2 h + 1 shares the 32-B sector 2 kk + h: the even lane loads its bytes 0-15 (its kb 0 | the odd lane's kb 0), the odd
        // lane 16-31 (the even lane's kb 1 | its kb 1); one exchange of 8 bytes completes both: whole sectors, half the loads
        // (8-byte loads per k bring every sector from L2 twice, half each time).
        const bool odd = tq & 1;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int4 v = ldw4(rp + 32 * (2 * kk + (tq >> 1)) + 16 * (tq & 1));
            const int s0 = __shfl_xor_sync(0xffffffffu, odd ? v.x : v.z, 1);
            const int s1 = __shfl_xor_sync(0xffffffffu, odd ? v.y : v.w, 1);
            r.q[2 * kk] = odd ? make_int2(s0, s1) : make_int2(v.x, v.y);
            r.q[2 * kk + 1] = odd ? make_int2(v.z, v.w) : make_int2(s0, s1);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.d = r.d; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        WV<T_IQ4NL>::Sc t;
        t.h = s.d;
        WV<T_IQ4NL>::wsc(t, k, fA, gA);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) { w0 = r.q[k].x; w1 = r.q[k].y; }
};

// Offset types (every main-layer ffn_down is on wv, these too): tile-major whole value
// blocks (16 x BS bytes per unit) and a tile-major meta stream (Q6_K / Q3_K: the loader's 36-B block pairs, IQ3_S: 16 B per
// block). Values are fed with offset 32 (MIN 2: the accumulator input is MAGIC - 32 qsum, wv_fill's min2 mx), so the
// weight is fA q'. Q6_K / Q3_K carry per-16 scales (S16): split MMAs only (X16), where h = tq >> 1 is the thread's chunk.
// frag / wsc take the thread's tq / h (MIN 2 types only: wv_sb).
__host__ __device__ constexpr bool wv_min2(int t) { return t == T_Q6K || t == T_Q3K || t == T_IQ3S; }
// the meta pair (Q6_K / Q3_K) of super-block sb of row r of the tile
__device__ __forceinline__ const uint8_t * wv_tpair(const Mat & m, int tile, int r, int sb) {
    return reinterpret_cast<const uint8_t *>(m.d) + (((size_t) tile * (m.K >> 9) + (sb >> 1)) * 16 + r) * (2 * Q6_META);
}
// Q6_K values 192 B per block: qlo[128] (Q4_K-order nibbles) | qhi[64] (8 B per unit u' = 2p + h: A's highs | B's, byte m bits
// 2j = element 4j + m of the 16-chunk); meta pair: sc[16] even | sc[16] odd | d even | d odd
template <> struct WV<T_Q6K> {
    static constexpr int MIN = 2, BS = Q6_BLK;
    static constexpr bool S16 = true;
    struct Raw { int sc[4]; int dd; int2 q[4]; int2 hb[4]; };
    struct Sc { int sc[4]; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q6_BLK);
        const uint8_t * mp = wv_tpair(m, tile, row, sb);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = ldw1(mp + 16 * (sb & 1) + 4 * k); // (meta is only 4-byte aligned)
        r.dd = (int) ((unsigned) ldw1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            r.q[p] = ldw2(blk + 32 * p + 8 * tq);
            r.hb[p] = ldw2(blk + 128 + 8 * (2 * p + (tq >> 1)));
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    // chunk c = 4p + 2 part + h of the block: byte c of sc[16]
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int h, float & fA) {
        const int c0 = 4 * (k >> 1) + 2 * (k & 1); // even: bytes c0, c0 + 1 lie in word c0 >> 2
        const unsigned w = (unsigned) s.sc[c0 >> 2] >> (8 * h);
        const int v = (int) (signed char) (w >> (8 * (c0 & 3)));
        fA = fmaf(s.d, __int_as_float(WV_MAGIC + v), s.nd);
    }
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        const unsigned hs = (k & 1) ? (unsigned) r.hb[k >> 1].y : (unsigned) r.hb[k >> 1].x;
        const int j0 = 2 * (tq & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | (int) (((hs >> (2 * j0)) & 0x03030303u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | (int) (((hs >> (2 * j0 + 2)) & 0x03030303u) << 4);
    }
};
// ---- FM Q6_K: every main-layer Q6_K matrix, fragment-ordered with the meta inline (wv_relayout) ----------------------
// Unit (16-row tile, super-block) = 3360 B = 16 x (192 + 18):
//   [0, 256)     sc: per g {row g sc[16] | row g + 8 sc[16]} (int8, natural 16-chunk order)
//   [256, 288)   d: per g {row g fp16 | row g + 8 fp16}
//   [288, 2336)  qlo: 4 groups p x 512 B, lane l = 4 g + tq: {row g | row g + 8} x {word A | word B}: A = the Q4_K-order nibble
//                bytes 32 p + 4 tq .. + 3 (chunk 0 elements 4 tq .. + 3 of sub-block 2 p in the low nibbles, 2 p + 1 in the
//                high ones), B = bytes 32 p + 16 + 4 tq .. + 3 (chunk 1)
//   [2336, 3360) qhi: 4 groups p x 256 B, lane l: {row g H | row g + 8 H}; byte m of H = the high 2 bits of element 4 tq + m
//                of (sub-block, chunk) (2p, 0) at bits 4-5, (2p + 1, 0) at 6-7, (2p, 1) at 0-1, (2p + 1, 1) at 2-3
// Thread tq supplies elements 4 tq .. + 3 of both 16-chunks of each sub-block (11 coalesced loads per unit; the TM layout
// takes 26 loads of 4-8 B). Two MMA forms on it, bit-identical to each other and to the TM split path (the same exact integer
// 16-chunk sums, the same float sequence per (row, token, chunk), chunk 0's and chunk 1's sums added last):
//   X16 (ffn_down's q16 activations): split m16n8k32, chunk 0 in k-slots 0-15 and chunk 1 in 16-31, B column n = token n & 3
//   of chunk n >> 2 (the split md / mx tables of the X16 fill);
//   q32 phases: m16n8k16 per 16-chunk over 8 tokens: the non-split md table + the fill's per-chunk qsum table (mx4).
// (Prefill: mmh's FM6 stage reads the same units, prefill_gemm.cuh.)
struct Q6FRaw { int4 sc[2]; int dd; int2 q[2][4]; int hi[2][4]; };
__device__ __forceinline__ void q6f_load(Q6FRaw & r, const Mat & m, int tile, int sb, int g, int tq) {
    const uint8_t * u = m.w + ((size_t) tile * (m.K >> 8) + sb) * Q6F_UNIT;
    const int l = 4 * g + tq;
    r.sc[0] = ldw4(u + 32 * g);
    r.sc[1] = ldw4(u + 32 * g + 16);
    r.dd = ldw1(u + Q6F_D + 4 * g);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int4 v = ldw4(u + Q6F_LO + 512 * p + 16 * l);
        r.q[0][p] = make_int2(v.x, v.y);
        r.q[1][p] = make_int2(v.z, v.w);
        const int2 h = ldw2(u + Q6F_HI + 256 * p + 8 * l);
        r.hi[0][p] = h.x;
        r.hi[1][p] = h.y;
    }
}
// the int8x4 operand (q + 32) of row rr (0: g, 1: g + 8), sub-block k, chunk c: elements 16 c + 4 tq .. + 3
__device__ __forceinline__ int q6f_frag(const Q6FRaw & r, int rr, int k, int c) {
    const int p = k >> 1, part = k & 1;
    const unsigned base = c ? (unsigned) r.q[rr][p].y : (unsigned) r.q[rr][p].x;
    const unsigned lo = (part ? base >> 4 : base) & 0x0f0f0f0fu;
    const unsigned H = (unsigned) r.hi[rr][p];
    const unsigned hp = c == 0 ? (part ? H >> 2 : H) : (part ? H << 2 : H << 4);
    return (int) (lo | (hp & 0x30303030u));
}
// d sc[cn] of row rr, 16-chunk cn = 2 k + c, computed as WV<T_Q6K>::wsc2 (fmaf form: exact)
__device__ __forceinline__ float q6f_fa(const Q6FRaw & r, float d, float nd, int rr, int cn) {
    const int4 s = r.sc[rr];
    const int w = (cn >> 2) == 0 ? s.x : (cn >> 2) == 1 ? s.y : (cn >> 2) == 2 ? s.z : s.w;
    const int v = (int) (signed char) ((unsigned) w >> (8 * (cn & 3)));
    return fmaf(d, __int_as_float(WV_MAGIC + v), nd);
}
__device__ __forceinline__ void mma16816(int (&d)[4], int a0, int a1, int b0, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// Q3_K values 96 B per block: unit u' = 2p + h: 2-bit lows at 8u' (A word | B word; byte m bits 2j = element 4j + m), highs
// at 64 + 4u' (byte m: bit j = A element 4j + m, bit 4 + j = B); meta as Q6_K (sc - 32 as int8, d). Fed u3 + 28 (offset 32).
template <> struct WV<T_Q3K> {
    static constexpr int MIN = 2, BS = Q3_BLK;
    static constexpr bool S16 = true;
    struct Raw { int sc[4]; int dd; int2 lo[4]; int hi[4]; };
    using Sc = WV<T_Q6K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q3_BLK);
        const uint8_t * mp = wv_tpair(m, tile, row, sb);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = ldw1(mp + 16 * (sb & 1) + 4 * k);
        r.dd = (int) ((unsigned) ldw1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.lo[p] = ldw2(blk + 8 * u);
            r.hi[p] = ldw1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int h, float & fA) { WV<T_Q6K>::wsc2(s, k, h, fA); }
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
        const unsigned x = (k & 1) ? (unsigned) r.lo[k >> 1].y : (unsigned) r.lo[k >> 1].x, hh = (unsigned) r.hi[k >> 1];
        const int j0 = 2 * (tq & 1), hs = 4 * (k & 1);
        w0 = (int) ((((x >> (2 * j0)) & 0x03030303u) | (((hh >> (j0 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
        w1 = (int) ((((x >> (2 * j0 + 2)) & 0x03030303u) | (((hh >> (j0 + 1 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
    }
};
// IQ3_S values 96 B per block: unit u' = 2p + h: the low bytes of A's 4 grid indices then B's at 8u', signs at 64 + 4u' (byte
// m: bit c = A element 4c + m, bit 4 + c = B); meta 16 B per block: qh[8] (byte u': index bit 8, A in bits 0-3, B in 4-7) |
// scales[4] (byte p: sub-block 2p | 2p + 1 << 4) | fp16 d | pad. Fed 32 +- grid (offset 32); per-32 scales (no S16).
template <> struct WV<T_IQ3S> {
    static constexpr int MIN = 2, BS = Q3_BLK;
    static constexpr bool S16 = false;
    struct Raw { int2 qh; int2 sd; int2 ix[4]; int sg[4]; };
    struct Sc { unsigned sc; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q3_BLK);
        const uint8_t * mp = wv_tmeta(m, tile, row, sb, IQ3S_META);
        r.qh = ldw2(mp);
        r.sd = ldw2(mp + 8);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.ix[p] = ldw2(blk + 8 * u);
            r.sg[p] = ldw1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.sc = (unsigned) r.sd.x;
        s.d = __half2float(__ushort_as_half((unsigned short) (r.sd.y & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int, float & fA) {
        const int v = 1 + 2 * (int) ((s.sc >> (4 * k)) & 0xfu); // nibble k = sub-block k
        fA = fmaf(s.d, __int_as_float(WV_MAGIC + v), s.nd);
    }
    // (the qh byte of unit u' is per thread: u' = 2p + (tq >> 1))
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int p = k >> 1, part = k & 1, u = 2 * p + (tq >> 1);
        const unsigned qhw = (unsigned) (u < 4 ? r.qh.x : r.qh.y) >> (8 * (u & 3));
        const unsigned x = part ? (unsigned) r.ix[p].y : (unsigned) r.ix[p].x, sv = (unsigned) r.sg[p];
        const int c0 = 2 * (tq & 1), hs = 4 * part;
        int w[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = c0 + i;
            const unsigned g = iq3s_grid(((x >> (8 * c)) & 0xffu) | (((qhw >> (c + hs)) & 1u) << 8));
            const unsigned mk = ((sv >> (c + hs)) & 0x01010101u) * 0xffu;
            w[i] = (int) (0x20202020u + (g ^ mk) - mk);
        }
        w0 = w[0];
        w1 = w[1];
    }
};

// the thread's rows g (r0) and g + 8 (r1) of super-block sb of the tile
template <int TYPE>
__device__ __forceinline__ void wv_load2(typename WV<TYPE>::Raw & r0, typename WV<TYPE>::Raw & r1, const Mat & m, int tile, int sb,
                                         int g, int tq) {
    if constexpr (wv_fm(TYPE)) {
        constexpr int BS = WV<TYPE>::BS;
        const uint8_t * u = wv_unit(m, tile, sb, BS);
        const int l = 4 * g + tq;
        int qo;
        if constexpr (TYPE == T_IQ4X) {
            const int4 h = ldw4(u + 16 * g);
            r0.h = make_int2(h.x, h.y);
            r1.h = make_int2(h.z, h.w);
            qo = 128;
        } else {
            r0.h = ldw4(u + 32 * g);
            r1.h = ldw4(u + 32 * g + 16);
            qo = 256;
        }
        if constexpr (TYPE == T_Q5K) {
            const int4 v = ldw4(u + 256 + 16 * l);
            r0.qh = make_int2(v.x, v.y);
            r1.qh = make_int2(v.z, v.w);
            qo = 768;
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int4 v = ldw4(u + qo + 512 * p + 16 * l);
            r0.q[p] = make_int2(v.x, v.y);
            r1.q[p] = make_int2(v.z, v.w);
        }
    } else {
        WV<TYPE>::load(r0, m, tile, g, sb, tq);
        WV<TYPE>::load(r1, m, tile, g + 8, sb, tq);
    }
}

// Shared-memory activations of a K chunk (wv_fill): the engine's unit layout (16-byte A / B chunks of the 32-element
// units, xa / xb, meta xm) with a token stride of US + 2 units, so the 8-byte B loads of a half-warp (4 tokens x 4 tq)
// hit distinct banks; md[(j * NG + n) * 4 + tq] = {dx_a, -12582912 dx_a, dx_b, -..dx_b} for the two tokens (a, b) of the
// thread's C columns in token group n and natural 32-block j of the chunk (split: of its 16-chunk), mx = (dx qsum)_a, _b.
struct WvXS {
    const int8_t * xa;
    const int8_t * xb;
    int ld; // bytes per token row
    const float4 * md;
    const float2 * mx;
    const float4 * mx4; // q6 fills: per (j, n, tq) the bits of MAGIC - 32 qsum of (token a, chunk 0), (b, 0), (a, 1), (b, 1)
};
template <int NG> struct WvAcc { float v[NG][4]; };
// a tile's running sums carried across the K sub-chunk fills of a batched pass wider than MAXT rows (wv_tile_st): a, and
// v1 the FM Q6_K k16 form's chunk-1 sums; finished once after the last sub-chunk (wv_st_fin), so every token's fmaf chain
// is the whole-K tile's: the same bits as a solo launch
template <int T, bool X16> struct WvSt {
    WvAcc<WvCfg<T, X16>::NG> a;
    float v1[!X16 ? WvCfg<T, X16>::NG : 1][4];
};

template <int US, int T, bool X16>
__host__ __device__ constexpr int wv_smem() {
    using C = WvCfg<T, X16>;
    // (+ the q32 fills' per-chunk qsum table mx4, 16 B per (j, n, tq))
    return 2 * C::NTOKP * (US + 2) * 16 + T * (US + 2) * 8 + US * C::NG * 4 * (!X16 ? 40 : 24);
}

// Fill units [u0, u0 + US) of T activation rows. norm: s_rs[t] = the RMS-norm scale of row t (as fill_xs).
// Also returns the dp4a view of the same units (token stride US + 2: rows_* with XP = 2).
// min2: mx = the MIN 2 accumulator inputs (bits of MAGIC - 32 qsum) instead of dx qsum (the matrix is an offset type)
// q6 (q32 fills): also the per-chunk qsum table mx4 of the FM Q6_K tiles' m16n8k16 form
template <int T, bool X16, int US>
__device__ __forceinline__ WvXS wv_fill(const QBuf & q, int u0, bool norm, SX * sxo = nullptr, bool min2 = false, bool q6 = false) {
    using C = WvCfg<T, X16>;
    constexpr int LDU = US + 2;
    static_assert(wv_smem<US, T, X16>() <= SM_MISC, "wv activations");
    uint8_t * const sm = smem;
    int4 * xa = reinterpret_cast<int4 *>(sm);
    int4 * xb = xa + C::NTOKP * LDU;
    uint2 * xm = reinterpret_cast<uint2 *>(xb + C::NTOKP * LDU);
    float4 * md = reinterpret_cast<float4 *>(xm + T * LDU);
    float2 * mx = reinterpret_cast<float2 *>(md + US * C::NG * 4);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < T * US; i += NT) {
        const int t = i / US, u = i - t * US;
        const size_t gi = (size_t) t * q.U + u0 + u;
        cp_async16(xa + t * LDU + u, q.xa + gi * 16);
        cp_async16(xb + t * LDU + u, q.xb + gi * 16);
    }
    for (int i = tid; i < T * US / 2; i += NT) {
        const int t = 2 * i / US, u = 2 * i - t * US;
        cp_async16(xm + t * LDU + u, q.meta + (size_t) t * q.U + u0 + u);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    float * s_rs = misc(sm);
    if (norm) { // normed consumers always read the K = D residual stream (U_D = 160 partial sums)
        if (warp < T) {
            float pp[U_D / 32];
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) pp[k] = __ldcg(q.ssq + warp * U_D + lane + 32 * k);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) acc += pp[k];
            acc = wsum(acc);
            if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
        }
    } else if (tid < T) {
        s_rs[tid] = 1.0f;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // md / mx from the staged meta (no second global round trip)
    for (int i = tid; i < US * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1; // chunk-local unit of the 64-group (hh = 0)
        float v[4];
        float2 x2;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (X16) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
            float dx = 0.f;
            int qs = 0;
            if (t < T) {
                const uint2 m0 = xm[t * LDU + ub + (hh < 0 ? 0 : hh)];
                const uint2 m1 = xm[t * LDU + ub + 1];
                const int sh = 16 * part; // (shifts: a runtime index into &m0 / &m1 would put them in local memory)
                dx = __half2float(__ushort_as_half((unsigned short) (m0.x >> sh)));
                const int q0 = (short) (m0.y >> sh), q1 = (short) (m1.y >> sh);
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -WV_FMAGIC * dx;
            (k ? x2.y : x2.x) = min2 ? __int_as_float(WV_MAGIC - 32 * qs) : dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
    float4 * const mx4 = reinterpret_cast<float4 *>(mx + US * C::NG * 4);
    if (!X16 && q6)
        for (int i = tid; i < US * C::NG * 4; i += NT) { // tokens 8 n + 2 tq + {0, 1} (the non-split md's), chunks 0 / 1
            const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
            const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1;
            int qv[2][2];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int t = 8 * n + 2 * tq + k;
#pragma unroll
                for (int c = 0; c < 2; c++) qv[k][c] = t < T ? reinterpret_cast<const short *>(&xm[t * LDU + ub + c].y)[part] : 0;
            }
            mx4[i] = make_float4(__int_as_float(WV_MAGIC - 32 * qv[0][0]), __int_as_float(WV_MAGIC - 32 * qv[1][0]),
                                 __int_as_float(WV_MAGIC - 32 * qv[0][1]), __int_as_float(WV_MAGIC - 32 * qv[1][1]));
        }
    __syncthreads();
    mark_fill();
    if (sxo) {
        sxo->xa = xa;
        sxo->xb = xb;
        sxo->xm = xm;
        sxo->U = LDU;
    }
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), LDU * 16, md, mx, mx4};
}

// wv_fill's view (the same addresses) rebuilt from its token stride ldu = US + 2 units, for the out-of-line wv_tile: a
// WvXS passed by value arrives as generic pointers (generic loads in the K loop), this one derives from smem (LDS)
template <int T, bool X16>
__device__ __forceinline__ WvXS wv_xs(int ldu) {
    using C = WvCfg<T, X16>;
    const int4 * xa = reinterpret_cast<const int4 *>(smem);
    const int4 * xb = xa + C::NTOKP * ldu;
    const uint2 * xm = reinterpret_cast<const uint2 *>(xb + C::NTOKP * ldu);
    const float4 * md = reinterpret_cast<const float4 *>(xm + T * ldu);
    const float2 * mx = reinterpret_cast<const float2 *>(md + (ldu - 2) * C::NG * 4);
    const float4 * mx4 = reinterpret_cast<const float4 *>(mx + (ldu - 2) * C::NG * 4);
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), ldu * 16, md, mx, mx4};
}

// one super-block (8 sub-blocks) of the tile from the thread's raw rows r0 (row g) and r1 (row g + 8); sbl = the
// super-block's index within the shared-memory chunk; xra / xrb = this thread's A / B unit row (+ 8 tq)
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv_sb(const typename WV<TYPE>::Raw & r0, const typename WV<TYPE>::Raw & r1, int sbl,
                                      const WvXS & xs, const int8_t * xra, const int8_t * xrb, bool bon,
                                      float (&acc)[WvCfg<T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using C = WvCfg<T, X16>;
    const int tq = threadIdx.x & 3;
    typename W::Sc s0, s1;
    W::scales(r0, s0);
    W::scales(r1, s1);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int a[4];
        float fA0, gA0, fA1, gA1;
        if constexpr (W::MIN == 2) { // offset types: the thread's tq / chunk h select its bytes / scales
            W::frag2(r0, k, tq, a[0], a[2]);
            W::frag2(r1, k, tq, a[1], a[3]);
            W::wsc2(s0, k, tq >> 1, fA0);
            W::wsc2(s1, k, tq >> 1, fA1);
            gA0 = gA1 = 0.f;
        } else {
            W::frag(r0, k, a[0], a[2]);
            W::frag(r1, k, a[1], a[3]);
            W::wsc(s0, k, fA0, gA0);
            W::wsc(s1, k, fA1, gA1);
        }
        const int j = 8 * sbl + k;
        const int8_t * xr = ((k & 1) ? xrb : xra) + 128 * sbl + 32 * (k >> 1);
#pragma unroll
        for (int n = 0; n < C::NG; n++) {
            int2 bb = make_int2(0, 0);
            if (bon) bb = *reinterpret_cast<const int2 *>(xr + C::TPG * n * xs.ld);
            const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
            int ci[4] = {WV_MAGIC, WV_MAGIC, WV_MAGIC, WV_MAGIC};
            float2 mx = make_float2(0.f, 0.f);
            if (W::MIN) mx = xs.mx[(j * C::NG + n) * 4 + tq];
            if (W::MIN == 2) { // MAGIC - 32 qsum: the offset's sum (wv_fill with min2)
                ci[0] = ci[2] = __float_as_int(mx.x);
                ci[1] = ci[3] = __float_as_int(mx.y);
            }
            int cc[4];
            mma16832(cc, a, bb.x, bb.y, ci);
            acc[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc[n][0]);
            acc[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc[n][1]);
            acc[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc[n][2]);
            acc[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc[n][3]);
            if (W::MIN == 1) {
                acc[n][0] = fmaf(-gA0, mx.x, acc[n][0]);
                acc[n][1] = fmaf(-gA0, mx.y, acc[n][1]);
                acc[n][2] = fmaf(-gA1, mx.x, acc[n][2]);
                acc[n][3] = fmaf(-gA1, mx.y, acc[n][3]);
            }
        }
    }
}

// Tile `tile` (rows 16 tile .. + 15) of m over super-blocks [sb0, sb0 + nsb) (the chunk in shared memory starts at sb0).
// C fragment: v[n][c] = row 16 tile + g + 8 (c >> 1), token TPG n + 2 tq + (c & 1) (split: tokens 4 n + 2 tq & 3 + (c & 1),
// valid in threads tq < 2). Out of line per (type, T, X16): each gets its own register allocation (128 cap).
// FM Q6_K tile (see "FM Q6_K"): X16 = the split m16n8k32 form, else m16n8k16 per 16-chunk (xs from a q6 fill)
template <int T, bool X16>
__device__ __forceinline__ void wv_q6f_acc(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs, float (&acc)[WvCfg<T, X16>::NG][4],
                                           float (&acc1)[X16 ? 1 : WvCfg<T, X16>::NG][4]) {
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, h = tq >> 1;
    // B: k16 token 8 n + g, elements 16 c + 4 tq of the sub-block; split token 4 n + (g & 3), chunk g >> 2 only
    const int tok = X16 ? (g & 3) : g, co = X16 ? 16 * (g >> 2) : 0;
    const int8_t * xra = xs.xa + tok * xs.ld + co + 4 * tq;
    const int8_t * xrb = xs.xb + tok * xs.ld + co + 4 * tq;
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        Q6FRaw r;
        q6f_load(r, m, tile, sb0 + sb, g, tq);
        const float d0 = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned) r.dd >> 16)));
        const float nd0 = -WV_FMAGIC * d0, nd1 = -WV_FMAGIC * d1;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int j = 8 * sb + k;
            const int8_t * xr = ((k & 1) ? xrb : xra) + 128 * sb + 32 * (k >> 1);
            const int w00 = q6f_frag(r, 0, k, 0), w10 = q6f_frag(r, 1, k, 0);
            const int w01 = q6f_frag(r, 0, k, 1), w11 = q6f_frag(r, 1, k, 1);
            if constexpr (!X16) {
                const float f00 = q6f_fa(r, d0, nd0, 0, 2 * k), f10 = q6f_fa(r, d1, nd1, 1, 2 * k);
                const float f01 = q6f_fa(r, d0, nd0, 0, 2 * k + 1), f11 = q6f_fa(r, d1, nd1, 1, 2 * k + 1);
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int b0 = *reinterpret_cast<const int *>(xr + 8 * n * xs.ld);
                    const int b1 = *reinterpret_cast<const int *>(xr + 8 * n * xs.ld + 16);
                    const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
                    const float4 mx = xs.mx4[(j * C::NG + n) * 4 + tq];
                    const int ci0[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    const int ci1[4] = {__float_as_int(mx.z), __float_as_int(mx.w), __float_as_int(mx.z), __float_as_int(mx.w)};
                    int c0[4], c1[4];
                    mma16816(c0, w00, w10, b0, ci0);
                    mma16816(c1, w01, w11, b1, ci1);
                    acc[n][0] = fmaf(f00, fmaf(md.x, __int_as_float(c0[0]), md.y), acc[n][0]);
                    acc[n][1] = fmaf(f00, fmaf(md.z, __int_as_float(c0[1]), md.w), acc[n][1]);
                    acc[n][2] = fmaf(f10, fmaf(md.x, __int_as_float(c0[2]), md.y), acc[n][2]);
                    acc[n][3] = fmaf(f10, fmaf(md.z, __int_as_float(c0[3]), md.w), acc[n][3]);
                    acc1[n][0] = fmaf(f01, fmaf(md.x, __int_as_float(c1[0]), md.y), acc1[n][0]);
                    acc1[n][1] = fmaf(f01, fmaf(md.z, __int_as_float(c1[1]), md.w), acc1[n][1]);
                    acc1[n][2] = fmaf(f11, fmaf(md.x, __int_as_float(c1[2]), md.y), acc1[n][2]);
                    acc1[n][3] = fmaf(f11, fmaf(md.z, __int_as_float(c1[3]), md.w), acc1[n][3]);
                }
            } else {
                const float fA0 = q6f_fa(r, d0, nd0, 0, 2 * k + h), fA1 = q6f_fa(r, d1, nd1, 1, 2 * k + h);
                const int a[4] = {w00, w10, w01, w11};
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int x = *reinterpret_cast<const int *>(xr + 4 * n * xs.ld);
                    const int b0 = (g >> 2) ? 0 : x, b1 = (g >> 2) ? x : 0;
                    const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
                    const float2 mx = xs.mx[(j * C::NG + n) * 4 + tq];
                    const int ci[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    int cc[4];
                    mma16832(cc, a, b0, b1, ci);
                    acc[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc[n][0]);
                    acc[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc[n][1]);
                    acc[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc[n][2]);
                    acc[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc[n][3]);
                }
            }
        }
    }
}
template <int T, bool X16>
__device__ __forceinline__ void wv_q6f_fin(float (&acc)[WvCfg<T, X16>::NG][4], const float (&acc1)[X16 ? 1 : WvCfg<T, X16>::NG][4]) {
    using C = WvCfg<T, X16>;
    if constexpr (!X16) { // chunk 0 + chunk 1 (the split form's tq < 2 sum: own chunk-0 sum + the partner's chunk 1)
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += acc1[n][c];
    } else {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += __shfl_xor_sync(0xffffffffu, acc[n][c], 2);
    }
}
template <int T, bool X16>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_q6f(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
    using C = WvCfg<T, X16>;
    WvAcc<C::NG> acc;
    float acc1[X16 ? 1 : C::NG][4]; // k16 form: chunk 1's sums (acc: chunk 0's)
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc.v[n][c] = 0.f;
#pragma unroll
    for (int n = 0; n < (X16 ? 1 : C::NG); n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc1[n][c] = 0.f;
    wv_q6f_acc<T, X16>(m, tile, sb0, nsb, xs, acc.v, acc1);
    wv_q6f_fin<T, X16>(acc.v, acc1);
    return acc;
}

// (the activations: wv_fill<T, X16, ldu - 2>'s, rebuilt here by wv_xs)
template <int TYPE, int T, bool X16>
__device__ __noinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile(const Mat m, int tile, int sb0, int nsb, int ldu) {
    const WvXS xs = wv_xs<T, X16>(ldu);
    if constexpr (TYPE == T_Q6K) {
        return wv_tile_q6f<T, X16>(m, tile, sb0, nsb, xs);
    } else {
    static_assert(!wv_s16(TYPE) || X16, "per-16 weight scales need the split MMAs");
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    WvAcc<C::NG> acc;
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc.v[n][c] = 0.f;
    const int tok = X16 ? (g & 3) : g;
    const int8_t * xra = xs.xa + tok * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + tok * xs.ld + 8 * tq;
    const bool bon = !X16 || (g >> 2) == (tq >> 1);
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        typename WV<TYPE>::Raw r0, r1;
        wv_load2<TYPE>(r0, r1, m, tile, sb0 + sb, g, tq);
        wv_sb<TYPE, T, X16>(r0, r1, sb, xs, xra, xrb, bon, acc.v);
    }
    if (X16) { // chunk 0 (tq 0-1) + chunk 1 (tq 2-3) of the same tokens
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc.v[n][c] += __shfl_xor_sync(0xffffffffu, acc.v[n][c], 2);
    }
    return acc;
    }
}

// xs: the phase's wv_fill view (wv_tile gets only its token stride)
template <int T, bool X16>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_any(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
    const int ldu = xs.ld / 16;
    if constexpr (!X16) // FM Q6_K in the q32 phases (m16n8k16 form; the phase's fill was q6)
        if (m.type == T_Q6K) return wv_tile<T_Q6K, T, false>(m, tile, sb0, nsb, ldu);
    if constexpr (X16) { // ffn_down: the offset types too
        switch (m.type) {
            case T_Q6K: return wv_tile<T_Q6K, T, X16>(m, tile, sb0, nsb, ldu);
            case T_Q3K: return wv_tile<T_Q3K, T, X16>(m, tile, sb0, nsb, ldu);
            case T_IQ3S: return wv_tile<T_IQ3S, T, X16>(m, tile, sb0, nsb, ldu);
            default: break;
        }
    }
    switch (m.type) {
        case T_Q4K: return wv_tile<T_Q4K, T, X16>(m, tile, sb0, nsb, ldu);
        case T_Q5K: return wv_tile<T_Q5K, T, X16>(m, tile, sb0, nsb, ldu);
        case T_IQ4NL: return wv_tile<T_IQ4NL, T, X16>(m, tile, sb0, nsb, ldu);
        case T_Q8_0: return wv_tile<T_Q8_0, T, X16>(m, tile, sb0, nsb, ldu);
        default: return wv_tile<T_IQ4X, T, X16>(m, tile, sb0, nsb, ldu);
    }
}


// ---- phases ----------------------------------------------------------------------------------------------------------
// Work split of a phase (bench k_wv): block b takes the tiles [ntile b / G, ntile (b + 1) / G) of the phase's tile space
// (G blocks per K chunk), its warps round-robin starting from the last warp, so the first warps, which also take the
// phase's dp4a items (alpha / beta, the hybrid's Q6_K / Q3_K / IQ3_S matrices: gemv_pairs over the low global warps),
// get fewer tiles. Every output depends only on its own row and token (T-invariant per tensor).

// the hybrid's dp4a matrices on the wv fill (rows_* XP = 2). No Q6_K case here or in resid_dp / rows16_dp_g
// (every main-layer Q6_K is FM: the loader places it and refuses anything else outside the wv layout)
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_dp(const Mat & m, int lo, int hi, Out out) {
    switch (m.type) {
        case T_Q3K: gemv_pairs<T, IT, T_Q3K, 2>(m, lo, hi, out); break;
        default: gemv_pairs<T, IT, T_IQ3S, 2>(m, lo, hi, out); break;
    }
}
template <int T>
__device__ __forceinline__ LaneT<T> rows16_dp(const Mat & m, int j) {
    switch (m.type) {
        case T_Q3K: return rows16<T, T_Q3K, 2>(m, j);
        default: return rows16<T, T_IQ3S, 2>(m, j);
    }
}

// out / ssm_out not in the wv layout: dp4a resid32 (T <= 5: any type; T >= 6: only the hybrid types)
template <int T, int IT>
__device__ __forceinline__ void resid_dp(const Mat & W, const QBuf & in, float * x, bool add, const float * wnext, uint8_t * sm) {
    if constexpr (T <= 5) {
        resid_any<T, IT>(W, in, x, add, wnext, sm);
    } else {
        switch (W.type) {
            case T_Q3K: resid32<T, IT, T_Q3K>(W, in, x, add, wnext, sm); break;
            default: resid32<T, IT, T_IQ3S>(W, in, x, add, wnext, sm); break;
        }
    }
}

// C fragment (v[0][c]: row g + 8 (c >> 1), token 2 tq + (c & 1)) -> lane r < 16 holds row r (lanes 16-31: copies)
template <int T>
__device__ __forceinline__ LaneT<T> wv_to_lane(const WvAcc<1> & a) {
    const int r = threadIdx.x & 15;
    LaneT<T> o;
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int src = 4 * (r & 7) + (t >> 1);
        const float x0 = __shfl_sync(0xffffffffu, a.v[0][t & 1], src);
        const float x1 = __shfl_sync(0xffffffffu, a.v[0][2 + (t & 1)], src);
        o.v[t] = (r >> 3) ? x1 : x0;
    }
    return o;
}

// non-split C fragment of tile `tile` -> dst[t * ld + 16 tile + row] = v * s_rs[t]
template <int T>
__device__ __forceinline__ void wv_store(float * dst, int ld, int tile, const WvAcc<1> & a, const float * s_rs) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int t = 2 * tq + (c & 1);
        if (t < T) dst[t * ld + 16 * tile + g + 8 * (c >> 1)] = a.v[0][c] * s_rs[t];
    }
}

// delta-net input projections: alpha / beta (fp32 path, as dn_proj), qkv -> conv ring and z (wv tiles, or dp4a pairs
// for a hybrid-type matrix)
// BAT (batched launches): row t's qkv goes to the conv ring of its own lane at its own position: rt (the pass's lanes) gives
// each row its lane, the conv-ring rows are looked up at the start (s_cvp); orphan rows (past the lanes) are not stored.
// No L2 warm-up of dn_core's inputs (they are per lane). Every matrix of the phase must be in the wv layout (host check).
template <int T, bool BAT = false>
__device__ __noinline__ void dn_proj_wv(const LayerW & W, int dl, int n, int dn_c, RowTab rt = RowTab{}) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    float ** const s_cvp = reinterpret_cast<float **>(misc(nullptr) + 64); // (BAT: per row, its conv-ring row or null)
    if constexpr (BAT) {
        if (threadIdx.x < T) {
            const int t = threadIdx.x;
            int r0;
            const int ln = rt_lane(rt, t, r0);
            float * cv = nullptr;
            if (ln >= 0) {
                const Bufs & V = cBs[view_main(ln)];
                const int p = ((volatile Ctl *) V.ctl)->n + t - r0;
                cv = V.conv + (size_t) dl * CONV_RING * QKV + (size_t) (p % CONV_RING) * QKV;
            }
            s_cvp[t] = cv;
        }
        // (published by wv_fill's barrier)
    }
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.qkv.type == T_Q6K || W.gate.type == T_Q6K);
    const float * s_rs = misc(nullptr);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH;
    // alpha / beta items. Upstream's dealing (gwarp: warps 0-3 (+ 4) of every block) puts them on warps that also take a
    // qkv / z tile below (tiles start at warp NW - 1 going down), so those warps finish after the others.
    // K1 (cB.k1, every matrix of the phase in the wv layout): block b takes the items [n_ab b / G, n_ab (b + 1) / G) and
    // deals them round-robin to its warps with one tile fewer than the others (warps 0 .. NW - 1 - nb % NW, nb = the block's
    // tiles: at 82 blocks the 3-4 warps without a tile, <= 2 items each). Each item computes the same row / K-chunk as
    // with upstream's dealing: bit-exact.
    int ab0 = gwarp(), ab1 = n_ab, astep = nwarps();
    if (cB.k1 && W.qkv.fm && W.gate.fm) {
        constexpr int NTL = QKV / 16 + VD / 16; // (the tile space below when both are FM)
        const int nb = NTL * (blockIdx.x + 1) / gridDim.x - NTL * blockIdx.x / gridDim.x;
        const int nl = NW - nb % NW; // warps 0 .. nl - 1 carry nb / NW tiles, the others one more
        ab1 = n_ab * (blockIdx.x + 1) / gridDim.x;
        ab0 = warp < nl ? n_ab * blockIdx.x / gridDim.x + warp : ab1;
        astep = nl;
    }
    for (int it = ab0; it < ab1; it += astep) {
        const int row = it % (2 * NVH), ch = it / (2 * NVH);
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        if (W.abq) {
            const int8_t * qr = W.abq + (size_t) row * D;
            const __half * dr = W.abd + (size_t) row * (D / 32);
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const int qw = ldw1(qr + k);
                const float d = __half2float(dr[k >> 5]);
                const float4 fn = __ldg(reinterpret_cast<const float4 *>(W.attn_norm + k));
                float4 w;
                w.x = ((float) (int8_t) (qw) * d) * fn.x;
                w.y = ((float) (int8_t) (qw >> 8) * d) * fn.y;
                w.z = ((float) (int8_t) (qw >> 16) * d) * fn.z;
                w.w = ((float) (int8_t) (qw >> 24) * d) * fn.w;
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        } else {
            const float * wr = W.ab + (size_t) row * D;
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const float4 w = ldw4f(wr + k);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<T>(cB.ab + ch * RMAX * 2 * NVH, 2 * NVH, row, acc);
    }
    int lo = n_ab;
    if (!W.qkv.fm) {
        gemv_pairs_dp<T, 5>(W.qkv, lo, lo + QKV / 2, PairOut{conv, QKV, n});
        lo += QKV / 2;
    }
    if (!W.gate.fm) gemv_pairs_dp<T, 5>(W.gate, lo, lo + VD / 2, PairOut{cB.z, VD, -1});
    const int nt_q = W.qkv.fm ? QKV / 16 : 0, ntile = nt_q + (W.gate.fm ? VD / 16 : 0);
    const int t0 = ntile * blockIdx.x / gridDim.x, t1 = ntile * (blockIdx.x + 1) / gridDim.x;
    for (int tile = t0 + (NW - 1 - warp); tile < t1; tile += NW) {
        const bool isq = tile < nt_q;
        const int lt = isq ? tile : tile - nt_q;
        const WvAcc<1> a = wv_tile_any<T, false>(isq ? W.qkv : W.gate, lt, 0, D / 256, xs);
        if (isq) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
                if constexpr (BAT) {
                    if (t < T && s_cvp[t]) s_cvp[t][r] = a.v[0][c] * s_rs[t];
                } else {
                    if (t < T) conv[((n + t) % CONV_RING) * QKV + r] = a.v[0][c] * s_rs[t];
                }
            }
        } else {
            wv_store<T>(cB.z, VD, lt, a, s_rs);
        }
    }
    // Warm L2 with what dn_core will read (as dn_proj)
    if (!BAT && blockIdx.x < NVH) {
        const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x;
        pf_l2(cB.dstate + ((size_t) dl * NVH + h) * 128 * 128 + tid * 32);
        if (tid == 0) {
            pf_l2(W.dt + (h & ~31));
            pf_l2(W.A + (h & ~31));
            pf_l2(W.snorm);
            pf_l2(W.snorm + 32);
            pf_l2(W.snorm + 64);
            pf_l2(W.snorm + 96);
        }
        if (tid < 384) {
            const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
            if ((tid & 7) == 0) pf_l2(W.conv + ch * 4);
            if ((tid & 31) == 0)
                for (int j = 0; j < 3; j++)
                    if (n - 3 + j >= 0) pf_l2(conv + ((n - 3 + j) % CONV_RING) * QKV + ch);
        }
        const float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
        for (int i = tid; i < (n - dn_c) * 8; i += NT) {
            const float * slot = ring + (size_t) ((dn_c + (i >> 3)) % KV_RING) * KVGB;
            const int c = (i & 7);
            pf_l2(c < 4 ? slot + kh * 128 + c * 32 : slot + KVGB_V + h * 128 + (c - 4) * 32);
        }
    }
}

// K6 (MK_K6 bit 4, EXACT): rows 16 lt + 8 rh .. + 7 of Q8_0 k / v tile lt (w = 1 k, 2 v), the whole K. The thread
// loads only its row g + 8 rh (half the tile's bytes per warp) and feeds it as both MMA rows (g and g + 8): an MMA row's
// int32 sums and its scale use only that row's data, so the C values of the first row (c < 2) are bit-identical to the
// whole tile's values of that row; the duplicates (c >= 2) are dropped. Same store as wv_store.
template <int T>
__device__ __noinline__ void attn_kv_rh(const Mat m, int w, int lt, int rh, const WvXS xs) {
    using C = WvCfg<T, false>;
    static_assert(C::NG == 1, "K6 row halves: one token group");
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    WvAcc<1> acc;
#pragma unroll
    for (int c = 0; c < 4; c++) acc.v[0][c] = 0.f;
    const int8_t * xra = xs.xa + g * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + g * xs.ld + 8 * tq;
#pragma unroll 1
    for (int sb = 0; sb < D / 256; sb++) {
        typename WV<T_Q8_0>::Raw r0;
        WV<T_Q8_0>::load(r0, m, lt, g + 8 * rh, sb, tq);
        wv_sb<T_Q8_0, T, false>(r0, r0, sb, xs, xra, xrb, true, acc.v);
    }
    const float * s_rs = misc(nullptr);
    float * const dst = w == 1 ? cB.kbuf : cB.vbuf;
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int t = 2 * tq + c;
        if (t < T) dst[t * KVD + 16 * lt + g + 8 * rh] = acc.v[0][c] * s_rs[t];
    }
}

// attention q / k / v projections (wv tiles of the wv-layout matrices, dp4a pairs of the hybrid's)
// pf false (batched launches): no L2 warm-up of attn_core's inputs (they are per lane)
template <int T>
__device__ __noinline__ void attn_proj_wv(const LayerW & W, int slot, int n, bool pf = true) {
    const int warp = threadIdx.x >> 5;
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.q.type == T_Q6K || W.k.type == T_Q6K || W.v.type == T_Q6K);
    const float * s_rs = misc(nullptr);
    int lo = 0;
    if (!W.q.fm) {
        gemv_pairs_dp<T, 5>(W.q, lo, lo + QG / 2, PairOut{cB.qbuf, QG, -1});
        lo += QG / 2;
    }
    if (!W.k.fm) {
        gemv_pairs_dp<T, 5>(W.k, lo, lo + KVD / 2, PairOut{cB.kbuf, KVD, -1});
        lo += KVD / 2;
    }
    if (!W.v.fm) gemv_pairs_dp<T, 5>(W.v, lo, lo + KVD / 2, PairOut{cB.vbuf, KVD, -1});
    const int nq = W.q.fm ? QG / 16 : 0, nk = W.k.fm ? KVD / 16 : 0, nv = W.v.fm ? KVD / 16 : 0;
    // K6: items = the q tiles, then the k tiles, then the v tiles; with bit 4 a Q8_0 k / v tile (87 KB vs 56 KB for a
    // Q5_K q tile: the phase's tail in the 10 Q8_0-v layers) is 2 items of 8 rows each (attn_kv_rh, exact). Dealing:
    // contiguous runs per block (upstream), or with bit 2 item i -> block i % G (the heavy k / v items spread over the blocks)
    const int k6 = cB.k6, ks = (k6 & 4) ? 2 : 1;
    const int sk = W.k.type == T_Q8_0 ? ks : 1, sv = W.v.type == T_Q8_0 ? ks : 1;
    const int nkk = nk * sk, ni = nq + nkk + nv * sv;
    int i0, i1, istep;
    if (k6 & 2) {
        i0 = blockIdx.x + gridDim.x * (NW - 1 - warp);
        i1 = ni;
        istep = gridDim.x * NW;
    } else {
        i0 = ni * blockIdx.x / gridDim.x + (NW - 1 - warp);
        i1 = ni * (blockIdx.x + 1) / gridDim.x;
        istep = NW;
    }
    for (int it = i0; it < i1; it += istep) {
        const int w = it < nq ? 0 : it < nq + nkk ? 1 : 2;
        const int j = it - (w == 0 ? 0 : w == 1 ? nq : nq + nkk), ns = w == 0 ? 1 : w == 1 ? sk : sv;
        const int lt = j / ns;
        if (ns > 1) {
            attn_kv_rh<T>(w == 1 ? W.k : W.v, w, lt, j - lt * ns, xs);
            continue;
        }
        const WvAcc<1> a = wv_tile_any<T, false>(w == 0 ? W.q : w == 1 ? W.k : W.v, lt, 0, D / 256, xs);
        wv_store<T>(w == 0 ? cB.qbuf : w == 1 ? cB.kbuf : cB.vbuf, w == 0 ? QG : KVD, lt, a, s_rs);
    }
    // Warm L2 with what attn_core reads first (as attn_proj)
    if (!pf) return;
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    int nch, CS;
    attn_dims(n + T, nch, CS);
    const size_t ctx = cB.ctx;
    const int rk = cB.kvq8 ? HD * cB.kvb[0][slot] / 8 : HD * 2, rv = cB.kvq8 ? HD * cB.kvb[1][slot] / 8 : HD * 2;
    const int lk = rk / 128, lv = rv / 128, lines = max(lk, lv);
    const uint8_t * Kc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[0][slot])
                                 : static_cast<const uint8_t *>(cB.kc) + (size_t) slot * NKV * ctx * rk;
    const uint8_t * Vc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[1][slot])
                                 : static_cast<const uint8_t *>(cB.vc) + (size_t) slot * NKV * ctx * rv;
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c0 = (item % nch) * CS, c1 = min(n, min(c0 + CS, c0 + 64));
        for (int i = threadIdx.x; i < (c1 - c0) * lines; i += NT) {
            const size_t pos = (size_t) g * ctx + c0 + i / lines;
            const int l = i % lines;
            if (l < lk) pf_l2(Kc + pos * rk + l * 128);
            if (l < lv) pf_l2(Vc + pos * rv + l * 128);
        }
    }
}

// q16_store of chunk j from lane layout (lanes < 16: row r of each token)
template <int T>
__device__ __forceinline__ void glu_store(int j, const float (&act)[T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t += 2) {
        const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
        const float v = lane < 16 ? act[t] : other;
        q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
    }
}

// C1: the ffn_gu -> ffn_down grid barrier is replaced by per-chunk waits. ffn_down (resid_wv) runs in WV_S split-K chunks of G = gridDim / WV_S blocks, and
// chunk kc reads only the gate / up tiles 272 kc .. 272 kc + 271; ffn_gu_wv deals exactly those tiles to chunk kc's
// blocks, each block counts its arrival on the chunk's counter, and resid_wv waits for its own chunk's G arrivals before
// it reads cB.qa. Bitwise exact (the same tiles and arithmetic; only which block computes a gate / up tile and when
// ffn_down starts change). Layers whose gate and up are both on dp4a at T <= 5 (ffn_gu: warp-cyclic rows, every block
// feeds every chunk) keep the barrier.
// Per-chunk readiness of cB.qa between ffn_gu_wv and ffn_down's resid_wv: chunk kc's counter is the word
// cB.bar + C1_W0 + GS_GO kc (own 128-byte line; zeroed once with the barrier words, then running on from the count
// C1_BASE the previous launch left: MK_C1E adds it). Each of the chunk's G
// producer blocks release-adds 1 after its gate / up tiles (c1_arrive); the chunk's consumers (the same G blocks)
// acquire-poll it until it reaches G x the number of split edges this launch has passed (s_c1n, thread 0 of each block
// of chunks 0 .. WV_S - 1; mk_cycle zeroes it). Cumulative, never reset within a launch: the next edge's adds come only
// after the PH_FFN_DOWN grid barrier, which every consumer reaches after its wait, so a count of G e is reached exactly
// when edge e's producers have all arrived.
static_assert((FF / 16) % WV_S == 0, "gate / up tiles per ffn_down chunk");
static_assert(SM_TOTAL + 32 <= 99 * 1024, "mk_cycle's dynamic + static shared memory (+ s_c1n)");
__shared__ unsigned s_c1n;
#define MK_C1E(x) (s_c1b + (x))
__device__ __forceinline__ void c1_arrive(int kc) {
    __syncthreads(); // (every warp's q16 stores of the block's tiles before thread 0's release)
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cB.bar + C1_W0 + GS_GO * kc) : "memory");
}
__device__ __forceinline__ void c1_wait(int kc, int G) {
    if (threadIdx.x == 0) {
        const unsigned e = MK_C1E(++s_c1n * (unsigned) G);
        const unsigned * w = cB.bar + C1_W0 + GS_GO * kc;
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(w) : "memory");
        } while (MK_BAR_LT(v, e));
    }
    __syncthreads();
}

// FFN gate / up: warp item = tile j of gate and of up -> silu(g) * u quantised per 16-chunk (chunk j) for ffn_down.
// c1: the tiles of ffn_down chunk kc go to that chunk's blocks (resid_wv: G = gridDim / WV_S blocks per chunk;
// blocks past WV_S G take none), and each block arrives on its chunk's counter after its tiles (no grid barrier follows).
template <int T>
__device__ __noinline__ void ffn_gu_wv(const LayerW & W, bool c1 = false) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tq = lane & 3;
    const int G1 = gridDim.x / WV_S, kc1 = blockIdx.x / G1;
    if (c1 && kc1 >= WV_S) return;
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.fg.type == T_Q6K || W.fu.type == T_Q6K);
    const float * s_rs = misc(nullptr);
    constexpr int NJ = FF / 16;
    int t0 = NJ * blockIdx.x / gridDim.x, t1 = NJ * (blockIdx.x + 1) / gridDim.x;
    if (c1) {
        constexpr int NJC = NJ / WV_S;
        const int bi = blockIdx.x - kc1 * G1;
        t0 = NJC * kc1 + NJC * bi / G1;
        t1 = NJC * kc1 + NJC * (bi + 1) / G1;
    }
    const bool gw = W.fg.fm, uw = W.fu.fm;
    for (int j = t0 + (NW - 1 - warp); j < t1; j += NW) {
        float act[T];
        if (gw && uw) { // GLU on the C fragments, one transpose
            const WvAcc<1> g = wv_tile_any<T, false>(W.fg, j, 0, D / 256, xs);
            const WvAcc<1> u = wv_tile_any<T, false>(W.fu, j, 0, D / 256, xs);
            WvAcc<1> a;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 2 * tq + (c & 1);
                const float s = t < T ? s_rs[t] : 0.f;
                a.v[0][c] = silu(g.v[0][c] * s) * (u.v[0][c] * s);
            }
            const LaneT<T> l = wv_to_lane<T>(a);
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? l.v[t] : 0.f;
        } else { // one of them on dp4a (rows16: lane r = row r, already scaled)
            LaneT<T> g, u;
            if (gw) {
                g = wv_to_lane<T>(wv_tile_any<T, false>(W.fg, j, 0, D / 256, xs));
#pragma unroll
                for (int t = 0; t < T; t++) g.v[t] *= s_rs[t];
            } else {
                g = rows16_dp<T>(W.fg, j);
            }
            if (uw) {
                u = wv_to_lane<T>(wv_tile_any<T, false>(W.fu, j, 0, D / 256, xs));
#pragma unroll
                for (int t = 0; t < T; t++) u.v[t] *= s_rs[t];
            } else {
                u = rows16_dp<T>(W.fu, j);
            }
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? silu(g.v[t]) * u.v[t] : 0.f;
        }
        glu_store<T>(j, act);
    }
    if (c1) c1_arrive(kc1);
}

// Output rows with the residual add (out: K = VD, ffn_down: K = FF, X16 = its per-16 activations), split-K WV_S chunks
// (G = gridDim / WV_S blocks per chunk). Each chunk's partial sums go to cB.wvpart; the last of the 2 WV_S (tile, chunk)
// arrivals at a 32-row group adds the chunks in order 0 .. WV_S - 1 (fixed: T-invariant), adds the residual and
// re-quantises it for the next RMS-normed consumer (weight wnext), like resid32's epilogue.
template <int T, bool X16, int K>
__device__ __noinline__ void resid_wv(const Mat Wm, const QBuf in, const float * wnext, int role) {
    constexpr int NSB = K / 256, CH = NSB / WV_S, US = 8 * CH;
    static_assert(NSB % WV_S == 0, "split-K chunks of whole super-blocks");
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    const int G = gridDim.x / WV_S, b = blockIdx.x;
    if (b >= WV_S * G) return;
    const int kc = b / G, bi = b - kc * G;
    if (role & 2) c1_wait(kc, G); // (C1: ffn_down after ffn_gu_wv's c1 dealing, no grid barrier: this chunk's tiles of in)
    const WvXS xs = wv_fill<T, X16, US>(in, US * kc, false, nullptr, X16 && wv_min2(Wm.type), !X16 && Wm.type == T_Q6K);
    constexpr int NTILE = D / 16;
    const int t0 = NTILE * bi / G, t1 = NTILE * (bi + 1) / G;
    unsigned * cnt = cB.wvcnt + (role & 1) * (D / 32); // (role bit 1: the C1 wait above)
    float * x = cB.x;
    for (int tile = t0 + (NW - 1 - warp); tile < t1; tile += NW) {
        const WvAcc<C::NG> a = wv_tile_any<T, X16>(Wm, tile, CH * kc, CH, xs);
        float * part = cB.wvpart + (size_t) kc * RMAX * D;
        if (!X16 || tq < 2) {
#pragma unroll
            for (int n = 0; n < C::NG; n++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = X16 ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                    if (t < T) __stcg(part + (size_t) t * D + 16 * tile + g + 8 * (c >> 1), a.v[n][c]);
                }
        }
        __threadfence();
        __syncwarp();
        unsigned old = 0;
        if (lane == 0) old = atomicAdd(cnt + (tile >> 1), 1u);
        old = __shfl_sync(0xffffffffu, old, 0);
        if (old == 2 * WV_S - 1) { // last arrival at this 32-row group
            __threadfence();
            const int j = tile >> 1, row = 32 * j + lane;
            if (cB.k2) {
                // K2: the K2-off loop below makes one dependent L2 round trip (+ q32_store's shuffle chains) per token,
                // serially. Here a batch of TB tokens issues all its partial / residual loads first, then runs the tokens'
                // epilogues (independent: their shuffle chains interleave). Same per-element order (((p0 + p1) + p2) + p3)
                // + x: bit-exact.
                constexpr int TB = MK_K2_TB < T ? MK_K2_TB : T;
                const float wn = wnext[row];
#pragma unroll // (T, TB compile-time: no runtime guards; x stores keep the batches apart)
                for (int tb = 0; tb < T; tb += TB) {
                    float p[TB][WV_S], xo[TB];
#pragma unroll
                    for (int u = 0; u < TB; u++)
                        if (tb + u < T) {
#pragma unroll
                            for (int k = 0; k < WV_S; k++) p[u][k] = __ldcg(cB.wvpart + ((size_t) k * RMAX + tb + u) * D + row);
                            xo[u] = __ldcg(x + (tb + u) * D + row);
                        }
#pragma unroll
                    for (int u = 0; u < TB; u++)
                        if (tb + u < T) {
                            float s = p[u][0];
#pragma unroll
                            for (int k = 1; k < WV_S; k++) s += p[u][k];
                            const float xv = s + xo[u];
                            x[(tb + u) * D + row] = xv;
                            q32_store(cB.qn, tb + u, j, xv * wn, xv * xv, true);
                        }
                }
            } else {
#pragma unroll 1
                for (int t = 0; t < T; t++) {
                    float s = __ldcg(cB.wvpart + (size_t) t * D + row);
#pragma unroll
                    for (int k = 1; k < WV_S; k++) s += __ldcg(cB.wvpart + ((size_t) k * RMAX + t) * D + row);
                    const float xv = s + __ldcg(x + t * D + row);
                    x[t * D + row] = xv;
                    q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
                }
            }
            if (lane == 0) cnt[j] = 0;
        }
    }
}

// out / ssm_out (K = VD) in the wv layout, residual add into x, re-quantised for the next normed consumer (role 0).
// (The offset types' out / ssm_out stay on dp4a resid32.)
template <int T>
__device__ __forceinline__ void resid_out_wv(const Mat & Wm, const float * wnext) {
    resid_wv<T, false, VD>(Wm, cB.qo, wnext, 0);
}

#if MK_PAR_WIDE
// ---- batched passes wider than MAXT rows (RT = RMAX, MKEngine::lanes_step) ----------------------------------------------
// 24 rows of K = D activations do not fit shared memory (6.4 KB a row): every wv phase fills the K range in super-block
// aligned sub-chunks (wv_fill_ck) and each warp carries its single item's sums across them (wv_tile_st, WvSt), finishing
// once at the end (wv_st_fin). A token's fmaf chain is the same as in the whole-K tile of a solo launch (the same super-blocks
// in the same order, the finishing shuffle / Q6F chunk-1 add after the last one): bit-identical rows at any RT. Every phase
// gives a warp at most one item at RMAX (attn_proj <= 1024 items, dn_proj 1024 tiles, ffn_gu 1088 gate / up pairs, resid_wv
// 320 tiles per split-K chunk on 82 SMs x 16 warps); host checks (slots_open) cover the configurations this does not run.

// sub-chunk size (super-blocks) of a fill of T rows: the largest whose activations + scale tables fit SM_MISC
__host__ __device__ constexpr int wv_smem_rt(int us, int T, bool X16) { // (wv_smem<us, T, X16>())
    return 2 * ((T + (X16 ? 3 : 7)) / (X16 ? 4 : 8) * (X16 ? 4 : 8)) * (us + 2) * 16 + T * (us + 2) * 8 +
           us * ((T + (X16 ? 3 : 7)) / (X16 ? 4 : 8)) * 4 * (!X16 ? 40 : 24);
}
template <int T, bool X16, int NSB>
__host__ __device__ constexpr int wv_csb() {
    int c = NSB;
    while (c > 1 && wv_smem_rt(8 * c, T, X16) > SM_MISC) c--;
    return c;
}
static_assert(wv_smem_rt(56, 24, false) == wv_smem<56, 24, false>() && wv_smem_rt(48, 24, true) == wv_smem<48, 24, true>(), "wv_smem_rt");

// the RMS-norm scales s_rs[t] of T rows (as wv_fill's): once per phase, published by the first fill's barrier
template <int T>
__device__ __forceinline__ void rs_rows(const QBuf & q) {
    float * s_rs = misc(nullptr);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int t = warp; t < T; t += NW) {
        float pp[U_D / 32];
#pragma unroll
        for (int k = 0; k < U_D / 32; k++) pp[k] = __ldcg(q.ssq + t * U_D + lane + 32 * k);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < U_D / 32; k++) acc += pp[k];
        acc = wsum(acc);
        if (lane == 0) s_rs[t] = rsqrtf(acc / (float) D + EPS);
    }
}

// wv_fill of units [u0, u0 + nu) (nu <= US) without the s_rs part; a barrier first (the previous sub-chunk may be in use)
template <int T, bool X16, int US>
__device__ __forceinline__ WvXS wv_fill_ck(const QBuf & q, int u0, int nu, bool min2, bool q6) {
    using C = WvCfg<T, X16>;
    constexpr int LDU = US + 2;
    static_assert(wv_smem<US, T, X16>() <= SM_MISC, "wv activations");
    uint8_t * const sm = smem;
    int4 * xa = reinterpret_cast<int4 *>(sm);
    int4 * xb = xa + C::NTOKP * LDU;
    uint2 * xm = reinterpret_cast<uint2 *>(xb + C::NTOKP * LDU);
    float4 * md = reinterpret_cast<float4 *>(xm + T * LDU);
    float2 * mx = reinterpret_cast<float2 *>(md + US * C::NG * 4);
    const int tid = threadIdx.x;
    __syncthreads();
    for (int i = tid; i < T * nu; i += NT) {
        const int t = i / nu, u = i - t * nu;
        const size_t gi = (size_t) t * q.U + u0 + u;
        cp_async16(xa + t * LDU + u, q.xa + gi * 16);
        cp_async16(xb + t * LDU + u, q.xb + gi * 16);
    }
    for (int i = tid; i < T * nu / 2; i += NT) {
        const int t = 2 * i / nu, u = 2 * i - t * nu;
        cp_async16(xm + t * LDU + u, q.meta + (size_t) t * q.U + u0 + u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
    __syncthreads();
    for (int i = tid; i < nu * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1;
        float v[4];
        float2 x2;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (X16) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
            float dx = 0.f;
            int qs = 0;
            if (t < T) {
                const uint2 m0 = xm[t * LDU + ub + (hh < 0 ? 0 : hh)];
                const uint2 m1 = xm[t * LDU + ub + 1];
                const int sh = 16 * part; // (as wv_fill: shifts, not a runtime index into &m0 / &m1)
                dx = __half2float(__ushort_as_half((unsigned short) (m0.x >> sh)));
                const int q0 = (short) (m0.y >> sh), q1 = (short) (m1.y >> sh);
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -WV_FMAGIC * dx;
            (k ? x2.y : x2.x) = min2 ? __int_as_float(WV_MAGIC - 32 * qs) : dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
    float4 * const mx4 = reinterpret_cast<float4 *>(mx + US * C::NG * 4);
    if (!X16 && q6)
        for (int i = tid; i < nu * C::NG * 4; i += NT) {
            const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
            const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1;
            int qv[2][2];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int t = 8 * n + 2 * tq + k;
#pragma unroll
                for (int c = 0; c < 2; c++) qv[k][c] = t < T ? reinterpret_cast<const short *>(&xm[t * LDU + ub + c].y)[part] : 0;
            }
            mx4[i] = make_float4(__int_as_float(WV_MAGIC - 32 * qv[0][0]), __int_as_float(WV_MAGIC - 32 * qv[1][0]),
                                 __int_as_float(WV_MAGIC - 32 * qv[0][1]), __int_as_float(WV_MAGIC - 32 * qv[1][1]));
        }
    __syncthreads();
    mark_fill();
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), LDU * 16, md, mx, mx4};
}

template <int T, bool X16>
__device__ __forceinline__ void wv_st_zero(WvSt<T, X16> & st) {
#pragma unroll
    for (int n = 0; n < WvCfg<T, X16>::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) st.a.v[n][c] = 0.f;
#pragma unroll
    for (int n = 0; n < (!X16 ? WvCfg<T, X16>::NG : 1); n++)
#pragma unroll
        for (int c = 0; c < 4; c++) st.v1[n][c] = 0.f;
}

// the generic tile's super-block loop (wv_tile's), on the caller's sums
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv_gen_acc(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs, float (&acc)[WvCfg<T, X16>::NG][4]) {
    static_assert(!wv_s16(TYPE) || X16, "per-16 weight scales need the split MMAs");
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int tok = X16 ? (g & 3) : g;
    const int8_t * xra = xs.xa + tok * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + tok * xs.ld + 8 * tq;
    const bool bon = !X16 || (g >> 2) == (tq >> 1);
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        typename WV<TYPE>::Raw r0, r1;
        wv_load2<TYPE>(r0, r1, m, tile, sb0 + sb, g, tq);
        wv_sb<TYPE, T, X16>(r0, r1, sb, xs, xra, xrb, bon, acc);
    }
}

// a sub-chunk fill's view (wv_fill_ck's layout of us units), rebuilt in the tile functions from the shared base: their loads
// stay shared-space and the phases keep no fill pointers live across the calls
template <int T, bool X16>
__device__ __forceinline__ WvXS wv_xs_ck(int us) {
    using C = WvCfg<T, X16>;
    const int ldu = us + 2;
    const int4 * xa = reinterpret_cast<const int4 *>(smem);
    const int4 * xb = xa + C::NTOKP * ldu;
    const float4 * md = reinterpret_cast<const float4 *>(reinterpret_cast<const uint2 *>(xb + C::NTOKP * ldu) + T * ldu);
    const float2 * mx = reinterpret_cast<const float2 *>(md + us * C::NG * 4);
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), ldu * 16, md, mx,
                reinterpret_cast<const float4 *>(mx + us * C::NG * 4)};
}

// super-blocks [sb0, sb0 + nsb) of tile `tile` (the current fill of us units), continuing a (out of line per type, as wv_tile)
template <int TYPE, int T, bool X16>
__device__ __noinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_st(const Mat m, int tile, int sb0, int nsb, int us, WvAcc<WvCfg<T, X16>::NG> a) {
    const WvXS xs = wv_xs_ck<T, X16>(us);
    if constexpr (TYPE == T_Q6K) {
        static_assert(X16, "the FM Q6_K k16 form carries v1: wv_tile_st6");
        float v1[1][4]; // (the split form has no chunk-1 sums)
        wv_q6f_acc<T, X16>(m, tile, sb0, nsb, xs, a.v, v1);
    } else
        wv_gen_acc<TYPE, T, X16>(m, tile, sb0, nsb, xs, a.v);
    return a;
}
// FM Q6_K in the q32 phases (k16 form), continuing both chunks' sums
template <int T>
__device__ __noinline__ WvSt<T, false> wv_tile_st6(const Mat m, int tile, int sb0, int nsb, int us, WvSt<T, false> st) {
    wv_q6f_acc<T, false>(m, tile, sb0, nsb, wv_xs_ck<T, false>(us), st.a.v, st.v1);
    return st;
}

// continue st with super-blocks [sb0, sb0 + nsb) of tile `tile`. Q6: m may be FM Q6_K (q32 phases; else st.v1 stays zero and
// out of the calls, which then carry only the 8 sums of st.a)
template <int T, bool X16, bool Q6>
__device__ __forceinline__ void wv_tile_any_st(const Mat & m, int tile, int sb0, int nsb, int us, WvSt<T, X16> & st) {
    if constexpr (Q6 && !X16)
        if (m.type == T_Q6K) {
            st = wv_tile_st6<T>(m, tile, sb0, nsb, us, st);
            return;
        }
    WvAcc<WvCfg<T, X16>::NG> & a = st.a;
    if constexpr (X16) {
        switch (m.type) {
            case T_Q6K: a = wv_tile_st<T_Q6K, T, X16>(m, tile, sb0, nsb, us, a); return;
            case T_Q3K: a = wv_tile_st<T_Q3K, T, X16>(m, tile, sb0, nsb, us, a); return;
            case T_IQ3S: a = wv_tile_st<T_IQ3S, T, X16>(m, tile, sb0, nsb, us, a); return;
            default: break;
        }
    }
    switch (m.type) {
        case T_Q4K: a = wv_tile_st<T_Q4K, T, X16>(m, tile, sb0, nsb, us, a); return;
        case T_Q5K: a = wv_tile_st<T_Q5K, T, X16>(m, tile, sb0, nsb, us, a); return;
        case T_IQ4NL: a = wv_tile_st<T_IQ4NL, T, X16>(m, tile, sb0, nsb, us, a); return;
        case T_Q8_0: a = wv_tile_st<T_Q8_0, T, X16>(m, tile, sb0, nsb, us, a); return;
        default: a = wv_tile_st<T_IQ4X, T, X16>(m, tile, sb0, nsb, us, a); return;
    }
}

// the tile's finished C fragment (wv_tile's ending: X16 chunk 0 + chunk 1; FM Q6_K k16 form chunk 0 + chunk 1)
template <int T, bool X16, bool Q6>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_st_fin(const WvSt<T, X16> & st, int type) {
    using C = WvCfg<T, X16>;
    WvAcc<C::NG> a = st.a;
    if constexpr (X16) {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) a.v[n][c] += __shfl_xor_sync(0xffffffffu, a.v[n][c], 2);
    }
    else if (Q6 && type == T_Q6K) {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) a.v[n][c] += st.v1[n][c];
    }
    return a;
}

// K6 bit 4 (exact Q8_0 k / v row halves, attn_kv_rh) on a sub-chunk (the current fill of us units), continuing a
template <int T>
__device__ __noinline__ WvAcc<WvCfg<T, false>::NG> attn_kv_rh_st(const Mat m, int lt, int rh, int sb0, int nsb, int us, WvAcc<WvCfg<T, false>::NG> a) {
    const WvXS xs = wv_xs_ck<T, false>(us);
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int8_t * xra = xs.xa + g * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + g * xs.ld + 8 * tq;
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        typename WV<T_Q8_0>::Raw r0;
        WV<T_Q8_0>::load(r0, m, lt, g + 8 * rh, sb0 + sb, tq);
        wv_sb<T_Q8_0, T, false>(r0, r0, sb, xs, xra, xrb, true, a.v);
    }
    return a;
}

// non-split C fragment of tile `tile`, all token groups -> dst[t * ld + 16 tile + row] = v * s_rs[t] (wv_store's values)
template <int T>
__device__ __forceinline__ void wv_store_n(float * dst, int ld, int tile, const WvAcc<WvCfg<T, false>::NG> & a, const float * s_rs) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < WvCfg<T, false>::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int t = 8 * n + 2 * tq + (c & 1);
            if (t < T) dst[t * ld + 16 * tile + g + 8 * (c >> 1)] = a.v[n][c] * s_rs[t];
        }
}

// non-split C fragment, all token groups -> lane r < 16 holds row r (wv_to_lane's layout)
template <int T>
__device__ __forceinline__ LaneT<T> wv_to_lane_n(const WvAcc<WvCfg<T, false>::NG> & a) {
    const int r = threadIdx.x & 15;
    LaneT<T> o;
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int n = t >> 3, tt = t & 7, src = 4 * (r & 7) + (tt >> 1);
        const float x0 = __shfl_sync(0xffffffffu, a.v[n][tt & 1], src);
        const float x1 = __shfl_sync(0xffffffffu, a.v[n][2 + (tt & 1)], src);
        o.v[t] = (r >> 3) ? x1 : x0;
    }
    return o;
}

// a hybrid (dp4a) matrix's rows 16 j .. + 15 for token group grp (rows 8 grp .. + 7) read from the QBuf in global memory:
// the per-lane unit order and warp sums of rows16 (the activations are the same values as its shared-memory copy)
template <int TYPE>
__device__ __noinline__ LaneT<8> rows16_g(const Mat m, int j, int grp) {
    const int lane = threadIdx.x & 31;
    const QBuf & q = cB.qn;
    SX sx;
    sx.xa = reinterpret_cast<int4 *>(q.xa + (size_t) 8 * grp * q.U * 16);
    sx.xb = reinterpret_cast<int4 *>(q.xb + (size_t) 8 * grp * q.U * 16);
    sx.xm = q.meta + (size_t) 8 * grp * q.U;
    sx.U = q.U;
    const float * s_rs = misc(nullptr) + 8 * grp;
    constexpr int R = 2;
    LaneT<8> res;
#pragma unroll
    for (int t = 0; t < 8; t++) res.v[t] = 0.f;
#pragma unroll 1
    for (int r = 0; r < 16; r += R) {
        float a[R][8];
        zero_acc<8, R>(a);
        mrows<8, 5, R, TYPE, 0>(m, 16 * j + r, sx, a);
        wsum_acc<8, R>(a);
#pragma unroll
        for (int k = 0; k < R; k++)
            if (lane == r + k) {
#pragma unroll
                for (int t = 0; t < 8; t++) res.v[t] = a[k][t] * s_rs[t];
            }
    }
    return res;
}
__device__ __forceinline__ LaneT<8> rows16_dp_g(const Mat & m, int j, int grp) {
    switch (m.type) {
        case T_Q3K: return rows16_g<T_Q3K>(m, j, grp);
        default: return rows16_g<T_IQ3S>(m, j, grp);
    }
}
template <int T>
__device__ __forceinline__ void rows16_dp_all(const Mat & m, int j, LaneT<T> & o) {
#pragma unroll 1
    for (int grp = 0; grp < T / 8; grp++) {
        const LaneT<8> p = rows16_dp_g(m, j, grp);
#pragma unroll
        for (int t = 0; t < 8; t++)
#pragma unroll
            for (int u = 0; u < T / 8; u++)
                if (u == grp) o.v[8 * u + t] = p.v[t];
    }
}

// _ck phases (MK_DEEP2 bit 2): L2 prefetch of tile `tile`'s units from super-block sb on, MK_CK_PFL 128-B lines per warp
// and matrix, issued before a sub-chunk fill so the warp's weight stream keeps running through the fill's barriers (the
// T <= MAXT phases fill once). Normal L2 priority: the lines are read once, within microseconds
#ifndef MK_CK_PFL
#define MK_CK_PFL 12
#endif
__device__ __forceinline__ void pf_l2n(const void * p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
template <bool X16>
__device__ __forceinline__ void wv_pf_next(const Mat & m, int tile, int sb) {
    const int lane = threadIdx.x & 31, ty = m.type;
    if (!(cB.deep2 & 2) || lane >= MK_CK_PFL) return;
    // unit (tile, sb) bytes: wv_unit / wv_tblk (16 rows x BS), FM Q6_K in the q32 phases q6f_load's Q6F_UNIT
    const int ub = ty == T_Q6K && !X16 ? Q6F_UNIT
                 : 16 * (ty == T_Q4K || ty == T_IQ4NL ? 144 : ty == T_Q5K ? 176 : ty == T_IQ4X ? 136 : ty == T_Q8_0 ? 256
                         : ty == T_Q6K ? Q6_BLK : Q3_BLK);
    const size_t nsb = (size_t) (m.K >> 8), off = ((size_t) tile * nsb + sb) * ub + 128 * lane;
    if (off < ((size_t) tile + 1) * nsb * ub) pf_l2n(m.w + off);
}

// attention q / k / v projections, RT rows (every matrix in the wv layout)
template <int T>
__device__ __noinline__ void attn_proj_ck(const LayerW & W) {
    using C = WvCfg<T, false>;
    constexpr int NSB = D / 256, CSB = wv_csb<T, false, NSB>(), NCH = (NSB + CSB - 1) / CSB, US = 8 * CSB;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    rs_rows<T>(cB.qn);
    const bool q6 = W.q.type == T_Q6K || W.k.type == T_Q6K || W.v.type == T_Q6K;
    const int k6 = cB.k6, ks = (k6 & 4) ? 2 : 1;
    const int nq = QG / 16, nk = KVD / 16, nv = KVD / 16;
    const int sk = W.k.type == T_Q8_0 ? ks : 1, sv = W.v.type == T_Q8_0 ? ks : 1;
    const int nkk = nk * sk, ni = nq + nkk + nv * sv;
    int it;
    if (k6 & 2) {
        it = blockIdx.x + gridDim.x * (NW - 1 - warp);
    } else {
        const int i0 = ni * blockIdx.x / gridDim.x, i1 = ni * (blockIdx.x + 1) / gridDim.x;
        it = i0 + (NW - 1 - warp) < i1 ? i0 + (NW - 1 - warp) : ni;
    }
    const bool have = it < ni;
    const int w = !have ? 0 : it < nq ? 0 : it < nq + nkk ? 1 : 2;
    const int j = have ? it - (w == 0 ? 0 : w == 1 ? nq : nq + nkk) : 0, ns = w == 0 ? 1 : w == 1 ? sk : sv;
    const int lt = j / ns, rh = j - lt * ns;
    const Mat & m = w == 0 ? W.q : w == 1 ? W.k : W.v;
    WvSt<T, false> st;
    wv_st_zero<T, false>(st);
#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
        const int s0 = NSB * c / NCH, s1 = NSB * (c + 1) / NCH;
        if (have) wv_pf_next<false>(m, lt, s0);
        wv_fill_ck<T, false, US>(cB.qn, 8 * s0, 8 * (s1 - s0), false, q6);
        if (have) {
            if (ns > 1) st.a = attn_kv_rh_st<T>(m, lt, rh, s0, s1 - s0, US, st.a);
            else wv_tile_any_st<T, false, true>(m, lt, s0, s1 - s0, US, st);
        }
    }
    if (!have) return;
    const float * s_rs = misc(nullptr);
    float * const dst = w == 0 ? cB.qbuf : w == 1 ? cB.kbuf : cB.vbuf;
    if (ns > 1) { // (the row half's first MMA row: c < 2)
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int t = 8 * n + 2 * tq + c;
                if (t < T) dst[t * KVD + 16 * lt + g + 8 * rh] = st.a.v[n][c] * s_rs[t];
            }
        return;
    }
    wv_store_n<T>(dst, w == 0 ? QG : KVD, lt, wv_st_fin<T, false, true>(st, m.type), s_rs);
}

// delta-net input projections, RT rows of several lanes (dn_proj_wv<T, true>'s stores: each row's qkv to its lane's conv
// ring, s_cvp; alpha / beta as dn_proj_wv)
template <int T>
__device__ __noinline__ void dn_proj_ck(const LayerW & W, int dl, RowTab rt) {
    using C = WvCfg<T, false>;
    constexpr int NSB = D / 256, CSB = wv_csb<T, false, NSB>(), NCH = (NSB + CSB - 1) / CSB, US = 8 * CSB;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    float ** const s_cvp = reinterpret_cast<float **>(misc(nullptr) + 64);
    if (threadIdx.x < T) {
        const int t = threadIdx.x;
        float * cv = nullptr;
        if (rt.nl == 0) { // (the deep pass: row t at the sequence's n + t)
            const int p = ((volatile Ctl *) cB.ctl)->n + t;
            cv = cB.conv + (size_t) dl * CONV_RING * QKV + (size_t) (p % CONV_RING) * QKV;
        } else {
            int r0;
            const int ln = rt_lane(rt, t, r0);
            if (ln >= 0) {
                const Bufs & V = cBs[view_main(ln)];
                const int p = ((volatile Ctl *) V.ctl)->n + t - r0;
                cv = V.conv + (size_t) dl * CONV_RING * QKV + (size_t) (p % CONV_RING) * QKV;
            }
        }
        s_cvp[t] = cv;
    }
    rs_rows<T>(cB.qn);
    const bool q6 = W.qkv.type == T_Q6K || W.gate.type == T_Q6K;
    constexpr int NTL = QKV / 16 + VD / 16, nt_q = QKV / 16;
    const int t0 = NTL * blockIdx.x / gridDim.x, t1 = NTL * (blockIdx.x + 1) / gridDim.x;
    const int tile = t0 + (NW - 1 - warp);
    const bool have = tile < t1;
    const bool isq = tile < nt_q;
    const int lt = isq ? tile : tile - nt_q;
    const Mat & m = isq ? W.qkv : W.gate;
    // alpha / beta items on the warps without a tile (dn_proj_wv's K1 dealing; the same rows / K-chunks: bit-exact)
    const int n_ab = AB_SPLIT * 2 * NVH;
    const int nb = t1 - t0, nlw = NW - nb % NW;
    int ab0 = gwarp(), ab1 = n_ab, astep = nwarps();
    if (cB.k1) {
        ab1 = n_ab * (blockIdx.x + 1) / gridDim.x;
        ab0 = warp < nlw ? n_ab * blockIdx.x / gridDim.x + warp : ab1;
        astep = nlw;
    }
    WvSt<T, false> st;
    wv_st_zero<T, false>(st);
    // the alpha / beta items' sub-chunk: MK_DEEP2 bit 4 the last (no fill barrier waits for them; the 16-row items take
    // twice the 8-row time), else the first
    const int abc = (cB.deep2 & 4) ? NCH - 1 : 0;
#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
        const int s0 = NSB * c / NCH, s1 = NSB * (c + 1) / NCH;
        if (have) wv_pf_next<false>(m, lt, s0);
        wv_fill_ck<T, false, US>(cB.qn, 8 * s0, 8 * (s1 - s0), false, q6);
        if (c == abc) {
            const float * s_rs = misc(nullptr);
            for (int it = ab0; it < ab1; it += astep) {
                const int row = it % (2 * NVH), ch = it / (2 * NVH);
                float acc[T];
#pragma unroll
                for (int t = 0; t < T; t++) acc[t] = 0.f;
                if (W.abq) {
                    const int8_t * qr = W.abq + (size_t) row * D;
                    const __half * dr = W.abd + (size_t) row * (D / 32);
                    for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                        const int qw = ldw1(qr + k);
                        const float d = __half2float(dr[k >> 5]);
                        const float4 fn = __ldg(reinterpret_cast<const float4 *>(W.attn_norm + k));
                        float4 wf;
                        wf.x = ((float) (int8_t) (qw) * d) * fn.x;
                        wf.y = ((float) (int8_t) (qw >> 8) * d) * fn.y;
                        wf.z = ((float) (int8_t) (qw >> 16) * d) * fn.z;
                        wf.w = ((float) (int8_t) (qw >> 24) * d) * fn.w;
#pragma unroll
                        for (int t = 0; t < T; t++) {
                            const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                            acc[t] += wf.x * xv.x + wf.y * xv.y + wf.z * xv.z + wf.w * xv.w;
                        }
                    }
                } else {
                    const float * wr = W.ab + (size_t) row * D;
                    for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                        const float4 wf = ldw4f(wr + k);
#pragma unroll
                        for (int t = 0; t < T; t++) {
                            const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                            acc[t] += wf.x * xv.x + wf.y * xv.y + wf.z * xv.z + wf.w * xv.w;
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
                store_lane_t<T>(cB.ab + ch * RMAX * 2 * NVH, 2 * NVH, row, acc);
            }
        }
        if (have) wv_tile_any_st<T, false, true>(m, lt, s0, s1 - s0, US, st);
    }
    if (!have) return;
    const float * s_rs = misc(nullptr);
    const WvAcc<C::NG> a = wv_st_fin<T, false, true>(st, m.type);
    if (isq) {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 8 * n + 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
                if (t < T && s_cvp[t]) s_cvp[t][r] = a.v[n][c] * s_rs[t];
            }
    } else {
        wv_store_n<T>(cB.z, VD, lt, a, s_rs);
    }
}

// ffn_gu_ck's sub-chunk fills: tile j's finished gate / up C fragments (out of line per variant: only the loop's values are
// live across its tile calls). Q6 (block-uniform: a FM Q6_K gate or up) carries the k16 form's chunk-1 sums; without it the
// calls carry 8 sums each
template <int NG> struct WvGU { WvAcc<NG> g, u; };
template <int T, bool Q6>
__device__ __noinline__ WvGU<WvCfg<T, false>::NG> ffn_gu_ck_sums(const LayerW & W, int j, bool have) {
    constexpr int NSB = D / 256, CSB = wv_csb<T, false, NSB>(), NCH = (NSB + CSB - 1) / CSB, US = 8 * CSB;
    const bool gw = W.fg.fm, uw = W.fu.fm;
    WvSt<T, false> sg, su;
    wv_st_zero<T, false>(sg);
    wv_st_zero<T, false>(su);
#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
        const int s0 = NSB * c / NCH, s1 = NSB * (c + 1) / NCH;
        if (have) {
            if (gw) wv_pf_next<false>(W.fg, j, s0);
            if (uw) wv_pf_next<false>(W.fu, j, s0);
        }
        wv_fill_ck<T, false, US>(cB.qn, 8 * s0, 8 * (s1 - s0), false, Q6);
        if (have) {
            if (gw) wv_tile_any_st<T, false, Q6>(W.fg, j, s0, s1 - s0, US, sg);
            if (uw) wv_tile_any_st<T, false, Q6>(W.fu, j, s0, s1 - s0, US, su);
        }
    }
    WvGU<WvCfg<T, false>::NG> r;
    r.g = wv_st_fin<T, false, Q6>(sg, W.fg.type);
    r.u = wv_st_fin<T, false, Q6>(su, W.fu.type);
    return r;
}

// FFN gate / up, RT rows (C1 dealing as ffn_gu_wv; a hybrid gate or up matrix: rows16 over token groups from global memory)
template <int T>
__device__ __noinline__ void ffn_gu_ck(const LayerW & W) {
    using C = WvCfg<T, false>;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tq = lane & 3;
    const int G1 = gridDim.x / WV_S, kc1 = blockIdx.x / G1;
    if (kc1 >= WV_S) return;
    constexpr int NJC = FF / 16 / WV_S;
    const int bi = blockIdx.x - kc1 * G1;
    const int t0 = NJC * kc1 + NJC * bi / G1, t1 = NJC * kc1 + NJC * (bi + 1) / G1;
    rs_rows<T>(cB.qn);
    const bool q6 = W.fg.type == T_Q6K || W.fu.type == T_Q6K;
    const int j = t0 + (NW - 1 - warp);
    const bool have = j < t1;
    const bool gw = W.fg.fm, uw = W.fu.fm;
    const WvGU<C::NG> gu = q6 ? ffn_gu_ck_sums<T, true>(W, j, have) : ffn_gu_ck_sums<T, false>(W, j, have);
    const WvAcc<C::NG> & ga = gu.g, & ua = gu.u;
    if (have) {
        const float * s_rs = misc(nullptr);
        float act[T];
        if (gw && uw) { // GLU on the C fragments, one transpose (ffn_gu_wv's)
            WvAcc<C::NG> a;
#pragma unroll
            for (int n = 0; n < C::NG; n++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * n + 2 * tq + (c & 1);
                    const float s = t < T ? s_rs[t] : 0.f;
                    a.v[n][c] = silu(ga.v[n][c] * s) * (ua.v[n][c] * s);
                }
            const LaneT<T> l = wv_to_lane_n<T>(a);
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? l.v[t] : 0.f;
        } else {
            LaneT<T> gl, ul;
            if (gw) {
                gl = wv_to_lane_n<T>(ga);
#pragma unroll
                for (int t = 0; t < T; t++) gl.v[t] *= s_rs[t];
            } else {
                rows16_dp_all<T>(W.fg, j, gl);
            }
            if (uw) {
                ul = wv_to_lane_n<T>(ua);
#pragma unroll
                for (int t = 0; t < T; t++) ul.v[t] *= s_rs[t];
            } else {
                rows16_dp_all<T>(W.fu, j, ul);
            }
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? silu(gl.v[t]) * ul.v[t] : 0.f;
        }
        glu_store<T>(j, act);
    }
    c1_arrive(kc1);
}

// out / ssm_out / ffn_down with the residual add, RT rows (resid_wv's split-K WV_S chunks; each chunk's K range in
// sub-chunk fills, the tile's sums carried across them: its partial is resid_wv's)
template <int T, bool X16, int K>
__device__ __noinline__ void resid_wv_ck(const Mat Wm, const QBuf in, const float * wnext, int role) {
    constexpr int NSB = K / 256, CH = NSB / WV_S, CSB = wv_csb<T, X16, CH>(), NCH = (CH + CSB - 1) / CSB, US = 8 * CSB;
    static_assert(NSB % WV_S == 0, "split-K chunks of whole super-blocks");
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    const int G = gridDim.x / WV_S, b = blockIdx.x;
    if (b >= WV_S * G) return;
    const int kc = b / G, bi = b - kc * G;
    unsigned * cnt = cB.wvcnt + (role & 1) * (D / 32);
    constexpr int NTILE = D / 16;
    const int t0 = NTILE * bi / G, t1 = NTILE * (bi + 1) / G;
    const int tile = t0 + (NW - 1 - warp);
    const bool have = tile < t1;
    if (have) wv_pf_next<X16>(Wm, tile, CH * kc); // (the weights do not wait for ffn_gu's chunk: fetched during c1_wait)
    if (role & 2) c1_wait(kc, G);
    const bool min2 = X16 && wv_min2(Wm.type), q6 = !X16 && Wm.type == T_Q6K;
    WvSt<T, X16> st;
    wv_st_zero<T, X16>(st);
#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
        const int s0 = CH * c / NCH, s1 = CH * (c + 1) / NCH;
        if (have && c > 0) wv_pf_next<X16>(Wm, tile, CH * kc + s0);
        wv_fill_ck<T, X16, US>(in, 8 * (CH * kc + s0), 8 * (s1 - s0), min2, q6);
        if (have) wv_tile_any_st<T, X16, true>(Wm, tile, CH * kc + s0, s1 - s0, US, st);
    }
    if (!have) return;
    const WvAcc<C::NG> a = wv_st_fin<T, X16, true>(st, Wm.type);
    float * x = cB.x;
    float * part = cB.wvpart + (size_t) kc * RMAX * D;
    if (!X16 || tq < 2) {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = X16 ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                if (t < T) __stcg(part + (size_t) t * D + 16 * tile + g + 8 * (c >> 1), a.v[n][c]);
            }
    }
    __threadfence();
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) old = atomicAdd(cnt + (tile >> 1), 1u);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old != 2 * WV_S - 1) return;
    __threadfence();
    const int jj = tile >> 1, row = 32 * jj + lane;
    // (resid_wv's K2 epilogue: per-element order (((p0 + p1) + p2) + p3) + x, batches of TB tokens)
    constexpr int TB = MK_K2_TB < 8 ? MK_K2_TB : 8;
    const float wn = wnext[row];
#pragma unroll 1
    for (int tb = 0; tb < T; tb += TB) {
        float p[TB][WV_S], xo[TB];
#pragma unroll
        for (int u = 0; u < TB; u++)
            if (tb + u < T) {
#pragma unroll
                for (int k = 0; k < WV_S; k++) p[u][k] = __ldcg(cB.wvpart + ((size_t) k * RMAX + tb + u) * D + row);
                xo[u] = __ldcg(x + (tb + u) * D + row);
            }
#pragma unroll
        for (int u = 0; u < TB; u++)
            if (tb + u < T) {
                float s = p[u][0];
#pragma unroll
                for (int k = 1; k < WV_S; k++) s += p[u][k];
                const float xv = s + xo[u];
                x[(tb + u) * D + row] = xv;
                q32_store(cB.qn, tb + u, jj, xv * wn, xv * xv, true);
            }
    }
    if (lane == 0) cnt[jj] = 0;
}
#endif // MK_PAR_WIDE
