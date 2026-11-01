// wv host side: load-time re-layout of the main-pass matrices (see wv.cuh). Included by qwen38_mk.cu inside
// its anonymous namespace, after parallel_rows.
#pragma once

// ---- host: load-time re-layout (exact byte permutation of the device layout) --------------------------------------
// w: N rows x rb bytes in the device layout of `type` (after the loader's repacks; may be the GGUF mapping itself);
// meta: its meta stream, meta_n bytes (N rows of meta_n / N bytes: Q8_0's scales, Q6_K / Q3_K / IQ3_S's meta) or none
// (meta_n 0). Writes the wv layout to out (N rb bytes; Q6_K: N rb + meta_n bytes with the meta inline) and a
// re-laid meta stream to out_meta. Returns the bytes left in the meta stream (0: none, or folded into out). N % 16 == 0.
// out / out_meta must not overlap w / meta. Every output byte is written.
size_t wv_relayout(int type, int N, int K, int rb, const uint8_t * w, const uint8_t * meta, size_t meta_n, uint8_t * out,
                   uint8_t * out_meta) {
    const int nb = K / 256;
    if (type == T_Q6K) { // FM Q6_K (wv.cuh "FM Q6_K"): units of Q6F_UNIT bytes, the meta stream inline
        const int mrow = nb * Q6_META;
        parallel_rows(N / 16, [&](int64_t tile) {
            for (int rr = 0; rr < 16; rr++) {
                const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
                for (int sb = 0; sb < nb; sb++) {
                    uint8_t * u = out + ((size_t) tile * nb + sb) * Q6F_UNIT;
                    const uint8_t * blk = w + (size_t) r * rb + (size_t) sb * Q6_BLK;
                    const uint8_t * mp = meta + (size_t) r * mrow + (sb >> 1) * 2 * Q6_META;
                    memcpy(u + 32 * g + 16 * half, mp + 16 * (sb & 1), 16);
                    memcpy(u + Q6F_D + 4 * g + 2 * half, mp + 32 + 2 * (sb & 1), 2);
                    for (int tq = 0; tq < 4; tq++) {
                        const int l = 4 * g + tq;
                        for (int p = 0; p < 4; p++) {
                            memcpy(u + Q6F_LO + 512 * p + 16 * l + 8 * half, blk + 32 * p + 4 * tq, 4);
                            memcpy(u + Q6F_LO + 512 * p + 16 * l + 8 * half + 4, blk + 32 * p + 16 + 4 * tq, 4);
                            uint8_t * hd = u + Q6F_HI + 256 * p + 8 * l + 4 * half;
                            for (int mm = 0; mm < 4; mm++) { // high 2 bits of element 4 tq + mm of (sub-block 2p + part, chunk c)
                                auto hb = [&](int part, int c) { return (blk[128 + 8 * (2 * p + c) + 4 * part + mm] >> (2 * tq)) & 3; };
                                hd[mm] = (uint8_t) ((hb(0, 0) << 4) | (hb(1, 0) << 6) | hb(0, 1) | (hb(1, 1) << 2));
                            }
                        }
                    }
                }
            }
        });
        return 0;
    }
    if (wv_fm(type)) {
        const int BS = type == T_Q4K ? 144 : type == T_Q5K ? 176 : 136;
        parallel_rows(N / 16, [&](int64_t tile) {
            for (int rr = 0; rr < 16; rr++) {
                const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
                const uint8_t * row = w + (size_t) r * rb;
                for (int sb = 0; sb < nb; sb++) {
                    uint8_t * u = out + ((size_t) tile * nb + sb) * 16 * BS;
                    if (type == T_IQ4X) { // source: the loader's 272-B block pairs (hdr0 | hdr1 | qs0 | qs1)
                        const uint8_t * pr = row + (sb >> 1) * 272;
                        const int odd = sb & 1;
                        memcpy(u + 16 * g + 8 * half, pr + 8 * odd, 8);
                        for (int tq = 0; tq < 4; tq++)
                            for (int p = 0; p < 4; p++)
                                memcpy(u + 128 + 512 * p + 16 * (4 * g + tq) + 8 * half, pr + 16 + 128 * odd + 32 * p + 8 * tq, 8);
                    } else {
                        const uint8_t * blk = row + (size_t) sb * BS;
                        memcpy(u + 32 * g + 16 * half, blk, 16);
                        const int qs = type == T_Q4K ? 16 : 48, qo = type == T_Q4K ? 256 : 768;
                        for (int tq = 0; tq < 4; tq++) {
                            const int l = 4 * g + tq;
                            if (type == T_Q5K) memcpy(u + 256 + 16 * l + 8 * half, blk + 16 + 8 * tq, 8);
                            for (int p = 0; p < 4; p++) memcpy(u + qo + 512 * p + 16 * l + 8 * half, blk + qs + 32 * p + 8 * tq, 8);
                        }
                    }
                }
            }
        });
        return 0;
    }
    // tile-major whole blocks of BS bytes (per super-block), the meta stream likewise (MB-byte entries)
    auto tm = [&](const uint8_t * src, uint8_t * dst, int rowb, int BS) {
        const int nbk = rowb / BS;
        parallel_rows(N / 16, [&](int64_t tile) {
            for (int rr = 0; rr < 16; rr++)
                for (int i = 0; i < nbk; i++)
                    memcpy(dst + (((size_t) tile * nbk + i) * 16 + rr) * BS, src + ((size_t) tile * 16 + rr) * rowb + (size_t) i * BS, BS);
        });
    };
    // Q4_K / IQ4_NL 144, Q5_K 176, Q8_0 256 (+ 16-B scale entries); ffn_down: Q6_K 192 / Q3_K 96 (+ 36-B meta
    // pairs), IQ3_S 96 (+ 16-B meta per block)
    const int BS = type == T_Q8_0 ? 256 : type == T_Q5K ? 176 : type == T_Q6K ? Q6_BLK : type == T_Q3K || type == T_IQ3S ? Q3_BLK : 144;
    const int MB = type == T_Q6K || type == T_Q3K ? 2 * Q6_META : 16;
    tm(w, out, rb, BS);
    if (meta_n) tm(meta, out_meta, (int) (meta_n / N), MB);
    return meta_n;
}

// ---- device re-layout (MK_LOAD_GPURL, default 1): the same permutation on the GPU, from the bytes as uploaded ---------
// The loader uploads a matrix in its pre-wv layout into a staging buffer and one of these kernels writes the wv layout into
// the matrix's own buffer (exact: the same bytes as wv_relayout; checked against the host path by MK_WDIGEST).
// - fragment-major types (wv_fm: IQ4_XS, Q4_K / Q5_K with MK_WV_FMK) move whole 8-byte words: their gather table is made by
//   running wv_relayout itself on a probe of 16 rows whose words hold their own (row, word) positions (wv_probe), so this
//   path follows any change of the host code;
// - tile-major types move whole blocks of 4-byte words (wv_tm_dev: the host's tm(), word for word);
// - FM Q6_K packs 2-bit fields: wv_q6f_dev is the host loop, line for line.
struct WvProbe {
    int G = 0;                 // super-blocks per gather group (IQ4_XS: 2, the loader's 272-B block pairs)
    int seg = 0;               // source bytes of one row per group (G * block bytes)
    std::vector<uint32_t> tab; // per output word of a group: row << 24 | source word in the row's group segment
    const uint32_t * dtab = nullptr;
};
inline bool wv_probe(int type, WvProbe & p) {
    if (!wv_fm(type)) return false;
    const int BS = type == T_Q4K ? 144 : type == T_Q5K ? 176 : 136;
    p.G = type == T_IQ4X ? 2 : 1;
    p.seg = p.G * BS;
    if (p.seg % 8) return false;
    const int ws = p.seg / 8, W = 16 * ws;
    std::vector<uint64_t> in((size_t) W), out((size_t) W, ~0ull);
    for (int r = 0; r < 16; r++)
        for (int j = 0; j < ws; j++) in[(size_t) r * ws + j] = ((uint64_t) r << 32) | (uint64_t) j;
    wv_relayout(type, 16, 256 * p.G, p.seg, reinterpret_cast<const uint8_t *>(in.data()), nullptr, 0,
                reinterpret_cast<uint8_t *>(out.data()), nullptr);
    p.tab.resize((size_t) W);
    std::vector<uint8_t> hit((size_t) W, 0);
    for (int j = 0; j < W; j++) {
        const uint64_t v = out[(size_t) j];
        const uint64_t r = v >> 32, s = v & 0xffffffffu;
        if (r >= 16 || s >= (uint64_t) ws || hit[(size_t) (r * ws + s)]) return false; // not a word permutation
        hit[(size_t) (r * ws + s)] = 1;
        p.tab[(size_t) j] = (uint32_t) (r << 24 | s);
    }
    return true;
}
__global__ void wv_fm_dev(const uint64_t * __restrict__ src, uint64_t * __restrict__ dst, const uint32_t * __restrict__ tab, int W,
                          int ngrp, int64_t rbw, int segw, int64_t total) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t gi = i / W;
        const int j = (int) (i - gi * W);
        const int64_t tile = gi / ngrp;
        const int grp = (int) (gi - tile * ngrp);
        const uint32_t t = tab[j];
        dst[i] = src[(tile * 16 + (t >> 24)) * rbw + (int64_t) grp * segw + (t & 0xffffff)];
    }
}
// tm(): dst block ((tile nbk + i) 16 + rr) = src row tile 16 + rr, block i (bsw words per block, rowbw words per row)
__global__ void wv_tm_dev(const uint32_t * __restrict__ src, uint32_t * __restrict__ dst, int64_t rowbw, int bsw, int nbk, int64_t total) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        int64_t q = i / bsw;
        const int j = (int) (i - q * bsw);
        const int rr = (int) (q % 16);
        q /= 16;
        const int b = (int) (q % nbk);
        const int64_t tile = q / nbk;
        dst[i] = src[(tile * 16 + rr) * rowbw + (int64_t) b * bsw + j];
    }
}
// one thread per (tile, super-block, row of the tile): that row's bytes of the Q6F_UNIT unit (wv_relayout's Q6_K loop)
__global__ void wv_q6f_dev(const uint8_t * __restrict__ w, const uint8_t * __restrict__ meta, uint8_t * __restrict__ out, int N, int nb, int rb) {
    const int mrow = nb * Q6_META;
    const int64_t items = (int64_t) (N / 16) * nb * 16;
    for (int64_t it = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; it < items; it += (int64_t) gridDim.x * blockDim.x) {
        const int rr = (int) (it % 16);
        const int64_t q = it / 16;
        const int sb = (int) (q % nb);
        const int64_t tile = q / nb;
        const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
        uint8_t * u = out + ((size_t) tile * nb + sb) * Q6F_UNIT;
        const uint8_t * blk = w + (size_t) r * rb + (size_t) sb * Q6_BLK;
        const uint8_t * mp = meta + (size_t) r * mrow + (sb >> 1) * 2 * Q6_META;
        for (int k = 0; k < 16; k++) u[32 * g + 16 * half + k] = mp[16 * (sb & 1) + k];
        for (int k = 0; k < 2; k++) u[Q6F_D + 4 * g + 2 * half + k] = mp[32 + 2 * (sb & 1) + k];
        for (int tq = 0; tq < 4; tq++) {
            const int l = 4 * g + tq;
            for (int p = 0; p < 4; p++) {
                for (int k = 0; k < 4; k++) u[Q6F_LO + 512 * p + 16 * l + 8 * half + k] = blk[32 * p + 4 * tq + k];
                for (int k = 0; k < 4; k++) u[Q6F_LO + 512 * p + 16 * l + 8 * half + 4 + k] = blk[32 * p + 16 + 4 * tq + k];
                uint8_t * hd = u + Q6F_HI + 256 * p + 8 * l + 4 * half;
                for (int mm = 0; mm < 4; mm++) {
                    const uint8_t * hq = blk + 128 + 8 * (2 * p) + mm;
                    const int h00 = (hq[0] >> (2 * tq)) & 3, h10 = (hq[4] >> (2 * tq)) & 3; // (part, c) = (0, 0), (1, 0)
                    const int h01 = (hq[8] >> (2 * tq)) & 3, h11 = (hq[12] >> (2 * tq)) & 3; // (0, 1), (1, 1)
                    hd[mm] = (uint8_t) ((h00 << 4) | (h10 << 6) | h01 | (h11 << 2));
                }
            }
        }
    }
}
// Which wv layouts have a device path, for a matrix of this shape (rb row bytes, meta_n meta bytes): host side, before the
// staging copy is queued. FM types: also makes the type's gather table (once; probes[type].dtab must then be uploaded).
inline bool wv_dev_ok(int type, int N, int K, int rb, size_t meta_n, std::map<int, WvProbe> & probes) {
    const int nb = K / 256;
    if (N % 16) return false;
    if (type == T_Q6K) return meta_n == (size_t) N * nb * Q6_META && nb % 2 == 0;
    if (wv_fm(type)) {
        auto it = probes.find(type);
        if (it == probes.end()) {
            WvProbe p;
            if (!wv_probe(type, p)) return false;
            it = probes.emplace(type, std::move(p)).first;
        }
        return nb % it->second.G == 0 && rb % 8 == 0 && rb == nb / it->second.G * it->second.seg;
    }
    const int BS = type == T_Q8_0 ? 256 : type == T_Q5K ? 176 : type == T_Q6K ? Q6_BLK : type == T_Q3K || type == T_IQ3S ? Q3_BLK : 144;
    const int MB = type == T_Q6K || type == T_Q3K ? 2 * Q6_META : 16;
    const size_t mrow = meta_n ? meta_n / N : 0;
    return rb % BS == 0 && BS % 4 == 0 && rb % 4 == 0 && (!meta_n || (meta_n % N == 0 && mrow % MB == 0 && MB % 4 == 0 && mrow % 4 == 0));
}
// Enqueues on st the device re-layout of the staged matrix sw (values, 8-byte aligned) / sm (meta) into out / out_meta, as
// wv_relayout writes it (after wv_dev_ok said yes). Returns the launch error.
inline cudaError_t wv_dev_launch(int type, int N, int K, int rb, const uint8_t * sw, const uint8_t * sm, size_t meta_n, uint8_t * out,
                                 uint8_t * out_meta, cudaStream_t st, const WvProbe * pp) {
    const int nb = K / 256, TPB = 256;
    auto grid = [](int64_t n) { return (unsigned) std::max<int64_t>(1, std::min<int64_t>(4096, (n + 255) / 256)); };
    if (type == T_Q6K) {
        wv_q6f_dev<<<grid((int64_t) (N / 16) * nb * 16), TPB, 0, st>>>(sw, sm, out, N, nb, rb);
        return cudaGetLastError();
    }
    if (wv_fm(type)) {
        const WvProbe & p = *pp;
        const int W = (int) p.tab.size(), ngrp = nb / p.G;
        const int64_t total = (int64_t) (N / 16) * ngrp * W;
        wv_fm_dev<<<grid(total), TPB, 0, st>>>(reinterpret_cast<const uint64_t *>(sw), reinterpret_cast<uint64_t *>(out), p.dtab, W, ngrp,
                                               rb / 8, p.seg / 8, total);
        return cudaGetLastError();
    }
    const int BS = type == T_Q8_0 ? 256 : type == T_Q5K ? 176 : type == T_Q6K ? Q6_BLK : type == T_Q3K || type == T_IQ3S ? Q3_BLK : 144;
    const int MB = type == T_Q6K || type == T_Q3K ? 2 * Q6_META : 16;
    const int64_t tw = (int64_t) N * rb / 4;
    wv_tm_dev<<<grid(tw), TPB, 0, st>>>(reinterpret_cast<const uint32_t *>(sw), reinterpret_cast<uint32_t *>(out), rb / 4, BS / 4, rb / BS, tw);
    if (meta_n) {
        const size_t mrow = meta_n / N;
        const int64_t tm = (int64_t) meta_n / 4;
        wv_tm_dev<<<grid(tm), TPB, 0, st>>>(reinterpret_cast<const uint32_t *>(sm), reinterpret_cast<uint32_t *>(out_meta), (int64_t) mrow / 4, MB / 4,
                                            (int) (mrow / MB), tm);
    }
    return cudaGetLastError();
}
