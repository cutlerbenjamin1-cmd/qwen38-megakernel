# Engine notes

How the engine works inside. This document began as open-jet's megakernel README (this repository started as that
directory, history included) and has been updated as the engine changed. Measurements are in
[BENCHMARKS.md](BENCHMARKS.md), the server in [SERVER.md](SERVER.md) and every setting in
[CONFIGURATION.md](CONFIGURATION.md); numbers quoted here carry their date.

A specialised CUDA inference engine for Qwen3.8-27B `Q4_K_M` and unsloth's `UD-Q4_K_XL` (llama.cpp arch `qwen35`:
48 Gated-DeltaNet + 16 gated-attention layers, one MTP block). It is tuned and tested on an RTX 3090 (sm_86, 82 SMs);
`make` also builds for other 24 GB GPUs (sm_89, sm_120), which are untested.

## Model files

Which model file runs on it (details in [The GGUF](#the-gguf)):

| Model file | Runs on the engine |
|---|---|
| unsloth's Qwen3.8-27B `UD-Q4_K_XL` | yes (tuned on it) |
| unsloth's Qwen3.8-27B `Q4_K_M` | yes |
| another Qwen3.8-27B `Q4_K_M` | when every tensor has a type listed under [The GGUF](#the-gguf); untested |
| fine-tunes of Qwen3.8-27B (other quantizers) | when every tensor has a type listed under [The GGUF](#the-gguf); four checked on 2026-10-18 |
| a file without the MTP layer | yes, without drafting (depth 0), or with `--mtp` and a separate MTP file of the same model |
| any other quant | refused by the loader, which names the first tensor it cannot read (use llama.cpp) |

## How it works

Decoding runs an entire speculative cycle in one persistent cooperative kernel launch:

```
main model over T = 1 + n_draft tokens (up to 8)  ->  lm_head + verification on device (greedy, or exact
speculative sampling)  ->  MTP catch-up rows for the accepted tokens  ->  up to n_draft MTP draft steps
```

The ~340 phases of a cycle (projections, DeltaNet core, attention, FFN, lm_head, MTP) are separated by a software grid
barrier; the host only reads back the accepted tokens. Prompts go through batched tensor-core kernels instead (see
[Prefill](#prefill)).

- **Wide verify** (`MK_WV=1 MK_MAXT=8`, the served build): the main pass's matrix products run as int8 tensor-core
  GEMMs (`mma.m16n8k32`, tokens on N; `wv.cuh`) on a weight layout made at load (fragment-ordered tiles, values
  unchanged), so a pass over 8 rows costs little more than a pass over 1. Every output is the same sequence of integer
  sums and float FMAs at any T. The lean build (`make MKFLAGS=`, `MK_MAXT=5`) reads the weights with dp4a GEMVs and
  verifies at most 5 rows (4 drafts).
- **Draft depth:** up to `-d` drafts per cycle. An on-device gate stops drafting early when the running product of the
  MTP head's own top-1 probabilities falls below a threshold (0.25-0.4, by draft index; at depth > 5 the deeper drafts
  have their own gates). With sampling, the drafts are sampled too. `MK_POLICY` overrides the defaults (a
  `key=value,...` list, documented at `MKEngine::set_policy`; the keys: [CONFIGURATION.md](CONFIGURATION.md)).
- **Greedy output does not depend on the draft depth:** the attention split-K partition depends only on the context
  length, and the verify GEMMs compute every row the same way at any T. With sampling, the token distribution is the
  one of sampling without drafts.

Speculation in numbers (2026-10-16, commit 7d67c5d, CLI, UD-Q4_K_XL, q8_0 KV; tokens / total time):

| prompts | drafts | tok/s | draft acceptance | tokens / cycle | ms / cycle |
|---|---|---|---|---|---|
| 4 short chat prompts, greedy | 0 | 44.9 | - | 1.00 | 22.3 |
| | 2 | 99.2 | 76.9% | 2.38 | 24.0 |
| | 7 | 128.3 | 50.9% | 3.43 | 26.7 |
| the same, sampled, thinking on (8 runs) | 7 | 124.0 | 51.5% | 3.30 | 26.6 |
| a 38k-token prompt, greedy | 0 | 41.7 | - | 1.00 | 24.0 |
| | 7 | 113.6 | 48.4% | 3.40 | 29.9 |
| copying text verbatim, short / at 38k | 7 | 257.7 / 235.3 | 95.1% / 97.9% | 7.51 | 29.1 / 31.9 |

A cycle that verifies 8 rows took ~29.1 ms against 22.3 ms for 1 row (short context), so the gain comes from accepted
drafts. Without drafts, llama.cpp b11548 decoded the same prompts at 42.0 tok/s (short) and 35.1 tok/s (38k) in that
session.

## Correctness

The output is not byte-identical to llama.cpp's: the arithmetic is ordered and rounded differently, so where two tokens
are close the other one can win, and a generation then continues differently. Measured against llama.cpp's own
next-token distributions (same GGUF; reference: llama.cpp b10246 generating token by token with its q8_0 KV cache;
13 sequences of this repo's code and docs, 13,299 positions scored after a 1024-token prompt each), UD-Q4_K_XL with
the q8_0 KV cache agrees on 98.737% of the top tokens with a mean KL divergence of 0.00119 nats (2026-10-17, the same
on 2026-10-20), closer than llama.cpp's own batched run (98.489% / 0.00166); most disagreements are near-ties. The
tables (the prefill-only and `mixed` paths, 32k to 160k context, Q4_K_M, image prompts) and the commands that
reproduce them are in [BENCHMARKS.md](BENCHMARKS.md#against-llamacpp).

`llama_ref` teacher-forces 2048-token chunks of the text (`--chunk`) and stores, per scored position, the 256 most
likely tokens (`logit_stats.h`); `--kld` compares an engine with such a file (`--ref-out` also writes its own).
`--score-path` selects the megakernel's batched prefill kernels (`prefill`, the default), the decode kernel (`decode`),
or the prefill kernels for the unscored prompt and the decode kernel for the scored positions (`mixed`, what a
generation runs).

Speculative verification is exact: greedy output is the main model's greedy choice at every draft depth, and with
sampling a draft is accepted with probability p(draft) and a rejected one is resampled from the rest of the
distribution, so the sampled distribution is the same as without drafts (chi-square and PIT tests).

The random numbers come from a counter-based generator keyed by the seed and the position: a row's acceptance and
resampling numbers by its position, a sampled draft's by the position it drafts. Each position is decided once, and a
position drafted again after a rejection before it draws the same number, but that number is independent of
everything that decided the tokens before it, so the output stays exact (2026-10-17: a CPU replica of the loop,
`tests/spec_rng_sim.py`, 396k re-drafted positions; the PIT test on 13,401 re-drafted rows of the engine at 7
drafts). One opt-in `MK_POLICY` setting needs a restriction: `chain=2` with sampled drafts runs without drop gates
(dropping a draft on its own probability selects which drafts get verified: biased).

**unsloth UD-Q4_K_XL** (`Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17.6 GB: Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS per tensor plus a
few Q3_K, IQ3_S and IQ4_NL) puts its extra bits where the decode spends its time: its FFN gate / up average 5.1 bits
per weight against Q4_K's 4.5, so a cycle reads more weight bytes than with the plain Q4_K_M. Its MTP block (Q6_K in
the file) is re-quantized to Q4_K at load ([The GGUF](#the-gguf)). In one session on 2026-10-14 the plain Q4_K_M
decoded faster at most context lengths ([BENCHMARKS.md](BENCHMARKS.md#q4_k_m-and-upstreams-scripts)).

## Build / run

Building the served configuration and the lean engine (`make MKFLAGS=`, up to 4 drafts): [BUILDING.md](BUILDING.md).
The CLI:

```sh
./qwen38_mk -m ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "Write quicksort in C" -n 512 \
            -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin --think --temp 1.0 --top-p 0.95 --top-k 20
```

Its flags, the environment variables the engine reads at load (KV presets, prefill chunk, exact reuse, checkpoint
slots, keep-warm, watchdog, draft policy, state files) and the measurement, A/B and debug switches, compile-time ones
included, are listed with their defaults in [CONFIGURATION.md](CONFIGURATION.md). Token ids read from files
(`--prompt-ids`, `--score`, a `--kld` reference's sequences and records) must lie in the vocabulary, and
`--kld-images` rows inside their sequence: anything else is refused with a message before it is used. After its
output the CLI leaves without freeing the engine's memory one buffer at a time (the process exit frees it; ~900
frees could hang on a failing driver).

### The GGUF

The kernels take the weight type as a template argument and dispatch on each matrix's type at run time
(`gemv_pairs_any`, `resid_any`, `ffn_gu`, prefill `gemm`), so every projection of the trunk and the MTP block can be
any of:

| type | how the kernels read it |
|---|---|
| Q4_K, Q5_K | as stored; in the wide-verify build the main-pass matrices are re-laid-out at load into fragment-ordered tiles (values unchanged) |
| Q6_K, Q8_0 | re-laid-out at load (values unchanged) |
| IQ4_XS | re-laid-out at load in pairs of 256-weight blocks, 272 B per pair (both block headers, then each block's index bytes in Q4_K order); 4-bit indices mapped to int8 weights with `__byte_perm` |
| Q3_K, IQ3_S | re-laid-out at load (values + scales streams, 114 / 112 B per 256 weights); the GEMV and the prefill GEMM feed exactly the integers and scales of the Q6_K re-encode they replace (`MK_REENC3=1`): bit-identical results at ~0.54x the bytes |
| IQ4_NL | re-laid-out at load into 144-byte blocks of 256 weights (the eight fp16 scales, then the index bytes); bit-identical to the Q8_0 re-encode it replaces (`MK_IQ4NL_Q8=1`) at 0.53x the bytes |

`token_embd` can be Q4_K, Q6_K or Q8_0: it stays in mapped pinned host memory (not in VRAM) as stored, and each lookup
(the decode embed phase, the MTP block's input, the prefill embed kernels) stages the token's row and dequantizes it the
way ggml does. `output` can be Q6_K or Q8_0 (Q8_0: ~0.3 GB more VRAM; lm_head and the draft head switch on the type
around the row reader). Q8_0 `ssm_alpha` /
`ssm_beta` (UD) stay int8 with their block scales and the kernels apply the norm weight on the fly; other types are
dequantized to F32 at load and folded with the norm weight into fp32 rows (the same values; `MK_AB_F32=1` forces this
path). The exact re-encodes keep every weight's value (both formats store a small integer times the same fp16 `d` and an
integer sub-block scale), and the loader checks each re-encoded or re-laid-out row bit for bit against ggml's
dequantization of the original. A file with any other type is refused, naming the first tensor that does not match.
Which files that covers: [Model files](#model-files).

The engine reads the file with its own parser (`gguf.h`, mmap), independently of libllama's loader (which reads only
the vocabulary, in parallel), so the parser checks every read against the file size: version 2 / 3, the kv and tensor
counts, string and array lengths (no nested arrays, as ggml), duplicate keys or tensor names, 1-4 dimensions of at
most 2^63 elements, the type and whole blocks per row, `general.alignment` (a power of two up to 1 MiB), and each
tensor's data (aligned, inside the file); sizes are computed without overflow. A malformed file is refused with an
error naming the field. A tensor's byte size always comes from its type and dimensions, and no dimension may be 0. The
loader then reads every tensor by the shape it expects (`File::get(name, dims)`, `gguf::shape_is`): each matrix is
exactly [K, N] (further dimensions 1), `token_embd` exactly [5120, 248320], the norms and vectors their exact sizes, so
a file with the right names but a short or reshaped tensor is refused, naming it, instead of being read past its end.
`tests/gguf_fuzz.cpp` (`make tests/gguf_fuzz`) feeds it malformed and randomly corrupted files (size and offset lies
included) under AddressSanitizer / UBSan, reads every byte of every tensor of the files that parse and tries the
shape-checked reads; UD-Q4_K_XL, UD-Q4_K_M and Q4_K_M parse to the same tensors and keys as before the checks
(2026-10-17). The vision projector (mmproj) is read through ggml's GGUF reader, which checks the same sizes; the CPU
encoder then checks every tensor's size against the projector's hyperparameters, and the hyperparameters and the types
of the keys it reads (a crafted file is refused with an error rather than ending the process).

The MTP draft block (`blk.64.*`, `nextn.eh_proj`) is the one lossy exception: its matrices stored wider than Q4_K
(Q5_K, Q6_K, Q8_0: all of them in UD-Q4_K_XL, `eh_proj` / `attn_v` / `ffn_down` in Q4_K_M) are re-quantized to Q4_K at
load (`ggml_quantize_chunk`, ~2.5 s of CPU for UD-Q4_K_XL, on a background thread during the load), so every MTP pass
streams less (a Q4_0 MTP block too, and IQ4_NL in builds with more than 4 drafts). Only the drafts change: they are
verified by the unchanged main model, so greedy output and the sampled output distribution stay exact. When it went in
(2026-10-12, before wide verify) it saved 107 MiB of VRAM on UD-Q4_K_XL, sped greedy decode up 1.5% and moved
acceptance by under a point. On the lean build `MK_MTP_Q4=0` keeps the stored types (Q4_0 repacked to Q8_0 exactly;
`MK_MTP_Q4=ab` loads both and the policy key `mtpq=0|1` switches between them, for A/B tests). Builds with more than 4
drafts refuse both: there every main-layer matrix of those types is in the tensor-core layout, so only the MTP block
would reach dp4a kernels for them, and those builds do not compile them; a matrix of a type their dp4a kernels do not
read is refused at load.

The MTP layer can come from another file (`--mtp FILE`, `MK_MTP=FILE`; C API `mk_new_mtp`), e.g. the separate
`mtp-*.gguf` files published for conversions that dropped it: its `blk.64.*` tensors replace the model file's (the
file's own `token_embd` / `output` copies are not used: the MTP block shares the model's). The file goes through the
same parser and loader; the load stops with an error naming the mismatch unless its architecture, embedding / FFN
sizes, head counts and dimensions, `block_count` (the MTP block at layer 64), vocabulary size and tokenizer match the
model and all 15 `blk.64` tensors are there. A model file without the layer (`block_count` 64), or `--no-mtp` /
`MK_NO_MTP=1`, runs without drafting: depth 0 whatever is asked (a notice, not an error), no MTP weights, draft head
or MTP KV slot (1/17 of the KV cache), and the prefill skips the MTP block's K / V. Greedy output equals the drafted
engine's at depth 0 bit for bit (checked on UD-Q4_K_XL, 2026-10-17).

The draft head gets the same treatment (`MK_DRAFT_Q4`, on by default; `MK_DRAFT_Q4=0` turns it off and saves its 90 MiB
of VRAM): the 32,768 rows of the draft vocabulary are re-quantized from `output`'s Q6_K to a Q4_K copy at load (0.8 s),
which every draft pass reads instead of the shared lm_head rows (0.69x the bytes, contiguous). Tokens added to the draft
vocabulary on the fly still come from lm_head. Greedy output and the sampled distribution are unchanged (the main head
verifies); the policy key `dq4=0|1` switches it at run time. Measured on 2026-10-17: +2.1% greedy decode on the short
bench prompts (4 x 512 tokens, 7 drafts; cycles and accepted drafts identical), no measurable change at 38k and 101.6k.

Gate and up projections of different types (25 of the 65 layers of UD-Q4_K_XL) are read each in its own type over the
shared activations in decode. In prefill one mixed-type fused GEMM computes both (warps 0-3 gate rows, warps 4-7 up
rows) with the same products, sums and `silu(gate) * up` as two GEMMs plus a GLU kernel, so it is bit-identical to them
without the fp32 round trip (`MK_PF_GLU_MIX=0`: the two-GEMM path). unsloth's Q4_K_M has no mixed pairs. A plain
`llama-quantize ... Q4_K_M` stores `ssm_out`, `ssm_alpha`, `ssm_beta` and `nextn.eh_proj` as Q4_K: alpha / beta are
dequantized and the other two run in their stored Q4_K.

To build a file with unsloth's mix from the BF16 GGUF (unsloth's imatrix optional):

```sh
llama-quantize --imatrix imatrix_unsloth.gguf \
    --tensor-type ssm_out=q5_k --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
    --tensor-type nextn.eh_proj=q8_0 \
    Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

Benchmarks and microbenchmarks (`bench/`): [BENCHMARKS.md](BENCHMARKS.md#reproducing) and
[CONFIGURATION.md](CONFIGURATION.md#tools-and-benchmarks).

## Server

`server/mk_server.py` serves the engine through the C API in `capi.cpp` (`libqwen38mk.so` / `qwen38mk.dll`). What
it offers and how it behaves (the three APIs, the conversation cache with exact reuse, checkpoints, pinned prefixes,
parking and side requests, keep-warm, the console, limits and security) is in [SERVER.md](SERVER.md), with the
engine calls behind each feature under
[Engine calls behind the server](SERVER.md#engine-calls-behind-the-server). The engine side, in short:

- **Checkpoints:** DeltaNet state snapshots saved inside the batched prefill into a pool of `MK_CK_SLOTS` [8] slots
  in pinned host memory; a rewind restores the last one at or before a divergence and prefills from there. Pinned
  checkpoints (`MKEngine::ck_pin`) are never evicted from the pool and travel with a parked conversation.
- **Exact reuse** (`MK_EXACT=1`): prefill rows that do not depend on where a chunk starts (attention over absolute
  key segments, checkpoints on the absolute 32-token scan grid) and a draft vocabulary rebuilt as a fresh prefill
  builds it, so a restored prefix plus a prefill equals a fresh prefill of the whole prompt, bit for bit.
- **Host store:** pinned 32 MiB chunks (`MK_STORE_CHUNK_MIB`) up to a budget. `MKEngine::park` copies a sequence
  (KV rows, window rings, recurrent state, token sequence, image spans, draft vocabulary, its newest checkpoints and
  its pinned ones) and `unpark` makes the live engine that sequence again, bit for bit. `side_begin` / `side_end`
  keep the live sequence aside while a short request runs from position 0: the recurrent state is copied out at
  once, KV and window-ring rows only just before a launch would overwrite them (1024 rows at a time), and
  everything is copied back afterwards.
- **Keep-warm:** after each request `capi.cpp` runs a one-block, 32-thread polling kernel for `MK_KEEP_WARM_S`
  seconds, each launch ending itself after 50 ms, so the GPU's clocks are still up when the next request comes.

## KV cache

fp16, or q8_0 (per 32 values, fp16 scale) when fp16 does not fit the context or `--kv q8` asks for it. On top of the
q8_0 cache, `MK_KV` can store each attention layer's K and V (and the MTP block's) at 4 to 8 bits, optionally
Hadamard-rotated, and `MK_KV_WIN=W` keeps the most recent >= W positions of every low-bit side as q8_0 in a ring as
well, so recent tokens are read at 8 bits. The window moves in steps of the attention's 640-position tile partition
(`ATT_BUCKET`): a tile is read from the ring when it starts at or after `ws = (ceil(L / 640) - 1 - ceil(W / 640)) x
640`, so every row of a launch and every draft depth sees the same split of tiles between the ring and the main buffer,
and greedy output stays independent of the draft depth. Each ring holds `(ceil(W / 640) + 1) x 640 + 32` rows per KV
head; the 32 take the draft rows a launch writes past its partition. Why the presets use W = 2048:
[CONFIGURATION.md](CONFIGURATION.md#choosing-the-window-and-rotation). The decode attention expands low-bit tiles to
int8 in shared memory and runs the same pipelined kernel as the q8_0 cache; the prefill attention reads them too. With
exact reuse, each checkpoint also holds the window rings as of its position. The `MK_KV` syntax and the presets C, D,
D', E and F: [CONFIGURATION.md](CONFIGURATION.md#kv-cache-presets); their accuracy, VRAM and speed:
[BENCHMARKS.md](BENCHMARKS.md#kv-cache-presets).

Each layer's cache and the RoPE table hold 2 x `MK_MAXT` rows (16 in the served build, ~0.6 MB in all) past the
context: with more than 5 drafts, the last cycles before "context full" write the MTP block's draft rows up to 4
positions past it (since 2026-10-17; before, those rows overwrote the first rows of the next head or of the next buffer,
layer 3's cache, and a reused prefix then decoded differently; the server, which stops 16 tokens before the end, never
got there, a CLI or C API run decoding to "context full" did).

In the per-layer scans made for the low-bit formats, K was ~2.3x as sensitive as V, and no setting that saved 0.5 GB
or more stayed within the KL-divergence noise of the q8_0 cache, so the presets are opt-in.

## Images (vision)

What the server accepts, how a prompt with images is built (as llama-server builds it), the limits, reuse and encode
times: [SERVER.md](SERVER.md#images). Inside, the vision tower runs on the CPU (no GPU memory: the 120k-context VRAM
is unchanged) and the engine takes its output rows in the batched prefill. `make vision` builds
`libqwen38vision.so` (+ `tools/mkv_cli`, `tools/vision_ref`); `make` builds it when llama.cpp has libmtmd.

- **Engine** (`MKImage`, `MKEngine::prefill(tokens, cks, ck_exact, images)`, `mk_prefill_img`): an image's positions
  take its fp32 rows instead of token embeddings (`pk_embed_img`; the MTP block takes them too, as llama.cpp does) and
  llama.cpp's Qwen-VL M-RoPE positions (IMROPE: t / row / column per frequency, the text after an image continuing at
  `+ max(nx, ny)`): rows of the RoPE table are rewritten for the image spans (gathered from the plain table), so no
  attention kernel changes and a text-only prompt runs exactly what it ran before (KLD logs byte-identical).
  Checkpoints, rewinds and exact reuse work across images; a rewind into an image re-feeds it.
- **Encoder** (`vision/mk_vision.h`, `--vision-backend auto|cpu|mtmd`): `cpu` = `vision/vit_cpu.cpp`, the tower in fp32
  with AVX2 / FMA kernels (packed 6x16 FMA GEMM micro-kernel, attention on the same kernel with an online softmax,
  exact tanh-GELU, weights widened from the file's type, a pool of pinned workers that take tasks dynamically;
  x86-64 with AVX2 + FMA: Zen 2-5, Haswell+; results independent of the thread count and the same with g++ and MSVC:
  no C library function whose last bit varies between libraries, the online softmax's e^x being glibc's `expf` algorithm
  in the encoder's own code, checked equal to glibc's on all 2^32 inputs); `mtmd` = llama.cpp's libmtmd
  encoder (any CPU); `auto` = `cpu` where it runs.

Accuracy against an fp64 forward of the checkpoint (`vision/vit_ref.py`, 5 images of 196 .. 1024 tokens; 2026-10-16):
relative L2 error of the rows, lowest row cosine:

| encoder | rel. error | min row cosine |
|---|---|---|
| vit_cpu, BF16 mmproj | 4e-6 .. 4e-5 | 1.000000 |
| libmtmd, BF16 mmproj (BF16 GEMM inputs, fp16 K / V, table GELU) | 4e-3 .. 2.7e-2 | 0.903 |
| libmtmd, Q8_0 mmproj | 1.3e-2 .. 8.4e-2 | 0.889 |

At the output (megakernel, q8 KV, teacher-forced on llama.cpp's distributions given the fp64 rows; `tools/vision_ref`,
`qwen38_mk --kld --kld-images`): mean KL with the fp64 rows 0.0006 .. 0.0016 (the text-only level, 0.0012), with
vit_cpu's rows the same, with libmtmd's BF16 rows up to 2.5x that, with Q8_0 rows up to 7x. The BF16 mmproj is the
one to use: vit_cpu widens either file to fp32, so a Q8_0 file is no faster, only less accurate.

## Video

How clips are decoded into frames (ffmpeg, as llama.cpp's mtmd-helper does), how the prompt is laid out (as
llama.cpp b10246 builds it), reuse, limits, cost and live feeds: [SERVER.md](SERVER.md#video). To the engine each
frame or frame pair is an image (the same M-RoPE positions, so the engine needs nothing new):

- **Encoder:** a lone frame is a still image (the patch embedding's two temporal kernels summed); a pair puts the first
  frame through the first kernel and the second through the other (`vit_cpu` `encode_pair`, `mkv_frames_load`).
  Against an fp64 forward (`vision/vit_ref.py`, 3 pairs of 220 .. 920 tokens): vit_cpu 5e-6 .. 1.1e-5, min row cosine
  1.000000; libmtmd 5.5e-3 .. 2.2e-2, 0.981: the same as for images.

## How it gets its speed

- **Weights stay in their GGUF k-quant encoding** (Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS, plus Q3_K / IQ3_S / IQ4_NL),
  read once per cycle regardless of T. Q6_K, Q8_0, IQ4_XS, IQ4_NL, Q3_K and IQ3_S are re-laid-out at load time (values
  unchanged) so every lane does aligned 16/8-byte loads; IQ4_XS indices become int8 weights through two `__byte_perm`
  table lookups per 4 bytes (llama.cpp's `get_int_from_table_16`).
- **Wide verify** (served build): the main-pass phases run int8 tensor-core GEMMs over the fragment-ordered weights
  (every lane load one 16-byte LDG of contiguous bytes; integer to float without I2F through a magic-number add), so
  the cost of a verify pass grows little from 1 to 8 rows.
- **dp4a GEMV with shared activations** (the lean build, and the matrices wide verify does not take): activations are
  int8 (per-16/32 scales, as llama.cpp's q8_1), each warp streams 2-4 weight rows at once and reuses every
  shared-memory activation load across them, so T = 4 costs almost the same memory time as T = 1. Weight loads bypass
  L1 and are marked L2 evict-first.
- **No launches, few round trips**: activations are re-quantised by the producing phase (including the next RMS norm's
  weight), so consumers only async-copy (`cp.async`) them.
- **Gated-DeltaNet rollback by lazy replay**: the recurrent state is committed only through the surely-accepted token;
  accepted drafts are replayed from a small (k, v, g, beta) ring at the start of the next cycle, so rejected drafts cost
  no state snapshots or extra state traffic.
- **Split barrier before the DeltaNet core** (`MK_K5=1`, served build): the 48 DeltaNet heads go to the first 48 blocks
  that finish the DeltaNet projections; they load the head's state and replay its accepted drafts before waiting for
  the rest. Bitwise exact; decode +1.1% at 38k and +0.75% at 101.6k context (A-B-A, 2026-10-17).
- **Per-chunk waits before ffn_down** (`MK_C1`, on in every `MK_WV` build): ffn_down runs in four split-K chunks, and
  each chunk's blocks compute exactly the gate / up tiles that chunk reads, count them on the chunk's own counter and
  wait only for that, instead of a grid barrier over all blocks. Bitwise exact; decode +0.4% at 38k, +0.6% at 101.6k and
  +0.4% on short prompts (A-B-A, 2026-10-20).
- **No per-launch copy or barrier reset** (`MK_EGRESS`, on in every `MK_WV` build): the kernel writes its control block
  to mapped host memory instead of a device-to-host copy per launch, the barrier counts carry over between launches
  instead of being zeroed, and new draft-vocabulary rows are uploaded asynchronously. Bitwise exact; decode on Linux
  +1.0..1.3% at 38k, +0.7..1.2% at 101.6k and +1.0..1.3% on short prompts (two A-B-As), natively on Windows +1.0% at
  38k, +0.8% at 101.6k and +1.0% on short prompts (2026-10-20); 2 MiB more VRAM. `-DMK_EGRESS=0` builds without it.
- **Attention**: split-K flash decoding with tensor-core MMAs (Q.K^T and P.V in fp16 with fp32 accumulation, online
  softmax) over 32-position K/V tiles; the chunk boundaries depend on the context length only, so greedy output does
  not depend on the draft depth. With the q8 cache (`attn_core_q8`) each block runs a three-deep software pipeline with
  one barrier per tile (S of tile j, softmax of tile j - 1 and P V of tile j - 2 on specialised warps), the raw q8 K
  and V tiles stream through cp.async rings and both MMA operands are converted from int8 in registers. Same
  arithmetic as the simple loop (bit-identical output), about twice its KV bandwidth at 100K context; when it went in
  (2026-10-12) decode gained 18.7% at 101.6K depth and 7.3% at 38K. Chunks of a single tile (context <= 640) keep the
  simple loop, which has no pipeline to fill; low-bit `MK_KV` slots run the pipelined kernel too (`attn_core_q8m`).
  At T > 5 (up to 48 query rows) one pass over K / V serves all rows (`attn_core_w48`).
- **MTP drafting on device**, with the draft lm_head restricted to the 32,768 most frequent tokens
  (`draft_vocab_32k.bin`, built by `tools/vocab_freq` from local code + docs; covers ~99.4% of tokens), plus the tokens
  of the current sequence that are missing from it, added on the fly (identifiers of pasted code). Verification always
  uses the full vocabulary, so this affects only the acceptance rate, never the output.
- **Sampling** keeps the 32 best logits of each verified row; a draft is accepted with probability p(draft) and a
  rejected one is resampled from the rest of the distribution, which keeps the sampled distribution exact.
- **L2 warm-up**: before each barrier, blocks prefetch (evict-last) what their next latency-bound phase reads first
  (DeltaNet state, conv weights, norm weights, RoPE rows, first K/V tile).

## Prefill

A prompt (all but its last token) is processed in chunks of up to 2048 tokens (`MK_PF_CAP`) by ordinary kernel
launches (`prefill.cuh`), which leave exactly the state the megakernel expects (trunk and MTP K/V caches, committed
DeltaNet state, conv ring, hidden-state history); the last prompt token then goes through the megakernel, which
produces the first token and the first MTP drafts. The same path appends to a running conversation
(`MKEngine::prefill` on a non-empty sequence, e.g. the next user turn): it starts at the committed DeltaNet position
and also re-feeds the last emitted token and any accepted drafts not yet folded into the recurrent state; the output is
token-identical to processing the whole conversation as one fresh prompt. The host queues each chunk's work before it
waits for the previous chunk, so the GPU does not idle between chunks (`MK_PF_SYNC=1`: the previous flow); on Windows
those waits use a high-resolution waitable timer, since a plain sleep ends at the next timer tick (15.6 ms by default).

- **GEMMs on fp16 tensor cores straight from the k-quants** (`prefill_gemm.cuh`): 128 x 128 tiles (narrower token
  tiles for short prompts), software-pipelined through cp.async stages; weights stay in their GGUF blocks and are
  dequantised to fp16 in registers (exact small integers via the `0x6400` trick, then one fp16 scale and min per
  sub-block); activations are fp16 with the elements of each 16-group permuted so that a 4-byte weight word is directly
  an MMA A fragment. Products accumulate in fp16 over k = 128, then in fp32. Measured inside a 101.6k prefill: 89-105
  TFLOPS, 108-117 in isolation (2026-10-12). FFN gate and up share one GEMM whose epilogue writes `silu(gate) * up` as
  the down projection's fp16 input. Stage hand-off per type (2026-10-18): in the Q5_K and IQ4_NL GEMMs each warp copies
  the weight bytes of its own rows, so the block barrier covers only the activation tiles (one per two stages); Q6_K,
  Q3_K, IQ3_S and seven of the mixed-type gate / up pairs use per-stage mbarriers (copies two stages ahead) instead of
  the block barrier (both need four cp.async stages); the other kernels keep one block barrier per stage. Same products
  in the same order: bit-identical outputs.
- **DeltaNet in chunked (WY) form on tensor cores**: per 32-token chunk, `A = tril(beta K K^T)` with the gate decays,
  `T = (I + A)^-1` (one warp, forward substitution), then `U = T beta V - T beta gamma (K S)`, `O = gamma Q S + P U` and
  `S <- gamma_last S + K^T (decay U)`. Value columns are independent, so a block owns (head, 64 columns) and each warp
  keeps its 128 x 8 slice of the state in fp32 MMA accumulators, turning it (and K S, U) into fp16 MMA operands in
  registers with `movmatrix`; gates are carried as log g, so tiny or zero gates are exact.
- **Attention**: causal flash attention over the KV cache on tensor cores (`prefill_attn.cuh`). A block takes 16 query
  positions of one KV head (6 q heads x 16 rows); two warps per q head split S by key positions and P V by head-dim
  halves, so 12 warps keep 64 fp32 accumulators each and K / V of the next phase load during the MMAs. P V accumulates
  in fp16 per 32-position tile, then in fp32. The q-tiles of the last, partial wave are split along the keys and merged.
  When this kernel went in (2026-10-11) the prefill of a 101.6K-token prompt went from 127.9 s to 90.3 s. With the q8
  cache (default), Q K^T runs on int8 tensor cores straight from the cache (`pk_attn8`): Q is quantized per 32 dims
  like the cache, each 32-dim block's int32 partial is scaled by q scale x k scale, the raw q8 K / V tiles stream
  through a cp.async ring and V is converted for the next tile while the current one computes (2026-10-12: 59 -> 84
  TFLOPS-equivalent in the microbenchmark at 100K; attention of a 101.6K-token prompt 38.2 s -> 25.8 s, no measurable
  KLD change). With `MK_EXACT=1` the attention works on absolute key segments, so a row does not depend on where its
  chunk starts.

## Limitations

- Specialised to this model's shapes (27B) and the weight types under [The GGUF](#the-gguf) (tested: unsloth's Q4_K_M
  and UD-Q4_K_XL); the loader rejects any other model or type. Tuned and tested on sm_86 (RTX 3090); built for sm_86,
  sm_89 and sm_120 GPUs with 24 GB or more. `make` builds for the installed GPU (`make ARCH=sm_89` to override).
- More than 4 drafts needs the served build and the q8 KV cache (`--kv q8`; low-bit `MK_KV` presets are fine).
- The decode kernel runs one block per SM and needs at least 80 SMs for more than 4 drafts (50 for up to 4): the
  engine refuses a smaller GPU at load, naming the requirement (an RTX 3090 has 82). On a GPU with more than 128 SMs it
  runs 128 blocks.
- Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order), and presence / frequency
  penalties on the generated tokens (applied first; greedy too; exact under speculative verification: a verify row also
  counts the drafts before it); no repetition penalty.
- Text, images and video (no audio; images and clips as base64; video needs ffmpeg; Qwen-VL projectors only); one
  request at a time; several conversations are kept in host memory between requests
  ([SERVER.md](SERVER.md#the-conversation-cache)), and an edit re-prefills from the last checkpoint before it; no slot
  save / restore.

## Files

- `qwen38_mk.cu` / `qwen38_mk.h` - loader, repacking, the megakernel and the `MKEngine` API
- `wv.cuh` / `wv_host.cuh` - wide verify: the int8 tensor-core verify GEMM and its load-time weight re-layout
- `prefill.cuh` / `prefill_gemm.cuh` / `prefill_attn.cuh` - batched prefill kernels, their quantised fp16 GEMM and
  the prefill attention
- `loader.cuh` - the host side of the model load (read-ahead, reusable buffers); `gguf.h` - minimal mmap GGUF reader
- `capi.cpp` - C API (engine + libllama tokenizer + keep-warm) for `libqwen38mk.so`
- `server/mk_server.py` - the HTTP server ([SERVER.md](SERVER.md)); `scripts/serve.sh` / `scripts/serve.bat` - the launchers
- `main.cu` - CLI (tokenization via libllama)
- `compat_win32.h` / `.cpp`, `vision/win32.h` / `.cpp`, `build.ps1` - the native Windows build
- `draft_vocab_32k.bin` - MTP draft-head token subset (int32 ids)
- `tools/vocab_freq.cpp` - builds a draft vocabulary from a list of text files
- `tools/llama_ref.cpp` / `logit_stats.h` - llama.cpp's next-token distributions and the comparison with them (KL
  divergence, perplexity, top-token agreement)
- `vision/mk_vision.{h,cpp}` - the CPU vision encoder's C API (`libqwen38vision.so`): image decoding, video frames and
  frame pairs, backends
- `vision/vit_cpu.{h,cpp}` - the vision tower in fp32 with AVX2 / FMA kernels (the `cpu` backend)
- `vision/mkv_cli.cpp` - encodes images from the command line (grid, time, rows)
- `vision/vit_check.cpp` - test: vit_cpu against libmtmd (pixels, rows, times; `--pair`: video frame pairs);
  `vision/vit_ref.py` - the fp64 reference
- `tools/vision_ref.cpp` - llama.cpp's distributions for an image prompt + the image rows it fed (`--kld-images`)
- `bench/` - benchmarks ([BENCHMARKS.md](BENCHMARKS.md#reproducing)); `tests/` - server tests (no GPU needed)
