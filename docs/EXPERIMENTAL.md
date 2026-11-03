# Experimental builds

Two opt-in source builds are in active development and on the list for the served build: the deep pass (up to 15 drafts
per step) and int8 decode attention. Neither is in the release packages; each builds from source with one flag
([BUILDING.md](BUILDING.md) has the rest of the build). The default build stays the served one: 7 drafts, every KV
preset, the full 262,144-token context.

Contents: [the deep pass](#the-deep-pass) - [int8 decode attention](#int8-decode-attention) - [both
together](#both-together) - [speed](#speed) - [how it was measured](#how-it-was-measured).

## The deep pass

The default build verifies at most 8 tokens per cycle (`-d` up to 7; the launchers use 7). A build with `MK_MAXD=16`
also takes `-d` 8 to 15 and verifies those 9 to 16 tokens in one 16-row pass (`main_forward_d`: the matrix products in
the 16-row phases of the `--agents` batched passes, attention as two 8-row calls). It is opt-in and in active
development: the served configuration and the release packages are the default build, so the deep pass needs a source
build, and its support for the low-bit KV presets is on the list.

- **Build:** `make -j2 MKFLAGS=-DMK_MAXD=16` (Linux / WSL2) or `.\build.ps1 -MkFlags "-DMK_MAXD=16"` (Windows). Both
  give the flag to the device object and the seven host objects alike, as the engine requires (it refuses at load a
  device object built with other settings than its host files), and a change of flags recompiles all of them, ptxas
  included ([its RAM and time](BUILDING.md#requirements)). `-DMK_PAR_RMAX` must stay at 16 or more (its default is 16).
  A server reports the build in `/props` (`build_info`: `-DMK_MAXD=16`).
- **Drafts 8-15 need** the q8_0 KV cache (`--kv q8`) with 8-bit K and V in the 16 attention layers: no `MK_KV` preset
  (C, D, D', E or F) and no `MK_KV_WIN` rings. Low-bit presets with the deep pass are on the list, not supported yet.
  They also need every main matrix but ffn gate / up in the wide-verify layout, and at least 80 SMs. With `-d` above 7
  and one of these missing, the load fails with `n_draft > 7 (the deep pass) needs <the first one missing>`. At 7 drafts
  or fewer none of this is checked: a deep-pass build loads with the presets as the default build does.
- **`scripts\serve.bat` above 240,000 tokens of context** sets preset F unless `MK_KV` is set, so with `MK_DRAFT`
  above 7 a deep-pass build fails to load there. Keep `MK_DRAFT` at 7 or below for contexts that need a preset (the
  full 262,144 tokens on a 24 GB card).
- **When a cycle takes the deep pass:** only when the MTP head's own probabilities expect at least 3 of the drafts past
  the 7th to be accepted (`MK_POLICY` key `tier`, [CONFIGURATION.md](CONFIGURATION.md#mk_policy-keys); `tier=0` drops
  this condition) and its 16 positions lie in one attention partition, as the 8-row pass's 8 must; other cycles verify
  up to 8 tokens, as the default build does. Parallel slots (`--agents`) that decode together in one batched pass verify
  at most 8 tokens each in either build; a slot that decodes alone (one busy agent, or `--agents 1`) steps as a single
  sequence and can take the deep pass.
- **Output:** the default build's. Greedy output at 0, 7 and 15 drafts, and a seeded sampled run, matched the default
  build's at 7 drafts on short prompts and on a 40.5k-token prompt (512 tokens), also with every cycle forced through
  the 16-row pass (2026-11-03). `MK_POLICY=stop=0,drop=0,droplast=0,tier=0 scripts/exactness.py -m <gguf> --depths
  0,7,15` runs the [exactness checks](BUILDING.md#checking-a-build) that way: every cycle at 15 drafts then takes the
  16-row pass.
- **Speed:** [below](#speed).
- **On the list:** the low-bit KV presets with drafts 8-15, and int8 attention in the 16-row pass ([both
  together](#both-together)).

## Int8 decode attention

The decode attention computes Q K^T in int8 for the attention layers whose K and V are both 8-bit (the q8_0 cache): each
query row is quantized per 32 dimensions as the cache's K is, and a score is the sum of the blocks' exact int32 dot
products times the two scales. One function computes the scores for every attention kernel, so the engine's exactness
holds with it on: the same output at every draft count, from parallel slots as from a single sequence. It is not
bit-identical to the default fp16 Q.

- **Build and run:** `make -j2 MKFLAGS=-DMK_ATT_I8=1` (Linux / WSL2) or `.\build.ps1 -MkFlags "-DMK_ATT_I8=1"`
  (Windows), then `MK_ATT_I8=1` in the environment at run time (the q8_0 KV cache; slots of a low-bit `MK_KV` preset
  keep fp16 Q). Without the variable an int8 build runs the default attention and gives the default build's output. A
  default build prints that `MK_ATT_I8` is ignored.
- **Cost to compile:** ptxas peaked at 26.1 GiB in 17 minutes, against 21.0 GiB in 12 for the default build (its eleven
  extra attention kernels). This is why it is not in the default build.
- **Accuracy** (2026-11-02, an earlier build of the same code): against llama.cpp's reference distributions on five
  reference sets, KL divergence within +0.00002 of the fp16 path, and top-1 agreement with llama.cpp as high as fp16's
  (100k context: 0.05 points lower). Its top token matched the fp16 path's at 99.27% of positions (99.24% in decode),
  under the 99.5% that was proposed as the bar. The bar is not settled; until it is, int8 attention stays opt-in.
- **Exactness** (2026-11-03): with `MK_ATT_I8=1`, greedy output identical at 0, 4 and 7 drafts and without the MTP
  layer, the same tokens as an earlier int8 build of this branch, and four parallel slots identical to each conversation
  alone. With the variable unset, the 40.5k-token prompt's 512 tokens matched the default build's.
- **Speed** ([below](#speed)): 12% less attention time per 8-row pass at 40k tokens of context (2026-11-02); decode
  +1.4% after a 40.5k-token prompt, but 2.7% slower on short prompts, about half of that from building it in.
- **On the list:** a settled accuracy bar, then int8 attention in the served build; the deep pass with it is below.

## Both together

`make -j2 MKFLAGS="-DMK_MAXD=16 -DMK_ATT_I8=1"` builds the deep pass with int8 attention
(`.\build.ps1 -MkFlags "-DMK_MAXD=16 -DMK_ATT_I8=1"` on Windows). ptxas peaked at 27.0 GiB in 17 minutes. With
`MK_ATT_I8=1` its greedy output at 0, 7 and 15 drafts matched an earlier deep build of this branch with int8 attention,
and with the variable unset it matched the deep build (2026-11-03). Speed: [below](#speed).

## Speed

Decode rates on an RTX 3090, two sessions on 2026-11-03 ([how](#how-it-was-measured)); each row against the default
build of the same session.

| Build | Drafts | Short prompts | vs default | After a 40.5k-token prompt | vs default |
|---|---|---|---|---|---|
| v2026.11.01's engine (default build) | 7 | 155.9 tok/s | -5.2% | 115.9 tok/s | -5.4% |
| **this release, default build** | 7 | **164.4 tok/s** | | **122.5 tok/s** | |
| deep pass | 15 | 168.8 tok/s | +2.7% | 121.9 tok/s | -0.5% |
| int8 attention (`MK_ATT_I8=1`) | 7 | 159.9 tok/s | -2.7% | 124.2 tok/s | +1.4% |
| *second session:* default build | 7 | 165.0 tok/s | | 122.1 tok/s | |
| int8 build, `MK_ATT_I8` unset | 7 | 162.7 tok/s | -1.4% | 122.0 tok/s | -0.1% |
| deep pass | 15 | 170.1 tok/s | +3.1% | 122.2 tok/s | +0.1% |
| deep pass and int8 attention | 15 | 163.4 tok/s | -1.0% | 124.3 tok/s | +1.8% |

- **The default build** of this release decodes 5.5% faster on short prompts and 5.7% faster after the 40.5k-token
  prompt than v2026.11.01's engine, and reads that prompt in 22.6 s instead of 23.3 s. Every experimental build reads it
  in the same 22.5-22.8 s: neither option changes prompt processing.
- **The deep pass** gains on short prompts (+2.7% and +3.1%); after the 40.5k-token prompt it runs at the default
  build's rate.
- **Int8 attention** gains after the long prompt (+1.4%, +1.8% with the deep pass) and loses on short prompts: -2.7%
  with it on, -1.4% with it only built in, so about half of that loss comes from the extra kernels in the build.

## How it was measured

An RTX 3090 (driver 610.62, 480 W power limit) under WSL2, the UD-Q4_K_XL model, the q8_0 KV cache, and the CLI with
the default draft policy, greedy: `qwen38_mk -m <gguf> --kv q8 -n 512 -d 7` (`-d 15` for the deep pass).

- **Short prompts:** the four prompts of `bench/bench.py` (`python3 bench/bench.py mk --draft N -n 512 --reps 2`, which
  passes `-c 8192`), the mean of the request rates.
- **The 40.5k-token prompt:** 40,509 tokens (the start of an earlier version of this engine's `qwen38_mk.cu`, then a
  question), `-c 65536`: the decode rate of the 512 tokens and the time to read the prompt.
- **Sessions:** each ran its builds interleaved, two rounds, with nothing else on the GPU; the rows are the means of the
  two rounds, which agreed within about 1%. A build's rate moves by a few percent from one session to another with GPU
  clocks and temperature ([BENCHMARKS.md](BENCHMARKS.md)), so compare rows within a session.
