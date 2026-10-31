# Benchmarks

Every number here was measured on one machine: an RTX 3090 (24 GB, 480 W limit), a Ryzen 9 5900XT and 64 GB of RAM,
Windows 11 with WSL2, CUDA 13.2, with unsloth's `Qwen3.8-27B-UD-Q4_K_XL.gguf` unless a section says otherwise. Each
result carries its date and the commit it was measured on.

How to read them:
- **Sessions.** The same binary runs several percent faster or slower from one session to the next with GPU temperature
  and clocks (up to ~11% between sessions on 2026-10-12). Absolute numbers are only compared within one session;
  changes made after a full run are measured A-B-A (old build, new build, old build again, in one session), never added
  onto older absolutes.
- **Rates.** Decode "tok/s" is generated tokens over decode time, as the server reports it (`timings`). The
  short-prompt rows are means of the per-request rates, as the client reports them; token-weighted rates (total tokens
  over total time) are given next to them.
- **VRAM** is `nvidia-smi`'s memory used on the whole card, other processes included, unless a row says otherwise.
  GiB = MiB / 1,024.
- **Other engines** are only compared where we ran them ourselves on this machine: llama.cpp b11548 (tuned) for speed,
  llama.cpp b10246 as the accuracy reference, and upstream open-jet at 72ba4a7.

Contents: [headline numbers](#headline-numbers) - [full run 2026-10-30](#full-run-2026-10-30) -
[parallel agents](#parallel-agents) - [KV cache presets](#kv-cache-presets) -
[Q4_K_M and upstream's scripts](#q4_k_m-and-upstreams-scripts) - [exactness](#exactness) -
[the conversation cache](#the-conversation-cache) - [keep-warm](#keep-warm) - [images and video](#images-and-video) -
[earlier full runs](#earlier-full-runs) - [reproducing](#reproducing).

## Headline numbers

The latest result in each area. The links lead to the details, and the previous full run is at the
[end of the file](#earlier-full-runs).

**Against llama.cpp** ([full run 2026-10-30](#full-run-2026-10-30)): UD-Q4_K_XL, q8_0 KV cache, the server at 120k
context. This engine: the release v2026.10.30 (6f3b257), 2026-10-30. llama.cpp: b11548's `llama-server`, tuned (q8_0
KV, 4 MTP drafts), the same client, prompts and model file, the best value of each row over three sessions on
2026-10-12 / 13.

| | llama.cpp | this engine |
|---|---|---|
| greedy decode, short prompts (4 x 512 tokens) | 82.7 tok/s | 156.7 tok/s |
| sampled decode, thinking on (8 runs) | 65.8 tok/s | 138.2 tok/s |
| decode at 38k / 101.6k context | 62.3 / 53.4 tok/s | 119.8 / 109.4 tok/s |
| read a 101.6k-token prompt from scratch | 101.0 s | 75.2 s |
| first token, short prompt | 124 ms | 66 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 323 ms | 176 ms |
| 282 benchmark questions: right, total time | 229, 2,727 s | 230, 1,676 s |
| VRAM, idle server at 120k context | 22.8 GiB | 21.3 GiB (preset D: 20.1, preset F: 20.0) |

**Everything else:**

| | result | measured | details |
|---|---|---|---|
| 4 parallel agents (`--agents 4`, Q4_K_M) | 380.1 tok/s in total on short prompts, 318.8 with a ~23k-token context each | 2026-10-30 | [parallel agents](#parallel-agents) |
| upstream open-jet 72ba4a7, its own scripts (Q4_K_M, 4 drafts each) | writing code 163.8 against 132.6 tok/s; with 30k tokens of code in context 118.1 against 86.9; prefill 1,878 against 1,528 tok/s | 2026-10-25 | [Q4_K_M and upstream's scripts](#q4_k_m-and-upstreams-scripts) |
| accuracy against llama.cpp generating token by token | top token agrees on 98.737% of positions, mean KL 0.00119 nats (llama.cpp's own batched run: 98.489%, 0.00166) | 2026-10-17, unchanged through v2026.10.30 | [against llama.cpp](#against-llamacpp) |
| KV presets D / F: KL at 100k context | 0.00226 / 0.00218, against q8_0's 0.00217 | 2026-10-25 | [KV cache presets](#kv-cache-presets) |
| the full 262,144-token context, preset F | a 260,896-token prompt in 340 s, decode at that depth 66.6 tok/s, 23,650 MiB on the card | 2026-10-25 | [the full context](#the-full-context) |
| greedy token ids at 0, 4 and 7 drafts | identical | each release | [speculation](#speculation) |
| Windows build against Linux build | identical output | each release | [Windows and Linux](#windows-and-linux) |
| reused, parked and restored requests (exact reuse) | bit for bit equal to the same request sent fresh | 2026-10-14 to 10-21 | [exact reuse](#exact-reuse) |

## Full run 2026-10-30

The release v2026.10.30 (6f3b257), its Linux build on WSL2: the server at 120,000 tokens of context as
`scripts/serve.sh` starts it (wide verify, 7 drafts, exact reuse, side requests and parked conversations on, prefill
chunk 1,024), driven by [bench/suite](../bench/suite/README.md) in one session. The llama.cpp column is b11548's
`llama-server`, tuned (q8_0 KV, 4 MTP drafts), driven by the same client with the same prompts and model file: the best
value of each row over three sessions (2026-10-12 and 10-13; the questions on 10-13).

| | llama.cpp (10-12 / 10-13) | this, q8_0 KV | this, preset D | this, preset F |
|---|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens), mean of request rates | 82.7 tok/s | 156.7 | 156.8 | 150.2 |
| the same, token-weighted | 76.2 | 133.2 | 133.4 | 127.3 |
| sampled decode, thinking on (8 runs), mean of request rates | 65.8 tok/s | 138.2 | 138.2 | 138.3 |
| the same, token-weighted | 63.9 | 132.0 | 132.0 | 130.6 |
| decode at 11.6k context (200 tokens) | 69.3 tok/s | 138.7 | 137.5 | 124.0 |
| decode at 38k context (256 tokens) | 62.3 tok/s | 119.8 | 102.8 | 101.8 |
| decode at 101.6k context (256 tokens) | 53.4 tok/s | 109.4 | 99.0 | 109.9 |
| read an 11.6k-token prompt | 8.24 s | 5.95 s | 6.06 s | 6.11 s |
| read a 38k-token prompt | 29.6 s | 23.1 s | 24.0 s | 24.2 s |
| read a 101.6k-token prompt from scratch | 101.0 s | 75.2 s | 78.1 s | 77.9 s |
| first token, short prompt | 124 ms | 66 ms | 65 ms | 66 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 323 ms | 176 ms | 184 ms | 185 ms |
| edit an earlier message in an 11.7k-token chat, prompt time | 329 ms | 77 ms | 76 ms | 77 ms |
| edit the end of a 9.1k-token system prompt, prompt time | 956 ms | 89 ms | 99 ms | 100 ms |
| VRAM, whole card, server idle | 22.8 GiB | 21.3 GiB | 20.1 GiB | 20.0 GiB |
| 282 benchmark questions: right, total time | 229, 2,727 s | 230, 1,676 s | 230, 1,716 s | 229, 1,714 s |

Notes:
- **The follow-up turn** re-reads the previous reply because exact reuse is on; `MK_EXACT_REUSE=0` appends to it
  instead (~80 ms on 2026-10-14, but not bit-identical to a fresh request).
- **The 101.6k-token prompt from scratch** ran on a separate server with side requests and parking off
  (`MK_SIDE_MAX=0 MK_PARK_MIB=0`): with them on, the client's cache-clearing request runs beside the cached
  conversation instead of evicting it, and the "from scratch" request would be served from the cache.
- **The presets' decode rows** are not like-for-like with q8_0's: preset D's replies differ from q8_0's from 2,048
  tokens on, preset F's from the start (its rotated K changes the window's rows too), and with the replies the share of
  accepted drafts changes. The time per engine cycle compares the kernels directly (reply time / (reply tokens / tokens
  per cycle), from the server's console):

  | context | q8_0: tokens per cycle, ms per cycle | preset D | preset F | D / F, ms per cycle against q8_0 |
  |---|---|---|---|---|
  | 11.6k (200 tokens) | 3.75, 27.0 | 3.83, 27.8 | 3.37, 27.1 | +2.8% / +0.5% |
  | 38k (256 tokens) | 3.53, 29.5 | 2.98, 29.0 | 3.06, 30.1 | -1.8% / +2.1% |
  | 101.6k (256 tokens) | 3.61, 33.0 | 3.40, 34.4 | 3.81, 34.7 | +4.2% / +5.1% |

  One reply each, timed to 0.01 s, so each derived value is good to about 0.5%. On the short prompts preset D's replies
  and cycles were q8_0's; preset F's cycles took 0.1% to 2.6% longer, and most of its lower greedy rate there comes from
  accepting fewer drafts (3.80 instead of 4.16 tokens per cycle on one of the four prompts).
- **The questions**, per set: q8_0 and preset D gave the same answers as on 2026-10-20: GSM8K 50 / 50, MMLU-Pro 47 / 70,
  ARC-C 47 / 50, MATH-500 29 / 40 (and 9 / 10 with thinking on), HumanEval 37 / 41, NIAH 9 / 9, LongBench v2 2 / 12.
  Preset F differs in the same three sets as on 2026-10-25 ([preset F](#preset-f)): MMLU-Pro 46, MATH-500 30,
  HumanEval 36. This client gives LongBench v2 answers 128 tokens, which cut 8 of its 12 answers on 2026-10-14; the
  262,144-token runs give them 1,024. The total time includes the side requests and parks the questions cause beside
  the cached long conversation.
- **VRAM:** the Windows desktop held 733 MiB of the card during this run. The engine's own memory, measured right after
  the run against the desktop alone (q8_0 at 120k, idle), was 21,041 MiB: 50 MiB less than 7322fbd, the code of the
  2026-10-20 run, whose whole-card figure (21.1 GiB) included a smaller desktop share (about 0.5 GB).
- **Conditions:** fans at 98%; GPU 46-68 C and SM clock 1,950-2,025 MHz (`nvidia-smi` at the start and end of each
  pass); memory clock 9,851 MHz.
- **Against the previous full run** ([2026-10-20, 7322fbd](#2026-10-20-7322fbd)): decode +0.4% to +1.6% for q8_0 and
  preset D; q8_0's time per cycle on the long-context replies 0.9% to 1.5% lower, with the same tokens per cycle; the
  prompt rows within 1.5% either way (none of the changes in between touched prefill); the questions 1,681 -> 1,676 s
  (q8_0) and 1,722 -> 1,716 s (D) with the same answers. Each change in between was measured on its own
  ([below](#changes-since-the-2026-10-20-run)).

### How the run was measured

- **Client:** [bench/suite](../bench/suite/README.md)'s `speed_client.py` against `/v1/chat/completions` without
  streaming: the requests of the 2026-10-12 / 13 llama.cpp sessions plus a clock warm-up. It reads the server's
  `timings` (prompt and decode time) and `usage` (cached tokens). One warm-up request, then ~3 s of decode, so the
  timed requests start at full clocks.
- **Short prompts:** four chat prompts (a Python prime check with doctests, how a hash map works, a 300-word story, a
  Flask-to-FastAPI migration plan): greedy without thinking, 512 tokens each; then sampled with thinking (temperature
  1.0, top_p 0.95, top_k 20) with seeds 11 and 22, up to 768 tokens each. First token: three one-word questions,
  `max_tokens` 1.
- **Long context:** an 11.6k-token excerpt of a 38k-token document with a summary request (greedy, thinking on, 200
  tokens) and a follow-up question; the whole 38k-token document, then a 101.6k-token one, each with a 256-token answer
  (greedy, no thinking). In the main pass the 101.6k prompt reuses the 38k one's prefix; the from-scratch row is the
  separate server described above.
- **Edits:** a three-turn chat over 30k characters of the document, then turn 2's user message edited; a 9.1k-token
  system prompt, then its last word changed. The row is the prompt time of the edited request.
- **Questions:** 282 items, greedy, zero-shot, the same prompts for both engines, thinking off except for the 10
  MATH-500 items with thinking; scored offline. The llama.cpp answers come from the same client on 2026-10-13.

### Changes since the 2026-10-20 run

Each change to the GPU code was measured against the build before it, the two builds alternating in one session, with
the CLI under Linux / WSL2 (UD-Q4_K_XL, q8_0 KV cache, 7 drafts unless noted). None of them changed the greedy token
ids or the accuracy report.

| change | effect | measured |
|---|---|---|
| preset F's own 5-bit prefill kernels (fad007f) | F's prompt time -3.4% at 101.6k, -1.5% at 38k (exact mode), its KL report byte-identical; the other caches' kernels unchanged | 2026-10-25 |
| parallel slots (`--agents`, f09ef59) | one conversation decodes 0.6% slower (UD-Q4_K_XL), 1.3% slower (Q4_K_M) | 2026-10-28 |
| compile trim (752368c): code no served setting reaches is no longer compiled; lm_head's penalties and output type became runtime switches | ms per cycle -0.28% on short prompts, unchanged at 38k and with 4 slots, ~+0.2% with a presence / frequency penalty (estimated); ptxas peak ~36 -> 20.2 GiB, build 28.5 -> 13 minutes | 2026-10-29 |
| the megakernel's stack frame 2,016 -> 816 bytes (df0d418) | 144 MiB less local memory reserved; speed unchanged at 38k | 2026-10-30 |
| local memory and spills, frame 816 -> 752 bytes (6f3b257) | ms per cycle -1.55% at 0 drafts, -0.91% at 38k, -1.19% at 101.6k; 4 slots +7.4% tok/s | 2026-10-30 |

## Parallel agents

`--agents N` serves up to four requests at once, each in a parallel slot of the engine with its own context; every slot
that is decoding runs in the same batched launch ([SERVER.md](SERVER.md#several-agents-at-once---agents)). Measured
with [bench/suite/parallel_client.py](../bench/suite/README.md) (the prompts, sampling, context text and measurements
of open-jet's `bench/parallel_bench.py`) on unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf`, right after the full run on
2026-10-30 (WSL2, v2026.10.30, `--agents 4 -d 7`, q8_0 KV cache, exact mode):

| test | result |
|---|---|
| 4 agents, short coding prompts: total decode (better of 2 runs) | 380.1 tok/s |
| 4 agents, each with its own ~23k-token context: total decode | 318.8 tok/s |
| the same: the slowest first token | 0.51 s |
| a ~20k-token prompt sent while another request decodes: its first token | 13.7 s |
| the other request meanwhile | 11.2 tok/s |
| 1 agent, short prompts (better of 2 runs) | 181.0 tok/s |
| 1 agent, ~23k-token context | 183.4 tok/s |
| VRAM at ready, whole card | 22,727 MiB (733 MiB of it the desktop's) |

- A total adds the requests' decode rates (512 tokens each, temperature 0.6).
- One agent's short-prompt rate follows how many sampled drafts it accepts: twelve passes that day ran 167 to 204 tok/s
  (median 187) at the same 26.2 to 26.9 ms per engine cycle, with 4.4 to 5.5 tokens per cycle.
- The "other request" row counts that request's streamed chunks while the long prompt is read; it varies between
  sessions (10.8 to 12.9 tok/s on 2026-10-28).
- Each slot's greedy reply is the one the request gets alone, also when its prompt arrives while others decode; seeded
  sampling beside other slots keeps its distribution, not its tokens
  ([SERVER.md](SERVER.md#several-agents-at-once---agents)).

## KV cache presets

`MK_KV` stores K and V below 8 bits, and `MK_KV_WIN=2048` keeps the most recent 2,048 positions in an 8-bit ring next
to the low-bit cache (the presets C, D, D', E and F and their syntax:
[CONFIGURATION.md](CONFIGURATION.md#kv-cache-presets)). While a whole sequence fits in the window, every attention read
comes from the 8-bit ring, so preset D's output is q8_0's: its short-prompt accuracy report is byte-identical to
q8_0's. Preset F stores K rotated, which changes the ring's rows too.

Accuracy against llama.cpp (the reference in [against llama.cpp](#against-llamacpp)), top-token agreement / mean KL in
nats. The fp16 cache, more precise than q8_0, marks the floor of the measurement:

| KV cache | short prompts (13,299 positions) | ~32k (4,094) | 64k (4,095) | 100k (4,095) |
|---|---|---|---|---|
| fp16 (2026-10-12 / 10-13) | 98.774% / 0.00113 | 99.096% / 0.00226 | 0.00161 | |
| q8_0 | 98.737% / 0.00119 (2026-10-17, 7322fbd 10-20) | 99.121% / 0.00219 (10-17) | 99.267% / 0.00163 (10-15) | 99.121% / 0.00217 (10-25, 0278d09) |
| D: `K*=6,V*=5` + window | = q8_0 (2026-10-20, 7322fbd) | 98.950% / 0.00236 (10-20, 7322fbd) | 99.243% / 0.00160 (10-13) | 99.121% / 0.00226 (10-25, 0278d09) |
| D': D + 8-bit MTP layer | = q8_0 (10-15, 7d67c5d) | 0.00236 (10-15) | 0.00161 (10-15) | |
| F: `K*=5r,V*=5` + window | 98.714% / 0.00113 (2026-10-25, 7322fbd) | 98.974% / 0.00242 (10-25, 7322fbd) | 99.341% / 0.00163 (10-25, 7322fbd) | 99.170% / 0.00218 (10-25, 0278d09) |
| C: `V*=5` + window | = q8_0 (10-13) | 99.072% / 0.00228 (10-13) | 99.316% / 0.00159 (10-13) | |
| 4-bit everywhere: `*=4`, no window | 0.00547 (10-15) | | | |

- The MTP layer's cache only feeds the drafts, so D and D' score the same wherever both were measured.
- The fp16 cache was no better than q8_0: on its 2026-10-12 / 13 builds q8_0 read 0.00113 on the short set and 0.00219
  at 32k (the reference itself was generated with a q8_0 cache).
- The short set's values moved by up to 0.00006 between builds as the kernels changed, which is why each cell carries
  its date. The 32k set's noise is about 4% either way.
- **Memory:** the KV cache at 120k / 262,144 context takes 3.25 / 6.97 GB with preset D and 2.99 / 6.39 GB with F
  (2026-10-25). Whole-card VRAM, speed and the 282 questions under each preset: the
  [full run](#full-run-2026-10-30).

### Preset F

K at 5 bits, stored Hadamard-rotated, and V at 5 bits, with the 2,048 window (`MK_KV='K*=5r,V*=5' MK_KV_WIN=2048`). In
the rounding emulator, rotated 5-bit K under the window read at the fp16 cache's floor (+4% at 32k, -2% at 64k). With
the real kernels (2026-10-25, 7322fbd):

- **Short prompts** fit in the window, but F's rotated K changes the ring's rows too, so its short-set KL (0.00113) is
  not q8_0's; the difference is within the 0.00006 the set moves between builds.
- **The 282 questions:** 229 right against q8_0's and D's 230 (2026-10-25, the 2026-10-20 client and settings; the
  2026-10-30 run gave the same 229, set for set). The answers match per set except three items: one more MATH-500
  problem right, one MMLU-Pro answer cut by its 1,024-token cap before it named a choice, and one HumanEval task wrong.
  F's answers match D's word for word on 167 of the 282 (all of ARC-C and the needles, 36 of 41 HumanEval tasks): with
  rotated K even a short prompt's rows differ from q8_0's, so longer answers take other words on the way.
- **A larger window did not help:** at 32k, F read 0.00242 / 0.00239 / 0.00245 / 0.00240 with `MK_KV_WIN` 2048 / 2688
  / 3328 / 4608 (rings of 114 / 137 / 159 / 204 MiB), all within the set's noise. Its excess over q8_0 sits in the
  older 5-bit rows.

### The full context

The model's full 262,144-token context with presets D and F, under WSL2 and natively on Windows (where the desktop
shares the card):

| | D, WSL2 (2026-10-20, 7322fbd) | F, WSL2 (2026-10-25, 7322fbd) | F, native Windows (2026-10-25) | D, native Windows, 512-token chunk (2026-10-24) |
|---|---|---|---|---|
| read a 260,896-token prompt from scratch | 326 s | 340 s | 340 s | 366 s |
| decode at that depth | 66 tok/s | 66.6 tok/s | 66.5 tok/s | 64.3 tok/s |
| follow-up turn, prompt time | 1.28 s | | 1.3 s | 1.6 s |
| VRAM, whole card | 24,080 MiB after the prompt | 23,650 MiB after the prompt | peak 23,622 of 24,576 MiB | peak 23,771 of 24,576 MiB |
| needle in a haystack (4 needles, asks one), at 190k and at the longest length that fits | 4 / 4 | 4 / 4 | found at 190k | found at 190k |
| LongBench v2 at that length (6 questions, 1,024 answer tokens) | 4 / 6 | 3 / 6 | | |

- **Natively on Windows** the default settings (1,024-token prefill chunk, the draft head's Q4_K copy) leave preset D
  just past what the Windows driver keeps resident at this length, and the engine pages: a 442-token reply decoded at
  16 tok/s against 82.5 under WSL2 (the same tokens and drafts). A 512-token chunk with the draft head on lm_head
  (0.17 GB less, the same output) keeps it resident: the same reply at 81.6 tok/s, and the last column. Preset F stays
  resident with the defaults, so `scripts\serve.bat` uses F above 240k unless `MK_KV` is set.
- **LongBench:** the item F missed was still reasoning toward the right choice when it reached the 1,024-token answer
  cap; D answered it in 658 tokens.
- F's WSL2 column ran on the generic low-bit prefill kernel, before F's own 5-bit kernels (fad007f, 3.4% faster at
  101.6k).

### 100k and 160k context

The 32k / 64k measurement, further out (2026-10-25, the v2026.10.25 Linux package, 0278d09): llama.cpp b10246 with its
q8_0 cache reads one sequence and stores its last 4,096 positions (`tools/llama_ref --chunk N --max-chunks 1
--score-from N-4096 -b 512`), and the engine's `mixed` path is scored against that. Two texts: `pf101k.txt` (code, the
text of the 32k / 64k sets) and War and Peace (`pg2600.txt` from `bench/suite/fetch.py`, line ends converted to LF, from
its first byte). Prose has flatter next-token distributions than code, so it reads higher KL at any cache precision,
q8_0 included: compare within a row. Top-token agreement / mean KL / perplexity:

| text, tokens (scored positions) | q8_0 | D | F |
|---|---|---|---|
| code, 100,352 (96,256-100,351); reference perplexity 1.6250 | 99.121% / 0.00217 / 1.6226 | 99.121% / 0.00226 / 1.6210 | 99.170% / 0.00218 / 1.6230 |
| War and Peace, 100,352 (96,256-100,351); 9.4888 | 96.996% / 0.00311 / 9.4830 | 96.874% / 0.00317 / 9.4884 | 96.825% / 0.00317 / 9.4898 |
| War and Peace, 163,840 (159,744-163,839); 7.7330 | 96.361% / 0.00467 / 7.7244 | 96.581% / 0.00468 / 7.7282 | 96.459% / 0.00472 / 7.7329 |
| the same 4,096 tokens behind 96,256 (the novel from token 63,488); 7.7474 | 96.606% / 0.00336 / 7.7318 | 96.825% / 0.00335 / 7.7339 | 96.850% / 0.00344 / 7.7296 |

- **D and F read q8_0's KL at every depth here**, within the 32k set's ~4% noise; the largest gap is D's +4% at 100k
  on code.
- **The KL grows with depth for every cache, q8_0 included:** the same 4,096 tokens read 0.0034 behind ~100k of context
  and 0.0047 behind ~160k for q8_0, D and F alike (the last two rows), so the growth is between this engine's arithmetic
  and llama.cpp's, not in the cache formats. For scale, llama.cpp's own distributions for those tokens differ by
  0.00864 between the two depths. The disagreements stay near-ties: at 160k the median lead of llama.cpp's token over
  the engine's is 0.09 nats, and 1 of the 140-149 disagreements exceeds 0.5 nats for each cache.
- **160k** is about as far as llama.cpp's q8_0 cache fits on a 24 GB card to make the reference (34 KB per token).
  This engine's q8_0 cache held the 163,904 positions in 6.06 GB (the `--kld` CLI under WSL2: no server, no draft
  head).

### The window size and rotation

Why the presets use a 2,048-position window and, except for F's K, unrotated rows. All rows below come from the
rounding emulator (2026-10-12 / 13), which stores the cache in each format and is scored like the real kernels; it
keeps exactly W recent positions at 8 bits, where the kernels round the window up to 640-position steps
([CONFIGURATION.md](CONFIGURATION.md#choosing-the-window-and-rotation)). KL is relative to q8_0's on the same build
(0.00113 on the short prompts, 0.00219 at 32k); the fp16 cache reads +0% / +3%, and the 32k set's noise is about 4%
either way. The low-bit loss of K and V sits almost entirely in the most recent ~0.5-2k positions.

The window. The short prompts are ~1k-token sequences, so any window of 1k or more holds all of them, and only the 32k
set can tell 1k from 2k:

| low-bit sides | short prompts: no window / 512 | 32k: no window / 512 / 1,024 / 2,048 |
|---|---|---|
| V at 5 bits, rotated | +27% / +6% | +16% / +5.5% / +4.6% / 0% |
| K at 6 bits, rotated | +12% / +2.7% | +4% / +7% / - / +3% |
| K at 6 and V at 5 bits, rotated | +42% / - | - / - / +6% / +2.7% |
| K at 6 and V at 5 bits (preset D) | - / - | - / - / - / +1.4% |

Attention sinks instead of a window (the first 4 positions kept at 8 bits) did nothing: K at 6 bits, rotated, read +11%
on the short prompts against +12% without.

Rotation helps without a window (on the short prompts, K at 6 bits read +23% plain and +12% rotated, V at 5 bits +40%
and +27%), but not under it: with the 2,048 window, K at 6 and V at 5 bits read +1.4% plain and +2.7% rotated at 32k,
the same within the set's noise, so preset D drops the rotation (no rotation of the queries, no rotating back of the
attention output). Other formats tried in the emulator (3-bit, codebooks, per-row scales, random signs, angle coding)
scored worse than the plain per-32-value grid.

## Q4_K_M and upstream's scripts

unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf` runs on the same engine. Its last full run was on 2026-10-14 (b3be928), in one
session with UD-Q4_K_XL (q8_0 KV, 7 drafts, 120k context, WSL2):

| | UD-Q4_K_XL | Q4_K_M |
|---|---|---|
| greedy decode, short prompts | 144.7 tok/s | 148.1 tok/s |
| decode at 11.6k / 38k / 101.6k context | 128.6 / 109.9 / 102.0 | 145.7 / 109.4 / 106.0 |
| read a 101.6k-token prompt from scratch | 79.0 s | 77.6 s |
| VRAM, whole card | 21.1 GB | 20.7 GB |
| 282 questions right | 230 | 231 |

The two files disagreed on 15 of the 282 answers and had the same perplexity. Accuracy against llama.cpp on Q4_K_M:
98.887% / 0.00127 (a wide-verify build with up to 4 drafts, 2026-10-13; llama.cpp batched on that file: 98.489% /
0.00184).

Upstream open-jet's own benchmark scripts at its master 72ba4a7 (`bench/code_bench.py`, `bench/think_bench.py`: their
megakernel halves, best of 2, greedy code runs, the thinking runs at temperature 1.0 with seeds 1000 / 1001), on
Q4_K_M, in one session on 2026-10-25 (WSL2): upstream's engine (`make qwen38_mk` with its default flags, 4 drafts, the
CLI's default KV cache) between two passes of this engine (7322fbd, the release's GPU code, q8_0 cache). The table gives
the lower of this engine's two passes, which agreed within 0.7%:

| test | open-jet 72ba4a7, 4 drafts | this engine, 4 drafts | this engine, 7 drafts |
|---|---|---|---|
| writing code (no context) | 132.6 tok/s | 163.8 tok/s | 184.9 tok/s |
| 30k tokens of code in context | 86.9 | 118.1 | 126.9 |
| short prompts (mean of 4) | 117.1 | 140.2 | 157.2 |
| reasoning mode (4 prompts x 2 seeds, 1,024 tokens) | 102.5 | 119.4 | 130.1 |
| prefill, 29.6k-token prompt | 1,528 tok/s | 1,878 tok/s | 1,877 tok/s |

Draft acceptance in reasoning mode: open-jet 55.6%, this engine 61.1% with 4 drafts and 51.8% with 7. Upstream has
since merged some of this engine's kernels (open-jet #24 to #27), so the comparison is with 72ba4a7 as it was.

## Exactness

### Speculation

- **Greedy output does not depend on the draft depth.** The attention's split-K partition depends only on the context
  length, and the verify GEMMs compute every row the same way whatever the number of rows, so the same prompt gives the
  same token ids at 0, 4 and 7 drafts. Every release is checked this way (v2026.10.30: the Windows build's ids at 0 and
  7 drafts equal the Linux build's at 7).
- **Sampling keeps the distribution of sampling without drafts.** A draft is accepted with probability p(draft) and a
  rejected one is resampled from the rest of the distribution. Checked with chi-square and PIT tests; the counter-based
  random numbers, keyed by seed and position, were checked with a CPU replica of the loop (396k re-drafted positions)
  and a PIT test on 13,401 re-drafted rows of the engine at 7 drafts (KS p = 0.70; 2026-10-17). Two opt-in policy
  settings that were biased were fixed or restricted then ([ENGINE.md](ENGINE.md#correctness)).
- **The MTP layer only drafts.** A model file without one runs at depth 0, and its greedy output equals the drafted
  engine's bit for bit (UD-Q4_K_XL, 2026-10-17).

### Exact reuse

With `--exact-reuse` (`MK_EXACT=1`; the launchers set it), a reused request is answered bit for bit like the same
request sent fresh. The test is a state digest after the prompt (every K / V row, every row of the 8-bit window ring,
the DeltaNet and convolution state), compared with a fresh prefill of the whole prompt.

| case | result | date |
|---|---|---|
| follow-up turns, edits, abandoned replies, 38k context, q8_0 and preset D', several chunk sizes | digest bitwise equal to fresh | 2026-10-14 |
| sampled follow-up / edit / restored park with a fixed seed (the draft vocabulary rebuilt as a fresh prefill builds it) | 9 / 9 equal to fresh (before the change, 5 of 9 differed) | 2026-10-20 |
| a side request beside a cached conversation, a park and restore | digest after the side request / restore bitwise equal to before; the next replies equal to the conversation alone on a fresh server, greedy and sampled (8 / 8) | 2026-10-19 |
| a new session resumed at a pinned checkpoint (live, parked, after a side request, after the conversation grew to 63k) | replies equal to the same request on a fresh server, q8_0 and preset D, greedy and sampled | 2026-10-21 |

What it costs (2026-10-14): a follow-up turn re-reads the previous reply (11.6k-token chat, 7 drafts: 167 instead of
78 ms; with a thinking reply kept in the history: 469 instead of 86 ms). Edits cost the same either way. A fresh
98k-token prompt took 1.3% longer, and restoring a checkpoint and reading 32 / 128 / 512 tokens after it took 2.6% and
1.6% less and 0.7% more time (A-B-A, q8_0). The accuracy report is the same within noise: 98.662% / 0.00118 on the
short set and 99.121% / 0.00231 at 32k with exact reuse, 98.654% / 0.00116 and 99.023% / 0.00232 without. The CLI
measurements in [against llama.cpp](#against-llamacpp) run without exact reuse (the CLI's default); the launchers'
server runs with it.

### Against llama.cpp

The engine is not bit-identical to llama.cpp: the arithmetic is ordered and rounded differently, so where two tokens are
close the other one can win. It is scored against llama.cpp's own next-token distributions: llama.cpp b10246 generating
token by token with its q8_0 KV cache, on 13 sequences of this repository's code and docs (13,299 positions scored after
a 1,024-token prompt each), with the 256 most likely tokens stored per position (`tools/llama_ref`, `--kld`).

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.8099) |
|---|---|---|---|
| llama.cpp, batched instead (2026-10-11) | 98.489% | 0.00166 | 2.8116 |
| this engine, prefill then decode kernels (`mixed`), q8_0 KV (2026-10-17) | 98.737% | 0.00119 | 2.8085 |
| this engine, prefill kernels only, q8_0 KV (2026-10-17) | 98.744% | 0.00124 | 2.8073 |

The `mixed` report has been the same since 2026-10-15: none of the GPU changes up to v2026.10.30 changed it, and the
native Windows builds give it byte for byte. Most disagreements are near-ties: the median lead of llama.cpp's token over
the engine's is 0.043 nats (llama.cpp batched: 0.051), and 1 of the 168 disagreements exceeds 0.5 nats. At long context
(same reference setup, q8_0 KV): 99.121% / 0.00219 over 4,094 positions of 2 sequences at ~32k (2026-10-17), 99.267% /
0.00163 over 4,095 positions of one 64k sequence (2026-10-15, 7d67c5d), and 99.121% / 0.00217 over 4,095 positions of
one 100k sequence (2026-10-25, 0278d09; more in [100k and 160k context](#100k-and-160k-context)).

### Windows and Linux

The Windows build (MSVC + nvcc) against the Linux build (g++ + nvcc) of the same source (fdad907), on the same model
file, 2026-10-16:

| check | Windows vs Linux |
|---|---|
| greedy token ids, 7 and 0 drafts | identical |
| `--kld` report against llama.cpp (13,299 positions) | identical |
| server replies: text, thinking, an image, an image follow-up turn | identical |
| ptxas registers and spills, every function | identical |
| decode (38k context, chat prompts) and prefill (38k tokens) | within 1.5% |
| image encode, 196 / 300 / 1,024 tokens | 0.59 / 0.81 / 3.56 s vs 0.50 / 0.80 / 3.95 s |

Every release since has passed the same identity checks natively, and since 2026-10-19 also the side-request and
parked-conversation scenarios under q8_0 and preset D and the title scenario under preset D (v2026.10.30: all identical
to the Linux build).

The image encoder's rows first differed between the two builds at fp32 rounding level: its attention took e^x from the
C library, and glibc's and Microsoft's `expf` differ in the last bit on 193,924 of the 2^32 inputs. Since 2026-10-18
the encoder computes it with glibc's algorithm on both, and the Windows rows equal the Linux rows bit for bit (57
images: stills and every frame pair of 4 video clips, at 7 / 8 / 30 / 32 threads).

## The conversation cache

How it works: [SERVER.md](SERVER.md). Costs (WSL2, q8_0 unless noted, host memory already allocated, 2026-10-19):

| operation | 12k tokens | 38k tokens | 101.6k tokens | 120k tokens |
|---|---|---|---|---|
| park a conversation in host memory | 31 ms | 62 ms (D: 63 ms) | 151 ms | 177 ms (4.99 GB) |
| restore it | 29 ms | 65 ms (D: 74-92 ms) | 152 ms | 178 ms |
| read it again instead | 7.4 s | 23.9 s | 78.7 s | 98.4 s |

- The first park of a large conversation pays for page-locked host memory at ~1.1 s per GB (5.7 s for 120k tokens); it
  is kept for reuse until the server exits.
- A side request (a client's title or summary request beside a long conversation) moves only the rows it overwrites:
  12 ms each way for a 2.7k-token side prompt, 20 / 19 ms for 8k (preset D: 17-22 ms), whatever the length of the cached
  conversation (2026-10-19). Its own conversation is parked afterwards: 10-15 ms for a title, ~10 ms back, 15-21 ms for
  ~400 tokens (2026-10-20).
- Two OpenCode 2.0.32 sessions working at the same time on native Windows (230k context, preset D) finished in 110 s
  with ~72k prompt tokens read, against 242 s and ~232k tokens with the host store off (2026-10-19).

### Pinned prefix checkpoints

A new session of a client that keeps its system prompt and tools resumes after them. Measured with each client's
captured first requests (preset D, 131,072 context, native Windows, 2026-10-21): the server before the change (538d441,
with 7322fbd's engine) against the change (merged as 2e659e0 + 4eb7b51). "Live": a 52-55k-token conversation of the
same client is cached; "parked": another 9.8k-token conversation has parked it. Times are the client's wall time,
including the park of the long conversation and ~2.5-3.5 s of first-use host-memory allocation.

| new session (prompt tokens) | | before: cached / time | after: cached / time |
|---|---|---|---|
| Claude Code, default settings (11,850) | live | 0 / 10.12 s | 8,288 / 5.92 s |
| | parked | 0 / 8.67 s | 8,288 / 3.98 s |
| Claude Code with its cache-friendly settings (11,809) | live | 0 / 9.96 s | 9,024 / 5.11 s |
| | parked | 0 / 8.28 s | 9,024 / 3.27 s |
| Codex CLI, `/v1/responses` (8,847) | live | 0 / 8.01 s | 8,672 / 3.44 s |
| | parked | 32 / 6.64 s | 8,672 / 2.00 s |
| Crush, another directory (11,168) | live | 0 / 9.30 s | 10,304 / 4.11 s |
| | parked | 32 / 7.67 s | 10,304 / 2.12 s |

### Replayed client traffic

The requests 9 clients sent during their test drives on 2026-10-21, replayed back to back against a native Windows
server (preset D, 230k context): ef4a6dd (the 2026-10-21 release) against the server before its pinned checkpoints and
side-request rules (538d441, the same engine). The server samples its own replies, so reply lengths differ slightly
from the live runs.

| | before (538d441) | ef4a6dd |
|---|---|---|
| 9 single-client sequences: time reading prompts | 885 s | 800 s |
| tokens read | 1,267k | 1,150k |
| first turns of new agent sessions (50 requests) | 173 s | 120 s |
| DeepSeek Harness / Crush / Qwen Code / Pi / Goose / Claude Code first turns | 29.8 / 30.5 / 62.3 / 8.4 / 5.7 / 27.9 s | 11.3 / 20.2 / 47.2 / 5.2 / 3.1 / 24.3 s |
| Open WebUI first turns | 92 s | 75 s |
| Claude Code + Open WebUI interleaved: tokens read, cache misses | 508k, 28 | 489k, 10 |
| DeepSeek Harness + Aider interleaved: tokens read, cache misses | 266k, 32 | 218k, 7 |

- **The cost:** every parked conversation carries its pinned checkpoint (+288 MiB with preset D), so fewer fit in the
  host store (Claude Code's sequence held 4 instead of 6, and two of its requests cached 4-6k fewer tokens).
- **Title requests** that quote a whole long chat (Open WebUI's default) are too long to run beside it. Since 2efba87
  they no longer cost the chat its place in the cache: replayed on 2efba87 (2026-10-24), five of the sequences spent
  698 s reading prompts instead of 856 s. The trade-off: a chat whose title request looks the same but is never
  continued keeps its place too, and in one sequence it crowded out the pinned system prompt of two new Claude Code
  sessions (11.9k tokens read instead of 3.5k, each). With Open WebUI's title and tags templates set as in
  [CLIENTS.md](CLIENTS.md) (quoting only the end of the chat), these requests are short and run beside the chat.

`--park-min` (among its uses: conversations of fewer tokens are not parked; [SERVER.md](SERVER.md)), both interleaved
sequences, ef4a6dd, fixed seeds:

| `--park-min` | parks | GiB copied | real conversations evicted | tokens read | time reading prompts |
|---|---|---|---|---|---|
| 1024 (the default) | 68 | 106.5 | 15 | 706,771 | 489 s |
| 4096 | 51 | 81.4 | 8 | 709,651 | 489 s |

4096 halved the evictions of real conversations and copied a quarter less for the same prompt time; the default stays
1024.

## Keep-warm

After each request the engine keeps the GPU's clocks up for `MK_KEEP_WARM_S` seconds (default 120). On native Windows
at 147054b, a 255-token reply after 65 s of idle started in ~65 ms instead of 391-397 ms and decoded at 80.2 instead of
61-63 tok/s (2026-10-17). Holding the clocks costs power: 164-173 W instead of 30-35 W at idle on this card
(2026-10-17).

## Images and video

On the Ryzen 9 5900XT (32 threads):
- **Images** (2026-10-16): a 196 / 300 / 1,024-token image encodes in 0.51 / 0.78 / 3.56 s back to back; llama.cpp's
  libmtmd on the same CPU took 0.80 / 1.29 / 6.07 s. A new image after the CPU has been idle takes ~1.2-1.5 s for 300
  tokens while the cores ramp up. The encoder's rows are within 4e-6 to 4e-5 (relative) of an fp64 forward of the
  vision tower (libmtmd: 4e-3 to 2.7e-2). On six image prompts, the mean KL against llama.cpp fed the same image rows is
  0.0006-0.0059 (text only: 0.0012). Prompt tokens equal llama-server's on 7 of 7 image prompts.
- **Video** (2026-10-18; frames sampled at 4 fps as llama.cpp does, two consecutive frames per grid of tokens): a 640 x
  360 clip costs ~450 prompt tokens per second of video (228 at 2 fps) and its frame pairs encode in 0.56 s each, about
  real time at 4 fps; 320 x 240: 167 tokens per second, 0.22 s per pair; 1280 x 720: 1,861 tokens per second, ~3.2 s
  per pair. The prompt token counts equal llama-server's on four clips, and the frame-pair rows are within 5e-6 to
  1.1e-5 of an fp64 forward (libmtmd's: 5.5e-3 to 2.2e-2).

The image and video encoder runs on the CPU, so it takes no VRAM: with an image projector loaded, the idle server's
VRAM at 120k context did not grow (2026-10-16).

## Earlier full runs

Older runs (2026-10-14 and 10-15) are in this file's history at fe21771. The 2026-10-14 run with charts and raw data,
and the gates behind each change up to 2026-10-16, are in the open-jet fork's
[perf docs](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf/perf), where this engine was developed
until it moved here.

### 2026-10-20 (7322fbd)

The same client and settings as the 2026-10-30 run, q8_0 and preset D, one session, fans at 100%. The llama.cpp column
of the [2026-10-30 table](#full-run-2026-10-30) applies here too.

| | this, q8_0 KV | this, preset D |
|---|---|---|
| greedy decode, short prompts (mean / token-weighted) | 155.1 / 131.7 tok/s | 155.5 / 132.1 |
| sampled decode, thinking on (mean / token-weighted) | 136.5 / 130.4 tok/s | 136.8 / 130.8 |
| decode at 11.6k / 38k / 101.6k context | 137.3 / 117.9 / 108.7 tok/s | 136.1 / 101.9 / 98.6 |
| read an 11.6k / 38k / 101.6k-token prompt | 5.98 / 23.2 / 74.5 s | 6.09 / 24.1 / 77.6 s |
| first token, short prompt | 66 ms | 65 ms |
| follow-up turn (11.8k chat) / edit a message / edit the system prompt, prompt time | 178 / 76 / 89 ms | 185 / 76 / 97 ms |
| VRAM, whole card, server idle | 21.1 GiB | 20.0 GiB |
| 282 questions | 230 right, 1,681 s | 230 right, 1,722 s |

Time per engine cycle, q8_0 / D: 27.4 / 28.2 ms at 11.6k, 29.9 / 29.2 at 38k, 33.3 / 34.5 at 101.6k. Preset D's edit
rows and the timing of its GSM8K and MMLU-Pro sets come from a second pass that evening.

## Reproducing

- **Server benchmarks:** [bench/suite](../bench/suite/README.md) has the clients behind the server tables (speed, the
  282 questions, the from-scratch prefill, the 262,144-token run), the scorer and the summary script. The datasets are
  not in this repository: `python -I bench/suite/fetch.py` downloads them and checks every item against the hashes of
  the items these runs used (`bench/suite/items.json`). HumanEval answers are executed only with `score.py
  --exec-humaneval`. The 2026-10-30 run is the suite as published, and its summary script made the table above;
  re-scoring the saved 2026-10-20 answers with the suite's scorer gives the same 230 right answers for q8_0 and
  preset D, item for item.
- **Parallel agents:** the parallel step of [bench/suite](../bench/suite/README.md) (`parallel_client.py`, its context
  fetched by `fetch.py`); without a server, `python3 bench/slots_bench.py <gguf> all` checks the engine's slots against
  each conversation alone and times them.
- **Speed, CLI:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (the four
  chat prompts through the CLI; it runs the CLI's default KV cache, so at most 4 drafts); `bench/code_bench.py` and
  `bench/think_bench.py` are upstream's scripts; `bench/*.cu` are kernel microbenchmarks.
- **Accuracy against llama.cpp:**

  ```sh
  make tools/llama_ref
  tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
  tools/llama_ref -m model.gguf --kld llama.ref                            # llama.cpp batched, for scale
  ./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8     # this engine
  # long context: one sequence of N tokens, its last 4,096 positions scored (here N = 100,352)
  tools/llama_ref -m model.gguf -f pf101k.txt --chunk 100352 --max-chunks 1 --score-from 96256 -b 512 --ref-out long.ref
  MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 ./qwen38_mk -m model.gguf --kld long.ref --score-path mixed --kv q8   # preset F
  ```

- **Draft-depth independence:** `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin
  --ids`, then the same with `-d 4` and `-d 0`: the ids must be identical.

More on each check: [BUILDING.md](BUILDING.md#checking-a-build).
