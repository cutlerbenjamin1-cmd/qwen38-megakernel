# qwen38-megakernel

A CUDA engine and server for **Qwen3.8-27B on one 24 GB NVIDIA GPU**, built and tuned on an RTX 3090. It decodes with
the model's own multi-token-prediction (MTP) head drafting up to 7 tokens per step, speaks the OpenAI, Anthropic and
OpenAI Responses APIs, so coding agents and chat UIs connect to it directly, and keeps their conversations cached
between requests. As the launchers run it, drafting, cache reuse and the choice of Windows or Linux never change a
token of its output.

Built on the megakernel of [open-jet](https://github.com/L-Forster/open-jet) by Louis Forster
([credits](#credits-and-license)).

## Exact output

Speculative decoding is lossless in theory, but in practice verifying several tokens in one pass rounds differently from
verifying one, so greedy output can change with the number of drafts; a reused cache, or the same source compiled by
another compiler, can drift in the same way. This engine is built so that none of these changes the output, and the
release checks test it:

- **Drafting never changes the answer.** Greedy token ids are identical at 0, 4 and 7 drafts. With sampling, the
  output keeps the distribution of sampling without drafts (chi-square and PIT tests). The MTP layer only proposes
  tokens, so greedy output is the same with any MTP layer or none.
- **A cached conversation answers like a fresh one.** With exact reuse (the launchers' default), a follow-up turn, an
  edited conversation, a new session resumed after its system prompt, a conversation restored from host memory and a
  conversation kept through a client's title request all continue bit for bit as the same request sent fresh (greedy,
  and sampled with the same seed). A probe compares every K / V row, ring row and DeltaNet / convolution state with a
  fresh prefill's.
- **Windows and Linux agree bit for bit.** The MSVC and g++ builds give the same token ids, the same accuracy report
  and the same server replies, and since 2026-10-18 the same image-encoder rows.

So an agent's run can be reproduced, an edit or a resumed session gets the answer a fresh request would get, and an odd
reply is never the speculation's or the cache's doing.

What is not bit-identical, and how close it is:
- **llama.cpp.** Its arithmetic is ordered and rounded differently. Scored against llama.cpp's own next-token
  distributions (13,299 positions of code and prose, llama.cpp generating token by token), the engine picks the same top
  token at 98.737% of positions with a mean KL divergence of 0.00119 nats; llama.cpp's own batched mode scores 98.489%
  / 0.00166 against the same reference (q8_0 KV cache; 2026-10-17, unchanged at 7322fbd on 2026-10-20).
- **The low-bit KV presets** ([below](#the-full-262k-context-on-24-gb)): opt-in and near-lossless, not identical to
  q8_0.
- **`MK_EXACT_REUSE=0`** makes follow-up turns faster by appending to the previous reply instead of reading it again.
- **The lean build** (`make MKFLAGS=`, up to 4 drafts) reads the weights with other kernels, so it rounds differently
  from the served build; the checks above run on the served build.

Evidence and the tests behind each line: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#exactness).

## Speed

One RTX 3090 and a Ryzen 9 5900XT, unsloth's UD-Q4_K_XL, the server at 120k context, q8_0 KV cache, 7 drafts, greedy
unless noted. This engine: the release v2026.10.30 (6f3b257), one session on 2026-10-30 (Linux / WSL2). llama.cpp:
b11548's `llama-server`, tuned (q8_0 KV, 4 MTP drafts), the same client, prompts and model file, the best value of each
row over three sessions on 2026-10-12 and 10-13.

| | llama.cpp | this engine |
|---|---|---|
| greedy decode, short prompts (4 x 512 tokens) | 82.7 tok/s | **156.7 tok/s** |
| sampled decode, thinking on (8 runs) | 65.8 tok/s | **138.2 tok/s** |
| decode at 38k tokens of context | 62.3 tok/s | **119.8 tok/s** |
| decode at 101.6k tokens of context | 53.4 tok/s | **109.4 tok/s** |
| read a 101.6k-token prompt | 101.0 s | **75.2 s** |
| first token, short prompt | 124 ms | **66 ms** |
| edit an earlier message in an 11.7k-token chat | 329 ms | **77 ms** |
| 282 benchmark questions (GSM8K, MMLU-Pro, ARC-C, MATH-500, HumanEval, NIAH, LongBench v2) | 229 right, 2,727 s | **230 right, 1,676 s** |

The short-prompt rows are means of per-request rates, as the client reports them; the full table, token-weighted
rates, the preset D and F columns and earlier runs are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Several agents at once (`--agents 4`, Q4_K_M, 2026-10-30): 380 tok/s in total on short prompts and 319 with a
~23k-token context each; one agent alone 183 tok/s with that context and 167-204 on short prompts, depending on how
many sampled drafts it accepts ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#parallel-agents)).

## Picks up where you left off

The server keeps more than the conversation in front of it, and with exact reuse everything it restores answers as the
same request sent fresh would:

- **Follow-ups and edits** resume from DeltaNet checkpoints (one at every message start and every 4,096 tokens). A
  follow-up turn in an 11.8k-token chat cost 176 ms of prompt time (the previous reply is read again, for exactness),
  editing an earlier message in an 11.7k-token chat 77 ms, editing the end of a 9.1k-token system prompt 89 ms
  (llama.cpp: 323, 329 and 956 ms; 2026-10-30).
- **New sessions** of the same client resume after its system prompt and tools: that checkpoint is pinned, so the
  checkpoint pool never evicts it, and it is parked with its conversation. A new Claude Code session reused 8,288 of
  its 11,850 prompt tokens and a new Codex CLI session 8,672 of 8,847, where the server before reused none
  (2026-10-21, preset D).
- **Several conversations** share the engine: a request for another conversation parks the current one in host
  memory, and it comes back when its next request arrives. A 12k / 38k / 120k-token conversation took 31 / 62 / 177 ms
  to park and 29 / 65 / 178 ms to restore, against 7.4 / 23.9 / 98.4 s to read it again (q8_0, 2026-10-19).
- **Title, tag and summary requests** run beside the session (side requests) instead of throwing its cache away.

Replaying the requests that 9 clients sent during their 2026-10-21 test drives (native Windows, preset D, 230k
context), the server with pinned checkpoints and the new side-request rules (ef4a6dd) spent 800 s reading prompts where
the server before them (538d441, same engine) spent 885 s, and the first turns of new agent sessions took 120 s
instead of 173 s. How it works and how to size it: [docs/SERVER.md](docs/SERVER.md).

## Works with your coding agent

The server implements OpenAI chat completions (`/v1/chat/completions`), the Anthropic Messages API (`/v1/messages`,
with token counting), the OpenAI Responses API (`/v1/responses`) and llama-server's endpoints (`/health`, `/props`,
`/v1/models`, `/slots`, `/metrics`, `/tokenize`, `/detokenize`, `/apply-template`), with streaming, tool calls streamed
as they are generated, thinking, images and video (encoded on the CPU, so they take no VRAM). These clients ran test
tasks against it (the agents: create a file, find and fix a failing test, follow-ups, questions about a large file, an
image):

| client | version tested | client | version tested |
|---|---|---|---|
| Claude Code (Anthropic Messages) | 2.1.306 | Goose | 1.55.0 |
| Codex CLI (OpenAI Responses) | 0.163.0 | Hermes Agent | v0.22.3 |
| Qwen Code | 0.25.3 | Aider | 0.86.2 |
| Pi | 1.3.0 | Open WebUI | 0.12.3 |
| DeepSeek Harness (`dsh`) | 0.2.1-rc.2 | OpenCode | 2.0.32 |
| Crush | v0.98.1 | | |

The others use chat completions. All ran on native Windows with preset D at 230k context, on 2026-10-21 (OpenCode on
2026-10-18 to 10-20). Each needs the base URL, any API key and the model id (the GGUF's file name); several also need
the context window or image support set by hand. Per-client settings and quirks: [docs/CLIENTS.md](docs/CLIENTS.md).

## The full 262k context on 24 GB

Qwen3.8-27B reads up to 262,144 tokens. With the default q8_0 KV cache, an idle server at 120k context and the desktop
took 21.3 GB of the card; **preset D** fits the whole 262,144 on 24 GB. It stores K at 6 bits and V at 5 bits, and
keeps the most recent 2,048 positions in an 8-bit ring, because that is where low-bit storage costs accuracy. It is
opt-in:

```sh
MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144
```

What it costs, from the 2026-10-30 run (v2026.10.30), with preset F beside it (KL: mean KL divergence against
llama.cpp, as above, measured on 2026-10-20 and 10-25):

| | q8_0 | preset D | preset F |
|---|---|---|---|
| card memory, idle server at 120k context | 21.3 GB | **20.1 GB** | **20.0 GB** |
| KL at 32k context (an fp16 cache, 2026-10-12: 0.00226) | 0.00219 | 0.00236 | 0.00242 |
| the 282 questions | 230 right | 230 right | 229 right |
| read a 101.6k-token prompt | 75.2 s | 78.1 s | 77.9 s |
| decode time per engine cycle, 11.6k to 101.6k context | | -2% to +4% | +0.5% to +5% |

At 262,144 tokens of context (2026-10-20) the card held 23.5 GB; a 260,896-token prompt was read in 326 s, decode at
that depth ran at 66 tok/s, a follow-up turn took 1.3 s, and 4 of 4 needle-in-a-haystack and 4 of 6 LongBench v2
questions at that length were right. Natively on Windows, with the desktop on the same card, D at the full context
leaves the engine just past what the driver keeps resident, so above 240k `scripts\serve.bat` uses preset F instead: K
at 5 bits, rotated, 0.58 GB smaller than D at 262k, KL equal to q8_0's at 64k and 100k and 10% above it at 32k (D: 8%).
There the prompt took 340 s and decode at that depth ran at 66 tok/s (2026-10-25). At 64k context the measured KL of D
with an 8-bit MTP layer (preset D') was 0.00161, against q8_0's 0.00163 and the fp16 cache's 0.00161 (2026-10-13 /
10-15); at 100k, q8_0, D and F read 0.00217, 0.00226 and 0.00218, and on 160k tokens of prose 0.00467, 0.00468 and
0.00472 (2026-10-25; [100k and 160k context](docs/BENCHMARKS.md#100k-and-160k-context)).

While a conversation is shorter than the window, preset D keeps all of it at 8 bits and its output is q8_0's. Past
the window, its KL at 32k is 8% above q8_0's (an fp16 cache, more precise than both, scores 3% above q8_0 on the same
set), it scored the same on the 282 questions, set by set, and it read prompts about 4% slower. Its lower
long-context decode rates in the full table come mostly from different reply text accepting fewer drafts (2.98 instead
of 3.53 tokens per cycle at 38k); per cycle it ran within the table's -2% to +4%.

Our recommendation: for long-context work, or to free a gigabyte for something else, turn it on; the cost we can
measure is small. It stays opt-in because it is not bit-identical to q8_0. Every preset and its numbers:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#kv-cache-presets).

## Quick start

**Prebuilt packages** for Windows and Linux are on the
[releases page](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases), built for RTX 30-series GPUs (sm_86;
RTX 40-series cards can run that code but are untested): unpack one, `pip install -r requirements.txt`, download the
model as below and run `scripts/serve.sh` or `scripts\serve.bat`. What they need:
[docs/BUILDING.md](docs/BUILDING.md#prebuilt-packages).

To build from source on Linux or WSL2 (CUDA 13.2, llama.cpp next to this repository for its tokenizer and image
library):

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout b10246
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build ../llama.cpp/build -j
make -j2                          # the served configuration: about 11 minutes, up to ~19 GiB of RAM in ptxas
pip install -r requirements.txt
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
scripts/serve.sh                  # http://127.0.0.1:8080: 120k context, 7 drafts, exact reuse, images on
```

Windows 10 / 11 (Visual Studio 2022 Build Tools, CUDA 13.2, CMake + Ninja, Python), in PowerShell from an "x64 Native
Tools" prompt, then the server from cmd:

```powershell
git clone https://github.com/ggml-org/llama.cpp ..\llama.cpp; git -C ..\llama.cpp checkout b10246
cmake -S ..\llama.cpp -B ..\llama.cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build ..\llama.cpp\build --target llama mtmd
.\build.ps1                       # uses $env:CUDA_PATH, or -Cuda <toolkit folder>
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir D:\models
```

```bat
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
set MK_PYTHON=%CD%\.venv\Scripts\python.exe
scripts\serve.bat
```

Then ask it something:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write quicksort in C."}], "max_tokens": 512}'
```

The launchers take the port and the context size as arguments (`scripts/serve.sh 8080 64000`) and everything else from
environment variables (`MK_HOST=0.0.0.0` for the LAN, `MK_KV` / `MK_KV_WIN` for preset D, ...). Building in detail:
[docs/BUILDING.md](docs/BUILDING.md). Every setting: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Requirements

- **GPU:** NVIDIA Ampere or newer with 24 GB and at least 80 SMs for more than 4 drafts (50 for up to 4). Tuned and
  tested on an RTX 3090 (sm_86); other architectures build but are untested.
- **CUDA 13.2** (older 12.x versions are untested), Linux, WSL2 or Windows 10 / 11 x64.
- **RAM:** ~20 GiB free to compile the served configuration; at run time, page-locked host memory for parked
  conversations up to `MK_PARK_MIB` (8 GiB by default, allocated as needed).
- **llama.cpp b10246** next to this repository, **Python 3.10+** with jinja2, an x86-64 CPU with AVX2 + FMA for the
  fast image encoder, and ffmpeg for video (optional).

Details: [docs/BUILDING.md](docs/BUILDING.md#requirements).

## Documentation

- [docs/BUILDING.md](docs/BUILDING.md): requirements, the Linux and Windows builds, model files, checking a build.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): every launcher setting, server option, environment variable and
  build flag.
- [docs/SERVER.md](docs/SERVER.md): the APIs, the conversation cache (exact reuse, checkpoints, parking, side
  requests), the console, limits and security.
- [docs/CLIENTS.md](docs/CLIENTS.md): settings for each coding agent and chat UI.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): every measurement, with its date, commit and method.
- [docs/ENGINE.md](docs/ENGINE.md): how the engine works inside.

## Credits and license

This engine started as the megakernel of [open-jet](https://github.com/L-Forster/open-jet) by Louis Forster
(L-Forster). Upstream wrote its design and first version, the first commits of this history (4d92fd2 to 0f40d41,
2026-10-09 to 10-11): the persistent kernel that runs a whole speculative decode cycle in one launch, MTP drafting with
on-device verification (greedy and exact speculative sampling), the batched tensor-core prefill, the Q4_K_M weight
kernels, the draft vocabulary, the C API, CLI and OpenAI-compatible server with prefix reuse, the benchmark scripts, and
the accuracy method used above (`tools/llama_ref`: KL against llama.cpp's token-by-token distributions, with
llama.cpp's batched run for scale). Evan Light contributed the load-time tensor-type check and the server warm-up
(open-jet PRs #14 and #16).

This repository is open-jet's `megakernel/` directory, split out with its history from our fork's
[`qwen38-perf`](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf) branch at 62c8d7a (2026-10-16), and
development continues here. Since upstream's version it gained UD-Q4_K_XL support, draft-depth-independent greedy
output, wide verification on int8 tensor cores with up to 7 drafts, the low-bit KV presets, exact reuse, parked
conversations, side requests and pinned checkpoints, the CPU vision encoder and video input, the Anthropic and
Responses APIs, and the native Windows build. Changes that went back upstream include UD-Q4_K_XL support (PR #19) and
`token_embd` in host memory with a configurable prefill chunk (PR #20), merged on 2026-10-12, and
draft-depth-independent greedy output, the pipelined decode attention, the int8 prefill attention and the low-bit KV
presets (PRs #24 to #27), merged on 2026-10-29. All of them, merged and open:
[our pull requests to open-jet](https://github.com/L-Forster/open-jet/pulls?q=is%3Apr+author%3Acutlerbenjamin1-cmd).

It also stands on [llama.cpp](https://github.com/ggml-org/llama.cpp): its tokenizer (libllama), its image library
(libmtmd: the fallback encoder and the reference tools), its GGUF quantization formats, whose dequantization the
kernels follow, and its prompt layout for images and video. The model files and the default sampling settings are
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s; the model, Qwen3.8-27B, is Qwen's.

Licensed under the GNU AGPL v3, like open-jet ([LICENSE](LICENSE)).
