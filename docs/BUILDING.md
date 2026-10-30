# Building

The same sources build on Linux / WSL2 (g++ + nvcc, `make`) and natively on Windows (MSVC + nvcc, `build.ps1`), and the
two builds give the same output bit for bit ([BENCHMARKS.md](BENCHMARKS.md#windows-and-linux)). Every build setting
named here is described in [CONFIGURATION.md](CONFIGURATION.md).

Contents: [prebuilt packages](#prebuilt-packages) - [requirements](#requirements) - [Linux and WSL2](#linux-and-wsl2) -
[Windows](#windows) - [model files](#model-files) - [checking a build](#checking-a-build) - [files](#files).

## Prebuilt packages

Each [release](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases) has a Windows x64 zip and a Linux
x86-64 tarball of the served configuration (`MKFLAGS` as below: wide verify, up to 7 drafts), built from the tagged
commit. Each holds the engine library, the CLI, the vision library, the tools, llama.cpp b10246's libraries (CPU only:
the engine uses its tokenizer and image library), the server, the launcher, the docs, the licenses
(`THIRD_PARTY_NOTICES.md`), `BUILD_INFO.txt` (commit, flags, toolchain) and `QUICKSTART.txt`. They need:

- **GPU:** the engine is compiled for sm_86 (RTX 30-series; tested on an RTX 3090) and needs at least 80 SMs for more
  than 4 drafts. Compute capability 8.9 GPUs (RTX 40-series) can run sm_86 code but are untested; RTX 50-series
  (sm_120) and other architectures need a source build.
- **Driver:** one that supports CUDA 13.2 (`nvidia-smi` shows "CUDA Version: 13.2" or higher). The CUDA runtime is
  linked into the engine; nothing else from CUDA is needed.
- **CPU:** x86-64 with AVX2 (llama.cpp's libraries in the packages use it).
- **Python** 3.10+ with `pip install -r requirements.txt`, for the server.
- **Linux:** x86-64 with glibc 2.38 or newer and GCC 13's libstdc++ (built on Ubuntu 24.04: Ubuntu 24.04, Debian 13,
  Fedora 39 and newer qualify). Under WSL2 the Windows driver provides CUDA.
- **Windows:** 10 / 11 x64. The Visual C++ runtime is included.

The model is not included: download it as under [model files](#model-files), then follow `QUICKSTART.txt` (the same
steps as the README's quick start, without the build).

The packages are made with `scripts/package_release.sh` (after `make`, against a llama.cpp built CPU-only and without
`-march=native`: `cmake -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON
-DGGML_F16C=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF`; it needs `patchelf`, which re-points the binaries at their own
folder) and `scripts\package_release.ps1` (after `build.ps1`; it bundles the Visual C++ runtime from Visual Studio
2022's redistributable folder).

## Requirements

- **GPU:** NVIDIA Ampere or newer (the kernels use sm_80 features) with 24 GB of VRAM. The persistent kernel runs one
  block per SM and needs at least 80 SMs for more than 4 drafts, 50 for up to 4; the engine checks this at load and
  names the shortfall. Tuned and tested on an RTX 3090 (sm_86, 82 SMs). `make` builds for the installed GPU's
  architecture (sm_89 or sm_120 too), but only sm_86 is tested.
- **CUDA:** 13.2 is what we build with; older 12.x versions are untested. The Windows build needs a driver that supports
  CUDA 13.2; it links the CUDA runtime statically, so at run time only the driver is needed.
- **OS:** Linux, WSL2, or Windows 10 / 11 x64.
- **RAM to compile:** most of the build is ptxas on the megakernel, one thread with a large heap. For the served
  configuration it peaked at 18.8 GiB both on Linux / WSL2, in a 10.5-minute `make -j2`, and natively on Windows, in
  a 12-minute `build.ps1` (2026-10-30, 6f3b257). The lean build (`MKFLAGS=`, up to 4 drafts) was last measured at
  about 19 GB (2026-10-17), before the served build was trimmed.
- **RAM at run time:** parked conversations and side requests use page-locked host memory, up to `--park-mib` /
  `MK_PARK_MIB` (default 8192 MiB; 1024 MiB for side requests alone when parking is off), allocated as it is needed and
  kept until the server exits. The token embedding table also stays in page-locked host memory instead of VRAM.
- **llama.cpp b10246**, built next to this repository (`../llama.cpp`). The engine uses its tokenizer (libllama, loaded
  for the vocabulary only); the image library uses libmtmd as its fallback encoder, and the reference tools use both.
- **Python 3.10+** with jinja2 (`requirements.txt`) for the server; pytest for the tests.
- **CPU for the fast image encoder:** x86-64 with AVX2 + FMA (x86-64-v3: Zen 2+, Haswell+). The encoder is compiled for
  it in its own object, so the library still loads on older CPUs, which then use libmtmd.
- **ffmpeg** for video input only (optional): on `PATH`, named by `MK_FFMPEG` / `--ffmpeg`, or the one bundled with
  `pip install imageio-ffmpeg`. Without it the server answers video requests with a 400.

## Linux and WSL2

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout b10246
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build ../llama.cpp/build -j
make -j2        # qwen38_mk (CLI), libqwen38mk.so (server), tools/, libqwen38vision.so
pip install -r requirements.txt
```

What `make` builds:
- `qwen38_mk`: the command-line engine (benchmarks, accuracy checks, `--ids`).
- `libqwen38mk.so`: the engine and tokenizer behind a C API (`capi.cpp`), loaded by `server/mk_server.py`.
- `tools/vocab_freq` (builds a draft vocabulary) and `tools/llama_ref` (llama.cpp's next-token distributions for the
  accuracy check).
- `libqwen38vision.so`, `tools/mkv_cli` and `tools/vision_ref` (the CPU image encoder and its tools), when
  `../llama.cpp/build/bin/libmtmd.so` exists, which is llama.cpp's default; `make vision` builds them explicitly.

Build settings:
- **The served configuration** is the default: `MKFLAGS=-DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1 -DMK_K5=1` (wide verify on
  int8 tensor cores, up to 7 drafts, the fragment-ordered Q6_K layout, the split barrier before the DeltaNet core).
- **The lean engine:** `make MKFLAGS=`. It verifies at most 5 rows (4 drafts) with dp4a GEMVs; run it with
  `MK_DRAFT=4` (the launchers default to 7).
- **Architecture:** `make` asks `nvidia-smi` for the first GPU's compute capability (sm_86 if it is missing); override
  it with `ARCH=sm_89` or similar.
- **Huge pages:** the Makefile sets `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, which lets ptxas's heap use transparent
  huge pages and runs it ~20% faster with the same machine code (glibc 2.35+; it needs THP in `madvise` or `always`
  mode, `/sys/kernel/mm/transparent_hugepage/enabled`; nothing changes where THP is off).
- **Reports:** `ptxas.log` lists registers and spills for every function. `build/mk_build_info.h` records the commit and
  the engine flags, which the server reports in `/props`.

Under WSL2, keep the model on the Linux filesystem: GGUF files on `/mnt/c` (or any Windows drive) load several times
slower. With the model on ext4 the server was ready ~12 s after start (2026-10-16). WSL2's memory comes out of the same
pool as Windows', so a compile on one side and a loaded model on the other can exhaust it: a native ptxas failed with a
memory allocation error while a WSL2 GPU job held ~20 GB (2026-10-16).

## Windows

Requirements on top of the list above:
- **Compiler:** Visual Studio 2022 Build Tools with the C++ x64 tools (MSVC v143). CUDA 13.2 does not accept VS 2026's
  compiler.
- **CUDA 13.2:** the installer, or NVIDIA's redistributable archives (`cuda_nvcc`, `cuda_cudart`, `cuda_cccl`,
  `cuda_crt`, `libnvvm`; `cuda_cuobjdump` for the checks) unpacked into one folder.
- **Tools:** CMake and Ninja for llama.cpp, and Python 3.10+.

Build, in PowerShell, from an "x64 Native Tools" prompt for VS 2022 (or after `vcvars64.bat`), with llama.cpp next to
this repository:

```powershell
git clone https://github.com/ggml-org/llama.cpp ..\llama.cpp; git -C ..\llama.cpp checkout b10246
cmake -S ..\llama.cpp -B ..\llama.cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build ..\llama.cpp\build --target llama mtmd
.\build.ps1 -Cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"   # the default is $env:CUDA_PATH
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

- **llama.cpp:** a CPU-only build is enough here: the engine uses only its tokenizer and, for images, libmtmd.
- **What `build.ps1` builds:** `qwen38mk.dll` (the server's engine), `qwen38_mk.exe` (CLI), `qwen38vision.dll` and
  `tools\*.exe`, with the Makefile's nvcc flags and `MKFLAGS`. It writes `ptxas.log` and copies llama.cpp's DLLs next
  to the outputs. It finds Visual Studio 2022 itself (vswhere) and sets up its environment.
- **Parameters:** `-MkFlags ""` builds the lean engine; `-Targets engine|vision|tools` builds one part (`vision`
  needs no CUDA compile); `-Arch sm_86` overrides the GPU architecture (default: the first GPU's); `-Llama` points at
  another llama.cpp build; `-Cuda` at the toolkit. An engine object that is newer than the sources and was built with
  the same flags is reused.
- **The Windows-only files** are `compat_win32.h` / `.cpp` and `vision/win32.h` / `.cpp` (POSIX / Win32 shims); the
  Linux build does not compile them.

Run (cmd):

```bat
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
set MK_PYTHON=%CD%\.venv\Scripts\python.exe
scripts\serve.bat                 :: http://127.0.0.1:8080, 120k context, q8_0 KV, 7 drafts
```

`scripts\serve.bat` takes the same settings as `scripts/serve.sh` ([CONFIGURATION.md](CONFIGURATION.md)). Images are
on when `mmproj-BF16.gguf` sits next to the model or `MK_MMPROJ` names it (`MK_MMPROJ=off` turns them off). From an
NVMe disk the server was ready ~7 s after start (2026-10-16).

## Model files

The engine reads Qwen3.8-27B GGUFs (llama.cpp architecture `qwen35`) with its own parser and loader, and checks every
tensor's type and shape at load.

```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
```

| file | status |
|---|---|
| unsloth's `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.6 GB) | the default; the engine is tuned on it |
| unsloth's `Qwen3.8-27B-Q4_K_M.gguf` | runs on the same engine ([numbers](BENCHMARKS.md#q4_k_m-and-upstreams-scripts)) |
| another quantizer's Qwen3.8-27B Q4_K_M | runs when every tensor's type is supported; untested |
| fine-tunes of Qwen3.8-27B (abliterated or uncensored conversions, ...) | run when every tensor's type is supported |
| a file without the MTP layer | runs without drafting, or with `--mtp` and a separate MTP file |
| any other quantization | refused at load, naming the first tensor it cannot read (use llama.cpp) |

- **Tensor types:** every projection of the trunk and the MTP layer can be Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, Q3_K, IQ3_S
  or IQ4_NL; `token_embd` Q4_K, Q6_K or Q8_0; `output` Q6_K or Q8_0. How each is read, and the few other cases:
  [ENGINE.md](ENGINE.md#the-gguf).
- **The MTP layer** (`blk.64`) is what the engine drafts with. Some conversions drop it. Without one the engine loads
  with a notice and decodes without drafting (depth 0, about a third of the drafted speed), with no MTP weights and no
  MTP KV slot; `--no-mtp` (`MK_NO_MTP=1`) does the same for a file that has one. `--mtp FILE` (`MK_MTP=FILE` in the
  environment) takes the MTP layer from another GGUF of the same model, such as a separate `mtp-*.gguf`; the engine
  checks that it belongs to the model (architecture, sizes, vocabulary) and re-quantizes its matrices to Q4_K like the
  built-in layer's. The MTP layer only proposes drafts, which the main model verifies: the output is the same with any
  MTP layer or none (greedy bit for bit), only the acceptance rate, and with it the speed, changes.
- **The vision projector:** use `mmproj-BF16.gguf`. The encoder widens any file to fp32, so a Q8_0 projector is no
  faster and is less accurate.
- **Chat template:** the one stored in the GGUF; `--chat-template-file` (`MK_CHAT_TEMPLATE_FILE` in the launchers)
  replaces it, as in llama-server ([SERVER.md](SERVER.md)).

## Checking a build

- **Server tests:** `python -m pytest -q tests` runs without a GPU, against fake engines (the CI runs it on Python 3.10,
  3.11 and 3.12).
- **GGUF parser:** `make tests/gguf_fuzz && tests/gguf_fuzz` feeds `gguf.h` malformed and randomly corrupted files under
  AddressSanitizer and UBSan (CPU only, Linux).
- **Draft-depth independence:** the same prompt at `-d 0`, `-d 4` and `-d 7` must give identical ids:
  `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin --ids`
- **Accuracy against llama.cpp:** `tools/llama_ref` writes a reference file, and
  `./qwen38_mk -m <gguf> --kld <ref> --score-path mixed --kv q8` scores the engine against it
  ([BENCHMARKS.md](BENCHMARKS.md#against-llamacpp) has the full recipe and the expected numbers).
- **Images:** `tools/mkv_cli` encodes images and writes their rows. `tools/vision_ref` writes llama.cpp's reference for
  an image prompt plus the rows it fed, and `qwen38_mk --kld <ref> --kld-images <rows>` scores it.
- **Speed:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (4 chat
  prompts through the CLI; it uses the CLI's default KV cache, so at most 4 drafts). `bench/*.cu` are kernel
  microbenchmarks.
- **Windows against Linux:** the ids of the depth check, the `--kld` report and server replies to the same requests
  should be identical between a Windows and a Linux build of the same commit.

## Files

- `qwen38_mk.cu`, `qwen38_mk.h`, `prefill*.cuh`, `wv*.cuh`, `loader.cuh`, `gguf.h`: the engine
- `capi.cpp`: the C API of `libqwen38mk.so` / `qwen38mk.dll` (and the keep-warm thread)
- `main.cu`: the `qwen38_mk` CLI; `logit_stats.h`: the accuracy comparison shared by the CLI and the tools
- `server/mk_server.py`: the HTTP server
- `scripts/serve.sh`, `scripts/serve.bat`: the launchers (Linux / WSL2, Windows)
- `Makefile`; `build.ps1`, `compat_win32.h` / `.cpp`, `vision/win32.h` / `.cpp`: the Windows build and its POSIX / Win32
  shims (compiled only on Windows)
- `vision/`: the CPU image encoder (`libqwen38vision.so`), its CLI and its checks
- `tools/`: llama.cpp reference tools and the draft-vocabulary builder
- `bench/`: benchmarks (upstream's scripts and kernel microbenchmarks)
- `tests/`: server tests, the GGUF fuzzer, the sampling replica (`spec_rng_sim.py`) and chat templates
- `draft_vocab_32k.bin`: the MTP draft head's token subset
- `docs/`: this documentation
