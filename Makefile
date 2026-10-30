LLAMA ?= ../llama.cpp
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50); sm_86 if nvidia-smi is missing
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
ARCH := sm_$(patsubst sm_%,%,$(ARCH))
# engine configuration: wide tensor-core verify, up to 7 drafts (-d 7), fragment-ordered Q6_K = the served build.
# ptxas peaks at ~19 GiB of RAM with it (2026-10-30); `make MKFLAGS=` builds the lean engine (up to 4 drafts: -d 4).
MKFLAGS ?= -DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1 -DMK_K5=1
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC $(MKFLAGS)
# Most of the build is ptxas on the megakernel (one thread, ~19 GiB heap). It runs ~20% faster when glibc's malloc asks
# for transparent huge pages, which this tunable does (glibc 2.35+; nothing changes where THP is off). Same machine code.
export GLIBC_TUNABLES ?= glibc.malloc.hugetlb=1

all: qwen38_mk libqwen38mk.so tools/vocab_freq tools/llama_ref $(if $(wildcard $(LLAMA)/build/bin/libmtmd.so),vision)

# CPU vision encoder (no CUDA, no GPU memory) for server/mk_server.py --mmproj: vision/mk_vision.cpp (C API, image
# decoding, libmtmd backend) + on x86-64 vision/vit_cpu.cpp (the AVX2 / FMA backend, compiled for x86-64-v3 in its
# own object so that the library still loads on older CPUs, which then use libmtmd; -ffp-contract=off keeps the
# preprocessing's arithmetic exactly llama.cpp's). Built when llama.cpp's build has libmtmd (LLAMA_BUILD_TOOLS, its
# default); `make vision` builds it explicitly.
vision: libqwen38vision.so tools/mkv_cli tools/vision_ref
VISION_INC = -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/tools/mtmd -I$(LLAMA)/vendor
VISION_LIBS = -L$(LLAMA)/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$(abspath $(LLAMA)/build/bin)
ifeq ($(shell uname -m),x86_64)
VIT_OBJ = vision/vit_cpu.o
VIT_DEF =
else
VIT_OBJ =
VIT_DEF = -DMKV_NO_VIT
endif

vision/vit_cpu.o: vision/vit_cpu.cpp vision/vit_cpu.h
	g++ -O3 -std=c++17 -fPIC -march=x86-64-v3 -mtune=znver3 -ffp-contract=off -pthread $(VISION_INC) -c $< -o $@

libqwen38vision.so: vision/mk_vision.cpp vision/mk_vision.h $(VIT_OBJ)
	g++ -O2 -std=c++17 -shared -fPIC -pthread $(VIT_DEF) $(VISION_INC) vision/mk_vision.cpp $(VIT_OBJ) $(VISION_LIBS) -o $@

tools/mkv_cli: vision/mkv_cli.cpp vision/mk_vision.h libqwen38vision.so
	g++ -O2 -std=c++17 -Ivision $< -L. -lqwen38vision -Wl,-rpath,'$$ORIGIN/..' $(VISION_LIBS) -o $@

# test: vit_cpu against libmtmd (pixels, rows, time) on images; uses libmtmd's internal clip API (not shipped)
tools/vit_check: vision/vit_check.cpp vision/vit_cpu.o
	g++ -O2 -std=c++17 -pthread $(VISION_INC) -Ivision $< vision/vit_cpu.o $(VISION_LIBS) -o $@

# llama.cpp's next-token distributions for an image prompt + the image rows it fed (qwen38_mk --kld ... --kld-images)
tools/vision_ref: tools/vision_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread $(VISION_INC) $< $(VISION_LIBS) -o $@

# ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu qwen38_mk.h gguf.h prefill.cuh prefill_gemm.cuh prefill_attn.cuh loader.cuh wv.cuh wv_host.cuh
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

qwen38_mk: qwen38_mk.o main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

# the build's commit and engine flags (capi.cpp mk_build_info, the server's /props); rewritten only when they change
build/mk_build_info.h: FORCE
	@mkdir -p build
	@printf '#define MK_BUILD_COMMIT "%s"\n#define MK_BUILD_FLAGS "%s"\n' \
		"$$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$$(git diff --quiet HEAD -- 2>/dev/null || echo -dirty)" \
		"$(strip $(MKFLAGS))" > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv $@.tmp $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libqwen38mk.so: qwen38_mk.o capi.cpp qwen38_mk.h build/mk_build_info.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -Ibuild -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

# llama.cpp's next-token distributions for the accuracy check (see "Correctness" in README.md)
tools/llama_ref: tools/llama_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

# test: the GGUF parser (gguf.h) on malformed files under AddressSanitizer / UBSan, CPU only: tests/gguf_fuzz
tests/gguf_fuzz: tests/gguf_fuzz.cpp gguf.h
	g++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-sanitize-recover=undefined $< -o $@

clean:
	rm -f qwen38_mk libqwen38mk.so qwen38_mk.o ptxas.log tools/vocab_freq tools/llama_ref libqwen38vision.so tools/mkv_cli tools/vision_ref tools/vit_check vision/vit_cpu.o tests/gguf_fuzz

.PHONY: all vision clean FORCE
