#!/usr/bin/env python3
"""GPU checks of the C API: engine errors and their roll-backs, penalties, policies (scripts/gate.sh runs them).
usage: capi_checks.py <libqwen38mk.so> <model.gguf> <draft_vocab> [case]

Without a case it runs every case in its own process and prints one verdict each: PASS, FAIL, or SKIP with the reason (a
case that cannot test what it is for on this machine). Exit status 1 when any case FAILed or crashed, else 0.

The out-of-memory cases never ask the device for more than its free VRAM, so they need no spill and give the same answer on
every run (on WDDM / WSL2 the driver's CUDA sysmem fallback lets an allocation past the VRAM succeed in system memory, charged
to Windows commit, so "ask for 30 GB and expect a failure" spilled 20-28 GB and failed only when Windows commit ran out):
  slots_oom / ctor_oom      the out-of-memory comes from the device's current memory pool, capped (cuMemPoolCreate maxSize)
                            far below the free VRAM: the engine's own cudaMallocAsync fails part way, on any driver.
  slots_devoom / ctor_devoom  a device out-of-memory: the VRAM is filled to a known margin (a cuMemAlloc sized from
                            cuMemGetInfo); the pool is capped at the margin + 1 GiB, so with the sysmem fallback on at most
                            1 GiB goes past the VRAM, and then the case SKIPs ("sysmem fallback active").
On WSL2 every device allocation is charged to Windows commit (also inside the VRAM): the devoom cases start only with
CAPI_COMMIT_MIN_GB (default 45) of Windows commit free. CAPI_DEVOOM=0 skips them."""
import ctypes
import os
import subprocess
import sys

LIB, MODEL, DV = sys.argv[1:4]
CASES = ("penalty", "slots_oom", "slots_fit", "ctor_oom", "slots_devoom", "ctor_devoom", "policy")
PROMPT = b"<|im_start|>user\nWrite a short poem about a lighthouse.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
vp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
SKIP_RC = 77  # a case's exit code for SKIP (automake's convention)
MiB = 1 << 20
GiB = 1 << 30
# the slots of the out-of-memory cases: 2 x 40000 positions, ~1.7 GB a slot on the 27B (q8 KV), so with ~2.5 GiB left the
# first slot is built and the second fails part way; the retry (2 x 4096, ~0.37 GB a slot) fits in what the first freed
SLOT_N, SLOT_CTX = 2, 40000
CAP_SLOTS = 2560 * MiB   # slots_oom: the pool cap
CAP_CTOR = 3 * GiB       # ctor_oom: the pool cap (engine 2 needs ~17 GB)
MARGIN = 2560 * MiB      # the devoom cases: free VRAM left beside the filler
OVER = 1 * GiB           # the devoom cases: the pool cap over the margin (the most that can go past the VRAM)
HEADROOM = 1536 * MiB    # free VRAM a capped case keeps outside its cap (non-pool allocations, the runtime)
LEFT_MIB = 64            # device memory a failed call may leave behind (MiB; rounding, the runtime's own)
CTX_MIB = 512            # device memory the first engine may leave behind after its free (MiB: the engine's module code
                         # and the context's local memory stay loaded in the process)


# ---------------------------------------------------------------------------------------------------------------- driver API
class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class PoolProps(ctypes.Structure):  # CUmemPoolProps (cuda.h, CUDA 12.2+: maxSize; 88 bytes)
    _fields_ = [("allocType", ctypes.c_int), ("handleTypes", ctypes.c_int), ("location", Loc),
                ("win32SecurityAttributes", ctypes.c_void_p), ("maxSize", ctypes.c_size_t), ("usage", ctypes.c_ushort),
                ("reserved", ctypes.c_ubyte * 54)]


class Drv:
    """the CUDA driver API on the device's primary context (the one the engine's runtime uses)"""

    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.dev = ctypes.c_int()
        self.ctx = ctypes.c_void_p()
        for r in (self.cu.cuInit(0), self.cu.cuDeviceGet(ctypes.byref(self.dev), 0),
                  self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), self.cu.cuCtxSetCurrent(self.ctx)):
            if r != 0:
                raise RuntimeError(f"CUDA driver API setup failed ({r})")
        self.dflt = ctypes.c_void_p()
        self.cu.cuDeviceGetDefaultMemPool(ctypes.byref(self.dflt), self.dev)
        self.pool = None

    def free(self):
        """free device memory, bytes (after the device is idle)"""
        self.cu.cuCtxSynchronize()
        fr, tot = ctypes.c_size_t(), ctypes.c_size_t()
        self.cu.cuMemGetInfo_v2(ctypes.byref(fr), ctypes.byref(tot))
        return fr.value

    def stats(self, pool=None):
        """the pool's reserved / used, current and high-water (MiB)"""
        p = pool or self.pool
        vals = []
        for a in (5, 6, 7, 8):  # CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, _HIGH, USED_MEM_CURRENT, _HIGH
            v = ctypes.c_uint64()
            self.cu.cuMemPoolGetAttribute(p, a, ctypes.byref(v))
            vals.append(v.value // MiB)
        return dict(zip(("reserved", "reserved_high", "used", "used_high"), vals))

    def _pool(self, nbytes):
        pr = PoolProps()
        pr.allocType, pr.handleTypes, pr.location.type, pr.location.id, pr.maxSize = 1, 0, 1, self.dev.value, nbytes
        pool = ctypes.c_void_p()
        return pool if self.cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(pr)) == 0 else None

    def cap_works(self):
        """does the driver enforce a pool's maxSize? (a 64 MiB pool must refuse one 128 MiB allocation; nothing past the
        free VRAM is asked for either way)"""
        pool = self._pool(64 * MiB)
        if not pool:
            return False
        p = ctypes.c_uint64()
        r = self.cu.cuMemAllocFromPoolAsync(ctypes.byref(p), ctypes.c_size_t(128 * MiB), pool, None)
        if r == 0:
            self.cu.cuMemFreeAsync(p, None)
        self.cu.cuCtxSynchronize()
        self.cu.cuMemPoolDestroy(pool)
        return r != 0

    def cap(self, nbytes):
        """a pool of at most nbytes on the device, made its current pool (the engine's cudaMallocAsync source). None when
        the driver does not enforce the cap (checked first, see cap_works) or the engine does not allocate from pools."""
        if os.environ.get("MK_CUDAMALLOC", "")[:1] == "1" or not self.cap_works():
            return None
        pool = self._pool(nbytes)
        if not pool:
            return None
        self.cu.cuDeviceSetMemPool(self.dev, pool)
        self.pool = pool
        return pool

    def trim(self):
        """the capped pool's free memory back to the device"""
        self.cu.cuCtxSynchronize()
        if self.pool:
            self.cu.cuMemPoolTrimTo(self.pool, ctypes.c_size_t(0))

    def uncap(self):
        """the default pool current again, the capped one destroyed (call after everything allocated from it is freed)"""
        self.cu.cuCtxSynchronize()
        self.cu.cuDeviceSetMemPool(self.dev, self.dflt)
        if self.pool:
            self.cu.cuMemPoolDestroy(self.pool)
            self.pool = None

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        return p if self.cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0 else None

    def release(self, p):
        if p:
            self.cu.cuMemFree_v2(p)


# ------------------------------------------------------------------------------------------------------------------ engine
def smi(field):
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def vram():
    return smi("memory.used")


def lib():
    L = ctypes.CDLL(LIB)
    L.mk_new2.restype, L.mk_new2.argtypes = vp, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    L.mk_last_error.restype = ctypes.c_char_p
    L.mk_free.argtypes = [vp]
    L.mk_set_penalty.argtypes = [vp, ctypes.c_float, ctypes.c_float]
    L.mk_prefill.argtypes = [vp, ip, ctypes.c_int]
    L.mk_step.argtypes = [vp, ip, ctypes.c_int]
    L.mk_slots_open2.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mk_slot_begin2.argtypes = [vp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                 ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]
    L.mk_slot_prefill.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    L.mk_slots_step.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]
    L.mk_tokenize.argtypes = [vp, ctypes.c_char_p, ctypes.c_int, ip, ctypes.c_int]
    return L


def err(L):
    return L.mk_last_error().decode(errors="replace")


def new(L, ctx=8192):
    return L.mk_new2(MODEL.encode(), ctx, 7, DV.encode(), 1, 0)  # (kv_mode 1: q8_0, which the slots need)


def ids(L, h):
    out = (ctypes.c_int * 512)()
    n = L.mk_tokenize(h, PROMPT, len(PROMPT), out, 512)
    assert n > 0, err(L)
    return (ctypes.c_int * n)(*out[:n]), n


def check(ok, what, L=None):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok or L is None else f" ({err(L)})"), flush=True)
    return ok


def note(s):
    print("     " + s, flush=True)


def skip(why):
    """the case cannot test what it is for here: SKIP (the caller returns None)"""
    print("SKIP " + why, flush=True)


def main_steps(L, h, k):
    out = (ctypes.c_int * 16)()
    return all(L.mk_step(h, out, 16) > 0 for _ in range(k))


def slot_steps(L, h, slots, k):
    sa, out, cnt = (ctypes.c_int * len(slots))(*slots), (ctypes.c_int * (16 * len(slots)))(), (ctypes.c_int * len(slots))()
    for _ in range(k):
        if L.mk_slots_step(h, sa, len(slots), out, 16, cnt) != 0 or not all(cnt[i] > 0 for i in range(len(slots))):
            return False
    return True


def begin(L, h, slot, toks, n):
    return L.mk_slot_begin2(h, slot, toks, n, 0.0, 1.0, 20, 0.0, 1, 0, 0) >= 0 and L.mk_slot_prefill(h, slot, 1 << 30) == 0


def is_oom(L):
    return "out of memory" in err(L)


NOCAP = ("the engine's allocations cannot be capped here: {}")


def nocap_why():
    if os.environ.get("MK_CUDAMALLOC", "")[:1] == "1":
        return NOCAP.format("MK_CUDAMALLOC=1 (one cudaMalloc per buffer, outside any memory pool), so the request is not bounded")
    return NOCAP.format("the driver does not enforce a memory pool's maxSize (a 128 MiB allocation from a 64 MiB pool succeeded)")


# ------------------------------------------------------------------------------------------------------------------- cases
def case_penalty(L):
    # penalties on the engine's own sequence, then slots: the lone-slot step used to apply them to the lane (exit 4) and a
    # slot reset used to clear the lane's missing counts (-1, slot failed)
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    toks, n = ids(L, h)
    ok = check(L.mk_set_penalty(h, 0.5, 0.2) == 0, "set_penalty 0.5 / 0.2", L)
    ok &= check(L.mk_prefill(h, toks, n) >= 0 and main_steps(L, h, 3), "own sequence: prefill + 3 penalised steps", L)
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "slots_open 2 x 4096", L)
    ok &= check(begin(L, h, 0, toks, n), "slot 0: begin + prefill", L)
    ok &= check(slot_steps(L, h, [0], 4), "slot 0 alone: 4 steps (the solo launch)", L)
    ok &= check(begin(L, h, 0, toks, n), "slot 0 again: begin (truncate / rewind) + prefill", L)
    ok &= check(begin(L, h, 1, toks, n) and slot_steps(L, h, [0, 1], 3), "slots 0 + 1: 3 batched steps", L)
    ok &= check(main_steps(L, h, 2), "own sequence after the slot calls: 2 penalised steps", L)
    L.mk_free(h)
    return ok


def slots_rollback(L, D, h, what, devoom=False):
    """slots_open(SLOT_N, SLOT_CTX) under the current limit: fails part way (out of memory), everything it allocated goes
    back, a retry of 2 x 4096 succeeds and runs. True / False; None when the open did not fail, or (devoom) when its
    allocations went past the free VRAM (the caller decides)."""
    f0 = D.free()
    r = L.mk_slots_open2(h, SLOT_N, SLOT_CTX, 0)
    e = err(L) if r < 0 else "opened"
    st = D.stats()
    note(f"slots_open {SLOT_N} x {SLOT_CTX}: {r} ({e}); capped pool after it (MiB): {st}; device free before it {f0 // MiB} MiB")
    if r > 0 or (devoom and st["reserved_high"] > (f0 + 64 * MiB) // MiB):
        return None
    ok = check(is_oom(L), f"slots_open {SLOT_N} x {SLOT_CTX} fails with out of memory ({what})", L)
    ok &= check(st["used_high"] >= 1024, f"it failed part way: {st['used_high']} MiB of it was allocated first (>= 1024: slot 0 built)")
    ok &= check(st["used"] == 0, f"roll-back: {st['used']} MiB of it still allocated (0)")
    D.trim()
    f1 = D.free()
    ok &= check((f0 - f1) // MiB < LEFT_MIB, f"roll-back: device free memory {(f0 - f1) // MiB} MiB below the value before the open (< {LEFT_MIB})")
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "then slots_open 2 x 4096 succeeds", L)
    toks, n = ids(L, h)
    ok &= check(begin(L, h, 0, toks, n) and slot_steps(L, h, [0], 3), "slot 0: begin + prefill + 3 steps", L)
    return ok


def case_slots_oom(L):
    # slots_open failing part way through its allocations rolls back: a second open succeeds (it used to say "slots are open
    # already" over the half-built lanes). The failure: the pool the engine allocates from is capped below the free VRAM.
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < CAP_SLOTS + HEADROOM:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine: the capped pool needs {(CAP_SLOTS + HEADROOM) // MiB}")
        return None
    if not D.cap(CAP_SLOTS):
        L.mk_free(h)
        skip(nocap_why())
        return None
    note(f"device free {free // MiB} MiB beside the engine; the engine's slots now come from a pool capped at {CAP_SLOTS // MiB} MiB")
    ok = slots_rollback(L, D, h, f"the pool's {CAP_SLOTS // MiB} MiB cap")
    if ok is None:
        ok = check(False, f"slots_open {SLOT_N} x {SLOT_CTX} fails (it opened inside a {CAP_SLOTS // MiB} MiB pool)")
    L.mk_free(h)
    D.uncap()
    return ok


def case_slots_fit(L):
    # slots that cannot fit in the free device memory are refused before anything is allocated (they used to be
    # allocated until the device ran out, and with the sysmem fallback went on into system memory). Bounded on any build:
    # the pool is capped below the free VRAM, so a build without the check fails at the cap instead of spilling.
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < CAP_SLOTS + HEADROOM:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine: the capped pool needs {(CAP_SLOTS + HEADROOM) // MiB}")
        return None
    if not D.cap(free - HEADROOM):
        L.mk_free(h)
        skip(nocap_why())
        return None
    r = L.mk_slots_open2(h, 4, 200000, 0)
    e = err(L) if r < 0 else "opened"
    st = D.stats()
    note(f"device free {free // MiB} MiB beside the engine; slots_open 4 x 200000 (~30 GB): {r} ({e}); capped pool after it (MiB): {st}")
    ok = check(r < 0, "slots_open 4 x 200000 fails")
    ok &= check(st["used_high"] == 0, f"it was refused before allocating anything ({st['used_high']} MiB allocated first)")
    ok &= check("of device memory" in e, "the error says what it needs and what is free")
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "then slots_open 2 x 4096 succeeds", L)
    toks, n = ids(L, h)
    ok &= check(begin(L, h, 0, toks, n) and slot_steps(L, h, [0], 3), "slot 0: begin + prefill + 3 steps", L)
    L.mk_free(h)
    D.uncap()
    return ok


def ctor_rollback(L, D, v0, h1, what, devoom=False, fill=None):
    """engine 2 under the current limit fails part way; everything it allocated goes back; engine 1's free gives everything
    back; engine 3 starts. None (devoom) when engine 2's allocations went past the free VRAM (the sysmem fallback)."""
    v1 = D.free()
    h2 = new(L)
    st = D.stats()
    e2 = err(L)
    D.trim()
    v2 = D.free()
    note(f"engine 2: {'up' if h2 else e2}; capped pool after it: {st}; device free MiB: before engine 1 {v0 // MiB}, "
         f"engine 1 up {v1 // MiB}, after engine 2's attempt {v2 // MiB}")
    if h2:
        L.mk_free(h2)
        D.uncap()
        D.release(fill)
        L.mk_free(h1)
        return check(False, f"engine 2 fails (it came up despite {what})")
    if devoom and st["reserved_high"] > (v1 + 64 * MiB) // MiB:
        D.uncap()
        D.release(fill)
        L.mk_free(h1)
        return "fallback", st
    ok = check(is_oom(L), f"engine 2 fails with out of memory ({what})", L)
    ok &= check(st["used_high"] >= 1024, f"it failed part way: {st['used_high']} MiB of it was allocated first (>= 1024)")
    ok &= check(st["used"] == 0, f"engine 2's pool allocations all freed ({st['used']} MiB left)")
    ok &= check((v1 - v2) // MiB < LEFT_MIB, f"engine 2's attempt left {(v1 - v2) // MiB} MiB of device memory behind (< {LEFT_MIB})")
    D.uncap()
    D.release(fill)
    vu = D.free()
    L.mk_free(h1)
    v3 = D.free()
    note(f"device free MiB: the capped pool (and the filler) gone {vu // MiB}, after freeing engine 1 {v3 // MiB}")
    ok &= check((v0 - v3) // MiB < CTX_MIB, f"after freeing engine 1: device free memory {(v0 - v3) // MiB} MiB below the start "
                f"(< {CTX_MIB}: module code, local memory)")
    h3 = new(L)
    ok &= check(bool(h3), "engine 3 starts", L)
    if h3:
        L.mk_free(h3)
    return ok


def case_ctor_oom(L):
    # a second engine that runs out of memory part way frees what it allocated (it used to keep it), the first one's free
    # gives everything back, and a third starts. The failure: engine 2 allocates from a pool capped below the free VRAM.
    D = Drv()
    v0 = D.free()
    h1 = new(L)
    if not check(bool(h1), "engine 1", L):
        return False
    free = D.free()
    if free < CAP_CTOR + HEADROOM:
        L.mk_free(h1)
        skip(f"only {free // MiB} MiB of device memory free beside engine 1: the capped pool needs {(CAP_CTOR + HEADROOM) // MiB}")
        return None
    if not D.cap(CAP_CTOR):
        L.mk_free(h1)
        skip(nocap_why())
        return None
    note(f"device free {free // MiB} MiB beside engine 1; engine 2 allocates from a pool capped at {CAP_CTOR // MiB} MiB")
    return ctor_rollback(L, D, v0, h1, f"the pool's {CAP_CTOR // MiB} MiB cap")


FALLBACK = ("sysmem fallback active: an OOM cannot be forced ({}: the allocations went on past the VRAM into system memory, "
            "the Windows driver's CUDA sysmem fallback). To run this case: NVIDIA Control Panel > Manage 3D settings > CUDA - "
            "Sysmem Fallback Policy = Prefer No Sysmem Fallback (WSL2 follows the Windows driver), or a native Linux driver")


def devoom_off():
    if os.environ.get("CAPI_DEVOOM", "1") == "0":
        skip("CAPI_DEVOOM=0: the device out-of-memory cases are off")
        return True
    return False


def case_slots_devoom(L):
    # as slots_oom, with a device out-of-memory: the VRAM filled to MARGIN free; the pool capped at MARGIN + OVER, so with
    # the sysmem fallback on at most OVER goes past the VRAM (the open then succeeds or stops at the cap: SKIP)
    if devoom_off():
        return None
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < MARGIN + 256 * MiB:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine (the case leaves {MARGIN // MiB})")
        return None
    fill = D.alloc(free - MARGIN)
    if not fill:
        L.mk_free(h)
        skip(f"the filler ({(free - MARGIN) // MiB} MiB, inside the free VRAM) could not be allocated")
        return None
    if not D.cap(D.free() + OVER):
        D.release(fill)
        L.mk_free(h)
        skip(nocap_why())
        return None
    note(f"VRAM filled to {D.free() // MiB} MiB free; the engine's slots now come from a pool capped {OVER // MiB} MiB above that")
    os.environ["MK_SLOT_SPILL"] = "1"  # (past slots_open's up-front check, so the device itself runs out)
    ok = slots_rollback(L, D, h, "the VRAM is full", devoom=True)
    del os.environ["MK_SLOT_SPILL"]
    st = D.stats()
    L.mk_free(h)
    D.uncap()
    D.release(fill)
    if ok is None:
        skip(FALLBACK.format(f"slots_open {SLOT_N} x {SLOT_CTX} got {st['reserved_high']} MiB of pool with {MARGIN // MiB} MiB of VRAM free"))
    return ok


def case_ctor_devoom(L):
    # as ctor_oom, with a device out-of-memory: the VRAM filled to MARGIN free beside engine 1, engine 2's pool capped at
    # MARGIN + OVER (with the sysmem fallback on, engine 2 stops at the cap after at most OVER past the VRAM: SKIP)
    if devoom_off():
        return None
    D = Drv()
    v0 = D.free()
    h1 = new(L)
    if not check(bool(h1), "engine 1", L):
        return False
    free = D.free()
    if free < MARGIN + 256 * MiB:
        L.mk_free(h1)
        skip(f"only {free // MiB} MiB of device memory free beside engine 1 (the case leaves {MARGIN // MiB})")
        return None
    fill = D.alloc(free - MARGIN)
    if not fill:
        L.mk_free(h1)
        skip(f"the filler ({(free - MARGIN) // MiB} MiB, inside the free VRAM) could not be allocated")
        return None
    if not D.cap(D.free() + OVER):
        D.release(fill)
        L.mk_free(h1)
        skip(nocap_why())
        return None
    note(f"VRAM filled to {D.free() // MiB} MiB free beside engine 1; engine 2 allocates from a pool capped {OVER // MiB} MiB above that")
    r = ctor_rollback(L, D, v0, h1, "the VRAM is full", devoom=True, fill=fill)
    if isinstance(r, tuple):
        skip(FALLBACK.format(f"engine 2's pool grew to {r[1]['reserved_high']} MiB with {MARGIN // MiB} MiB of VRAM free"))
        return None
    return r


def case_policy(L):
    # a constructor that throws after the engine is built (a bad MK_POLICY) deletes it (it used to leak the whole engine)
    v0 = vram()
    os.environ["MK_POLICY"] = "garbage"  # (no key=value: set_policy throws after the Impl is built)
    h = new(L)
    v1 = vram()
    note(f"VRAM MiB: before {v0}, after the failed start {v1}; engine: {'up' if h else err(L)}")
    ok = check(not h, "MK_POLICY=garbage: the engine refuses to start")
    ok &= check(v1 - v0 < 1024, f"the failed start left {v1 - v0} MiB behind (< 1024: the CUDA context)")
    if h:
        L.mk_free(h)
    del os.environ["MK_POLICY"]
    h = new(L)
    ok &= check(bool(h), "without it the engine starts", L)
    if h:
        L.mk_free(h)
    return ok


def win_commit_free_gb():
    """Windows commit free (GB) from WSL / Windows, None where it cannot be read (native Linux)"""
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                            "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1)"],
                           capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def commit_gate(c):
    """the devoom cases may go up to OVER past the VRAM: on Windows / WSL2 they start only with CAPI_COMMIT_MIN_GB (default
    45) of Windows commit free, waiting up to 120 s for it (an earlier case's teardown). A reason to SKIP, or None."""
    need = float(os.environ.get("CAPI_COMMIT_MIN_GB", "45"))
    if not c.endswith("devoom") or need <= 0 or os.environ.get("CAPI_DEVOOM", "1") == "0":
        return None
    for _ in range(25):
        g = win_commit_free_gb()
        if g is None or g >= need:
            return None
        subprocess.run(["sleep", "5"])
    return f"commit guard: Windows commit free {g} GB < {need} GB (CAPI_COMMIT_MIN_GB) after 120 s"


if __name__ == "__main__":
    if len(sys.argv) > 4:
        r = globals()["case_" + sys.argv[4]](lib())
        sys.stdout.flush()
        os._exit(SKIP_RC if r is None else 0 if r is True else 1)  # (no interpreter teardown after the engine is gone)
    verdict = {}
    for c in CASES:
        print(f"== {c}", flush=True)
        why = commit_gate(c)
        if why:
            print("SKIP " + why, flush=True)
            verdict[c] = "SKIP"
            print(f"-- {c}: SKIP (not started)", flush=True)
            continue
        r = subprocess.run([sys.executable, __file__, LIB, MODEL, DV, c], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = r.stdout.splitlines()
        print("\n".join(ln for ln in lines if ln.startswith(("PASS", "FAIL", "SKIP", "     ", "Traceback", "[mk] error"))
                        or "Error" in ln or "exiting with code" in ln or "out of memory" in ln))
        fails = sum(ln.startswith("FAIL") for ln in lines)
        if r.returncode == SKIP_RC and not fails:
            verdict[c] = "SKIP"
        elif r.returncode == 0 and not fails:
            verdict[c] = "PASS"
        else:
            verdict[c] = "FAIL"
        sig = f", killed by signal {-r.returncode}" if r.returncode < 0 else ""
        print(f"-- {c}: {verdict[c]} (exit {r.returncode}{sig})", flush=True)
    failed = [c for c in CASES if verdict[c] == "FAIL"]
    skipped = [c for c in CASES if verdict[c] == "SKIP"]
    passed = [c for c in CASES if verdict[c] == "PASS"]
    print(f"CAPI_CHECKS_DONE: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped of {len(CASES)}"
          + (f"; FAILED: {' '.join(failed)}" if failed else "") + (f"; SKIPPED: {' '.join(skipped)}" if skipped else ""))
    sys.exit(1 if failed else 0)
