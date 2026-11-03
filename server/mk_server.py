#!/usr/bin/env python3
"""OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF, or the one given by --chat-template-file / --chat-template (llama-server's options), with
enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs and tools via `tools`;
reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured `tool_calls` (streamed as generated).
Sampling: temperature, top_p, top_k (1..32; <= 0: 32, the engine's maximum), min_p, seed; stop strings
(`stop`).  Successive requests of a conversation reuse the processed tokens: the new prompt is
matched against the sequence so far and only the rest is prefilled. When the new prompt diverges inside
what the engine already processed, it rewinds to the last DeltaNet checkpoint at or before the
divergence (saved during earlier prefills: at each generation prompt, each message start and every
--ck-every tokens; a pool of MK_CK_SLOTS, default 8).
POST /v1/messages (+ /v1/messages/count_tokens): the Anthropic Messages API over the same chat path.
POST /v1/responses (+ /v1/responses/input_tokens): the OpenAI Responses API over the same chat path, stateless (the client
sends the whole conversation each time, as Codex CLI does).
GET /health (503 "Loading model" until the model is loaded), /v1/models, /props, /slots, /metrics; POST /tokenize,
/detokenize, /apply-template, /v1/chat/completions/input_tokens (llama-server's shapes).  One request at a time; with
--agents N up to N at once, each in a parallel slot of the engine (its reply as when it runs alone; docs/SERVER.md).
After each request the engine keeps the GPU's clocks up for MK_KEEP_WARM_S seconds (default 120, 0 = off), so a
reply after a pause starts at full speed.
Limits (environment): MK_MAX_BODY_MB (request body, Content-Length or chunked, default 128), MK_MAX_IMAGES (images per
request, default 32),
MK_MAX_VIDEO_FRAMES (video frames per request, default 256), MK_TIMEOUT_S (a connection that sends and takes nothing
for this long is closed, default 600, 0 = never); --api-key / MK_API_KEY requires that key on every request but /health;
the image library refuses images over MK_IMAGE_MAX_PIXELS (default 100,000,000) before decoding them. A request
whose client disconnects stops at the next prefill chunk (2048 tokens) or engine cycle. A connection the server closes
while the client is still sending (an early 4xx for a body it does not read) is half-closed and drained first (bounded),
so the client reads the answer instead of a connection reset.
"""
import argparse
import base64
import binascii
import codecs
import collections
import contextlib
import ctypes
import errno
import hashlib
import hmac
import ipaddress
import itertools
import json
import os
import queue
import random
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK_MAX = 32
# sampling defaults (a request's absent / null fields): Unsloth's thinking-mode preset for Qwen3.8
DEF_TEMP, DEF_TOP_P, DEF_TOP_K, DEF_MIN_P = 1.0, 0.95, 20, 0.0
# request limits; main() reads them from the environment
MAX_BODY_MB = 128  # MK_MAX_BODY_MB: request body (several screenshots as base64 data: URLs fit)
MAX_IMAGES = 32  # MK_MAX_IMAGES: images per request
MAX_STOP, MAX_STOP_LEN = 64, 1024  # stop strings per request, characters per stop string
MAX_VIDEO_FRAMES = 256  # MK_MAX_VIDEO_FRAMES: video frames per request (all of its videos; 64 s at 4 fps)
IMAGE_MAX_PIXELS = 100_000_000  # MK_IMAGE_MAX_PIXELS (the vision library's too): larger video frames are not decoded
VIDEO_DECODE_S = 300  # a video's ffmpeg decode is stopped after this many seconds (400)
TIMEOUT_S = 600  # MK_TIMEOUT_S: a read or write stalled this long closes the connection (llama-server's --timeout)
VIDEO_TS_MS = 5000  # llama.cpp's video timestamp interval (mtmd_helper_video_init_params_default)
# lingering close (HTTPServer.shutdown_request): what a client still sends after the server's last answer is read and
# dropped for at most LINGER_S seconds / LINGER_BYTES bytes, LINGER_IDLE_S seconds without data (nginx's defaults)
LINGER_S, LINGER_IDLE_S, LINGER_BYTES = 30.0, 5.0, 1 << 30
# the engine and encoder libraries next to the repo root: qwen38mk.dll / qwen38vision.dll on Windows
LIB_MK, LIB_VISION = ("qwen38mk.dll", "qwen38vision.dll") if os.name == "nt" else ("libqwen38mk.so", "libqwen38vision.so")


class Console:
    """stderr: log lines, and on a terminal one status line under them that is rewritten in place while a request runs
    (a carriage return and spaces, no escape sequences: works in any console). A log line erases the status line and it
    is drawn again below it. Off a terminal (a log file, a pipe) only the log lines are written."""

    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""  # the status line ("": none)
        self.shown = 0  # columns it occupies on the screen

    @staticmethod
    def tty():
        try:
            return sys.stderr.isatty()
        except (AttributeError, ValueError):  # (no stderr, a closed one)
            return False

    def _erase(self, s):
        if self.shown:
            s.write("\r" + " " * self.shown + "\r")
            self.shown = 0

    def _draw(self, s):
        if self.text:
            # one row less than the width: a line that fills the row would wrap, and \r only returns to the last row
            t = self.text[:max(shutil.get_terminal_size((120, 24)).columns - 1, 20)]
            s.write("\r" + t + " " * max(self.shown - len(t), 0))
            self.shown = max(self.shown, len(t))

    def line(self, text):
        with self.lock:
            s = sys.stderr
            if s is None:
                return
            live = self.tty()
            if live:
                self._erase(s)
            s.write(text + "\n")
            if live:
                self._draw(s)
            s.flush()

    def status(self, text):
        """sets the status line ("": removes it); nothing off a terminal"""
        with self.lock:
            s = sys.stderr
            if s is None or text == self.text or not self.tty():
                return
            self.text = text
            if not text:
                self._erase(s)
            else:
                self._draw(s)
            s.flush()


CON = Console()


def log(*a):
    CON.line(" ".join(["[mk-server]", time.strftime("%H:%M:%S"), *map(str, a)]))


def secs(s):
    """a duration for the console: 0.42 s, 7.3 s, 2 min 05 s"""
    if s < 10:
        return f"{s:.2f} s"
    if s < 60:
        return f"{s:.1f} s"
    return f"{int(s // 60)} min {int(s % 60):02d} s"


def plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, mtp=None):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_set_penalty": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_prefill_ck": (ctypes.c_int, [vp, ip, ctypes.c_int, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
        }
        opt = {  # checkpoint pool (libraries before it: one checkpoint)
            "mk_prefill_cks": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]),
            "mk_checkpoints": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_rewind_to": (ctypes.c_int, [vp, ctypes.c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.pool = all(hasattr(L, name) for name in opt)
        if self.pool:
            for name, (res, args) in opt.items():
                f = getattr(L, name)
                f.restype, f.argtypes = res, args
        # images (libraries before it: text only)
        self.images = hasattr(L, "mk_prefill_img")
        if self.images:
            L.mk_prefill_img.restype = ctypes.c_int
            L.mk_prefill_img.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ip, ip, ip,
                                         ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            L.mk_n_embd.restype, L.mk_n_embd.argtypes = ctypes.c_int, []
        dv = draft_vocab.encode() if draft_vocab else None
        if mtp is not None:  # --mtp FILE / --no-mtp ("none"); otherwise the model file's MTP layer (or MK_MTP / MK_NO_MTP)
            if not hasattr(L, "mk_new_mtp"):
                raise RuntimeError("--mtp / --no-mtp need a newer libqwen38mk (mk_new_mtp)")
            L.mk_new_mtp.restype = vp
            L.mk_new_mtp.argtypes = [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, cp]
            self.h = L.mk_new_mtp(model.encode(), ctx, n_draft, dv, kv_mode, mtp.encode())
        else:
            self.h = L.mk_new(model.encode(), ctx, n_draft, dv, kv_mode)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.has_mtp = True  # False: no MTP layer, plain decoding at depth 0
        if hasattr(L, "mk_has_mtp"):
            L.mk_has_mtp.restype, L.mk_has_mtp.argtypes = ctypes.c_int, [vp]
            self.has_mtp = L.mk_has_mtp(self.h) == 1
        self.ctx = ctx
        self._pieces = {}
        # MK_EXACT engines (exact reuse): prefill rows do not depend on the chunking; see Server.load_prompt
        self.exact = False
        if hasattr(L, "mk_exact"):
            L.mk_exact.restype, L.mk_exact.argtypes = ctypes.c_int, [vp]
            self.exact = L.mk_exact(self.h) == 1
        # keep-warm (MK_KEEP_WARM_S; libraries before it: none)
        self._warm = hasattr(L, "mk_keep_warm")
        if self._warm:
            L.mk_keep_warm.restype, L.mk_keep_warm.argtypes = ctypes.c_double, [vp, ctypes.c_int]
        # prefill cancel (libraries before it: a disconnect during a prefill is noticed when it ends)
        self._cancel = hasattr(L, "mk_cancel")
        if self._cancel:
            L.mk_cancel.restype, L.mk_cancel.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        # prefill progress for the console (libraries before it: elapsed time only)
        self._progress = hasattr(L, "mk_prefill_progress")
        if self._progress:
            L.mk_prefill_progress.restype, L.mk_prefill_progress.argtypes = None, [vp, ip, ip]
        # decode statistics; the library's build (commit, engine flags; libraries before it: unknown)
        self._stats = hasattr(L, "mk_stats")
        if self._stats:
            lp = ctypes.POINTER(ctypes.c_long)
            L.mk_stats.restype, L.mk_stats.argtypes = None, [vp, lp, lp, lp]
        # conversation state in host memory: side requests and parked conversations (libraries before it: neither)
        self.store = hasattr(L, "mk_park")
        if self.store:
            ll, llp, ci = ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int
            for name, res, args in (("mk_store_budget", ci, [vp, ll]), ("mk_store_reserve", ci, [vp, ll]),
                                    ("mk_store_info", ci, [vp, llp, ci]), ("mk_kv_sizes", None, [vp, llp, llp, llp, llp]),
                                    ("mk_side_begin", ci, [vp]), ("mk_side_end", ci, [vp]), ("mk_side_drop", ci, [vp]),
                                    ("mk_side_active", ci, [vp]), ("mk_park", ci, [vp, ci]), ("mk_unpark", ci, [vp, ci]),
                                    ("mk_park_drop", ci, [vp, ci]), ("mk_park_need", ll, [vp, ci])):
                f = getattr(L, name)
                f.restype, f.argtypes = res, args
        # a side request's own conversation parked after it, and continued inside a later one (libraries before: neither)
        self.side_keep = self.store and hasattr(L, "mk_side_end_park") and hasattr(L, "mk_side_unpark")
        if self.side_keep:
            L.mk_side_end_park.restype, L.mk_side_end_park.argtypes = ctypes.c_int, [vp, ctypes.c_int, ip]
            L.mk_side_unpark.restype, L.mk_side_unpark.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        # pinned checkpoints: a prompt's shared prefix (a client's system prompt and tools) survives the pool's thinning
        # and parking (libraries before them: nothing is pinned, parked conversations keep their newest checkpoints only)
        self.ck_pins = hasattr(L, "mk_ck_pin") and hasattr(L, "mk_ck_pinned")
        if self.ck_pins:
            L.mk_ck_pin.restype, L.mk_ck_pin.argtypes = ctypes.c_int, [vp, ip, ctypes.c_int]
            L.mk_ck_pinned.restype, L.mk_ck_pinned.argtypes = ctypes.c_int, [vp, ip, ctypes.c_int]
        # parallel slots (--agents: SlotScheduler; open-jet's slot API, MKEngine::slots_open; libraries before them: none)
        self.slots = all(hasattr(L, n) for n in ("mk_slots_open2", "mk_slot_begin2", "mk_slot_prefill", "mk_slots_step",
                                                 "mk_slots_step_mixed"))
        if self.slots:
            ci, cf = ctypes.c_int, ctypes.c_float
            for name, args in (("mk_slots_open2", [vp, ci, ci, ci]),
                               ("mk_slot_begin2", [vp, ci, ip, ci, cf, cf, ci, cf, ctypes.c_ulonglong, ci, ci]),
                               ("mk_slot_prefill", [vp, ci, ci]), ("mk_slots_step", [vp, ip, ci, ip, ci, ip]),
                               ("mk_slots_step_mixed", [vp, ci, ci, ip, ci, ip, ci, ip, ip])):
                f = getattr(L, name)
                f.restype, f.argtypes = ci, args
        self.n_vocab = 0  # (libraries before mk_n_vocab: unknown)
        if hasattr(L, "mk_n_vocab"):
            L.mk_n_vocab.restype, L.mk_n_vocab.argtypes = ctypes.c_int, [vp]
            self.n_vocab = L.mk_n_vocab(self.h)
        # tokenizer flags and token attributes (libraries before them: special tokens always parsed, no BOS / EOS)
        self.tok_flags = hasattr(L, "mk_tokenize2") and hasattr(L, "mk_token_attr") and self.n_vocab > 0
        if self.tok_flags:
            L.mk_tokenize2.restype = ctypes.c_int
            L.mk_tokenize2.argtypes = [vp, cp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            L.mk_token_attr.restype, L.mk_token_attr.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        self._controls = None
        self.build = None
        if hasattr(L, "mk_build_info"):
            L.mk_build_info.restype, L.mk_build_info.argtypes = ctypes.c_char_p, []
            raw = L.mk_build_info().decode(errors="replace")
            try:
                self.build = json.loads(raw)
            except ValueError:
                self.build = {"raw": raw}

    def stats(self):
        """(decode cycles, drafted tokens, accepted drafts) since the load, or None (a library without mk_stats)"""
        if not self._stats:
            return None
        c, d, a = ctypes.c_long(), ctypes.c_long(), ctypes.c_long()
        self.L.mk_stats(self.h, ctypes.byref(c), ctypes.byref(d), ctypes.byref(a))
        return c.value, d.value, a.value

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text, special=True, add_special=False):
        """text -> token ids. special: the text of control tokens (<|im_start|>, ...) becomes those tokens, else plain text;
        add_special: the vocabulary's BOS / EOS where it adds them (llama_tokenize's parse_special / add_special;
        other than the defaults they need mk_tokenize2: ValueError without it)"""
        b = text.encode()
        if special and not add_special:
            def run(out, cap):
                return self.L.mk_tokenize(self.h, b, len(b), out, cap)
        elif self.tok_flags:
            def run(out, cap):
                return self.L.mk_tokenize2(self.h, b, len(b), out, cap, int(add_special), int(special))
        else:
            raise ValueError("parse_special false / add_special need a newer libqwen38mk (mk_tokenize2)")
        cap = min(len(b) + 16, 1 << 20)  # (a longer text is tokenized twice: the first call returns its count)
        out = (ctypes.c_int * cap)()
        n = run(out, cap)
        if -2 ** 31 < n < 0:  # (-count: a longer buffer; INT_MIN: an error)
            cap = -n
            out = (ctypes.c_int * cap)()
            n = run(out, cap)
        if n < 0:
            raise self._err("tokenize")
        return list(out[:n])

    def controls(self):
        """{text: id} of the vocabulary's control (and unknown) tokens: the ones whose text tokenize(special=False) keeps
        as text (user-defined ones such as <think> are tokens either way); {} without mk_token_attr. Read once (one call
        per token id, ~0.1 s)."""
        if self._controls is None:
            self._controls = {}
            if self.tok_flags:
                attr = self.L.mk_token_attr
                for t in range(self.n_vocab):
                    a = attr(self.h, t)
                    if a > 0 and a & 0x9:  # (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN)
                        text = self.piece(t).decode(errors="replace")
                        if text:
                            self._controls.setdefault(text, t)
        return self._controls

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def keep_warm(self, on=True):
        """on: keep the GPU's clocks up for MK_KEEP_WARM_S seconds from now (the engine stops the load before its next
        GPU work); off: stop now. Returns the window in seconds (0: off)."""
        return self.L.mk_keep_warm(self.h, 1 if on else 0) if self._warm else 0.0

    @contextlib.contextmanager
    def cancel_when(self, gone, every=0.05):
        """runs the block (a prefill call) next to a watcher thread that polls gone() every `every` seconds: once it
        returns True, the engine stops the prefill at its next chunk boundary (mk_cancel: a host flag) and the call
        raises PrefillCancelled, with the engine back at its last checkpoint before the prefill"""
        if not self._cancel or gone is None:
            yield
            return
        self.L.mk_cancel(self.h, 0)
        done = threading.Event()

        def watch():
            while not done.wait(every):
                if gone():
                    self.L.mk_cancel(self.h, 1)
                    return
        th = threading.Thread(target=watch, name="mk-cancel-watch", daemon=True)
        th.start()
        try:
            yield
        finally:
            done.set()
            th.join()
            self.L.mk_cancel(self.h, 0)

    def prefill_progress(self):
        """(done, total) positions of the running prefill's batched part, callable from any thread while it runs; None
        for a library without mk_prefill_progress"""
        if not self._progress:
            return None
        d, t = ctypes.c_int(), ctypes.c_int()
        self.L.mk_prefill_progress(self.h, ctypes.byref(d), ctypes.byref(t))
        return d.value, t.value

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def set_penalty(self, presence, frequency):
        if self.L.mk_set_penalty(self.h, presence, frequency) < 0:
            raise self._err("penalty")

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        """cks: more checkpoint positions (absolute; the engine moves them onto its scan grid), ck_at stays exact.
        images: (pos, nx, ny, rows address) of each image the fed positions overlap (absolute positions)"""
        arr = (ctypes.c_int * len(toks))(*toks)
        if images:
            k = len(images)
            ca = (ctypes.c_int * max(len(cks), 1))(*cks)
            pos, nx, ny = ((ctypes.c_int * k)(*[im[j] for im in images]) for j in range(3))
            rows = (ctypes.c_void_p * k)(*[im[3] for im in images])
            r = self.L.mk_prefill_img(self.h, arr, len(toks), ca, len(cks), ck_at, pos, nx, ny, rows, k)
        elif self.pool and cks:
            ca = (ctypes.c_int * len(cks))(*cks)
            r = self.L.mk_prefill_cks(self.h, arr, len(toks), ca, len(cks), ck_at)
        else:
            r = self.L.mk_prefill_ck(self.h, arr, len(toks), ck_at)
        if r == -2:  # (mk_cancel)
            raise PrefillCancelled(self.L.mk_last_error().decode())
        if r < 0:
            raise self._err("prefill")
        return r

    def step(self):
        out = (ctypes.c_int * 16)()
        n = self.L.mk_step(self.h, out, 16)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0

    def checkpoints(self):
        if not self.pool:
            ck = self.checkpoint_pos()
            return [ck] if ck >= 0 else []
        n = self.L.mk_checkpoints(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_checkpoints(self.h, out, n)
        return list(out[:n])

    def rewind_to(self, pos):
        if not self.pool:
            return pos == self.checkpoint_pos() and self.rewind_to_checkpoint()
        return self.L.mk_rewind_to(self.h, pos) == 0

    def ck_pin(self, positions):
        """pins, for each position P, the checkpoint at the last position in (P - 32, P]: one there now, else the one the
        next prefill saves there (MKEngine::ck_pin). Pinned checkpoints are never evicted from the pool and are parked
        with the conversation (besides its newest --park-cks)"""
        arr = (ctypes.c_int * max(len(positions), 1))(*positions)
        if self.L.mk_ck_pin(self.h, arr, len(positions)) < 0:
            raise self._err("checkpoint pin")

    def pinned(self):
        """positions of the pinned checkpoints (ascending)"""
        n = self.L.mk_ck_pinned(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_ck_pinned(self.h, out, n)
        return list(out[:n])

    # ---- host store: side requests, parked conversations (MKEngine::store_budget, side_begin, park) ----
    def store_budget(self, nbytes):
        if self.L.mk_store_budget(self.h, nbytes) < 0:
            raise self._err("host store")

    def store_info(self):
        """{budget, allocated, used, parked, last_bytes, last_ms, side_ms, kv_token_bytes, side_park_bytes, side_park_ms}"""
        out = (ctypes.c_longlong * 10)()
        n = self.L.mk_store_info(self.h, out, 10)
        keys = ("budget", "allocated", "used", "parked", "last_bytes", "last_us", "side_us", "kv_token_bytes",
                "side_park_bytes", "side_park_us")
        d = dict(zip(keys, out[:n]))
        d["last_ms"], d["side_ms"] = d.pop("last_us", 0) / 1e3, d.pop("side_us", 0) / 1e3
        d["side_park_ms"] = d.pop("side_park_us", 0) / 1e3
        return d

    def kv_sizes(self):
        """bytes: (per KV token, the window rings, the recurrent state, one checkpoint)"""
        v = [ctypes.c_longlong() for _ in range(4)]
        self.L.mk_kv_sizes(self.h, *[ctypes.byref(x) for x in v])
        return tuple(x.value for x in v)

    def side_begin(self):
        """True: the sequence is kept aside (the engine looks reset); False: no room in the host store"""
        r = self.L.mk_side_begin(self.h)
        if r < 0:
            raise self._err("side request")
        return r == 0

    def side_end(self):
        """True: the kept sequence is back bit for bit; False: it was dropped (the store ran full: the engine is reset)"""
        r = self.L.mk_side_end(self.h)
        if r < 0:
            raise self._err("side request")
        return r == 1

    def side_end_park(self, max_cks):
        """side_end, with the side request's own conversation parked first: (kept back, its entry id or None). Kept back
        False: the kept sequence was lost (the store ran full) and the side request's conversation stays live"""
        pid = ctypes.c_int(0)
        r = self.L.mk_side_end_park(self.h, max_cks, ctypes.byref(pid))
        if r < 0:
            raise self._err("side request")
        return r == 1, pid.value or None

    def side_unpark(self, pid):
        """right after side_begin: the side request continues parked conversation pid (its entry is freed); False: no
        room in the host store for what that moves aside (nothing changed)"""
        r = self.L.mk_side_unpark(self.h, pid)
        if r < 0:
            raise self._err("side request")
        return r == 0

    def side_active(self):
        return self.L.mk_side_active(self.h) == 1

    def side_drop(self):
        self.L.mk_side_drop(self.h)

    def park(self, max_cks):
        """a snapshot of the sequence in the host store: its id, or None (no room)"""
        r = self.L.mk_park(self.h, max_cks)
        if r == -3:
            return None
        if r < 0:
            raise self._err("park")
        return r or None

    def unpark(self, pid):
        if self.L.mk_unpark(self.h, pid) < 0:
            raise self._err("restore")

    def park_drop(self, pid):
        self.L.mk_park_drop(self.h, pid)

    def park_need(self, max_cks):
        return self.L.mk_park_need(self.h, max_cks)

    # ---- parallel slots (--agents: SlotScheduler drives them from its thread) ----
    SLOT_PER = 16  # tokens a slot can emit in one cycle that the output arrays hold (1 + its drafts, at most MAXT)

    def slots_open(self, n, ctx):
        """opens n slots of ctx positions each (0: the most that fit, leaving MK_SLOT_RESERVE_MIB of the device memory
        free); returns a slot's positions"""
        r = self.L.mk_slots_open2(self.h, n, ctx, 0)
        if r < 0:
            raise self._err("parallel slots")
        return r

    def slot_begin(self, slot, toks, temp, top_p, top_k, min_p, seed):
        """slot <- the prompt and its sampling, reusing what the slot's sequence has (truncated at the divergence, else
        back to its newest checkpoint before it, else from scratch); returns the reused tokens: slot_prefill feeds the
        rest"""
        arr = (ctypes.c_int * len(toks))(*toks)
        r = self.L.mk_slot_begin2(self.h, slot, arr, len(toks), temp, top_p, top_k, min_p, seed, 0, 0)
        if r < 0:
            raise self._err("slot prompt")
        return r

    def slot_prefill(self, slot, budget):
        """feeds up to `budget` of the slot's prompt tokens; returns how many are left (0: the slot decodes from the next
        cycle on, which emits its first token). Budget 0: the rest is abandoned (the slot keeps what was fed)"""
        r = self.L.mk_slot_prefill(self.h, slot, budget)
        if r < 0:
            raise self._err("slot prefill")
        return r

    def _slot_tokens(self, n, out, cnt):
        return [list(out[i * self.SLOT_PER:i * self.SLOT_PER + cnt[i]]) for i in range(n)]

    def slots_step(self, slots):
        """one engine cycle of the slots (one batched launch; a lone slot: the solo megakernel): the tokens each one
        emitted ([]: its context is full)"""
        n = len(slots)
        sa, out, cnt = (ctypes.c_int * n)(*slots), (ctypes.c_int * (n * self.SLOT_PER))(), (ctypes.c_int * n)()
        if self.L.mk_slots_step(self.h, sa, n, out, self.SLOT_PER, cnt) < 0:
            raise self._err("decode")
        return self._slot_tokens(n, out, cnt)

    def slots_step_mixed(self, pslot, budget, slots):
        """slots_step with the next `budget` tokens of slot pslot's prompt in the same pass: (the slots' tokens, the
        prompt tokens left)"""
        n = len(slots)
        sa, out, cnt = (ctypes.c_int * n)(*slots), (ctypes.c_int * (n * self.SLOT_PER))(), (ctypes.c_int * n)()
        rem = ctypes.c_int()
        if self.L.mk_slots_step_mixed(self.h, pslot, budget, sa, n, out, self.SLOT_PER, cnt, ctypes.byref(rem)) < 0:
            raise self._err("decode")
        return self._slot_tokens(n, out, cnt), rem.value


class Vision:
    """ctypes binding of vision/mk_vision.h (libqwen38vision.so): the CPU image encoder (llama.cpp's libmtmd), plus a
    cache of decoded / encoded images by content hash, so a conversation re-sending its images encodes each once.
    A request pins its images (image() .. release()): the cache never frees an image a request still holds, so it can
    hold more than --image-cache images while a request runs."""

    class Image:
        def __init__(self, vision, handle, key):
            self.v, self.h, self.key = vision, handle, key
            nx, ny = ctypes.c_int(), ctypes.c_int()
            self.n = vision.L.mkv_image_grid(handle, ctypes.byref(nx), ctypes.byref(ny))
            self.nx, self.ny = nx.value, ny.value
            self.encode_s = None
            self.pins = 0  # requests holding it (Vision.image / Vision.release)

        def rows(self):
            """address of the [n][n_embd] fp32 rows, encoding on the first call"""
            L = self.v.L
            if not self.h:
                raise RuntimeError("image encode: the image was freed")
            if self.encode_s is None:
                if L.mkv_image_encode(self.v.ctx, self.h) != 0:
                    raise RuntimeError(f"image encode: {L.mkv_last_error().decode()}")
                self.encode_s = L.mkv_image_encode_ms(self.h) / 1e3
            return L.mkv_image_rows(self.h)

        def free(self):
            if self.h:
                self.v.L.mkv_image_free(self.h)
                self.h = None

    class Video:
        """a decoded video in llama.cpp's layout: parts [str | Image], each Image one frame or a frame pair (Qwen-VL's
        temporal patch of 2: one grid of tokens); owned by Vision.videos (its images are pinned like images)"""

        def __init__(self, key, parts, n_frames):
            self.key, self.parts, self.n_frames = key, parts, n_frames
            self.images = [p for p in parts if not isinstance(p, str)]

        def free(self):
            for im in self.images:
                im.free()

    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("min_tokens", ctypes.c_int), ("max_tokens", ctypes.c_int),
                    ("flash_attn", ctypes.c_int), ("verbose", ctypes.c_int), ("backend", ctypes.c_int)]

    BACKENDS = {"auto": 0, "mtmd": 1, "cpu": 2}
    frames_ok = False  # the library has video frames (mkv_frames_load)
    videos = None      # key -> Video, least recently used first (__init__)
    video_cache_max = 0

    def __init__(self, lib, mmproj, model, threads=0, min_tokens=0, max_tokens=0, cache=8, backend="auto", video_cache=2):
        L = self.L = ctypes.CDLL(lib)
        vp, cp = ctypes.c_void_p, ctypes.c_char_p
        for name, res, args in (
                ("mkv_params_default", Vision.Params, []), ("mkv_new", vp, [cp, cp, Vision.Params]),
                ("mkv_free", None, [vp]), ("mkv_last_error", cp, []), ("mkv_n_embd", ctypes.c_int, [vp]),
                ("mkv_n_threads", ctypes.c_int, [vp]), ("mkv_backend", ctypes.c_int, [vp]),
                ("mkv_image_load", vp, [vp, cp, ctypes.c_size_t]),
                ("mkv_image_grid", ctypes.c_int, [vp, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]),
                ("mkv_image_encode", ctypes.c_int, [vp, vp]), ("mkv_image_rows", vp, [vp]),
                ("mkv_image_encode_ms", ctypes.c_double, [vp]), ("mkv_image_free", None, [vp])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        # video frames (libraries before it: images only)
        self.frames_ok = hasattr(L, "mkv_frames_load")
        if self.frames_ok:
            L.mkv_frames_load.restype = vp
            L.mkv_frames_load.argtypes = [vp, cp, cp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        p = L.mkv_params_default()
        p.n_threads, p.min_tokens, p.max_tokens, p.backend = threads, min_tokens, max_tokens, Vision.BACKENDS[backend]
        self.ctx = L.mkv_new(mmproj.encode(), model.encode(), p)
        if not self.ctx:
            raise RuntimeError(f"vision: {L.mkv_last_error().decode()}")
        self.n_embd = L.mkv_n_embd(self.ctx)
        self.n_threads = L.mkv_n_threads(self.ctx)
        self.backend = {1: "libmtmd", 2: "vit_cpu"}.get(L.mkv_backend(self.ctx), "?")
        self.cache = {}  # key -> Image, least recently used first
        self.cache_max = max(1, cache)
        self.videos = {}  # key -> Video, least recently used first
        self.video_cache_max = max(0, video_cache)

    def image(self, data):
        """the Image of these file bytes (decoded and preprocessed; encoded lazily by rows()), pinned: the caller
        releases it (release) when its request is done"""
        key = hashlib.sha256(data).hexdigest()
        im = self.cache.pop(key, None)
        if im is None:
            h = self.L.mkv_image_load(self.ctx, data, len(data))
            if not h:
                raise BadRequest(f"image: {self.L.mkv_last_error().decode()}")
            im = Vision.Image(self, h, key)
        self.cache[key] = im
        im.pins += 1
        return im

    def video(self, data, decoder, fps, max_tokens, max_frames):
        """the Video of these file bytes (decoded by decoder (VideoDecoder) at fps, frames resized to at most max_tokens
        tokens (0: the image limit); its images are preprocessed, encoded lazily by rows()), pinned like image(). A
        video of more than max_frames frames is refused (BadRequest) while it decodes."""
        key = f"{hashlib.sha256(data).hexdigest()}/{fps!r}/{max_tokens}"
        v = self.videos.pop(key, None)
        if v is not None and v.n_frames > max_frames:
            self.videos[key] = v
            raise BadRequest(f"video of {v.n_frames} frames at {fps:g} fps: over the {max_frames} frames left for this "
                             f"request (MK_MAX_VIDEO_FRAMES {MAX_VIDEO_FRAMES})")
        if v is None:
            parts, n = [], 0
            try:
                for kind, x in video_parts(decoder.frames(data, fps, max_frames), fps):
                    if kind == "text":
                        parts.append(x)
                        continue
                    (w, h, a), b = x[0], x[1]
                    n += 2 if b else 1
                    hd = self.L.mkv_frames_load(self.ctx, a, b[2] if b else None, w, h, max_tokens)
                    if not hd:
                        raise BadRequest(f"video frame: {self.L.mkv_last_error().decode()}")
                    parts.append(Vision.Image(self, hd, f"{key}/{len(parts)}"))
            except BaseException:
                for im in parts:
                    if not isinstance(im, str):
                        im.free()
                raise
            v = Vision.Video(key, parts, n)
        self.videos[key] = v
        for im in v.images:
            im.pins += 1
        return v

    def release(self, images):
        """unpins the images of a request (image() / video() once per pin), then frees the least recently used unpinned
        images beyond --image-cache and videos beyond --video-cache"""
        for im in images:
            im.pins -= 1
        extra = len(self.cache) - self.cache_max
        for key in [k for k, im in self.cache.items() if im.pins <= 0][:max(extra, 0)]:
            self.cache.pop(key).free()
        extra = len(self.videos or ()) - self.video_cache_max
        for key in [k for k, v in (self.videos or {}).items() if all(im.pins <= 0 for im in v.images)][:max(extra, 0)]:
            self.videos.pop(key).free()

    def warm_up(self):
        """one encode of a 64 x 64 grey PNG (not cached): the first encode after the engine's load pays ~0.5 s once
        (every weight streamed in again, the workers' scratch), here at startup instead of in the first image request"""
        w = h = 64
        raw = b"".join(bytes([0]) + bytes([128, 128, 128]) * w for _ in range(h))

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        png = (bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        hd = self.L.mkv_image_load(self.ctx, png, len(png))
        if not hd or self.L.mkv_image_encode(self.ctx, hd) != 0:
            raise RuntimeError(f"vision warm-up: {self.L.mkv_last_error().decode()}")
        self.L.mkv_image_free(hd)


def image_bytes(url):
    """the file bytes of an OpenAI image_url (a data: URL, as llama-server takes them by default)"""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise BadRequest("image_url: only data: URLs (base64) are supported")
    head, sep, payload = url.partition(",")
    if not sep or ";base64" not in head:
        raise BadRequest("image_url: expected a base64 data: URL")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"image_url: bad base64 ({e})")


# ---------------------------------------------------------------------------------------------
# video: frames from ffmpeg, laid out as llama.cpp's mtmd does (mtmd-helper.cpp video helpers + mtmd_tokenize)


class VideoBytes(bytes):
    """the file bytes of a video content part (normalize_messages' media list holds images as plain bytes)"""


VIDEO_PARTS = ("input_video", "video_url")  # llama-server's / vLLM's content part types


def video_bytes(part):
    """the file bytes of a video content part: llama-server's {"type": "input_video", "input_video": {"data": base64}}
    (or {"url": ...}), or {"type": "video_url", "video_url": {"url": ...}}; raw base64 or a base64 data: URL (no remote
    or file URLs)"""
    t = part.get("type")
    v = part.get(t)
    s = (v.get("data") or v.get("url")) if isinstance(v, dict) else v
    if not isinstance(s, str) or not s:
        raise BadRequest(f"{t}: expected base64 data")
    if s.startswith("data:"):
        head, sep, s = s.partition(",")
        if not sep or ";base64" not in head:
            raise BadRequest(f"{t}: expected a base64 data: URL")
    elif re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        raise BadRequest(f"{t}: only base64 data (or a base64 data: URL) is supported, not {s.split(':')[0]}:// URLs")
    try:
        data = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"{t}: bad base64 ({e})")
    if not data:
        raise BadRequest(f"{t}: empty video")
    return VideoBytes(data)


def f32(x):
    """x rounded to float32 (llama.cpp computes the video timestamps in float)"""
    return struct.unpack("f", struct.pack("f", x))[0]


def video_timestamp(k, fps, nxt):
    """mtmd-helper's timestamp text after frame k (0-based) at fps (float32) given the next threshold nxt (ms):
    (text or None, the next threshold); float32 arithmetic as there"""
    elapsed = f32(f32(f32(k) / fps) * 1000.0)
    if elapsed < nxt:
        return None, nxt
    s = f32(elapsed / 1000.0)
    minutes = int(f32(s / 60.0))
    sec = f32(s - f32(minutes * 60.0))
    return f"[{minutes}m{sec:.2f}s]", f32(nxt + VIDEO_TS_MS)


def video_parts(frames, fps):
    """a video's parts as llama.cpp lays them out: ("text", "Video:"), then each frame with a timestamp text
    ("[0m5.00s]") after every frame that reaches the next 5 s mark (frame 0 included); consecutive frames of one size
    (no text between) merge in pairs ("frames", (a, b)), a run's odd last frame stays alone ("frames", (a, None)).
    frames: (width, height, rgb bytes) in order; fps: the sampling rate (float32 in the timestamps)."""
    fps = f32(fps)
    yield "text", "Video:"
    nxt, pend = 0.0, None
    for k, fr in enumerate(frames):
        if pend is not None and pend[:2] != fr[:2]:  # (frames of different sizes do not merge)
            yield "frames", (pend, None)
            pend = None
        if pend is None:
            pend = fr
        else:
            yield "frames", (pend, fr)
            pend = None
        ts, nxt = video_timestamp(k, fps, nxt)
        if ts:
            if pend is not None:
                yield "frames", (pend, None)
                pend = None
            yield "text", ts
    if pend is not None:
        yield "frames", (pend, None)


class FrameTooLarge(ValueError):
    pass


def read_ppm(f, max_pixels=None):
    """the next binary PPM (P6, maxval 255) of a stream: (width, height, rgb bytes), None at its end; a frame of more
    than max_pixels pixels is refused from its header (FrameTooLarge), before its data is read"""
    def token():
        t = b""
        while True:
            c = f.read(1)
            if not c:
                return t or None
            if c == b"#" and not t:  # (a comment line)
                while c and c not in b"\r\n":
                    c = f.read(1)
                continue
            if c.isspace():
                if t:
                    return t
                continue
            t += c
            if len(t) > 16:
                raise ValueError("bad PPM header")
    magic = token()
    if magic is None:
        return None
    w, h, mx = token(), token(), token()
    if magic != b"P6" or not (w and h and mx) or not (w.isdigit() and h.isdigit()) or mx != b"255":
        raise ValueError("bad PPM header")
    w, h = int(w), int(h)
    if not (0 < w <= 65535 and 0 < h <= 65535):
        raise ValueError(f"bad PPM size {w} x {h}")
    if max_pixels and w * h > max_pixels:
        raise FrameTooLarge(f"frame of {w} x {h} pixels: over the limit of {max_pixels} pixels (MK_IMAGE_MAX_PIXELS)")
    data = f.read(w * h * 3)
    if len(data) != w * h * 3:
        raise ValueError("truncated frame")
    return w, h, data


def find_ffmpeg(explicit=None):
    """the ffmpeg executable for video input: explicit (--ffmpeg) or MK_FFMPEG (a file, or a directory with it), else
    ffmpeg on PATH, else the imageio-ffmpeg package's bundled binary (pip install imageio-ffmpeg); None: none found"""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    p = explicit or os.environ.get("MK_FFMPEG")
    if p:
        p = os.path.join(p, exe) if os.path.isdir(p) else p
        return p if os.path.isfile(p) else None
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # optional
        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.isfile(p) else None
    except Exception:  # (not installed, or no binary for this platform)
        return None


class VideoDecoder:
    """video file bytes -> RGB24 frames through an ffmpeg executable, sampled by ffmpeg's fps filter as llama.cpp's
    mtmd-helper does (`-vf fps=F`, rgb24), but read as a PPM stream (each frame carries its size: no ffprobe needed).
    Frames over MK_IMAGE_MAX_PIXELS are refused as images are: read_ppm checks each frame's header before reading it,
    and ffmpeg's decoder gets `-max_pixels` (its default is INT_MAX pixels: a crafted clip of 40,000 x 40,000 frames
    would be decoded, 4.8 GB of rgb24 each) with MAX_PIXELS_SLACK more, as decoders check their row-aligned size
    (1100 x 1000 counts as 1152 x 1000), so that every frame within the limit decodes."""
    MAX_PIXELS_SLACK = 1 << 23  # (a 64-column and 64-row alignment of frames up to 65535 x 65535)
    # the demuxers ffmpeg may use (-format_whitelist; "mov,mp4,m4a,..." matches by any of its names): media containers
    # that read nothing but the file (mov's external tracks, enable_drefs, are off by default). Not DASH / HLS /
    # concat: a manifest or playlist opens other files and URLs (ffmpeg 6's dash outside -protocol_whitelist too)
    FORMATS = "mov,mp4,matroska,webm,avi,mpegts,mpeg,ogg,flv,gif"

    def __init__(self, exe):
        self.exe = exe
        self.version = ""

    def check(self, timeout=20):
        """runs `ffmpeg -version`; the first line of its output (RuntimeError if it does not run)"""
        try:
            r = subprocess.run([self.exe, "-hide_banner", "-version"], stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"{self.exe}: {e}")
        if r.returncode != 0:
            raise RuntimeError(f"{self.exe} -version: exit {r.returncode}")
        self.version = (r.stdout.decode(errors="replace").splitlines() or ["?"])[0]
        return self.version

    def frames(self, data, fps, max_frames, timeout=None):
        """yields (width, height, rgb bytes) per frame; BadRequest for a video ffmpeg cannot decode, one of more than
        max_frames frames (when the next frame arrives) or a decode over `timeout` seconds"""
        timeout = VIDEO_DECODE_S if timeout is None else timeout
        with tempfile.TemporaryDirectory(prefix="mkvideo") as d:
            # (FORMATS only: a "video" that is a manifest, a playlist or a concat list cannot make ffmpeg open other
            # files or URLs; the file protocol only and a name without an extension stay as a second layer)
            path = os.path.join(d, "input")
            with open(path, "wb") as f:
                f.write(data)
            cmd = [self.exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
                   "-format_whitelist", self.FORMATS,
                   "-max_pixels", str(min(IMAGE_MAX_PIXELS + self.MAX_PIXELS_SLACK, 2 ** 31 - 1)), "-i", path,
                   "-an", "-sn", "-vf", f"fps={fps:.6f}", "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24", "pipe:1"]
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as e:
                raise BadRequest(f"video: cannot run ffmpeg ({e})")
            err = bytearray()

            def drain():  # (stderr in a thread: a chatty ffmpeg must not block on a full pipe)
                for line in p.stderr:
                    err.extend(line)
                    del err[:-4096]
            th = threading.Thread(target=drain, name="mk-ffmpeg-stderr", daemon=True)
            th.start()
            killed = threading.Event()

            def kill():
                killed.set()
                p.kill()
            timer = threading.Timer(timeout, kill)
            timer.start()
            n = 0
            try:
                while True:
                    try:
                        fr = read_ppm(p.stdout, IMAGE_MAX_PIXELS)
                    except FrameTooLarge as e:
                        raise BadRequest(f"video: {e}")
                    except ValueError as e:
                        if killed.is_set():
                            break
                        raise BadRequest(f"video: bad frame stream from ffmpeg ({e})")
                    if fr is None:
                        break
                    n += 1
                    if n > max_frames:
                        raise BadRequest(f"video of more than {max_frames} frames at {fps:g} fps (MK_MAX_VIDEO_FRAMES "
                                         f"{MAX_VIDEO_FRAMES} per request)")
                    yield fr
            finally:
                timer.cancel()
                if p.poll() is None:
                    p.kill()
                p.stdout.close()
                p.wait()
                th.join(5)
                if not th.is_alive():
                    p.stderr.close()
            if killed.is_set():
                raise BadRequest(f"video: decoding took over {timeout} s")
            big = re.search(r"Picture size (\d+)x(\d+) exceeds specified max pixel count", err.decode(errors="replace"))
            if big:  # (-max_pixels: ffmpeg did not decode a frame, also after others of a smaller size)
                raise BadRequest(f"video: frames of about {big[1]} x {big[2]} pixels: over the limit of {IMAGE_MAX_PIXELS} "
                                 f"pixels (MK_IMAGE_MAX_PIXELS)")
            fmt = re.search(r"\[([\w,]+) @ \w+\] Format not on whitelist", err.decode(errors="replace"))
            if fmt and n == 0:  # (-format_whitelist: a manifest, a playlist, another container)
                raise BadRequest(f"video: {fmt[1][:40]} input is refused (the containers read: "
                                 f"{self.FORMATS.replace(',', ', ')})")
            if n == 0:
                msg = err.decode(errors="replace").strip().splitlines()
                raise BadRequest("video: no frames decoded" + (f" ({msg[-1][:300]})" if msg else ""))


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


# llama.cpp's CHATML_TEMPLATE_SRC: `--chat-template chatml`, and the template of a GGUF that has none
CHATML_TEMPLATE = ("{%- for message in messages -%}\n"
                   "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n"
                   "{%- endfor -%}\n"
                   "{%- if add_generation_prompt -%}\n"
                   "  {{- '<|im_start|>assistant\\n' -}}\n"
                   "{%- endif -%}")
# the conversation a template given at startup must render (llama-server renders it at startup too)
TEMPLATE_EXAMPLE = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}, {"role": "user", "content": "How are you?"}]


def chat_template_source(kind, value):
    """the chat template of --chat-template-file PATH (kind "file": the file's text, UTF-8; CRLF line ends are read as
    LF, as llama-server on Windows reads it, and a byte-order mark is dropped) or --chat-template TEMPLATE (kind
    "inline": the Jinja source itself; "chatml" = llama.cpp's ChatML template), checked at startup: it must compile
    and render a short conversation with its messages in it. ValueError names the problem."""
    opt = "--chat-template-file" if kind == "file" else "--chat-template"
    if kind == "file":
        try:
            with open(value, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise ValueError(f"{opt}: cannot read {value}: {e.strerror or e}") from None
        try:
            src = raw.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError as e:
            raise ValueError(f"{opt}: {value} is not UTF-8 text ({e.reason} at byte {e.start})") from None
        what = value
    else:
        src, what = (CHATML_TEMPLATE if value == "chatml" else value), "the template"
    try:
        tmpl = make_template(src)
    except jinja2.exceptions.TemplateSyntaxError as e:
        raise ValueError(f"{opt}: {what} does not compile: {e.message} (line {e.lineno})") from None
    try:
        text = tmpl.render(messages=TEMPLATE_EXAMPLE, add_generation_prompt=True)
    except Exception as e:  # (raise_exception, an undefined name, or a Python error inside an expression)
        raise ValueError(f"{opt}: {what} fails on a short conversation (system, user, assistant, user): "
                         f"{type(e).__name__}: {e}") from None
    if TEMPLATE_EXAMPLE[-1]["content"] not in text:
        raise ValueError(f"{opt}: {what} does not render the conversation's messages: not a chat template"
                         + (" (the only built-in name is chatml)" if kind == "inline" else ""))
    return src


class ChatTemplateArg(argparse.Action):
    """--chat-template-file / --chat-template: one setting, the last one given wins (as in llama-server)"""

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, ("file" if option_string == "--chat-template-file" else "inline", value))


class Prompt(str):
    """a rendered prompt (LiteralTemplate.render) whose request content had special-token text: lit = (the prompt
    rendered with each such text as a placeholder character, {ord(placeholder): text}, the LiteralTemplate), from which
    Server.tokenize_prompt tokenizes those texts as plain text"""
    lit = None


class LiteralTemplate:
    """the chat template, rendering special-token text inside the request's messages and tools as text (--content-specials
    text). render() renders the request as it is (the prompt text is unchanged: /apply-template, token counts), and when
    that content has control-token text (<|im_end|>, <|im_start|>, ...: the vocabulary's control tokens) once more with
    each such text replaced by a private-use character the prompt does not contain. If that render, with the texts put
    back, is the same prompt (the template treats the characters as it treats the texts), the result is a Prompt telling
    tokenize_prompt which special-token texts came from the request: those become plain text (a tool result with
    "<|im_end|>\\n<|im_start|>system" cannot forge a turn), the template's own stay special tokens. Otherwise the request
    keeps llama-server's behaviour (logged)."""

    def __init__(self, template, controls):
        self.template, self.controls = template, controls  # controls: {text: token id} (Engine.controls)
        self.ctl_re = re.compile("|".join(re.escape(t) for t in sorted(controls, key=len, reverse=True)))
        self.warned = False

    def render(self, **ctx):
        text = self.template.render(**ctx)
        found = {}  # control text -> its placeholder
        free = (chr(c) for c in range(0x10FFFD, 0xFFFFF, -1) if chr(c) not in text)  # (plane 16: private use)

        def place(m):
            if m.group() not in found:
                found[m.group()] = next(free)
            return found[m.group()]

        def mark(x):
            if isinstance(x, str):
                return self.ctl_re.sub(place, x)
            if isinstance(x, dict):
                return {mark(k): mark(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [mark(v) for v in x]
            return x
        marked = dict(ctx, **{k: mark(ctx[k]) for k in ("messages", "tools") if k in ctx})
        if not found:
            return text
        back = {ord(p): t for t, p in found.items()}
        lit = self.template.render(**marked)
        if lit.translate(back) != text:
            if not self.warned:
                self.warned = True
                log("special-token text in a request's content: the chat template renders it differently once marked, so "
                    "it is tokenized as special tokens (as llama-server does); not logged again")
            return text
        p = Prompt(text)
        p.lit = (lit, back, self)
        return p


class BadRequest(Exception):
    pass


class ContextOverflow(BadRequest):
    """a prompt that does not fit the context (the reply needs RESERVE tokens): llama-server's exceed_context_size_error
    (400, with n_prompt_tokens and n_ctx, its message), and on /v1/messages Anthropic's "prompt is too long: N tokens
    > M maximum" (the text Claude Code reads before it compacts the conversation). at_least: counting stopped early
    (no more images decoded for a prompt that cannot run)"""
    RESERVE = 16

    def __init__(self, n_prompt, n_ctx, at_least=False):
        self.n_prompt, self.n_ctx = n_prompt, n_ctx
        n = f"at least {n_prompt}" if at_least else n_prompt
        super().__init__(f"request ({n} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it")
        self.anthropic = f"prompt is too long: {n_prompt} tokens > {n_ctx - self.RESERVE - 1} maximum"


class ClientGone(Exception):
    """the client closed its connection: the request stops, nothing is sent"""


# a client that is gone: closed, reset, or stalled past MK_TIMEOUT_S (a socket timeout is not a ConnectionError)
DISCONNECTS = (ConnectionError, TimeoutError)


class PrefillCancelled(Exception):
    """the engine stopped a prefill at a chunk boundary (Engine.cancel_when); nothing of it was kept"""


MEDIA_MARKER = "<__media__>"  # llama-server's (mtmd_default_marker)


def normalize_messages(messages, media=None):
    """media (a list: the server has a vision encoder): image_url parts and video parts (input_video / video_url)
    become MEDIA_MARKER and their file bytes are appended to media in order (videos as VideoBytes). A message with
    media is flattened to one string as llama-server does (text parts joined by newlines, no newline next to an image
    or video), so the prompt around each is llama-server's."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise BadRequest("messages: each message must be an object")
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text", "image_url", *VIDEO_PARTS):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text, image_url, "
                                     f"input_video and video_url only)")
            kinds = {p.get("type") for p in content if isinstance(p, dict)}
            if kinds & {"image_url", *VIDEO_PARTS}:
                if media is None:
                    what = "image" if "image_url" in kinds else "video"
                    raise BadRequest(f"{what} input is not supported: start the server with --mmproj")
                text, after_media = "", False
                for part in content:
                    if part.get("type") in VIDEO_PARTS:
                        media.append(video_bytes(part))
                        text += MEDIA_MARKER
                        after_media = True
                    elif part.get("type") == "image_url":
                        iu = part.get("image_url")
                        media.append(image_bytes(iu.get("url") if isinstance(iu, dict) else iu))
                        text += MEDIA_MARKER
                        after_media = True
                    else:
                        if not after_media and text:
                            text += "\n"
                        text += str(part.get("text") or "")
                        after_media = False
                m["content"] = text
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


# Qwen's XML tool calls: <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>.
# A call counts only with both closing tags: one cut by max_tokens (or left open) is not a call. A value runs to the next
# </parameter> (as llama.cpp's parser reads it), so it may contain any other markup, <tool_call> included.
TOOL_OPEN = "<tool_call>"
TOOL_HEAD_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>")
TOOL_END_RE = re.compile(r"</function>\s*</tool_call>")
PARAM_HEAD_RE = re.compile(r"<parameter=([^>\n]+)>")
TAG_NAME_RE = re.compile(r"[^>\n]*")  # (a name that may still be completing: nothing that ends it yet)
SPACE_RE = re.compile(r"\s*")


class ToolCallParser:
    """Qwen's tool calls in the generated text (from its first <tool_call>), parsed as the text arrives. `calls`: the
    complete calls in OpenAI's shape. With events, feed() also returns each call as it is generated: ("start", index,
    id, name) once its function name is complete, then ("args", index, fragment) pieces of its arguments' JSON text: a
    string parameter's value as it is generated (JSON-escaped), any other value whole at its end (it is decoded as JSON
    there), the closing brace with the call's closing tags, so the arguments become valid JSON only when the call is
    complete. A complete call's fragments add up to its arguments byte for byte (a parameter given twice: both values
    are sent, the arguments hold the last one, as a JSON parser reads the sent text). A parameter is a string when
    its schema type is "string", or it has no type and does not start like a JSON value.
    one (parallel_tool_calls false): no call is started in the stream after a complete one. Decisions do not depend on
    how the text is split into pieces: a tag that may still be completing waits for more text."""

    def __init__(self, tools, events=False, one=False):
        self.schemas = {}
        for t in tools or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            props = (fn.get("parameters") or {}).get("properties") if isinstance(fn, dict) else None
            self.schemas[fn.get("name") if isinstance(fn, dict) else None] = props if isinstance(props, dict) else {}
        self.events, self.one = events, one
        self.text, self.pos, self.state = "", 0, "seek"
        self.calls = []
        self.end = 0  # where the last complete call ends in text
        self.started = 0  # calls started in the stream
        self.open = []  # where the calls started in the stream that did not complete begin
        self.cur = None  # the call being read

    def feed(self, text, final=False):
        """the stream events of text (all of them by the final call)"""
        self.text += text
        out = []
        step = {"seek": self._seek, "head": self._head, "between": self._between, "value": self._value}
        while self.state != "done" and step[self.state](final, out):
            pass
        if final:
            self.state = "done"
        return out

    def unparsed(self, streamed=False):
        """the tool-call markup after the last complete call that is not a call itself (cut by max_tokens, left open or
        malformed), from its <tool_call> to the end; empty if there is none. streamed: only the part before a call that
        was started in the stream (the client has that one as a call already)"""
        u = self.text.find(TOOL_OPEN, self.end)
        if u < 0:
            return ""
        cut = min((s for s in self.open if s >= u), default=len(self.text)) if streamed else len(self.text)
        return self.text[u:cut]

    def _seek(self, final, out):
        i = self.text.find(TOOL_OPEN, self.pos)
        if i < 0:
            self.pos = max(self.pos, len(self.text) - len(TOOL_OPEN) + 1)
            return False
        self.start, self.pos, self.state = i, i, "head"
        return True

    def _head(self, final, out):
        t = self.text
        m = TOOL_HEAD_RE.match(t, self.start)
        if m is None:
            j = SPACE_RE.match(t, self.start + len(TOOL_OPEN)).end()
            if not final and ("<function=".startswith(t[j:]) if len(t) - j < 10 else
                              t.startswith("<function=", j) and TAG_NAME_RE.fullmatch(t, j + 10)):
                return False  # (it may still become a call)
            self.pos, self.state = self.start + len(TOOL_OPEN), "seek"  # (not a call)
            return True
        name = m.group(1).strip()
        props = self.schemas.get(name, {})
        self.cur = c = {"start": self.start, "name": name, "id": "call_" + uuid.uuid4().hex[:24], "props": props,
                        "args": {}, "n": 0, "index": None}
        if self.events and not (self.one and self.calls):
            c["index"], self.started = self.started, self.started + 1
            out.append(("start", c["index"], c["id"], name))
        self.pos, self.state = m.end(), "between"
        return True

    def _between(self, final, out):
        """after the function name or a parameter: the next parameter or the closing tags (other text is skipped, a new
        <tool_call> makes this one not a call)"""
        t, c = self.text, self.cur
        hits = [(j, tag) for tag in ("<parameter=", "</function>", TOOL_OPEN) for j in (t.find(tag, self.pos),) if j >= 0]
        if not hits:
            if final:
                return self._drop()  # (left open)
            self.pos = max(self.pos, len(t) - 10)
            return False
        j, tag = min(hits)
        if tag == TOOL_OPEN:  # (a new call before this one's closing tags: this one was left open)
            self._drop()
            self.pos, self.state = j, "seek"
            return True
        if tag == "</function>":
            m = TOOL_END_RE.match(t, j)
            if m is not None:
                self._arg("}" if c["n"] else "{}", out)
                self.calls.append({"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}})
                self.cur, self.end, self.pos, self.state = None, m.end(), m.end(), "seek"
                return True
            k = SPACE_RE.match(t, j + 11).end()
            if not final and len(t) - k < 12 and "</tool_call>".startswith(t[k:]):
                self.pos = j
                return False
            self.pos = j + 11
            return True
        m = PARAM_HEAD_RE.match(t, j)
        if m is None:
            if not final and TAG_NAME_RE.fullmatch(t, j + 11):
                self.pos = j
                return False
            self.pos = j + 11
            return True
        vs = m.end()
        if vs == len(t):  # (a newline right after the tag is not part of the value)
            if final:
                return self._drop()
            self.pos = j
            return False
        vs += t[vs] == "\n"
        key = m.group(1).strip()
        spec = c["props"].get(key)
        typ = spec.get("type") if isinstance(spec, dict) else None
        c.update(key=key, vstart=vs, vscan=vs, vsent=vs, kind="string" if typ == "string" else None if typ is None else "json")
        self.state = "value"
        return True

    def _value(self, final, out):
        t, c = self.text, self.cur
        k = t.find("</parameter>", c["vscan"])
        if k < 0:
            if final:  # (a stream gets the string's text to the end, the call stays incomplete)
                self._send_value(len(t), False, out)
                return self._drop()
            c["vscan"] = max(c["vstart"], len(t) - 11)
            self._send_value(max(c["vstart"], len(t) - 12), False, out)  # (a newline and the closing tag may follow)
            return False
        end = k - 1 if k > c["vstart"] and t[k - 1] == "\n" else k
        self._send_value(end, True, out)
        self.pos, self.state = k + len("</parameter>"), "between"
        return True

    def _send_value(self, upto, done, out):
        """the value's text up to `upto` (done: its end): a string's as it comes, another value's at its end"""
        t, c = self.text, self.cur
        if c["kind"] is None:  # (no type in the schema: decided by its first character)
            head = t[c["vstart"]:upto].strip()[:1]
            if not head and not done:
                return
            c["kind"] = "json" if head in "[{0123456789-tfn" else "string"
        key = json.dumps(c["key"], ensure_ascii=False)
        if c["kind"] == "string":
            if c["vsent"] == c["vstart"] and not c.get("opened"):
                c["opened"] = True
                self._arg((", " if c["n"] else "{") + key + ': "', out)
                c["n"] += 1
            if upto > c["vsent"]:
                self._arg(json.dumps(t[c["vsent"]:upto], ensure_ascii=False)[1:-1], out)
                c["vsent"] = upto
            if done:
                self._arg('"', out)
                c["args"][c["key"]] = t[c["vstart"]:upto]
                c["opened"] = False
            return
        if not done:
            return
        val = t[c["vstart"]:upto]
        try:
            val = json.loads(val)
        except (ValueError, RecursionError):  # (not JSON: the text itself)
            pass
        self._arg((", " if c["n"] else "{") + key + ": " + json.dumps(val, ensure_ascii=False), out)
        c["n"] += 1
        c["args"][c["key"]] = val

    def _arg(self, frag, out):
        if self.cur["index"] is not None:
            out.append(("args", self.cur["index"], frag))

    def _drop(self):
        """the call being read is not a call: a new <tool_call> before its closing tags, or (returns False: the end)
        left open at the end of the text, whose unclosed value or tag may hold anything (no call is looked for in it)"""
        if self.cur["index"] is not None:
            self.open.append(self.cur["start"])
        self.cur, self.state = None, "seek"
        return False


def unparsed_tool_call(text):
    """the tool-call markup after the last complete call that is not a call itself (ToolCallParser.unparsed)"""
    p = ToolCallParser([])
    p.feed(text, final=True)
    return p.unparsed()


def parse_tool_calls(text, tools):
    """the complete tool calls in text (ToolCallParser)"""
    p = ToolCallParser(tools)
    p.feed(text, final=True)
    return p.calls


def _off(limit, above=False):
    """a check that a sampler field has its 'off' value: None, or <= limit (above: >= limit)"""
    def ok(v):
        if v is None:
            return True
        try:
            return float(v) >= limit if above else float(v) <= limit
        except (TypeError, ValueError):
            return False
    return ok


# llama.cpp sampler fields the engine does not have (it samples with temperature, top_k, top_p, min_p and the presence /
# frequency penalties): accepted only with the value that turns them off (llama.cpp's web UI sends them all)
SAMPLER_OFF = {"mirostat": _off(0), "typical_p": _off(1, True), "typ_p": _off(1, True), "tfs_z": _off(1, True),
               "xtc_probability": _off(0), "dry_multiplier": _off(0), "top_n_sigma": _off(0), "dynatemp_range": _off(0),
               "min_tokens": _off(0), "n_probs": _off(0)}


# llama-server's --reasoning-format values: auto = deepseek (reasoning_content, streamed too: the default),
# deepseek-legacy (the content keeps the thinking block too; a stream only there), none (the thinking stays in the content)
REASONING_FORMATS = ("auto", "deepseek", "deepseek-legacy", "none")
# the chat_template_kwargs the server hands to the template
TEMPLATE_KWARGS = ("enable_thinking", "preserve_thinking", "reasoning_effort")
SOFT_SWITCH = re.compile(r"(/no_think|/think|<think>\s*</think>)\s*$")


def soft_switch(messages):
    """Qwen3's soft switches as a client sends them (Qwen3.8's template ignores them, so the server applies them): the
    last user message's text, else the leading system / developer message's, ending in /no_think (or an empty
    <think></think> block, as after the last user text or as a trailing assistant message) -> False, in /think -> True;
    None: no switch. The prompt keeps the text as sent."""
    def text(m):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):  # (parts: the last text one)
            c = next((p["text"] for p in reversed(c) if isinstance(p, dict) and isinstance(p.get("text"), str)), None)
        return c if isinstance(c, str) else ""

    def switch(s):
        m = SOFT_SWITCH.search(s)
        return None if m is None else m.group(1) == "/think"
    msgs = [m for m in messages if isinstance(m, dict)]
    if msgs and msgs[-1].get("role") == "assistant" and re.fullmatch(r"\s*<think>\s*</think>\s*", text(msgs[-1])):
        return False
    user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    s = switch(text(user)) if user is not None else None
    if s is None and msgs and msgs[0].get("role") in ("system", "developer"):
        s = switch(text(msgs[0]))
    return s


# llama.cpp's default sampler chain (its `samplers` field: names, or a string of their letters); the engine applies
# penalties, top_k, top_p, min_p and temperature in this order
SAMPLER_CHAIN = ("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature")
SAMPLER_NAMES = {**{a: n for n in SAMPLER_CHAIN + ("infill", "adaptive_p") for a in (n, n.replace("_", "-"), n.replace("_", ""))},
                 **dict(zip("edskypmxti", SAMPLER_CHAIN + ("infill",))), "nucleus": "top_p", "temp": "temperature",
                 "typ": "typ_p"}


def check_samplers(v, penalties):
    """a request's `samplers` is accepted when it asks for what the engine does: top_k, top_p, min_p and temperature in
    llama.cpp's order (the other samplers in their default places: their parameters are checked to be off), penalties
    too when a penalty is set; unknown names are dropped, as llama.cpp does"""
    if isinstance(v, str):
        names = [SAMPLER_NAMES.get(c) for c in v]
    elif isinstance(v, list) and all(isinstance(n, str) for n in v):
        names = [SAMPLER_NAMES.get(n.lower()) for n in v]
    else:
        raise BadRequest("samplers: an array of sampler names or a string of sampler letters")
    names = [n for n in names if n]
    order = [SAMPLER_CHAIN.index(n) if n in SAMPLER_CHAIN else -1 for n in names]
    need = {"top_k", "top_p", "min_p", "temperature"} | ({"penalties"} if penalties else set())
    if min(order, default=0) < 0 or order != sorted(set(order)) or not need <= set(names):
        raise BadRequest(f"samplers: {v!r} is not supported: the megakernel samples with "
                         f"{', '.join(n for n in SAMPLER_CHAIN if n in need)} in llama.cpp's order (its default chain)")


def check_params(body):
    """400 for the request fields the server cannot honour (they would change the output and need engine work, e.g.
    logprobs, n > 1, constrained sampling): never ignored silently. Harmless ones (user, metadata, store, cache_prompt,
    llama-server's n_keep / n_discard (no context shift), response_fields / return_tokens (its /completion only),
    n_cache_reuse, sse_ping_interval, verbose) are accepted; samplers only in the order the engine samples."""
    def bad(field, why):
        raise BadRequest(f"{field}: {why}")
    if not isinstance(body.get("messages"), list):
        bad("messages", "an array of messages is required")
    for k in ("n", "n_cmpl"):
        if body.get(k) not in (None, 1):
            bad(k, "one choice per request (the engine runs one sequence): send several requests instead")
    if body.get("id_slot") not in (None, -1, 0):
        bad("id_slot", "the server has one slot, 0 (-1: any)")
    if body.get("n_indent") not in (None, 0):
        bad("n_indent", "not supported (llama-server's infill option: it ends a reply on indentation)")
    t = body.get("t_max_predict_ms")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or t != t):
        bad("t_max_predict_ms", "a number of milliseconds (<= 0: no limit)")
    if body.get("samplers") is not None:
        try:
            pen = any(float(body.get(k) or 0) for k in ("presence_penalty", "frequency_penalty"))
        except (TypeError, ValueError):
            pen = False  # (sampling() refuses it)
        check_samplers(body["samplers"], pen)
    if body.get("logprobs") or body.get("top_logprobs") not in (None, 0):
        bad("logprobs", "token log-probabilities are not available from the megakernel (they need engine support)")
    if body.get("logit_bias"):
        bad("logit_bias", "not supported by the megakernel's sampler")
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type", "text") == "text"):
        bad("response_format", "only {\"type\": \"text\"}: JSON modes need constrained sampling (a grammar), which the "
                               "megakernel does not have")
    for k in ("json_schema", "grammar"):
        if body.get(k):
            bad(k, "constrained sampling (a grammar) is not supported by the megakernel")
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        bad("tool_choice", "\"auto\" or \"none\": forcing a tool call (required / a named function) needs constrained "
                           "sampling, which the megakernel does not have")
    if body.get("tools") is not None and not isinstance(body["tools"], list):
        bad("tools", "an array of tools")
    if body.get("functions") or body.get("function_call"):
        bad("functions", "the legacy function-calling fields are not supported: use tools")
    if body.get("ignore_eos"):
        bad("ignore_eos", "not supported by the megakernel")
    if body.get("audio") or "audio" in (body.get("modalities") or []):
        bad("audio", "audio output is not supported")
    if body.get("add_generation_prompt") is False or body.get("continue_final_message"):
        bad("add_generation_prompt", "continuing the final message is not supported")
    for k, ok in SAMPLER_OFF.items():
        if not ok(body.get(k)):
            bad(k, f"{body.get(k)!r} is not supported: the megakernel samples with temperature, top_k, top_p, min_p "
                   "and the presence / frequency penalties only")
    if body.get("reasoning_format") is not None and body["reasoning_format"] not in REASONING_FORMATS:
        bad("reasoning_format", f"one of {', '.join(REASONING_FORMATS)}")
    if body.get("chat_template_kwargs") is not None and not isinstance(body["chat_template_kwargs"], dict):
        bad("chat_template_kwargs", "an object")


def stop_strings(v, field="stop"):
    """the stop strings of a request: a string or a list of strings (empty ones dropped, as llama-server does)"""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise BadRequest(f"{field} must be a string or a list of strings")
    v = [s for s in v if s]
    if len(v) > MAX_STOP or any(len(s) > MAX_STOP_LEN for s in v):
        raise BadRequest(f"{field}: at most {MAX_STOP} strings of at most {MAX_STOP_LEN} characters")
    return v


class StopMatcher:
    """Cuts the generated text at the first occurrence of a stop string, as llama-server does: on the raw text (reasoning,
    content and tool-call markup alike), the stop string and everything after it dropped. Text that could still become
    the start of a stop string is held back until the next piece decides it (or the end releases it)."""

    def __init__(self, stops):
        self.stops = stops
        self.held = ""
        self.word = None  # the stop string found

    def _partial(self, buf):
        """length of the longest suffix of buf that is a proper prefix of a stop string"""
        best = 0
        for s in self.stops:
            j = buf.find(s[0], max(0, len(buf) - len(s) + 1))  # (scanned from the longest candidate)
            while 0 <= j and len(buf) - j > best:
                if s.startswith(buf[j:]):
                    best = len(buf) - j
                    break
                j = buf.find(s[0], j + 1)
        return best

    def feed(self, text, final=False):
        """the text that can go out now; after a match (self.word set) nothing more"""
        if not self.stops or self.word is not None:
            return text if self.word is None else ""
        buf = self.held + text
        # (a match cannot start in text released before: its start would have been held)
        at = -1
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (at < 0 or i < at):
                at, self.word = i, s
        if at >= 0:
            self.held = ""
            return buf[:at]
        keep = 0 if final else self._partial(buf)
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool calls (from the first <tool_call>:
    ToolCallParser). calls: the tool calls' stream events too, as ("tool_call", delta) with delta in the shape of an
    OpenAI stream's delta.tool_calls[] entry; one: parallel_tool_calls false. inline (reasoning_format): "only" (none):
    the thinking block stays in the content as generated, with the <think> the prompt opened it with; "both"
    (deepseek-legacy): there and as reasoning."""

    def __init__(self, thinking, tools, calls=False, one=False, inline=None):
        self.mode = "reasoning" if thinking else "content_start"
        self.inline, self.opened = inline if thinking else None, False
        self.tools = tools
        self.tc = ToolCallParser(tools, calls, one) if tools else None
        self.buf = ""
        self.tool_text = ""
        self.tool_sep = ""  # the whitespace between the content and the first <tool_call>
        self.content_seen = False

    @property
    def calls(self):
        """the complete tool calls (OpenAI's shape)"""
        return self.tc.calls if self.tc else []

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}, and ("tool_call", delta) (calls)"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    self._think(out, self.buf[:i], True)
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content" if self.inline else "content_start"  # (inline: the text as generated)
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    self._think(out, self.buf[:len(self.buf) - keep], False)
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        head = self.buf[:i].rstrip()
                        out.append(("content", head))
                        self.tool_sep = self.buf[len(head):i]
                        self.buf = self.buf[i:]
                        self.mode = "tools"
                        continue
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: everything from the first <tool_call>
            self.tool_text += self.buf
            for ev in self.tc.feed(self.buf, final):
                if ev[0] == "start":
                    out.append(("tool_call", {"index": ev[1], "id": ev[2], "type": "function",
                                              "function": {"name": ev[3], "arguments": ""}}))
                else:
                    out.append(("tool_call", {"index": ev[1], "function": {"arguments": ev[2]}}))
            self.buf = ""
            break
        out = [(k, t) for k, t in out if t]
        self.content_seen = self.content_seen or any(k == "content" for k, _ in out)
        return out

    def _think(self, out, text, end):
        """reasoning text (end: up to its </think>)"""
        if self.inline != "only":
            out.append(("reasoning", text))
        if self.inline:
            out.append(("content", ("" if self.opened else "<think>\n") + text + ("</think>" if end else "")))
            self.opened = True

    def unparsed(self, streamed=False):
        """the tool-call markup that is not a complete call (ToolCallParser.unparsed), as the rest of the content: the
        client gets the text, as from llama-server, instead of a call it would execute half-written. streamed: without
        a call the stream started (its client has it as a call whose arguments never became valid JSON)"""
        rest = self.tc.unparsed(streamed) if self.tc else ""
        return (self.tool_sep if self.content_seen else "") + rest if rest else ""


# ---------------------------------------------------------------------------------------------
# Anthropic Messages API (POST /v1/messages, /v1/messages/count_tokens): translated to the chat request above and back,
# so one code path generates for both (as llama-server's server_chat_convert_anthropic_to_oai)

# Claude Code's system prompt starts with a billing header whose cch= value changes on every request: normalized (as
# llama-server does), or the prompt would differ from its first tokens on and no prefix could be reused
BILLING_RE = re.compile(r"^(x-anthropic-billing-header:.*?cch=)[^;\s]{5};")


def _anthropic_image(src):
    if not isinstance(src, dict):
        raise BadRequest("image: a source object is required")
    if src.get("type") == "base64":
        return f"data:{src.get('media_type') or 'image/jpeg'};base64,{src.get('data') or ''}"
    if src.get("type") == "url":
        return str(src.get("url") or "")  # (refused by image_bytes: only data: URLs)
    raise BadRequest(f"image source type {src.get('type')!r} is not supported (base64)")


def _anthropic_document(b, notes):
    """a document block's text: plain-text and content sources (title and context first); None for the others (PDF,
    URL, file: skipped with a note)"""
    src = b.get("source") if isinstance(b.get("source"), dict) else {}
    if src.get("type") == "text":
        text = str(src.get("data") or "")
    elif src.get("type") == "content" and isinstance(src.get("content"), (str, list)):
        c = src["content"]
        text = c if isinstance(c, str) else "\n".join(
            str(x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    else:
        notes.append(("anthropic-doc-" + str(src.get("type")), f"Anthropic document source {src.get('type')!r} is not "
                       "supported (text and content sources are): the document is skipped"))
        return None
    return "\n\n".join(s for s in (str(b.get("title") or ""), str(b.get("context") or ""), text) if s)


def _skipped_block(t, notes, where="content"):
    notes.append((f"anthropic-block-{t}", f"Anthropic {where} block type {t!r} is not supported: skipped (as llama-server)"))


def _tool_result(content, notes):
    """a tool_result block's content as an OpenAI tool message's: a string, or parts when it has images"""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise BadRequest("tool_result content: a string or an array of blocks")
    parts = []
    for c in content:
        if not isinstance(c, dict):
            raise BadRequest("tool_result content: blocks must be objects")
        t = c.get("type")
        if t == "text":
            parts.append({"type": "text", "text": str(c.get("text") or "")})
        elif t == "image":
            parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(c.get("source"))}})
        elif t == "document":
            text = _anthropic_document(c, notes)
            if text is not None:
                parts.append({"type": "text", "text": text})
        else:
            _skipped_block(t, notes, "tool_result content")
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "\n".join(p["text"] for p in parts)


def takes_mid_system(template):
    """whether a chat template renders a system message after the first message (Qwen's raise "System message must be
    at the beginning")"""
    try:
        return "MIDSYS" in template.render(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                                                     {"role": "system", "content": "MIDSYS"}], add_generation_prompt=True)
    except Exception:
        return False


def anthropic_to_chat(body, notes=None, mid_system=True):
    """an Anthropic Messages request as the server's chat request. system: a string or text blocks (joined by blank
    lines); content blocks text / image / thinking (-> reasoning_content) / tool_use (-> tool_calls) / tool_result (->
    tool messages, before the rest of their message: they answer the previous turn's calls) / document (plain-text and
    content sources -> text); other block types are skipped, as llama-server does; tools input_schema -> parameters,
    any tool type (Anthropic's typed client and server tools too) as a function with its input_schema or none, as
    llama-server does; tool_choice auto / none / any / tool; stop_sequences -> stop; thinking enabled / adaptive ->
    thinking on, disabled or absent -> off (Anthropic's default; chat_template_kwargs win). notes: gets (key, message)
    for what was skipped or converted (the console's one-time warnings). mid_system False (the chat template takes a
    system message only at the start: takes_mid_system): a system message inside the conversation becomes a user
    message in its place"""
    notes = [] if notes is None else notes
    if not isinstance(body.get("messages"), list):
        raise BadRequest("messages: an array of messages is required")
    msgs = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            texts = [system]
        elif isinstance(system, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in system):
            texts = [str(b.get("text") or "") for b in system]
        else:
            raise BadRequest("system: a string or an array of text blocks")
        msgs.append({"role": "system", "content": "\n\n".join(BILLING_RE.sub(r"\1fffff;", t) for t in texts)})
    for m in body["messages"]:
        # (a system message inside the conversation: Claude Code sends some; the chat template decides whether it can
        # render one there, as llama-server leaves it to the template)
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
            raise BadRequest("messages: each message needs the role user, assistant or system")
        role, content = m["role"], m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise BadRequest("message content: a string or an array of content blocks")
        parts, calls, results, thinking = [], [], [], ""
        for b in content:
            if not isinstance(b, dict):
                raise BadRequest("message content: content blocks must be objects")
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": str(b.get("text") or "")})
            elif t == "image":
                parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(b.get("source"))}})
            elif t == "thinking":
                thinking += str(b.get("thinking") or "")
            elif t == "redacted_thinking":
                pass
            elif t == "tool_use":
                calls.append({"id": str(b.get("id") or ""), "type": "function", "function": {
                    "name": str(b.get("name") or ""), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
            elif t == "tool_result":
                results.append({"role": "tool", "tool_call_id": str(b.get("tool_use_id") or ""),
                                "content": _tool_result(b.get("content"), notes)})
            elif t == "document":
                text = _anthropic_document(b, notes)
                if text is not None:
                    parts.append({"type": "text", "text": text})
            else:  # (server tool blocks, search results, container uploads, ...)
                _skipped_block(t, notes)
        msgs += results
        if parts or calls or thinking:
            nm = {"role": role, "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text"
                  else (parts or "")}
            if calls:
                nm["tool_calls"] = calls
            if thinking:
                nm["reasoning_content"] = thinking
            msgs.append(nm)
    if not mid_system:  # (Claude Code sends its environment and skills list as a system message after the first one)
        for i, m in enumerate(msgs):
            if i and m["role"] == "system":
                m["role"] = "user"
                notes.append(("anthropic-mid-system", "Anthropic system messages inside the conversation are rendered as "
                              "user messages in their place: the chat template takes a system message only at the start"))
    out = {"messages": msgs, "stream": bool(body.get("stream"))}
    out["max_tokens"] = body.get("max_tokens") if body.get("max_tokens") is not None else 4096  # (llama-server's)
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences") is not None:
        out["stop"] = stop_strings(body["stop_sequences"], "stop_sequences")
    tools = body.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise BadRequest("tools: an array of tools")
        out["tools"] = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                raise BadRequest(f"tools: each tool needs a name, not {str(t)[:80]!r}")
            if t.get("type") not in (None, "custom"):  # (e.g. bash_20250124, web_search_20250305)
                notes.append((f"anthropic-tool-{t['type']}", f"Anthropic tool type {t['type']!r} ({t['name']}) is offered "
                              f"to the model as a plain function{'' if t.get('input_schema') else ' without parameters'}"
                              " (as llama-server): a call to it comes back as a tool_use block for the client"))
            out["tools"].append({"type": "function", "function": {
                "name": str(t["name"]), "description": t.get("description") or "", "parameters": t.get("input_schema") or {}}})
    tc = body.get("tool_choice")
    if tc is not None:
        kind = tc.get("type") if isinstance(tc, dict) else None
        if kind in ("any", "tool"):
            raise BadRequest(f"tool_choice {kind}: forcing a tool call needs constrained sampling, which the megakernel "
                             "does not have (auto, none)")
        if kind not in ("auto", "none"):
            raise BadRequest("tool_choice: {\"type\": \"auto\"} or {\"type\": \"none\"}")
        out["tool_choice"] = kind
        if tc.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    if body.get("mcp_servers"):
        raise BadRequest("mcp_servers: not supported (the client runs its tools)")
    think = body.get("thinking")
    kw = dict(body.get("chat_template_kwargs") or {})
    if think is not None and not (isinstance(think, dict) and think.get("type") in ("enabled", "adaptive", "disabled")):
        raise BadRequest("thinking: {\"type\": \"enabled\" | \"adaptive\" | \"disabled\"}")
    kw.setdefault("enable_thinking", bool(think) and think["type"] != "disabled")
    out["chat_template_kwargs"] = kw
    return out


def anthropic_stop_reason(finish, stop_word):
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls":
        return "tool_use"
    return "stop_sequence" if stop_word is not None else "end_turn"


def anthropic_usage(usage):
    reused = usage["prompt_tokens_details"]["cached_tokens"]
    return {"input_tokens": usage["prompt_tokens"] - reused, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": reused, "output_tokens": usage["completion_tokens"]}


def anthropic_tool_use(call):
    try:
        args = json.loads(call["function"]["arguments"])
    except ValueError:
        args = {}
    return {"type": "tool_use", "id": "toolu_" + call["id"].removeprefix("call_"), "name": call["function"]["name"],
            "input": args if isinstance(args, dict) else {}}


ANTHROPIC_ERRORS = {400: "invalid_request_error", 404: "not_found_error", 413: "request_too_large",
                    500: "api_error", 501: "api_error", 503: "overloaded_error"}


# ---------------------------------------------------------------------------------------------
# OpenAI Responses API (POST /v1/responses, /v1/responses/input_tokens): translated to the chat request above and back,
# as the Anthropic Messages API is (llama-server's server_chat_convert_responses_to_chatcmpl). Stateless: the client
# sends the whole conversation every time (store false, as Codex does); nothing is stored, so previous_response_id,
# conversation and item_reference are a 400.

# a reasoning item's encrypted_content (include reasoning.encrypted_content): this prefix + the reasoning text, zlib +
# base64. Not secret (the item has the text as reasoning_text content too): it brings the reasoning back for clients that
# return only encrypted_content, so the next prompt renders the turn as it was generated
RESP_REASONING = "mk-reasoning-1:"
RESP_KEEPALIVE_S = 10.0  # a stream's response.in_progress events during a long prefill (clients time out idle streams)
# bytes all of a request's encrypted_content may inflate to: the model's full context x its longest token (set in main)
REASONING_MAX_BYTES = 262144 * 256


def reasoning_seal(text):
    return RESP_REASONING + base64.b64encode(zlib.compress(text.encode(), 6)).decode()


def reasoning_open(s, budget=None):
    """this server's encrypted_content as text, else None; budget: [bytes left for the request], past it a BadRequest"""
    if not isinstance(s, str) or not s.startswith(RESP_REASONING):
        return None
    budget = [REASONING_MAX_BYTES] if budget is None else budget
    d = zlib.decompressobj()
    try:  # (max_length budget + 1 > 0: zlib reads 0 as no limit)
        raw = d.decompress(base64.b64decode(s[len(RESP_REASONING):], validate=True), budget[0] + 1)
    except (binascii.Error, zlib.error, ValueError):
        return None
    if len(raw) > budget[0]:
        raise BadRequest(f"reasoning encrypted_content: the request's reasoning inflates to more than the model's full "
                         f"context could hold ({REASONING_MAX_BYTES} bytes in all)")
    budget[0] -= len(raw)
    if not d.eof or d.unconsumed_tail or d.unused_data:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return None


def _resp_content(content, where):
    """a message item's or function output's content (a string, or input_text / output_text / refusal / input_image
    parts) as chat content: one string (text parts joined as the chat template joins a part list: nothing between),
    or with images the chat parts (normalize_messages flattens them as llama-server does)"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest(f"{where}: a string or an array of content parts")
    parts = []
    for p in content:
        if not isinstance(p, dict):
            raise BadRequest(f"{where}: content parts must be objects")
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(p.get("text") or "")})
        elif t == "refusal":
            parts.append({"type": "text", "text": str(p.get("refusal") or "")})
        elif t == "input_image":
            url = p.get("image_url")
            if not url and p.get("file_id"):
                raise BadRequest("input_image: file_id is not supported (the server stores no files): send the image as "
                                 "a data: URL in image_url")
            parts.append({"type": "image_url", "image_url": {"url": str((url.get("url") if isinstance(url, dict) else url)
                                                                     or "")}})
        elif t == "input_file":
            raise BadRequest("input_file is not supported: send the file's text as input_text (images as input_image)")
        else:
            raise BadRequest(f"{where}: content part type {t!r} is not supported (input_text, output_text, refusal, "
                             "input_image)")
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "".join(p["text"] for p in parts)


def responses_tools(tools, notes):
    """a Responses request's tools as chat tools: (tools, {function name: its namespace}, {custom tool names}).
    function -> function; namespace -> its functions (plain names; a call to one gets its namespace back); custom
    (freeform input) -> a function with one string parameter "input"; the others (web_search, file_search, mcp, ...:
    tools the server would have to run) are dropped with a note"""
    out, ns, custom = [], {}, set()

    def add(t, space=None):
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest(f"tools: a {t.get('type')} tool needs a name")
        if any(o["function"]["name"] == name for o in out):
            raise BadRequest(f"tools: two tools are named {name!r} (namespaces are flattened)")
        f = {"name": name, "description": str(t.get("description") or "")}
        if t.get("type") == "custom":
            fmt = t.get("format") if isinstance(t.get("format"), dict) else {}
            what = "The tool's input, free-form text"
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                what += " in this " + (f"{fmt['syntax']} " if fmt.get("syntax") else "") + f"grammar:\n{fmt['definition']}"
            f["parameters"] = {"type": "object", "properties": {"input": {"type": "string", "description": what}},
                               "required": ["input"]}
            custom.add(name)
        else:
            f["parameters"] = t.get("parameters") if isinstance(t.get("parameters"), dict) else {}
        if space:
            ns[name] = space
        out.append({"type": "function", "function": f})

    for t in tools:
        if not isinstance(t, dict):
            raise BadRequest("tools: each tool must be an object")
        kind = t.get("type")
        if kind in ("function", "custom"):
            add(t)
        elif kind == "namespace":
            for sub in t.get("tools") or []:
                if not isinstance(sub, dict) or sub.get("type") not in ("function", "custom"):
                    raise BadRequest(f"tools: namespace {t.get('name')!r}: function or custom tools only")
                add(sub, str(t.get("name") or ""))
        else:
            notes.append((f"responses-tool-{kind}", f"Responses tool type {kind!r} is not supported (the server runs no "
                          "tools): it is not offered to the model"))
    return out, ns, custom


def responses_to_chat(body, notes=None, mid_system=True):
    """a Responses API request as the server's chat request: (chat body, {"ns": function namespaces, "custom": custom
    tool names}: what the output items need). input: a string (one user message) or items: messages (roles user,
    assistant, system, developer -> system), reasoning (-> reasoning_content: its reasoning_text content, else this
    server's encrypted_content, else its summary), function_call / custom_tool_call (-> tool_calls) and their outputs
    (-> tool messages); the assistant-side items of one turn become one assistant message in generation order.
    instructions and the leading system / developer messages become one system message (their trimmed texts joined by
    newlines, as Qwen's templates merge them); a later one becomes a user message in its place when the template takes
    a system message only at the start (mid_system False). Whatever would change the output and cannot be honoured is
    a BadRequest (stored state, forced tool calls, JSON output, logprobs, files). notes: gets (key, message) for what was
    dropped or converted (the console's one-time warnings)"""
    notes = [] if notes is None else notes

    def bad(field, why):
        raise BadRequest(f"{field}: {why}")
    for k in ("previous_response_id", "conversation"):
        if body.get(k):
            bad(k, "not supported: the server stores no responses (send the whole conversation in input, with store "
                   "false)")
    if body.get("background"):
        bad("background", "not supported: the server stores no responses")
    if body.get("prompt"):
        bad("prompt", "stored prompt templates are not supported")
    inp = body.get("input")
    items = [{"type": "message", "role": "user", "content": inp}] if isinstance(inp, str) else inp
    if not isinstance(items, list):
        bad("input", "a string or an array of input items is required")
    msgs = []
    budget = [REASONING_MAX_BYTES]  # (one decompression budget for all of the request's encrypted_content)
    if body.get("instructions") is not None:
        if not isinstance(body["instructions"], str):
            bad("instructions", "a string")
        msgs.append({"role": "system", "content": body["instructions"]})
    # stage of the last message when it is an assistant message: 0 reasoning, 1 text, 2 tool calls (the order the model
    # generates them in, one reasoning and one text each); an item that cannot come next starts a new assistant message
    stage = [None]

    def turn(s):
        if stage[0] is None or stage[0] > s or (stage[0] == s and s < 2):
            msgs.append({"role": "assistant", "content": ""})
        stage[0] = s
        return msgs[-1]

    def other(m):
        msgs.append(m)
        stage[0] = None

    for it in items:
        if not isinstance(it, dict):
            bad("input", "input items must be objects")
        t = it.get("type") or ("message" if "role" in it else None)
        if t == "message":
            role = it.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                bad("input", f"message role {role!r}: user, assistant, system or developer")
            content = _resp_content(it.get("content"), "input message content")
            if role == "assistant":
                turn(1)["content"] = content
            else:
                other({"role": "system" if role == "developer" else role, "content": content})
        elif t == "reasoning":
            parts = it.get("content") if isinstance(it.get("content"), list) else []
            texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("type") == "reasoning_text"]
            text = "".join(texts) if texts else reasoning_open(it.get("encrypted_content"), budget)
            if text is None:
                summary = it.get("summary") if isinstance(it.get("summary"), list) else []
                text = "\n\n".join(str(p.get("text") or "") for p in summary if isinstance(p, dict))
            turn(0)["reasoning_content"] = text
        elif t in ("function_call", "custom_tool_call"):
            if not isinstance(it.get("name"), str) or not it.get("name"):
                bad("input", f"a {t} item needs a name")
            args = it.get("arguments") if t == "function_call" else json.dumps({"input": str(it.get("input") or "")},
                                                                                ensure_ascii=False)
            turn(2).setdefault("tool_calls", []).append({"id": str(it.get("call_id") or ""), "type": "function", "function": {
                "name": it["name"], "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)}})
        elif t in ("function_call_output", "custom_tool_call_output"):
            other({"role": "tool", "tool_call_id": str(it.get("call_id") or ""),
                   "content": _resp_content(it.get("output") if it.get("output") is not None else "", f"{t} output")})
        elif t == "item_reference":
            bad("input", "item_reference is not supported: the server stores no items (send the items themselves)")
        else:  # (built-in tool calls the server never makes: web_search_call, local_shell_call, compaction, ...)
            notes.append((f"responses-item-{t}", f"Responses input item type {t!r} is not supported: skipped"))
    # one system message at the start (Qwen's templates merge leading system / developer messages this way; the original
    # template takes only the first)
    lead = 0
    while lead < len(msgs) and msgs[lead]["role"] == "system" and isinstance(msgs[lead]["content"], str):
        lead += 1
    if lead > 1:
        msgs[:lead] = [{"role": "system", "content": "\n".join(
            s for s in (m["content"].strip() for m in msgs[:lead]) if s)}]
    if not mid_system:
        for i, m in enumerate(msgs):
            if i and m["role"] == "system":
                m["role"] = "user"
                notes.append(("responses-mid-system", "Responses system / developer messages inside the conversation are "
                              "rendered as user messages in their place: the chat template takes a system message only "
                              "at the start"))
    out = {"messages": msgs, "stream": bool(body.get("stream"))}
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    # (temperature, top_p: the API's; the others: the chat path's extensions, as llama-server passes the body through)
    for k in ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
              "chat_template_kwargs", "top_logprobs"):
        if body.get(k) is not None:
            out[k] = body[k]
    include = body.get("include") or []
    if not isinstance(include, list):
        bad("include", "an array of strings")
    if "message.output_text.logprobs" in include:
        bad("include", "message.output_text.logprobs: token log-probabilities are not available from the megakernel")
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            bad("text", "an object")
        fmt = text.get("format")
        if fmt is not None and not (isinstance(fmt, dict) and fmt.get("type", "text") == "text"):
            bad("text.format", "only {\"type\": \"text\"}: JSON output needs constrained sampling (a grammar), which the "
                               "megakernel does not have")
        if text.get("verbosity") not in (None, "medium"):
            notes.append(("responses-verbosity", "Responses text.verbosity is not supported (the model has no such "
                          "control): the reply is generated as usual"))
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            bad("reasoning", "an object")
        if isinstance(reasoning.get("effort"), str) and reasoning["effort"]:
            out["reasoning_effort"] = reasoning["effort"]  # (Server.render: none = thinking off, minimal = low)
        if (reasoning.get("summary") or reasoning.get("generate_summary")) not in (None, "none"):
            notes.append(("responses-summary", "Responses reasoning summaries are not generated: the reasoning is "
                          "returned whole as reasoning_text content (Codex shows it with show_raw_agent_reasoning)"))
    tools = body.get("tools")
    info = {"ns": {}, "custom": set()}
    if tools:
        if not isinstance(tools, list):
            bad("tools", "an array of tools")
        out["tools"], info["ns"], info["custom"] = responses_tools(tools, notes)
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        bad("tool_choice", "\"auto\" or \"none\": forcing a tool call (required / a named or allowed tool) needs "
                           "constrained sampling, which the megakernel does not have")
    if tc is not None:
        out["tool_choice"] = tc
    if body.get("parallel_tool_calls") is False:
        out["parallel_tool_calls"] = False
    return out, info


def responses_usage(usage, reasoning_tokens=0):
    """the chat usage as a Response's (input_tokens counts the reused ones, as OpenAI's does)"""
    return {"input_tokens": usage["prompt_tokens"],
            "input_tokens_details": {"cached_tokens": usage["prompt_tokens_details"]["cached_tokens"], "cache_write_tokens": 0},
            "output_tokens": usage["completion_tokens"], "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": usage["total_tokens"]}


def responses_reasoning(text, rid, sealed, status="completed"):
    item = {"type": "reasoning", "id": rid, "summary": [], "content": [{"type": "reasoning_text", "text": text}],
            "encrypted_content": reasoning_seal(text) if sealed else None, "status": status}
    return item


def responses_message(text, mid, status="completed"):
    return {"type": "message", "id": mid, "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]}


def responses_call(call_id, name, arguments, info, status="completed"):
    """a tool call as a function_call item (with its namespace), or a custom_tool_call for a custom tool (its input:
    the call's "input" argument)"""
    key = call_id.removeprefix("call_")
    if name in info["custom"]:
        try:
            inp = json.loads(arguments).get("input")
        except (ValueError, AttributeError):
            inp = None
        return {"type": "custom_tool_call", "id": "ctc_" + key, "call_id": call_id, "name": name,
                "input": inp if isinstance(inp, str) else "", "status": status}
    item = {"type": "function_call", "id": "fc_" + key, "call_id": call_id, "name": name, "arguments": arguments,
            "status": status}
    if name in info["ns"]:
        item["namespace"] = info["ns"][name]
    return item


def response_object(rid, created, body, model, status, output, usage=None, error=None, incomplete=None):
    """a Response object (the request's settings echoed as OpenAI does; store false: nothing is kept)"""
    r = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    text = body.get("text") if isinstance(body.get("text"), dict) else {}
    return {"id": rid, "object": "response", "created_at": created, "status": status, "background": False,
            "completed_at": int(time.time()) if status == "completed" else None, "error": error,
            "incomplete_details": incomplete, "instructions": body.get("instructions"),
            "max_output_tokens": body.get("max_output_tokens"), "max_tool_calls": body.get("max_tool_calls"),
            "model": model, "output": output, "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
            "previous_response_id": None, "reasoning": {"effort": r.get("effort"), "summary": r.get("summary")},
            "service_tier": "default", "store": False,
            "temperature": body["temperature"] if body.get("temperature") is not None else DEF_TEMP,
            "text": {"format": {"type": "text"}, "verbosity": text.get("verbosity") or "medium"},
            "tool_choice": body.get("tool_choice") or "auto", "tools": body.get("tools") or [],
            "top_p": body["top_p"] if body.get("top_p") is not None else DEF_TOP_P,
            "truncation": body.get("truncation") or "disabled", "usage": usage, "user": body.get("user"),
            "metadata": body.get("metadata") or {}}


# ---------------------------------------------------------------------------------------------


class FifoLock:
    """the engine's lock: requests get it in the order they arrive (threading.Lock wakes an arbitrary waiter)"""

    def __init__(self):
        self._cv = threading.Condition()
        self._tickets = itertools.count()
        self._turn = 0
        self._held = False

    def acquire(self):
        with self._cv:
            t = next(self._tickets)
            self._cv.wait_for(lambda: self._turn == t and not self._held)
            self._held = True
        return True

    def release(self):
        with self._cv:
            self._held = False
            self._turn += 1
            self._cv.notify_all()

    def locked(self):
        return self._held

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


class Parked:
    """a conversation parked in the engine's host store (Engine.park): what the server matches prompts against. side: a
    side request's own conversation (Engine.side_end_park); low: one that is likely never continued (a summary request's
    and the conversation it summarised: Server.cache_switch); both kinds are evicted before the others. tools: its last
    request had tools (None: unknown). pins: its pinned checkpoints (in cks too: Server.ck_pins)"""

    def __init__(self, pid, seq, cks, committed, img_spans, gen, nbytes, side=False, low=False, tools=None, pins=()):
        self.id, self.seq, self.cks, self.committed = pid, seq, cks, committed
        self.img_spans, self.gen, self.nbytes, self.t, self.side = img_spans, gen, nbytes, time.time(), side
        self.low, self.tools, self.pins = low, tools, tuple(pins)


SIDE_STORE_MIB = 1024  # host store of side requests when parking is off (--park-mib 0)
SIDE_REPLY_MAX = 1024  # with parking on, a side request with tools asks for at most this many tokens (cache_switch)
SIDE_REPLY_ROOM = 8192  # store room a side request takes for the rows of a reply it does not cap (Server.side_start)
SIDE_PARK_CKS = 1  # checkpoints kept with a side request's conversation (its generation prompt's: where a follow-up resumes)
SIDE_KEEP_MAX = 4  # side requests' conversations kept at most (the oldest goes first): most are titles, never continued
# a side request without tools that quotes at least --park-min tokens and answers at least this many (thinking aside) is
# taken for a summary of the conversation beside it (titles, tags, follow-ups and commit messages answer in tens)
SUMMARY_ANSWER_MIN = 256
PIN_MIN = 1024  # a shared prefix shorter than this is not pinned (Server.ck_pins: it is prefilled again in under ~0.5 s)


def mib(n):
    return f"{n / 2**20:,.0f} MiB"


SLOT_OWN_CTX = 1024  # --agents: the positions of the engine's own sequence (the warm-up's; the slots have their own)


def common_prefix(a, b):
    """the length of the common prefix of two token lists"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SlotJob:
    """a request in the slot scheduler (--agents): its prompt, sampling and token limit go in; each engine cycle's tokens
    come out through q (a list per cycle; []: the reply cannot go on, its slot's context is full; an exception: the
    engine failed). state: queued (waiting for a free slot), prefill, decode, done"""

    POLL_S = 0.05  # how often a request waiting for its tokens asks whether its client is still there

    def __init__(self, sched, toks, sp, max_new, rid=0):
        self.sched, self.toks, self.sp, self.max_new, self.rid = sched, toks, sp, max_new, rid
        self.q = queue.Queue()
        self.state, self.slot, self.reused, self.left, self.n = "queued", None, 0, len(toks), 0
        self.closed = False
        self.t_submit, self.t_begin = time.time(), None
        self.gone, self.rest = (lambda: False), []

    def close(self):
        """the request is done with the job (its reply ended, it failed, or its client left): the scheduler drops it
        from the queue, abandons its prefill, or takes its slot out of the next cycle"""
        if not self.closed:
            self.closed = True
            self.sched.wake()

    def _get(self, tick=None):
        """the next item of q; None once gone() is True (the job is closed)"""
        while True:
            try:
                item = self.q.get(timeout=self.POLL_S)
            except queue.Empty:
                if self.gone():
                    self.close()
                    return None
                if tick is not None:
                    tick()
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    def prompt(self, gone, progress=None):
        """waits until the prompt is processed: (the reply's first token, the prompt tokens the slot reused). gone() (the
        client left) is asked every POLL_S meanwhile: PrefillCancelled once it is True. progress(cached, processed,
        total): a request's return_progress (Server.prompt_progress's events), called when the prefill starts, as it
        goes (every 0.5 s at most) and when it is done"""
        self.gone = gone
        total, seen = len(self.toks), [None, 0.0]

        def tick():
            if progress is None or self.state != "prefill":
                return
            done, now = total - self.left, time.monotonic()
            if seen[0] is None or (done > seen[0] and now - seen[1] >= 0.5):
                seen[:] = [done, now]
                progress(self.reused, done, total)
        first = self._get(tick)
        if first is None:
            raise PrefillCancelled("while it waited for a free slot" if self.state == "queued" else
                                   f"at {total - self.left:,} of {total:,} tok")
        if not first:  # (Server._generate leaves the reply room in the context before it submits)
            raise RuntimeError("the slot had no room for the reply")
        if progress is not None:
            if seen[0] is None:
                progress(self.reused, self.reused, total)
            progress(self.reused, total, total)
        self.rest = first[1:]
        return first[0], self.reused

    def step(self):
        """the tokens of the job's next engine cycle; None: the client left meanwhile (the job is closed); []: the slot's
        context is full"""
        if self.rest:
            out, self.rest = self.rest, []
            return out
        return self._get()


class SlotScheduler:
    """--agents N: one thread owns the engine's parallel slots (MKEngine::slots_open: up to N conversations, each with
    its own KV cache and recurrent state) and decodes every admitted request together, one engine cycle for all of them
    (mk_slots_step: one batched launch; a lone slot runs the solo megakernel); each request thread (Server._generate) gets
    the tokens of its cycles through its SlotJob. Requests wait for a free slot in arrival order and take the one whose
    sequence shares the most with the prompt, less what that discards of the slot's last prompt (a conversation's next
    turn goes back to its own slot and prefills only its new tokens; an empty slot goes before another conversation's).
    While slots decode, a waiting prompt is fed in chunks, one per cycle in the same pass as their step
    (mk_slots_step_mixed, MK_PF_MIX rows, default 256; MK_PF_MIX=0: MK_PF_CHUNK tokens between the cycles, default 512,
    0: the whole prompt), so they keep decoding while a long prompt is read; while none decodes, MK_PF_SOLO tokens per call
    (default 0: the whole prompt). With MK_EXACT the chunks change no row (the engine feeds every chunk but a prompt's
    last as batched rows, each ending on the 32-token scan grid: up to 31 tokens off the budget), so a prompt that
    arrives while others decode gets the reply it gets alone. A closed job is dropped from the queue, its prefill abandoned at the next chunk, or its slot left out of the
    next cycle. Open-jet's SlotScheduler (--parallel / --agents), with jobs that can be closed at any point.

    With MK_EXACT (exact reuse; --agents sets it), a prompt never reuses the rows a slot decoded: those depend on how many
    drafts each cycle verified, which depends on the slots decoding beside it. A prompt that reaches into the slot's last
    reply first takes the slot back to a checkpoint inside its last prompt (slot_begin with that prompt's tokens), as
    Server.load_prompt restores a checkpoint with --exact-reuse.

    The engine's keep-warm load (MK_KEEP_WARM_S) is armed when the scheduler goes idle and stopped before its next GPU
    work: the slot calls do not stop it themselves."""

    def __init__(self, eng, n, ctx):
        self.eng, self.n, self.ctx = eng, n, ctx
        self.exact = bool(getattr(eng, "exact", False))
        self.mix = env_int("MK_PF_MIX", 256, 0)
        self.chunk = env_int("MK_PF_CHUNK", 512, 0)
        self.solo = env_int("MK_PF_SOLO", 0, 0)
        self.cv = threading.Condition()
        self.pending = collections.deque()  # jobs waiting for a slot (under cv)
        self.free = list(range(n))
        self.filling, self.active = {}, {}  # slot -> job: its prompt is fed (the oldest first) / it decodes
        self.seq = [[] for _ in range(n)]  # each slot's tokens as the engine holds them (prompt, reply): the slot choice
        self.held = [0] * n  # the length of each slot's last prompt (what another conversation's prompt would discard)
        self.last = [None] * n  # each slot's latest job (GET /slots)
        self.warm = False  # the keep-warm load is armed
        self.cycles = 0
        threading.Thread(target=self.run, name="mk-slots", daemon=True).start()

    def submit(self, toks, sp, max_new, rid=0):
        """a request: toks, sp = (temp, top_p, top_k, min_p, seed), at most max_new tokens; returns its SlotJob"""
        job = SlotJob(self, toks, sp, max_new, rid)
        with self.cv:
            self.pending.append(job)
            self.cv.notify()
        return job

    def wake(self):
        with self.cv:
            self.cv.notify()

    def counts(self):
        """(requests waiting for a slot, requests in a slot)"""
        return len(self.pending), len(self.filling) + len(self.active)

    def full(self):
        """a request sent now would wait for a slot"""
        return len(self.pending) >= len(self.free)

    def snapshot(self):
        """each slot's state for GET /slots (read from the HTTP threads while the scheduler runs)"""
        out = []
        for s in range(self.n):
            j = self.last[s]
            st = j.state if j is not None and j.state in ("prefill", "decode") else "idle"
            out.append({"state": st, "id_task": j.rid if j else -1, "n_prompt_tokens": len(j.toks) if j else 0,
                        "n_cached": j.reused if j else 0, "prefill_left": j.left if st == "prefill" else 0,
                        "n_decoded": j.n if j else 0, "n_past": len(self.seq[s])})
        return out

    def _end(self, job):
        job.state = "done"
        job.q.put([])  # (a request still waiting for tokens: the reply ends here)

    def _finish(self, slot):
        job = self.active.pop(slot)
        self.free.append(slot)
        self._end(job)

    def _fail(self, slot, job, e):
        """the engine failed on the job's slot: the request gets the error; the slot's state is unknown (its next prompt
        is matched against nothing)"""
        job.state = "done"
        job.q.put(e)
        self.seq[slot], self.held[slot] = [], 0
        self.free.append(slot)

    def _prefilled(self, slot, job):
        """the job's prompt is in: it decodes from the next cycle on"""
        del self.filling[slot]
        job.max_new = min(job.max_new, self.ctx - len(job.toks) - ContextOverflow.RESERVE)
        if job.max_new <= 0:
            self.free.append(slot)
            self._end(job)
        else:
            job.state = "decode"
            self.active[slot] = job

    def run(self):
        while True:
            try:
                self.cycle()
            except Exception as e:  # (a fault of the scheduler's own: its requests get it, the next ones are served)
                log(f"slot scheduler: {e!r}")
                traceback.print_exc()
                for s in list(self.filling):
                    self._fail(s, self.filling.pop(s), e)
                for s in list(self.active):
                    self._fail(s, self.active.pop(s), e)

    def cycle(self):
        """admits waiting requests to free slots, feeds one prompt chunk, runs one engine cycle of the decoding slots"""
        eng = self.eng
        with self.cv:
            while True:
                for job in [j for j in self.pending if j.closed]:  # (its client left while it waited)
                    self.pending.remove(job)
                    self._end(job)
                if self.pending or self.filling or self.active:
                    break
                if not self.warm:  # idle: the GPU's clocks stay up for MK_KEEP_WARM_S, as after a request one at a time
                    self.warm = True
                    eng.keep_warm(True)
                self.cv.wait()
            admit = []
            while self.pending and self.free:
                job = self.pending.popleft()
                # the slot holding this conversation; else the one whose loss costs least (an empty one before evicting
                # another conversation for the few template tokens it shares): reused - discarded tokens of the slot's
                # last prompt (open-jet counts its reply too, which the conversation's own next turn re-renders: then a
                # long reply could send that turn to another conversation's slot)
                slot = max(self.free, key=lambda s: 2 * common_prefix(self.seq[s], job.toks) - self.held[s])
                self.free.remove(slot)
                admit.append((slot, job))
        if self.warm:
            self.warm = False
            eng.keep_warm(False)
        for slot, job in admit:
            try:
                p = self.held[slot]
                if self.exact and p and common_prefix(self.seq[slot], job.toks) >= p:  # (exact reuse: see the class)
                    eng.slot_begin(slot, job.toks[:p], *job.sp)
                job.reused = eng.slot_begin(slot, job.toks, *job.sp)
            except Exception as e:  # (reported to the request; the others go on)
                self._fail(slot, job, e)
                continue
            job.slot, job.left, job.t_begin, job.state = slot, len(job.toks) - job.reused, time.time(), "prefill"
            self.seq[slot], self.held[slot], self.last[slot] = list(job.toks), len(job.toks), job
            self.filling[slot] = job
        for slot in [s for s, j in self.active.items() if j.closed]:
            self._finish(slot)
        stepped = None  # (slots, their tokens) of this cycle
        if self.filling:  # one chunk of the oldest admitted prompt
            slot = next(iter(self.filling))
            job = self.filling[slot]
            try:
                if job.closed:  # (the slot keeps what was fed)
                    eng.slot_prefill(slot, 0)
                    del self.filling[slot]
                    self.seq[slot] = job.toks[:len(job.toks) - job.left]
                    self.held[slot] = len(self.seq[slot])
                    self.free.append(slot)
                    self._end(job)
                else:
                    if self.active and self.mix > 0:
                        slots = sorted(self.active)
                        out, job.left = eng.slots_step_mixed(slot, self.mix, slots)
                        stepped = (slots, out)
                    else:
                        job.left = eng.slot_prefill(slot, (self.chunk if self.active else self.solo) or 1 << 30)
                    if job.left == 0:
                        self._prefilled(slot, job)
            except Exception as e:
                self.filling.pop(slot, None)
                self._fail(slot, job, e)
                if stepped is None and self.active and self.mix > 0:  # (the cycle failed with it)
                    for s in sorted(self.active):
                        self._fail(s, self.active.pop(s), e)
        if stepped is None:
            if not self.active:
                return
            slots = sorted(self.active)
            try:
                stepped = (slots, eng.slots_step(slots))
            except Exception as e:
                for s in slots:
                    self._fail(s, self.active.pop(s), e)
                return
        self.cycles += 1
        for s, toks in zip(*stepped):
            job = self.active[s]
            if not toks:  # (its context is full: the reply ends at its length)
                self._finish(s)
                continue
            self.seq[s] += toks
            job.n += len(toks)
            job.q.put(toks)
            if job.n >= job.max_new or any(eng.is_eog(t) for t in toks):
                self._finish(s)


class Server:
    vision = None   # Vision (--mmproj), else text only
    decoder = None  # VideoDecoder (an ffmpeg was found): video input with the vision encoder
    video_ok = False  # video input works in this process (mmproj loaded, frame-pair encoder, ffmpeg): /props
    img_spans = ()  # the images of the engine's sequence: (pos, n, key); replaced, never mutated
    stop_word = None  # the stop string that ended the last reply (None: it ended otherwise)
    last_prompt = (0, 0)  # the running request's prompt: (tokens, reused tokens), once it is processed
    reasoning_tokens = 0  # the last reply's tokens generated inside its thinking (a Response's reasoning_tokens)
    answer_tokens = 0  # the running (or last) reply's tokens of its answer (not thinking, not a tool call)
    rids = itertools.count(1)  # request numbers for the console
    rid = 0  # the running (or last) request's number: "#rid" on its console lines
    req = ("chat", None, 0.0)  # its (route, model field, start time), set by the handler when it gets the engine
    cache_note = ""  # how load_prompt matched the last prompt against the engine's sequence (the console's prompt line)
    # host cache (Engine.store): parked conversations by id, least recently parked first (None: no store); the running
    # side request's kept state (None: none) and whether its own prompt was processed (side_ready: its conversation can be
    # parked after it); the live sequence's generation prompt position (-1: none; a request whose prompt matches up to it
    # continues the live conversation)
    parked = None
    store_mib = 0
    side = None
    side_ready = False
    live_gen = -1
    # whether the live conversation's last request had tools (None: unknown), and why the live conversation is parked
    # with low priority when another one replaces it ("": it is not); cache_switch decides next_low for the request it
    # runs, which becomes live_low once its prompt is processed
    live_tools = None
    live_low = next_low = ""
    # the parked conversation cache_switch marked low because the running request (no tools) replaced it: debris_finish
    # takes the mark back if the reply's answer is shorter than a summary's (None: no such mark)
    debris = None
    asst_marker = ()  # the tokens of "<|im_start|>assistant": where a prompt's first reply starts (Server.first_reply)
    # (parked, bytes in use, bytes allocated, their token counts, how many are side requests' own, how many have low
    # priority): /metrics and /props read it
    store_gauge = (0, 0, 0, (), 0, 0)
    cache_pre = ""  # what the host cache did for the running request (before cache_note on the prompt line)
    tick = (0, 0.0, 0.0, 0)  # the running reply's (prompt tokens, prompt ms, its start, generated tokens): timings_now
    # --agents (SlotServer): the slot scheduler (None: one request at a time on the engine's own sequence); the running
    # request's return_progress callback (its handler sets it), slot job, slot sampling and wait for a free slot (seconds)
    sched = None
    on_progress = None
    job = None
    slot_sp = None
    queued_s = 0.0

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, self.engine_ctx(args), args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv], "none" if args.no_mtp else args.mtp)
        if not self.eng.has_mtp:
            log("no MTP layer: plain decoding (depth 0)")
        # --chat-template-file / --chat-template (read and checked in main) replace the GGUF's template everywhere it is
        # used: chat and Anthropic requests, /apply-template, token counts, /props; a GGUF without one gets ChatML
        src = getattr(args, "chat_template_src", None)
        self.template_src = src if src is not None else self.eng.chat_template()
        if not self.template_src:
            log("the GGUF has no chat template: using ChatML (as llama-server; --chat-template-file to give one)")
            self.template_src = CHATML_TEMPLATE
        self.template = make_template(self.template_src)
        self.setup_specials(args)
        im = self.eng.tokenize("<|im_start|>")
        self.im_start = im[0] if len(im) == 1 else None  # checkpoint at the start of the generation prompt
        if self.im_start is not None:
            self.asst_marker = tuple(self.eng.tokenize("<|im_start|>assistant"))
        self.sys_role = self.eng.tokenize("system")  # (a prompt that opens with a system message: ck_pins)
        if getattr(args, "ck_pin", 0) and not getattr(self.eng, "ck_pins", False):
            log(f"pinned checkpoints off: {os.path.basename(args.lib)} has no mk_ck_pin (rebuild it): a new session "
                f"re-reads a client's system prompt once the cached conversation has grown or was parked")
        if args.mmproj:
            if not self.eng.images:
                raise RuntimeError(f"{args.lib} has no image input (mk_prefill_img): rebuild it")
            self.vision = Vision(args.vision_lib, args.mmproj, args.model, args.vision_threads, args.image_min_tokens,
                                 args.image_max_tokens, args.image_cache, args.vision_backend, args.video_cache)
            if self.vision.n_embd != self.eng.L.mk_n_embd():
                raise RuntimeError(f"mmproj rows of {self.vision.n_embd} for a model of {self.eng.L.mk_n_embd()}")
            ids = [self.eng.tokenize(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError("the vocabulary has no <|vision_start|> / <|vision_end|> / <|image_pad|> tokens")
            self.vis_start, self.vis_end, self.img_pad = (i[0] for i in ids)
            self.setup_video(args)
        self.warm_up()
        self.setup_store(args)
        self.model_id = os.path.basename(args.model)
        self.lock = FifoLock()
        self.warned = set()
        self.setup_slots(args)
        vis = (f", vision on the CPU ({os.path.basename(args.mmproj)}, {self.vision.backend}, {self.vision.n_threads} threads)"
               if self.vision else "")
        self.t_start = time.time()
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts{vis}")
        self.warm_s = self.eng.keep_warm()  # (MK_KEEP_WARM_S: the first reply starts at full clocks too)

    def setup_specials(self, args):
        """--content-specials text (the default; env MK_CONTENT_SPECIALS): special-token text inside the request's
        messages and tools is tokenized as plain text (LiteralTemplate, tokenize_prompt); parse: as the special tokens,
        as llama-server does"""
        if getattr(args, "content_specials", "parse") != "text":
            log("special-token text in messages is tokenized as special tokens (--content-specials parse, as llama-server)")
            return
        controls = self.eng.controls()
        if not controls:
            log(f"special-token text in messages is tokenized as special tokens: {os.path.basename(args.lib)} has no "
                f"mk_tokenize2 / mk_token_attr (rebuild it)")
            return
        self.template = LiteralTemplate(self.template, controls)

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and then requests fail with a CUDA out-of-memory error. Warming up here moves that
        failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        self.eng.prefill(toks)
        self.eng.step()
        self.eng.reset()
        if self.vision:
            self.vision.warm_up()

    def engine_ctx(self, args):
        """the positions of the engine's own sequence (SlotServer: the slots have their own)"""
        return args.ctx

    def setup_slots(self, args):
        """(SlotServer: opens the engine's parallel slots and starts their scheduler)"""

    def setup_store(self, args):
        """host memory for side requests and parked conversations (--side-max, --park-mib; an engine without the host
        store: neither)"""
        side, park = getattr(args, "side_max", 0), getattr(args, "park_mib", 0)
        if not getattr(self.eng, "store", False) or not (side > 0 or park > 0):
            if (side > 0 or park > 0) and not getattr(self.eng, "store", False):
                log(f"side requests / parked conversations off: {args.lib} has no host store (mk_park): rebuild it")
            return
        self.parked = collections.OrderedDict()
        self.store_mib = max(park, SIDE_STORE_MIB if side > 0 else 0)
        self.eng.store_budget(self.store_mib << 20)
        pt, rings, state, ck = self.eng.kv_sizes()
        log(f"host cache (allocated on first use, at most {self.store_mib:,} MiB): side requests "
            f"{f'up to {side:,} tok' if side > 0 else 'off'}, parked conversations "
            f"{f'up to {park:,} MiB, {args.park_cks} checkpoints each' if park > 0 else 'off'} ({pt:,} B per token, "
            f"{mib(state)} recurrent state, {mib(ck)} per checkpoint)")
        if side > 0 and park > 0 and not self.side_keep():
            log(f"a side request's own conversation is not kept: {os.path.basename(args.lib)} has no mk_side_end_park "
                f"(rebuild it): its next turn is prefilled in full")

    def side_keep(self):
        """a side request's own conversation is parked after it (evicted before the other parked conversations) and a
        later bounded request continuing it runs as a side request again (needs parking on and the library's support)"""
        a = self.args
        return self.parked is not None and a.side_max > 0 and a.park_mib > 0 and getattr(self.eng, "side_keep", False)

    # ---- host cache: side requests and parked conversations, switched at request boundaries ----
    def resume_point(self, seq, committed, cks, cached_spans, toks, spans):
        """how load_prompt would continue `toks` (image spans `spans`) on the cached `seq`: (common prefix, the image
        position that cut it or None, the position it resumes at, 'append' | 'checkpoint' | 'reset')"""
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)
        match, img_cut = lcp, None
        for p, _, _ in set(spans) ^ set(cached_spans):
            if p < lcp:
                lcp, img_cut = p, p
        if seq and lcp >= committed and not self.eng.exact:
            return match, img_cut, lcp, "append"
        ck = max((p for p in cks if 0 < p <= lcp), default=0)
        return match, img_cut, ck, "checkpoint" if ck else "reset"

    def cache_switch(self, toks, spans, first_turn, max_tokens=-1, tools=True):
        """before load_prompt's own matching: brings back a parked conversation that this prompt continues better than
        the live one (parking the live one), or runs the request as a side request, or parks the live conversation when
        the request would discard it. Requests switch only here, between replies. With parking on, a side request is a
        one-shot: a first turn without tools (a client's title, tags, commit message, session name, summary; every
        agent harness sends its sessions' first turns with tools) whatever its max_tokens, or one whose reply is bounded
        (0 < max_tokens <= SIDE_REPLY_MAX); its own conversation is parked after it (side_finish), so a chat without
        tools continues as side requests, and any request without tools that would displace a conversation with tools
        runs beside it too. A new agent session's first turn (tools, a long reply) parks the live conversation like any
        conversation, so its next turns find it cached."""
        eng, a = self.eng, self.args
        self.cache_pre, self.next_low, self.debris = "", "", None
        seq = eng.sequence()
        live = self.resume_point(seq, eng.committed(), eng.checkpoints(), self.img_spans, toks, spans)
        lost = len(seq) - live[2]  # cached positions the normal path would discard
        cont = self.live_gen >= 0 and live[0] >= self.live_gen  # (diverges in the live one's generation prompt / reply)
        # a side request's own conversation that this one-shot prompt continues (it matches up to that conversation's
        # generation prompt): a side request again, resumed there, while the live conversation stays aside
        side_ok = (a.side_max > 0 and len(toks) <= a.side_max and not cont and lost >= max(2 * len(toks), a.park_min)
                   and (a.park_mib <= 0 or not tools or 0 < max_tokens <= SIDE_REPLY_MAX))
        if side_ok and self.side_keep() and self.parked:
            best = None
            for e in self.parked.values():
                if not e.side or e.gen < 0:
                    continue
                r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                if r[0] >= e.gen and r[2] > live[2] and (best is None or r[2] > best[1][2]):
                    best = (e, r)
            if best:
                self.side_start(seq, len(toks), max_tokens, resume=best[0])
                return
        if a.park_mib > 0 and self.parked:
            best = None
            for e in self.parked.values():
                r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                if r[2] >= live[2] + a.park_min and (best is None or r[2] > best[1][2]):
                    best = (e, r)
            if best:
                self.park_live(seq, "another conversation came back", keep=best[0].id, low=self.live_low)
                self.unpark(best[0])
                e, r = best
                if r[2] in e.pins and r[0] < e.gen:  # (a new session that shares only its system prompt: it stays parked)
                    self.park_live(eng.sequence(), "a new session reuses its system prompt",
                                   low="it was parked with low priority" if e.low else "")
                return
        # a first turn, or (parking on) a request without tools that would displace a conversation with tools: an agent
        # session stays live while a chat without tools or a client's summary (the conversation as messages) runs
        # beside it
        if side_ok and (first_turn or (not tools and self.live_tools and a.park_mib > 0)):
            # a summary of the live conversation (a long one without tools that answers at length: side_finish): if the
            # next request replaces the live conversation instead of continuing it, it was the summarised one, parked
            # with low priority
            after = "replaced right after a summary request beside it" if (
                not tools and len(toks) >= a.park_min and a.park_mib > 0) else ""
            self.side_start(seq, len(toks), max_tokens, after=after)
            return
        if a.park_mib > 0 and not cont and lost >= a.park_min:
            k = self.first_reply(toks)
            if k is not None and live[0] >= k + len(self.asst_marker):
                # everything up to the cached conversation's first reply is unchanged: this is that conversation with
                # its later history edited (a client moving or dropping a block, clearing old tool results, a chat
                # message edited); the request resumes from the engine's state either way, and the copy a park would
                # keep is never continued
                log(f"#{self.rid} not parked: the request continues the cached conversation with its history edited "
                    f"(the prompt matches {live[0]:,} of the {len(seq):,} cached tok, past its first reply at {k:,})")
                return
            # a request without tools that replaces a conversation whose last request had tools is how clients summarise
            # a conversation (compaction); the compacted conversation then replaces the summary's: both copies are
            # likely dead, parked with low priority (evicted before real parked conversations). Clients' other one-shots
            # that quote a conversation (titles, tags) look the same until they answer: the replaced conversation's mark
            # holds only for a summary's answer (debris_finish); the one-shot's own conversation stays low either
            # way, and takes no room from a conversation of normal priority when it is parked (park_live)
            debris = "" if tools or not self.live_tools else "replaced by a request without tools"
            mark = bool(debris) and not self.live_low  # (provisional: debris_finish may take it back)
            e = self.park_live(seq, "a new conversation", low=self.live_low or debris, provisional=mark)
            self.debris = e if mark else None
            self.next_low = debris and "the conversation of a request without tools that replaced one with tools"

    def first_reply(self, toks):
        """the position of the prompt's first assistant message (its <|im_start|>assistant tokens; a prompt without
        one: its generation prompt), or None"""
        m = self.asst_marker
        if not m:
            return None
        k = 0
        try:
            while True:
                k = toks.index(m[0], k)
                if tuple(toks[k:k + len(m)]) == m:
                    return k
                k += 1
        except ValueError:
            return None

    def store_update(self):
        """refreshes store_gauge (on the request thread: /metrics and /props never call into the engine)"""
        info = self.eng.store_info()
        self.store_gauge = (len(self.parked), info["used"], info["allocated"], tuple(len(e.seq) for e in self.parked.values()),
                            sum(e.side for e in self.parked.values()), sum(e.low for e in self.parked.values()))
        return info

    def make_room(self, need, budget, keep=None, side_only=False):
        """evicts parked conversations, side requests' own ones and low-priority ones first, each tier least recently
        parked first (never `keep`, the one about to be restored; side_only: only those), until `need` more bytes fit in
        `budget`; False: they do not"""
        while self.eng.store_info()["used"] + need > budget:
            pid = next((p for p, e in self.parked.items() if p != keep and (e.side or e.low)), None)
            if pid is None and not side_only:
                pid = next((p for p in self.parked if p != keep), None)
            if pid is None:
                return False
            e = self.parked[pid]
            kind = ", a side request" if e.side else ", low priority" if e.low else ""
            self.evict(pid, f"least recently parked{kind}) to make room")
        return True

    def evict(self, pid, why):
        e = self.parked.pop(pid)
        self.eng.park_drop(pid)
        self.count(park_evictions=1)
        self.store_update()
        log(f"#{self.rid} evicted the parked conversation of {len(e.seq):,} tok ({mib(e.nbytes)}, {why}")

    def park_live(self, seq, why, keep=None, low="", provisional=False):
        """a snapshot of the live conversation into the host store (the engine keeps it too); keep: a parked one that must
        stay (it is restored next); low: why it is parked with low priority ("": it is not), provisional: that mark may
        be taken back (Server.debris). Returns its Parked entry (None: not parked)"""
        eng, a = self.eng, self.args
        if len(seq) < a.park_min:
            return
        need, budget = eng.park_need(a.park_cks), a.park_mib << 20
        if need > budget:
            log(f"#{self.rid} not parked: the cached {len(seq):,} tok need {mib(need)} of host memory "
                f"(--park-mib {a.park_mib:,})")
            return
        if low and not provisional:
            # a low-priority conversation would be evicted before any normal one: as a side request's own conversation
            # (side_finish), it only takes room that side and other low-priority ones leave, and evicts nothing when
            # that is not enough
            spare = budget - eng.store_info()["used"] + sum(e.nbytes for p, e in self.parked.items()
                                                             if p != keep and (e.side or e.low))
            if need > spare or not self.make_room(need, budget, keep, side_only=True):
                log(f"#{self.rid} not parked: the cached {len(seq):,} tok ({mib(need)}) have low priority ({low}) and "
                    f"do not fit in the host store without evicting a conversation of normal priority")
                return
        else:
            self.make_room(need, budget, keep)
        # (the engine parks the newest --park-cks checkpoints and the pinned ones: Engine.ck_pin)
        pins = eng.pinned() if a.park_cks > 0 and getattr(eng, "ck_pins", False) else []
        cks = sorted(set(sorted(eng.checkpoints())[-a.park_cks:]) | set(pins)) if a.park_cks > 0 else []
        pid = eng.park(a.park_cks)
        if pid is None:
            log(f"#{self.rid} not parked: no room in the host store for the cached {len(seq):,} tok ({mib(need)})")
            return
        e = Parked(pid, seq, cks, eng.committed(), self.img_spans, self.live_gen, need, low=bool(low),
                   tools=self.live_tools, pins=pins)
        self.parked[pid] = e
        info = self.store_update()
        self.count(parks=1)
        why += f", low priority: {low}" if low else ""
        log(f"#{self.rid} parked the cached conversation ({why}): {len(seq):,} tok, {mib(info['last_bytes'])} copied in "
            f"{info['last_ms']:.0f} ms | {len(self.parked)} parked, {mib(info['used'])} of {a.park_mib:,} MiB")
        return e

    def unpark(self, e):
        """the engine becomes the parked conversation e (its entry is freed)"""
        eng = self.eng
        try:
            eng.unpark(e.id)
        except RuntimeError:  # (an engine error: the entry goes too, with its host memory)
            self.parked.pop(e.id, None)
            eng.park_drop(e.id)
            self.store_update()
            raise
        del self.parked[e.id]
        self.img_spans, self.live_gen, self.live_tools, self.live_low = e.img_spans, e.gen, e.tools, ""
        info = self.store_update()
        self.count(park_restores=1)
        self.cache_pre = f"restored a parked conversation ({len(e.seq):,} tok); "
        log(f"#{self.rid} restored the parked conversation of {len(e.seq):,} tok: {mib(info['last_bytes'])} copied in "
            f"{info['last_ms']:.0f} ms | {len(self.parked)} parked, {mib(info['used'])} in use")

    def side_start(self, seq, n, max_tokens=-1, resume=None, after=""):
        """runs the request as a side request: the live sequence stays cached (Engine.side_begin; side_finish brings it
        back after the reply). resume: a side request's parked conversation it continues (Engine.side_unpark); after:
        the low-priority reason the live conversation gets when it is back ("": its own)"""
        eng = self.eng
        # room for what the request may overwrite: its prompt and its reply (max_tokens, else SIDE_REPLY_ROOM: past that
        # the engine saves more rows while there is room, else the kept sequence is dropped), and the checkpoint blocks
        # of the live one in the slots it saves into
        pt, rings, state, ck = eng.kv_sizes()
        reply = min(max_tokens, SIDE_REPLY_ROOM) if max_tokens > 0 else SIDE_REPLY_ROOM
        self.make_room(state + rings + pt * min(len(seq), n + reply) + 2 * ck, self.store_mib << 20,
                       keep=resume.id if resume else None)
        if not eng.side_begin():
            log(f"#{self.rid} side request: no room in the host store: the cached {len(seq):,} tok are not kept")
            return
        self.side = (len(seq), self.img_spans, self.live_gen, self.live_tools, self.live_low, after)
        self.side_ready = False
        self.count(side_requests=1)
        self.cache_pre = f"side request: the cached {len(seq):,} tok are kept aside; "
        if resume is not None:
            if not eng.side_unpark(resume.id):
                log(f"#{self.rid} side request: no room in the host store to continue its parked conversation of "
                    f"{len(resume.seq):,} tok")
                return
            del self.parked[resume.id]
            self.img_spans, self.live_gen, self.live_tools = resume.img_spans, resume.gen, resume.tools
            info = self.store_update()
            self.count(side_resumed=1)
            self.cache_pre += f"its parked conversation ({len(resume.seq):,} tok) restored in {info['last_ms']:.0f} ms; "

    def side_finish(self):
        """after a side request (also one that failed or whose client left): the kept sequence comes back; with
        side_keep(), the side request's own conversation is parked first (when its prompt was processed and it has a
        checkpoint to resume from)"""
        if self.side is None:
            return
        n, spans, gen, tools, low, after = self.side
        self.side, eng, a = None, self.eng, self.args
        ok, pid, own, room = False, None, None, True
        try:
            if eng.side_active():
                if self.side_keep() and self.side_ready and eng.checkpoints():
                    sides = [p for p, e in self.parked.items() if e.side]  # (least recently parked first)
                    for p in sides[:max(0, len(sides) - SIDE_KEEP_MAX + 1)]:
                        self.evict(p, f"the oldest of {SIDE_KEEP_MAX} kept side request conversations)")
                    own = (list(eng.sequence()), sorted(eng.checkpoints())[-SIDE_PARK_CKS:], eng.committed(), self.img_spans,
                           self.live_gen, eng.park_need(SIDE_PARK_CKS))
                    # (no room: other kinds of parked conversations are not evicted for it)
                    room = own[5] <= a.park_mib << 20 and self.make_room(own[5], a.park_mib << 20, side_only=True)
                if own is not None and room:
                    ok, pid = eng.side_end_park(SIDE_PARK_CKS)
                else:
                    ok = eng.side_end()
        except RuntimeError as e:
            log(f"#{self.rid} side request: {e}")
            eng.side_drop()  # (whatever is left of it: the engine starts over)
            own = None
        info = self.store_update()
        kept = ""
        if pid is not None:
            self.parked[pid] = Parked(pid, *own, side=True, tools=self.live_tools)
            info = self.store_update()
            self.count(side_kept=1)
            kept = (f"; its own conversation ({len(own[0]):,} tok) parked: {mib(info.get('side_park_bytes', 0))} in "
                    f"{info.get('side_park_ms', 0.0):.0f} ms | {len(self.parked)} parked, {mib(info['used'])} in use")
        elif own is not None and ok:
            kept = "; its own conversation was not parked (no room in the host store)"
        if ok:
            self.img_spans, self.live_gen, self.live_tools = spans, gen, tools
            self.live_low = after if after and self.answer_tokens >= SUMMARY_ANSWER_MIN else low
            log(f"#{self.rid} side request done: the cached {n:,} tok are back ({mib(info['last_bytes'])}: kept in "
                f"{info['side_ms']:.0f} ms, back in {info['last_ms']:.0f} ms){kept}")
        elif own is not None and room:  # (side_end_park: the kept sequence was lost, the side request's conversation stays)
            self.count(side_dropped=1)
            log(f"#{self.rid} side request: the cached {n:,} tok were dropped (the host store ran full): the side "
                f"request's own conversation ({len(own[0]):,} tok) stays cached")
        else:
            self.img_spans, self.live_gen, self.live_tools, self.live_low = [], -1, None, ""
            self.count(side_dropped=1)
            log(f"#{self.rid} side request: the cached {n:,} tok were dropped (the host store ran full, or an engine "
                f"error): the next request prefills its prompt in full")

    def debris_finish(self, replied):
        """after a request without tools that parked the conversation with tools it replaced with low priority
        (cache_switch: likely a summary of it, the compacted conversation comes next): a reply whose answer is shorter
        than a summary's (SUMMARY_ANSWER_MIN tokens, thinking aside, as side_finish counts) gives that conversation, if
        it is still parked, normal priority again (a client's title or tags task quoting it). The mark stays after a
        summary's answer and without a reply (replied False: an error, or the client left): clients cut long summaries
        and move on or retry, and a retry continues the cut request's prompt, which parks nothing and decides nothing"""
        e, self.debris = self.debris, None
        if e is None or not replied or self.answer_tokens >= SUMMARY_ANSWER_MIN or self.parked.get(e.id) is not e:
            return
        e.low = False
        self.store_update()
        log(f"#{self.rid} the parked conversation of {len(e.seq):,} tok keeps normal priority: the request answered in "
            f"{self.answer_tokens:,} tok (< {SUMMARY_ANSWER_MIN}: not a summary)")

    def setup_video(self, args):
        """video input (input_video / video_url parts): needs the vision library's frame pairs (mkv_frames_load) and an
        ffmpeg (find_ffmpeg); without them video requests get a 400 and video_ok stays False"""
        if args.video_max_tokens and self.vision.backend != "vit_cpu":
            raise RuntimeError("--video-max-tokens needs the cpu vision backend (libmtmd resizes frames as images)")
        if not self.vision.frames_ok:
            log(f"video input off: {args.vision_lib} has no video frames (mkv_frames_load): rebuild it")
            return
        exe = find_ffmpeg(args.ffmpeg)
        if not exe:
            log("video input off: no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if not (args.ffmpeg or os.environ.get("MK_FFMPEG")) else
                f"video input off: ffmpeg not found at {args.ffmpeg or os.environ.get('MK_FFMPEG')}")
            return
        dec = VideoDecoder(exe)
        try:
            dec.check()
        except RuntimeError as e:
            log(f"video input off: {e}")
            return
        self.decoder, self.video_ok = dec, True
        log(f"video input: {exe} ({dec.version}), {args.video_fps:g} fps, frames of at most "
            f"{args.video_max_tokens or args.image_max_tokens or 'the default'} tokens, {MAX_VIDEO_FRAMES} frames per request")

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    # ---- reporting: /props, /v1/models, /slots, /metrics ----
    METRICS = ("requests", "requests_bad", "requests_failed", "requests_cancelled", "deferred", "prompt_tokens",
               "prompt_tokens_reused", "prompt_seconds", "tokens_predicted", "predicted_seconds", "decode_cycles",
               "draft_tokens", "draft_tokens_accepted", "side_requests", "side_dropped", "side_kept", "side_resumed", "parks",
               "park_restores", "park_evictions")
    _mlock = threading.Lock()
    t_start = 0.0
    warm_s = 0.0

    def count(self, **kw):
        """adds to the counters of /metrics"""
        with self._mlock:
            m = self.__dict__.setdefault("metrics", dict.fromkeys(self.METRICS, 0))
            for k, v in kw.items():
                m[k] += v

    def counters(self):
        with self._mlock:
            return dict(self.__dict__.get("metrics") or dict.fromkeys(self.METRICS, 0))

    def busy(self):
        return self.lock.locked()

    def processing(self):
        """requests the engine works on (/metrics)"""
        return int(self.busy())

    def n_vocab(self):
        return getattr(self.eng, "n_vocab", 0)

    def prompt_tokens(self, body):
        """the token count of the request's prompt (rendered with the chat template, images as their tokens), without
        generating: /v1/messages/count_tokens, /v1/chat/completions/input_tokens"""
        text, _, media = self.render(body)
        if not media:
            return len(self.tokenize_prompt(text, media)[0])
        with self.lock:  # (the image cache belongs to the running request too)
            toks, images = self.tokenize_prompt(text, media)
            self.vision.release([im for _, im in images])
        return len(toks)

    def engine_info(self):
        """what the server knows for sure about its engine: arguments, environment, the library's own answers"""
        a, e, v = self.args, self.eng, self.vision
        g = lambda o, k, d=None: getattr(o, k, d)  # noqa: E731 (fake engines in the tests have fewer attributes)
        return {
            "build": g(e, "build"), "lib": os.path.basename(str(g(a, "lib", ""))),
            "n_ctx": a.ctx, "kv": g(a, "kv"),  # (the KV format asked for; "auto" is resolved inside the engine)
            "mtp": g(e, "has_mtp", True), "mtp_file": g(a, "mtp"),
            "draft_depth": g(a, "draft") if g(e, "has_mtp", True) else 0,
            "draft_vocab": os.path.basename(a.draft_vocab) if isinstance(g(a, "draft_vocab"), str) else None,
            "exact_reuse": bool(g(e, "exact", False)),
            "checkpoints": {"pool": bool(g(e, "pool", False)), "every": g(a, "ck_every"), "messages": bool(g(a, "ck_msgs")),
                            "pinned": bool(g(a, "ck_pin") and g(e, "ck_pins", False)),
                            "slots": int(os.environ.get("MK_CK_SLOTS") or 8)},
            "keep_warm_s": self.warm_s, "prefill_cancel": bool(g(e, "_cancel", False)),
            "host_cache": None if self.parked is None else {
                "budget_mib": self.store_mib, "side_max": g(a, "side_max"), "park_mib": g(a, "park_mib"),
                "park_min": g(a, "park_min"), "park_checkpoints": g(a, "park_cks"), "parked": self.store_gauge[0],
                "parked_tokens": list(self.store_gauge[3]), "parked_side": self.store_gauge[4],
                "parked_low": self.store_gauge[5],
                "side_conversations": bool(self.side_keep()), "used_mib": round(self.store_gauge[1] / 2**20, 1),
                "allocated_mib": round(self.store_gauge[2] / 2**20, 1)},
            "vision": None if v is None else {
                "mmproj": os.path.basename(str(g(a, "mmproj", ""))), "backend": g(v, "backend"),
                "threads": g(v, "n_threads"), "image_min_tokens": g(a, "image_min_tokens"),
                "image_max_tokens": g(a, "image_max_tokens"), "image_cache": g(v, "cache_max")},
            "limits": {"max_body_mb": MAX_BODY_MB, "max_images": MAX_IMAGES, "max_stop": MAX_STOP,
                       "image_max_pixels": int(os.environ.get("MK_IMAGE_MAX_PIXELS") or 100_000_000)},
            "env": {k: "(set)" if k == "MK_API_KEY" and val else val  # (not the key: /props ends up in bug reports)
                    for k, val in sorted(os.environ.items()) if k.startswith("MK_")},
            "busy": self.busy(), "uptime_s": round(time.time() - self.t_start, 1) if self.t_start else 0.0,
        }

    def props(self):
        """the fields of llama-server's GET /props that the server has, plus `engine` (engine_info)"""
        b = getattr(self.eng, "build", None) or {}
        params = {"temperature": DEF_TEMP, "top_k": DEF_TOP_K, "top_p": DEF_TOP_P, "min_p": DEF_MIN_P, "seed": -1,
                  "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": -1, "n_predict": -1, "stop": [],
                  "stream": False, "n_probs": 0}
        return {
            "default_generation_settings": {"n_ctx": self.args.ctx, "params": params},
            "total_slots": 1, "model_alias": self.model_id, "model_path": getattr(self.args, "model", None),
            "modalities": {"vision": self.vision is not None, "video": bool(getattr(self, "video_ok", False)),
                           "audio": False},
            "media_marker": MEDIA_MARKER, "endpoint_slots": True, "endpoint_props": False, "endpoint_metrics": True,
            "chat_template": getattr(self, "template_src", None),
            "build_info": f"{b.get('commit', 'unknown')} {b.get('flags', '')}".strip() if b else "unknown",
            "is_sleeping": False, "engine": self.engine_info(),
        }

    def models(self):
        """GET /v1/models: the one model, with llama-server's meta where the server has it"""
        meta = {"n_ctx": self.args.ctx}
        try:
            meta["size"] = os.path.getsize(self.args.model)
        except (OSError, TypeError):
            pass
        if getattr(self.eng, "images", False):
            meta["n_embd"] = self.eng.L.mk_n_embd()
        if self.n_vocab():
            meta["n_vocab"] = self.n_vocab()
        caps = ["completion"] + (["multimodal"] if self.vision is not None else [])
        return {"object": "list",
                "data": [{"id": self.model_id, "object": "model", "created": int(self.t_start), "owned_by": "openjet",
                          "meta": meta}],
                "models": [{"name": self.model_id, "model": self.model_id, "type": "model", "capabilities": caps,
                            "details": {"format": "gguf"}}]}

    def slots(self):
        """GET /slots: the one slot (the llama-server fields the server has)"""
        return [{"id": 0, "n_ctx": self.args.ctx, "is_processing": self.busy(),
                 "speculative": bool(getattr(self.eng, "has_mtp", True)) and bool(getattr(self.args, "draft", 0))}]

    def metrics_text(self):
        """GET /metrics: Prometheus text; llama-server's names (llamacpp:) where they mean the same, megakernel: else"""
        m = self.counters()
        cyc, drafted, accepted = m["decode_cycles"], m["draft_tokens"], m["draft_tokens_accepted"]
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed (prefilled; reused ones not).",
             m["prompt_tokens"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt process time", m["prompt_seconds"]),
            ("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", m["tokens_predicted"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", m["predicted_seconds"]),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.",
             m["prompt_tokens"] / m["prompt_seconds"] if m["prompt_seconds"] else 0.0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.",
             m["tokens_predicted"] / m["predicted_seconds"] if m["predicted_seconds"] else 0.0),
            ("llamacpp:requests_processing", "gauge", "Number of requests processing.", self.processing()),
            ("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", m["deferred"]),
            ("megakernel:prompt_tokens_reused_total", "counter", "Prompt tokens reused from the engine's sequence.",
             m["prompt_tokens_reused"]),
            ("megakernel:decode_cycles_total", "counter", "Engine decode cycles (one verify + MTP draft each).", cyc),
            ("megakernel:draft_tokens_total", "counter", "MTP draft tokens produced.", drafted),
            ("megakernel:draft_tokens_accepted_total", "counter", "MTP draft tokens accepted.", accepted),
            ("megakernel:requests_total", "counter", "Generation requests received.", m["requests"]),
            ("megakernel:requests_bad_total", "counter", "Generation requests answered with a 4xx.", m["requests_bad"]),
            ("megakernel:requests_failed_total", "counter", "Generation requests that failed (5xx).",
             m["requests_failed"]),
            ("megakernel:requests_cancelled_total", "counter", "Generation requests whose client disconnected.",
             m["requests_cancelled"]),
            ("megakernel:side_requests_total", "counter", "Side requests (the cached sequence kept in host memory).",
             m["side_requests"]),
            ("megakernel:side_requests_dropped_total", "counter", "Side requests after which the cached sequence was lost.",
             m["side_dropped"]),
            ("megakernel:side_conversations_parked_total", "counter", "Side requests whose own conversation was parked.",
             m["side_kept"]),
            ("megakernel:side_conversations_resumed_total", "counter",
             "Side requests that continued a parked side request's conversation.", m["side_resumed"]),
            ("megakernel:parks_total", "counter", "Conversations parked in host memory.", m["parks"]),
            ("megakernel:park_restores_total", "counter", "Parked conversations brought back.", m["park_restores"]),
            ("megakernel:park_evictions_total", "counter", "Parked conversations evicted (host memory budget).",
             m["park_evictions"]),
            ("megakernel:parked_conversations", "gauge", "Conversations parked in host memory now.", self.store_gauge[0]),
            ("megakernel:host_cache_bytes", "gauge", "Host memory in use by parked conversations.", self.store_gauge[1]),
        ]
        return "".join(f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v:g}\n" if isinstance(v, float) else
                       f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v}\n" for n, t, h, v in rows)

    def render(self, body):
        """(prompt text, thinking, image file bytes in prompt order: one MEDIA_MARKER in the text each)"""
        kw = dict(body.get("chat_template_kwargs") or {})
        eff = body.get("reasoning_effort")  # (OpenAI's top-level field; chat_template_kwargs win)
        if isinstance(eff, str) and eff:
            if eff == "none":  # (as llama-server)
                kw.setdefault("enable_thinking", False)
            else:  # (the template knows low / medium / high / xhigh)
                if eff in ("max", "ultra"):  # (Codex offers them for its own models: the template's highest instead)
                    self.warn_once(f"effort-{eff}", f"reasoning effort {eff!r} is taken as 'xhigh' (the chat "
                                                    f"template's highest)")
                    eff = "xhigh"
                kw.setdefault("reasoning_effort", {"minimal": "low"}.get(eff, eff))
        if "enable_thinking" not in kw:  # (Qwen3's soft switches, which Qwen3.8's template ignores: the request's own)
            soft = soft_switch(body.get("messages") or [])
            if soft is not None:
                self.warn_once("soft-switch", "a request's trailing /no_think or /think (last user message, else the "
                                              "system message) or an empty <think></think> after the last user text "
                                              "turns its thinking off / on (chat_template_kwargs and reasoning_effort "
                                              "none win)")
                kw["enable_thinking"] = soft
        media = [] if self.vision else None
        ctx = dict(messages=normalize_messages(body.get("messages") or [], media), add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k, v in (getattr(self.args, "chat_template_kwargs", None) or {}).items():  # (--chat-template-kwargs: under them)
            kw.setdefault(k, v)
        for k in TEMPLATE_KWARGS:
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        n_img = sum(1 for d in media or [] if not isinstance(d, VideoBytes))
        if n_img > MAX_IMAGES:
            raise BadRequest(f"{n_img} images in one request: at most {MAX_IMAGES} (MK_MAX_IMAGES)")
        if n_img < len(media or []) and not self.video_ok:
            raise BadRequest("video input is not available: " + (
                "the server found no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if self.vision is not None and self.vision.frames_ok else "the vision library has no video frames"))
        if media and text.count(MEDIA_MARKER) != len(media):
            raise BadRequest(f"the rendered prompt has {text.count(MEDIA_MARKER)} media markers for {len(media)} images / videos")
        return text, thinking, media or []

    def tokenize_prompt(self, text, media):
        """the prompt's token ids and its images [(pos, Vision.Image)]: each MEDIA_MARKER becomes <|vision_start|>, the
        image's rows (<|image_pad|> placeholders, nx * ny of them), <|vision_end|>; the text between is tokenized piece
        by piece, as mtmd_tokenize does. A video's marker becomes its parts (Vision.video: texts, and frames / frame
        pairs each like an image). The images are pinned (Vision.image / video): the caller releases them. Special-token
        text from the request's content (a Prompt of LiteralTemplate) is tokenized as text (tokenize_text)."""
        lit = getattr(text, "lit", None)
        if not media:
            return self.tokenize_text(lit[0] if lit else text, lit), []
        toks, images = [], []
        segs = (lit[0] if lit else text).split(MEDIA_MARKER)  # (the same media markers: LiteralTemplate checked it)
        frames_left = MAX_VIDEO_FRAMES

        def add_image(im, j=None):  # (j: its slot in images, else appended)
            toks.append(self.vis_start)
            if j is None:
                images.append((len(toks), im))
            else:
                images[j] = (len(toks), im)
            toks.extend([self.img_pad] * im.n)
            toks.append(self.vis_end)
        try:
            for i, seg in enumerate(segs):
                if seg:
                    toks += self.tokenize_text(seg, lit)
                if i < len(media):
                    if isinstance(media[i], VideoBytes):
                        # llama.cpp's layout: "Video:", frames / frame pairs as images, timestamp texts between (each
                        # text tokenized on its own, as mtmd_tokenize does)
                        v = self.vision.video(media[i], self.decoder, self.args.video_fps, self.args.video_max_tokens,
                                              frames_left)
                        j = len(images)
                        images.extend((None, im) for im in v.images)  # (all pinned now: released below on an error)
                        frames_left -= v.n_frames
                        for part in v.parts:
                            if isinstance(part, str):
                                toks += self.eng.tokenize(part)
                            else:
                                add_image(part, j)
                                j += 1
                    else:
                        add_image(self.vision.image(media[i]))
                    if len(toks) + ContextOverflow.RESERVE >= self.args.ctx:  # (no more images decoded for it)
                        raise ContextOverflow(len(toks), self.args.ctx, at_least=True)
        except BaseException:
            self.vision.release([im for _, im in images])
            raise
        return toks, images

    def tokenize_text(self, text, lit=None):
        """the token ids of rendered prompt text (a piece between media). With lit (Prompt.lit) the text comes from the
        prompt rendered with the request content's special-token texts as placeholders: it is split at the template's
        own control tokens (kept), and each piece between them is tokenized with its texts put back, as plain text where
        it has any (parse_special false: the piece is one run of text, as llama.cpp tokenizes the text between special
        tokens, so those texts never become control tokens). Pieces without any, and every prompt without such text
        in its content, are tokenized with control-token text parsed as special tokens (Engine.tokenize's default)."""
        if not lit:
            return self.eng.tokenize(text)
        _, back, lt = lit
        toks, pos = [], 0
        for m in itertools.chain(lt.ctl_re.finditer(text), [None]):
            piece = text[pos:m.start() if m else len(text)]
            if piece:
                orig = piece.translate(back)
                toks += self.eng.tokenize(orig, special=False) if orig != piece else self.eng.tokenize(piece)
            if m:
                toks.append(lt.controls[m.group()])
                pos = m.end()
        return toks

    def ck_positions(self, toks, start):
        """checkpoints for a prefill of toks[start:] besides the generation prompt: each message start and every
        --ck-every tokens (absolute positions; the engine keeps the most useful ones in its pool)"""
        cks = set()
        every = self.args.ck_every
        if every > 0:
            cks.update(range((start // every + 1) * every, len(toks) - 1, every))
        if self.args.ck_msgs and self.im_start is not None:
            cks.update(k for k in range(start + 1, len(toks) - 1) if toks[k] == self.im_start)
        return sorted(cks)

    def ck_pins(self, toks, lcp, spans):
        """checkpoints the engine pins for this prompt (--ck-pin; Engine.ck_pin: never evicted from the pool, parked with
        the conversation): the end of its system block (the start of the first message after it), which every session
        of the same client shares, tools included; and, when the prompt leaves the cached conversation (lcp) or a parked
        one inside that block, the point where it does: a client whose system prompt ends with per-session text (a date,
        a directory, a session id) shares up to there with its other sessions. That one comes first (it serves every
        session of the client). A prompt without a system message, or a prefix under PIN_MIN tokens: none."""
        role = getattr(self, "sys_role", None)
        if (not getattr(self.args, "ck_pin", 0) or not getattr(self.eng, "ck_pins", False) or self.im_start is None
                or not role or toks[:1 + len(role)] != [self.im_start] + role):
            return []
        end = next((k for k in range(1 + len(role), len(toks) - 1) if toks[k] == self.im_start), 0)
        if end < PIN_MIN:
            return []
        left = [lcp]
        if lcp < end:
            for e in (self.parked or {}).values():
                if not e.side:
                    r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                    left.append(r[0] if r[1] is None else r[1])
        cut = max((x for x in left if PIN_MIN <= x < end), default=0)
        return [cut, end] if cut else [end]

    @contextlib.contextmanager
    def prefill_status(self, n_new):
        """the console's status line during a prefill call: the engine's progress (mk_prefill_progress), its rate and the
        time left, every 0.25 s from a watcher thread (nothing for a prefill shorter than that, or off a terminal)"""
        if not CON.tty():
            yield
            return
        stop, t0 = threading.Event(), time.monotonic()
        prog = getattr(self.eng, "prefill_progress", None)

        def show():
            while not stop.wait(0.25):
                CON.status(self.prefill_text(n_new, prog() if prog else None, time.monotonic() - t0))
        th = threading.Thread(target=show, name="mk-status", daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()

    @contextlib.contextmanager
    def prompt_progress(self, cached, total):
        """a request's return_progress (self.on_progress set): on_progress(cached, processed, total) when the prefill
        starts, from a watcher thread every 0.5 s while it runs (the engine's mk_prefill_progress; ctypes releases the
        GIL) and when it is done, as llama-server's prompt_progress events. The watcher is joined before the request
        thread goes on, so the two never write to the client at once."""
        cb = getattr(self, "on_progress", None)
        if cb is None:
            yield
            return
        prog, stop, seen = getattr(self.eng, "prefill_progress", None), threading.Event(), [cached]

        def watch():
            while not stop.wait(0.5):
                p = prog() if prog else None
                if p and p[1] > 0 and cached + p[0] > seen[0]:
                    seen[0] = cached + p[0]
                    cb(cached, seen[0], total)
        cb(cached, cached, total)
        th = threading.Thread(target=watch, name="mk-progress", daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()
        cb(cached, total, total)

    def timings_now(self):
        """timings_per_token: the timings of the reply so far (llama-server's fields)"""
        prompt_n, prompt_ms, t1, n_gen = self.tick
        ms = max((time.time() - t1) * 1e3, 1e-6)
        return {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": n_gen, "predicted_ms": ms,
                "predicted_per_second": n_gen * 1e3 / ms}

    def prefill_text(self, n_new, prog, el):
        """prog: the engine's (done, total) of the batched part, or None (a library without it: elapsed time only)"""
        if prog and prog[1] > 0:
            done, total = prog
            s = f"#{self.rid} prefill {done:,} / {total:,} tok ({100 * done // total}%)"
            if done:
                rate = done / max(el, 1e-9)
                s += f", {rate:,.0f} tok/s, ~{secs((total - done) / rate)} left"
            return f"{s} | {secs(el)}"
        return f"#{self.rid} prefill {n_new:,} tok | {secs(el)}"

    def decode_text(self, n_gen, phase, steps, stepped, el):
        s = f"#{self.rid} decoding {n_gen:,} tok, {n_gen / max(el, 1e-9):.1f} tok/s"
        if steps:
            s += f", {stepped / steps:.1f} tok/cycle"
        ph = ", ".join(f"{k} {v:,}" for k, v in phase.items() if v)
        return s + (f" | {ph}" if ph else "") + f" | {secs(el)}"

    def load_prompt(self, toks, images=(), gone=None, first_turn=False, max_tokens=-1, tools=True):
        """brings the engine to `toks` processed, reusing what it has; returns (first token, reused count).
        images: [(pos, Vision.Image)] of the prompt. gone (the client left: () -> bool) is watched during the prefill,
        which then stops at a chunk boundary with PrefillCancelled (Engine.cancel_when). first_turn: the prompt has no
        earlier assistant reply; max_tokens: the request's (-1: none); tools: the request has tools (not tool_choice
        "none"); all three for cache_switch's side request rule"""
        eng = self.eng
        self.cache_pre = ""
        if self.parked is not None:  # (the host cache: --side-max / --park-mib)
            self.cache_switch(toks, [(p, im.n, im.key) for p, im in images], first_turn, max_tokens, tools)
        seq = eng.sequence()
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        match, img_cut = lcp, None
        # rewind checkpoint at the last <|im_start|> (the generation prompt): a follow-up turn re-renders this turn
        # from there (thinking mode: <think>\n\n</think> instead of <think>\n), so it must not sit at the prompt's end
        gen = -1
        if self.im_start is not None:
            for k in range(len(toks) - 1, -1, -1):
                if toks[k] == self.im_start:
                    gen = k
                    break
        # images: their rows' placeholder ids are alike, so the processed sequence matches only up to the first image
        # that is not the same image at the same position in both
        spans = [(p, im.n, im.key) for p, im in images]
        for p, _, _ in set(spans) ^ set(self.img_spans):
            if p < lcp:
                lcp, img_cut = p, p

        def feed(start):  # prefill toks[start:] with the images it overlaps (encoded now, each once: Vision's cache)
            fed = [(p, im) for p, im in images if p + im.n > start]
            if any(im.encode_s is None for _, im in fed):
                # the CPU encodes leave the GPU idle (up to seconds): keep its clocks up until the prefill's first launch
                eng.keep_warm()
            imgs, armed = [], time.monotonic()
            todo, k = sum(1 for _, im in fed if im.encode_s is None), 0
            for p, im in fed:  # (a video's frame pairs encode for seconds to minutes: a client that left stops them)
                if imgs and im.encode_s is None:
                    if gone is not None and gone():
                        raise PrefillCancelled(f"the client left during the image encodes ({len(imgs)} of {len(fed)} done)")
                    if time.monotonic() - armed > 30:  # (keep-warm re-armed through long encodes)
                        eng.keep_warm()
                        armed = time.monotonic()
                if im.encode_s is None:
                    k += 1
                    CON.status(f"#{self.rid} encoding image {k} / {todo} on the CPU ({im.nx}x{im.ny} tok)")
                imgs.append((p, im.nx, im.ny, im.rows()))
            # (a side request saves the generation prompt's checkpoint only: where its own conversation's follow-up resumes)
            cks = self.ck_positions(toks, start) if self.side is None else []
            pins = self.ck_pins(toks, lcp, spans) if self.side is None else []
            if pins:
                eng.ck_pin(pins)
            with eng.cancel_when(gone), self.prefill_status(len(toks) - start), self.prompt_progress(start, len(toks)):
                if imgs:
                    return eng.prefill(toks[start:], gen, cks, imgs), start
                return eng.prefill(toks[start:], gen, cks), start

        self.img_spans = []  # (until the prefill succeeds)
        had = f"the prompt matches {match:,} of the {len(seq):,} cached tok" + (
            f" (to the image at {img_cut:,}, which differs)" if img_cut is not None else "")
        # exact reuse (an MK_EXACT engine, --exact-reuse): never append to the sequence. Its last prompt row and the
        # reply were processed by the decode kernels, whose arithmetic differs from the batched prefill's; restoring
        # the last checkpoint at or before the divergence (all of them sit inside earlier batched prefills, on the
        # absolute 32-token scan grid) and prefilling from there computes what a prefill from 0 computes, bit for bit.
        if seq and lcp >= eng.committed() and not eng.exact:
            eng.truncate(lcp)
            self.cache_note = f"{had}: continued from there"
            r = feed(lcp)
        else:
            # the last checkpoint at or before the divergence (one inside a kept image: the image is fed again)
            ck = max((p for p in eng.checkpoints() if 0 < p <= lcp), default=0)
            if ck and eng.rewind_to(ck):
                self.cache_note = f"{had}: resumed at the checkpoint at {ck:,}"
                r = feed(ck)
            else:
                eng.reset()
                self.cache_note = f"CACHE MISS: {had}, no checkpoint before that" if seq else "nothing cached"
                r = feed(0)
        self.img_spans = spans
        self.live_gen, self.live_tools, self.live_low = gen, tools, self.next_low
        self.side_ready = self.side is not None
        self.cache_note = self.cache_pre + self.cache_note
        return r

    def sampling(self, body):
        """sets the request's sampling on the engine (sampling_params); returns it as text for the console"""
        temp, top_p, top_k, min_p, seed, pen, desc = self.sampling_params(body)
        self.eng.set_penalty(*pen)
        self.eng.set_sampling(temp, top_p, top_k, min_p, seed)
        return desc

    def sampling_params(self, body):
        """the request's sampling: (temp, top_p, top_k, min_p, seed, (presence, frequency penalty), text for the
        console), greedy (temperature <= 0) as the engine takes it.
        top_k: absent -> DEF_TOP_K; <= 0 ("off" for llama.cpp / OpenAI-style clients) -> TOPK_MAX, the engine's
        maximum; above TOPK_MAX -> TOPK_MAX with a one-time warning"""
        try:
            temp = float(body["temperature"] if body.get("temperature") is not None else DEF_TEMP)
            top_p = float(body["top_p"] if body.get("top_p") is not None else DEF_TOP_P)
            top_k = int(body["top_k"]) if body.get("top_k") is not None else DEF_TOP_K
            min_p = float(body["min_p"] if body.get("min_p") is not None else DEF_MIN_P)
            seed = body.get("seed")
            seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("temperature, top_p, top_k, min_p and seed must be numbers")
        if top_k <= 0:
            top_k = TOPK_MAX
        elif top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        # presence / frequency penalty, vLLM's semantics: tokens generated in this response so far (not the prompt)
        pen = []
        for k in ("presence_penalty", "frequency_penalty"):
            try:
                v = float(body.get(k) or 0)
            except (TypeError, ValueError):
                raise BadRequest(f"{k} must be a number")
            if not -2.0 <= v <= 2.0:
                raise BadRequest(f"{k} must be in [-2, 2]")
            pen.append(v)
        try:
            rep = float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("repetition_penalty must be a number")
        if rep != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            temp, top_p, top_k, min_p = 0.0, 1.0, TOPK_MAX, 0.0
            desc = "greedy"
        else:
            desc = f"temp {temp:g} top_p {top_p:g} top_k {top_k} min_p {min_p:g}"
            if body.get("seed") is not None and int(body["seed"]) >= 0:
                desc += f" seed {seed}"
        if any(pen):
            desc += f" presence {pen[0]:g} frequency {pen[1]:g}"
        return temp, top_p, top_k, min_p, seed, tuple(pen), desc

    def start_reply(self, toks, max_new, images, gone, first_turn, asked, tools):
        """processes the prompt (load_prompt): (the reply's first token, the prompt tokens reused, the engine cycle: a
        function returning the next tokens). SlotServer: in a slot, by the slot scheduler's thread"""
        tok, reused = self.load_prompt(toks, images, gone, first_turn, asked, tools)
        return tok, reused, self.eng.step

    def eng_stats(self):
        """the engine's (decode cycles, drafted, accepted) counters, or None. They move in its decode cycles only and
        restart at a reset (a full re-prefill in load_prompt): read after the prompt, the difference is the reply's"""
        return self.eng.stats() if getattr(self.eng, "stats", None) is not None else None

    def status(self, text):
        """the console's status line for the running request (SlotServer: none, its requests run at once)"""
        CON.status(text)

    def request_done(self):
        """after each generation request (the handler): the status line goes, the keep-warm window restarts"""
        CON.status("")
        self.eng.keep_warm()  # (MK_KEEP_WARM_S: the window restarts after every request)

    def generate(self, body, emit, gone=lambda: False):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas, and in a
        stream (body["stream"]) with ('tool_call', delta.tool_calls[] entry) as the tool calls are generated.
        Returns (finish_reason, tool_calls, usage, timings). emit returning False, or gone() returning True (the client
        closed its connection; asked before the prefill, during it by a watcher thread (the prefill stops at its next
        chunk) and before each engine cycle) stops it with ClientGone."""
        check_params(body)
        self.answer_tokens = 0
        self.status(f"#{self.rid} preparing the prompt (template, tokens, media)")
        text, thinking, media = self.render(body)
        toks, images = self.tokenize_prompt(text, media)
        replied = False
        try:
            out = self._generate(body, emit, gone, thinking, toks, images, media)
            replied = True
            return out
        finally:
            if images:  # (the request's images stay pinned until here)
                self.vision.release([im for _, im in images])
            self.side_finish()  # (a side request: the cached sequence comes back, also after an error or a disconnect)
            self.debris_finish(replied)
            if self.job is not None:  # (--agents: its slot leaves the next cycle, if the reply did not end there)
                self.job.close()
                self.job = None

    def _generate(self, body, emit, gone, thinking, toks, images, media=()):
        max_new = next((body[k] for k in ("max_tokens", "max_completion_tokens", "n_predict") if body.get(k) is not None), -1)
        try:
            max_new = int(max_new)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("max_tokens must be an integer")
        if max_new == 0:  # (OpenAI's and Anthropic's minimum is 1)
            raise BadRequest("max_tokens must be at least 1 (-1: up to the context)")
        asked = max_new
        limit = self.args.ctx - len(toks) - ContextOverflow.RESERVE
        if limit <= 0:
            raise ContextOverflow(len(toks), self.args.ctx)
        max_new = limit if max_new <= 0 else min(max_new, limit)
        # tool_choice "none": the tools stay in the prompt, calls are not parsed (returned as text), as llama-server does
        tools = (body.get("tools") or []) if body.get("tool_choice") != "none" else []
        stopper = StopMatcher(stop_strings(body.get("stop")))
        self.stop_word = None
        samp = self.sampling(body)
        rid = self.rid
        route, model, t_req = self.req
        n_vid = sum(1 for d in media if isinstance(d, VideoBytes))
        what = [f"{len(toks):,} tok", plural(len(body.get("messages") or []), "message")]
        if tools:
            what.append(plural(len(tools), "tool"))
        if len(media) > n_vid:
            what.append(plural(len(media) - n_vid, "image"))
        if n_vid:
            what.append(plural(n_vid, "video"))
        how = ["thinking" if thinking else "no thinking"] + ([f"max_tokens {asked:,}"] if asked > 0 else [])
        log(f"#{rid} {route}{' stream' if body.get('stream') else ''}"
            f"{f' {str(model)[:48]!r}' if isinstance(model, str) and model else ''}: "
            f"{', '.join(what)} | {', '.join(how)} | {samp}")
        if gone():  # (it left while the request waited for the engine)
            log(f"#{rid} the client disconnected before its prompt was processed")
            raise ClientGone()
        self.status(f"#{rid} prefill")
        t0 = time.time()
        enc0 = [im.encode_s for _, im in images]
        first_turn = not any(isinstance(m, dict) and m.get("role") == "assistant" for m in body.get("messages") or [])
        try:
            tok, reused, step = self.start_reply(toks, max_new, images, gone, first_turn, asked, bool(tools))
        except PrefillCancelled as e:  # (the engine kept nothing of it)
            log(f"#{rid} the client disconnected during the prefill of {len(toks):,} tok: stopped after "
                f"{secs(time.time() - t0)} ({e})")
            raise ClientGone()
        t1 = time.time()
        t0 += self.queued_s  # (--agents: the wait for a free slot is not prompt time)
        st0 = self.eng_stats()

        def spent():
            st1 = self.eng.stats() if st0 is not None else None
            d = tuple(b - a for a, b in zip(st0, st1)) if st0 and st1 else (0, 0, 0)
            self.count(decode_cycles=d[0], draft_tokens=d[1], draft_tokens_accepted=d[2])
            return d
        self.last_prompt = (len(toks), reused)  # (an Anthropic stream's message_start reports it)
        enc = [im.encode_s for (_, im), e in zip(images, enc0) if e is None and im.encode_s is not None]
        n_new, enc_s = len(toks) - reused, sum(enc)
        pf_s = max(t1 - t0 - enc_s, 1e-9)
        line = (f"#{rid} prompt: {reused:,} cached + {n_new:,} prefilled in {secs(pf_s)} ({n_new / pf_s:,.0f} tok/s)"
                f" | {self.cache_note}")
        if images:
            grids = sorted({f"{im.nx}x{im.ny}" for _, im in images})
            line += (f" | images: {len(images)} ({', '.join(grids[:3])}{', ...' if len(grids) > 3 else ''} tok grids), "
                     f"{len(enc)} encoded on the CPU in {secs(enc_s)}")
        self.status(f"#{rid} decoding")
        log(line)
        # a stream gets the tool calls as they are generated (ToolCallParser's events). reasoning_format (OpenAI route; an
        # Anthropic reply has its thinking blocks): none -> the thinking stays in the content, deepseek-legacy -> both
        # (a stream: in the content only, as llama-server)
        fmt = "auto" if route == "messages" else body.get("reasoning_format") or getattr(self.args, "reasoning_format", "auto")
        inline = {"none": "only", "deepseek-legacy": "only" if body.get("stream") else "both"}.get(fmt)
        parser = OutputParser(thinking, tools, bool(body.get("stream")), body.get("parallel_tool_calls") is False, inline)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, pending, aborted, dropped = 0, "length", [tok], False, 0
        # t_max_predict_ms (llama-server's): past it, the reply ends at the next token with a newline (finish "length")
        t_max, timed_out = float(body.get("t_max_predict_ms") or 0) / 1e3, False
        self.tick = (n_new, (t1 - t0) * 1e3, t1, 0)  # (timings_now)
        phase = {"thinking": 0, "answer": 0, "tool call": 0}  # generated tokens by the part of the reply they are in
        steps = stepped = 0  # engine cycles, the tokens they produced
        live, shown = CON.tty(), time.monotonic()
        while n_gen < max_new and not aborted:
            if not pending:
                if gone():  # once per engine cycle: a closed connection stops the GPU work here
                    aborted = True
                    break
                pending = step()
                if not pending:  # (--agents: the client left during the cycle (None), or the slot's context is full)
                    aborted = pending is None
                    break
                steps += 1
                stepped += len(pending)
                if live and time.monotonic() - shown >= 0.25:
                    shown = time.monotonic()
                    self.status(self.decode_text(n_gen, phase, steps, stepped, time.time() - t1))
            t = pending.pop(0)
            n_gen += 1
            if self.eng.is_eog(t):
                finish = "stop"
                break
            mode0 = parser.mode
            self.tick = self.tick[:3] + (n_gen,)
            piece = dec.decode(self.eng.piece(t))
            for kind, delta in parser.feed(stopper.feed(piece)):
                if emit(kind, delta) is False:
                    aborted = True
                    break
            phase["thinking" if mode0 == "reasoning" else "tool call" if parser.mode == "tools" else "answer"] += 1
            if stopper.word is not None:
                # a stop string: the reply ends at it, and the tokens after this one (the rest of the cycle's accepted
                # drafts) are dropped. The engine's sequence keeps them, as after max_tokens: the next request finds
                # the divergence (load_prompt) and continues from the reply that was sent.
                finish, dropped = "stop", len(pending)
                break
            if t_max > 0 and "\n" in piece and time.time() - t1 > t_max:  # (the rest of the cycle: as after a stop)
                finish, dropped, timed_out = "length", len(pending), True
                break
        if aborted:  # (the engine's state is that of a reply cut by max_tokens: the next request reuses it as usual)
            spent()
            log(f"#{rid} the client disconnected after {n_gen:,} generated tok ({secs(time.time() - t1)}): stopped")
            raise ClientGone()
        for kind, delta in parser.feed(stopper.feed(dec.decode(b"", final=True), final=True), final=True):
            emit(kind, delta)
        self.stop_word = stopper.word
        self.reasoning_tokens, self.answer_tokens = phase["thinking"], phase["answer"]
        calls = parser.calls
        if body.get("parallel_tool_calls") is False:
            calls = calls[:1]  # (the first call only)
        rest = parser.unparsed()
        shown = parser.unparsed(streamed=True) if body.get("stream") else rest
        if shown:
            emit("content", shown)
        # a call cut by max_tokens is text (a stream's client has it as a call it started, arguments incomplete), and the
        # reply ends with "length" (complete calls before it are still sent)
        if calls and not (rest and finish == "length"):
            finish = "tool_calls"
        t2 = time.time()
        dt = max(t2 - t1, 1e-9)
        # (one token is the prefill's: no decode rate)
        parts = [f"{n_gen:,} tok in {secs(dt)}" + (f" = {n_gen / dt:.1f} tok/s" if n_gen > 1 else "")]
        ph = ", ".join(f"{k} {v:,}" for k, v in phase.items() if v)
        if ph:
            parts.append(ph)
        _, drafted, accepted = spent()
        if steps:  # (MTP: each engine cycle verifies the drafts and keeps the accepted ones plus one token)
            spec = f"{stepped / steps:.2f} tok/cycle"
            if drafted > 0:
                spec += f", {100 * accepted / drafted:.0f}% of {drafted:,} drafts accepted"
            parts.append(spec)
        parts.append(f"finish {finish}" + (f" ({plural(len(calls), 'call')})" if calls else "")
                     + (f" (stop string; {dropped} tok of the cycle dropped)" if stopper.word is not None else "")
                     + (f" (t_max_predict_ms; {dropped} tok of the cycle dropped)" if timed_out else ""))
        if t_req:
            parts.append(f"request {secs(t2 - t_req)}")
        log(f"#{rid} reply: " + " | ".join(parts))
        self.count(prompt_tokens=len(toks) - reused, prompt_tokens_reused=reused, prompt_seconds=t1 - t0,
                   tokens_predicted=n_gen, predicted_seconds=t2 - t1)
        usage = {"prompt_tokens": len(toks), "completion_tokens": n_gen, "total_tokens": len(toks) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        timings = {"prompt_n": len(toks) - reused, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - t1) * 1e3, "predicted_per_second": n_gen / max(t2 - t1, 1e-9)}
        return finish, calls, usage, timings


class PerRequest:
    """a Server attribute that belongs to the running request: in a SlotServer each request thread has its own (Server's
    class value until the thread sets one)"""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, srv, owner=None):
        if srv is None:
            return self
        return getattr(srv.local, self.name, getattr(Server, self.name))

    def __set__(self, srv, value):
        setattr(srv.local, self.name, value)


class NoLock(contextlib.nullcontext):
    """SlotServer.lock: requests do not take turns on the engine (the slot scheduler's thread runs it)"""

    @staticmethod
    def locked():
        return False


class SlotServer(Server):
    """--agents N: up to N requests are served at once, each in a parallel slot of the engine (SlotScheduler), with the
    reply it gets when it runs alone. Everything above the engine is Server's: the request threads render, tokenize,
    parse, stream and report as a request running alone does, with their per-request state of their own (PerRequest).
    What the slots cannot serve is refused with a 400 (images and video: slot prompts are token ids; the presence /
    frequency penalties: slots sample with temperature, top_p, top_k, min_p and seed); the features of the engine's
    one sequence (side requests, parked conversations, pinned checkpoints) are off."""

    stop_word = PerRequest()
    last_prompt = PerRequest()
    reasoning_tokens = PerRequest()
    answer_tokens = PerRequest()
    rid = PerRequest()
    req = PerRequest()
    tick = PerRequest()
    cache_note = PerRequest()
    cache_pre = PerRequest()
    on_progress = PerRequest()
    job = PerRequest()
    slot_sp = PerRequest()
    queued_s = PerRequest()

    def __init__(self, args):
        self.local = threading.local()
        if args.mmproj:
            log("--mmproj ignored: --agents serves text only (images and video need the server without --agents)")
            args.mmproj = None
        super().__init__(args)

    def engine_ctx(self, args):
        return SLOT_OWN_CTX

    def setup_store(self, args):
        """(no host cache: side requests and parked conversations move the engine's one sequence aside)"""

    def setup_slots(self, args):
        if not getattr(self.eng, "slots", False):
            raise RuntimeError(f"--agents needs parallel slots: {os.path.basename(args.lib)} has no mk_slots_open2 "
                               f"(rebuild it)")
        args.ctx = self.eng.slots_open(args.agents, args.ctx)
        self.lock = NoLock()
        self.sched = SlotScheduler(self.eng, args.agents, args.ctx)
        s = self.sched
        kv = " ".join(f"{k}={os.environ[k]}" for k in ("MK_KV", "MK_KV_WIN") if os.environ.get(k))
        if s.mix:
            feed = f"{s.mix} rows of a waiting prompt ride along with each decode cycle"
        elif s.chunk:
            feed = f"a waiting prompt is fed {s.chunk} tokens at a time between the decode cycles"
        else:
            feed = "a waiting prompt is fed whole between two decode cycles (the other slots wait meanwhile)"
        if s.solo:
            feed += f", {s.solo} tokens at a time while no slot decodes"
        log(f"--agents {args.agents}: {plural(args.agents, 'slot')} of {args.ctx:,} positions each (q8_0 KV{', ' + kv if kv else ''}"
            f"{', MK_EXACT: prefill rows independent of the chunking' if self.eng.exact else ''}); {feed}; side requests, "
            f"parked conversations, pinned checkpoints and images are off; presence / frequency penalties are refused")

    def status(self, text):
        pass

    def request_done(self):
        """(the keep-warm window is the slot scheduler's: armed when it goes idle)"""

    def busy(self):
        return self.sched.full()

    def processing(self):
        return self.sched.counts()[1]

    def counters(self):
        m = super().counters()
        m["deferred"] = self.sched.counts()[0]  # (requests waiting for a free slot)
        m["decode_cycles"] = self.sched.cycles  # (engine cycles of all slots together; drafts are not counted per slot)
        return m

    def eng_stats(self):
        return None  # (the engine counts the cycles of every slot together)

    def props(self):
        p = super().props()
        p["total_slots"] = self.sched.n
        return p

    def engine_info(self):
        info = super().engine_info()
        waiting, running = self.sched.counts()
        s = self.sched
        info["agents"] = {"slots": s.n, "slot_ctx": self.args.ctx, "running": running, "waiting": waiting,
                          "prefill_mix_rows": s.mix, "prefill_chunk": s.chunk, "prefill_solo": s.solo, "cycles": s.cycles}
        return info

    def slots(self):
        """GET /slots: llama-server's fields for each slot, and its state in the scheduler"""
        spec = bool(getattr(self.eng, "has_mtp", True)) and bool(getattr(self.args, "draft", 0))
        return [{"id": k, "n_ctx": self.args.ctx, "is_processing": d["state"] != "idle", "speculative": spec, **d}
                for k, d in enumerate(self.sched.snapshot())]

    def render(self, body):
        for m in body.get("messages") or []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", *VIDEO_PARTS) for p in c):
                raise BadRequest("image and video input are not available with --agents: the parallel slots take text "
                                 "prompts only (run the server without --agents for them)")
        return super().render(body)

    def sampling(self, body):
        temp, top_p, top_k, min_p, seed, pen, desc = self.sampling_params(body)
        for k, v in zip(("presence_penalty", "frequency_penalty"), pen):
            if v:
                raise BadRequest(f"{k} is not available with --agents: the parallel slots sample with temperature, "
                                 f"top_p, top_k, min_p and seed only")
        self.slot_sp = (temp, top_p, top_k, min_p, seed)
        return desc

    def start_reply(self, toks, max_new, images, gone, first_turn, asked, tools):
        job = self.job = self.sched.submit(toks, self.slot_sp, max_new, self.rid)
        tok, reused = job.prompt(gone, self.on_progress)
        self.queued_s = job.t_begin - job.t_submit
        self.cache_note = f"slot {job.slot}" + (f" (after {secs(self.queued_s)} in the queue)" if self.queued_s >= 0.01 else "")
        return tok, reused, job.step


POST_ROUTES = {"/v1/chat/completions": "chat", "/chat/completions": "chat", "/v1/messages": "messages",
               "/v1/messages/count_tokens": "count_tokens", "/tokenize": "tokenize", "/detokenize": "detokenize",
               "/apply-template": "apply_template", "/v1/chat/completions/input_tokens": "input_tokens",
               "/chat/completions/input_tokens": "input_tokens", "/v1/responses": "responses", "/responses": "responses",
               "/v1/responses/input_tokens": "responses_input_tokens", "/responses/input_tokens": "responses_input_tokens"}
# llama-server's other POST routes (and OpenAI's compaction route): 501 (not supported here) instead of 404
UNSUPPORTED = {"/completion", "/completions", "/v1/completions", "/v1/responses/compact", "/responses/compact",
               "/infill", "/embedding", "/embeddings", "/v1/embeddings", "/rerank", "/reranking",
               "/v1/rerank", "/v1/reranking", "/v1/audio/transcriptions", "/audio/transcriptions", "/lora-adapters",
               "/v1/chat/completions/control", "/models", "/v1/streams/lookup", "/tools", "/cors-proxy"}

# ---- the request gate: Host (DNS rebinding) and Origin (other sites' web pages) checks, CORS for listed origins ----
HOST_NAME = r"(?:\[[0-9A-Za-z:._~%-]+\]|[0-9A-Za-z._~%!$&'()*+,;=-]+)"  # (a reg-name or a bracketed IPv6 address)
AUTHORITY_RE = re.compile(rf"({HOST_NAME})(?::([0-9]{{0,5}}))?")
ORIGIN_RE = re.compile(rf"([A-Za-z][A-Za-z0-9+.-]*)://({HOST_NAME})(?::([0-9]{{0,5}}))?/?")
DEFAULT_PORTS = {"http": 80, "https": 443}
CORS_METHODS = "GET, POST, OPTIONS"
CORS_MAX_AGE = 86400  # seconds a browser may keep a preflight's answer (every request is still checked)
HEADER_LIST_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+(?:[ \t]*,[ \t]*[!#$%&'*+.^_`|~0-9A-Za-z-]+)*")


def is_ip(host):
    """host (as in a URL: IPv6 in brackets) is an IP address literal"""
    try:
        ipaddress.ip_address(host[1:-1] if host.startswith("[") and host.endswith("]") else host)
        return True
    except ValueError:
        return False


def parse_origin(origin):
    """(scheme, host, port) of an Origin header / --cors-origin value, lower case, port None for a scheme without a
    default one; None if it is not an origin ("null" -> ("null", "", None): only a listed "null" matches it)"""
    o = origin.strip()
    if o == "null":
        return "null", "", None
    m = ORIGIN_RE.fullmatch(o)
    if not m or (m.group(3) and int(m.group(3)) > 65535):
        return None
    scheme, host = m.group(1).lower(), m.group(2).lower()
    return scheme, host, int(m.group(3)) if m.group(3) else DEFAULT_PORTS.get(scheme)


def origin_key(origin):
    """the origin as browsers serialize it (scheme://host[:port], without the scheme's default port), or None"""
    p = parse_origin(origin)
    if p is None or p[0] == "null":
        return p and "null"
    scheme, host, port = p
    return f"{scheme}://{host}" + (f":{port}" if port is not None and port != DEFAULT_PORTS.get(scheme) else "")


class Gate:
    """Which requests the server answers (main sets HTTPServer.gate; without one, every request).
    Host: localhost, an IP address, the --host name or an --allow-host one (any port), or none (HTTP/1.0). A web page
    that points its own site's name at this machine (DNS rebinding) is refused: it would be same-origin with the server
    and could read the replies, /props and /metrics.
    Origin (browsers send it with every POST and every cross-site request): the server's own (Origin's host and port ==
    Host's, 80 without a port) or an --cors-origin one; pages of any other origin (other sites, another program's port
    on this machine, file: pages) get a 403, including the simple requests (a text/plain POST) a page may send without a
    preflight. Requests without an Origin are not from a web page (curl, the SDKs, agents) and are answered. An
    admitted origin gets CORS headers (GatedHandler).
    API key (--api-key): every request but /health and CORS preflights must carry it (authorized)."""

    def __init__(self, host="127.0.0.1", allow_hosts=(), cors_origins=(), api_key=None):
        self.api_key = api_key.encode() if api_key else None
        names = [h.strip().lower().rstrip(".") for h in allow_hosts if h.strip()]
        self.any_host = "*" in names
        self.hosts = {"localhost"} | {h for h in names if h != "*"}
        host = (host or "").strip().lower().rstrip(".")
        if host and not is_ip(host):
            self.hosts.add(host)
        self.origins = set()
        for o in cors_origins:
            if not o.strip():
                continue
            if o.strip() == "*":
                raise ValueError("--cors-origin / MK_CORS_ORIGINS: list the origins to allow (any origin is not supported)")
            key = origin_key(o)
            if key is None:
                raise ValueError(f"--cors-origin / MK_CORS_ORIGINS: {o.strip()[:100]!r} is not an origin "
                                 f"(scheme://host[:port], e.g. http://localhost:5173)")
            self.origins.add(key)

    def check(self, headers):
        """(None, the origin to send CORS headers to, or None) for a request to answer; (the 403's message, None) for
        one to refuse"""
        hosts, auth = headers.get_all("Host") or [], None
        if len(hosts) > 1:
            return "more than one Host header", None
        if hosts:
            m = AUTHORITY_RE.fullmatch(hosts[0].strip())
            if not m:
                return f"bad Host header {hosts[0].strip()[:100]!r}", None
            name = m.group(1).lower().rstrip(".")
            auth = name, int(m.group(2)) if m.group(2) else None
            if not (self.any_host or is_ip(name) or name in self.hosts):
                return (f"Host {name[:100]} is not allowed (web pages must not reach the server under other names): "
                        f"start the server with --allow-host {name[:100]} (env MK_ALLOW_HOSTS) to accept it"), None
        origins = headers.get_all("Origin") or []
        if not origins:
            return None, None
        if len(origins) > 1:
            return "more than one Origin header", None
        origin, p = origins[0].strip(), parse_origin(origins[0])
        if p is not None and (origin_key(origin) in self.origins or self.same(p, auth)):
            return None, origin
        shown = origin_key(origin) if p is not None else repr(origin[:100])
        return (f"Origin {shown} is not allowed (requests from web pages of other origins are refused): start the server "
                f"with --cors-origin {shown} (env MK_CORS_ORIGINS) to accept it"), None

    def authorized(self, command, path, headers):
        """the request may be answered: no API key set, /health or a CORS preflight (browsers send no credentials with
        those), or the key as `Authorization: Bearer <key>` (OpenAI's clients) or `x-api-key: <key>` (Anthropic's)"""
        if self.api_key is None or command == "OPTIONS" or path.split("?")[0] in ("/health", "/v1/health"):
            return True
        auth = (headers.get("Authorization") or "").strip()
        given = [auth[7:].strip()] if auth[:7].lower() == "bearer " else []
        given.append((headers.get("x-api-key") or "").strip())
        return any(hmac.compare_digest(g.encode(), self.api_key) for g in given if g)

    @staticmethod
    def same(p, auth):
        """the origin p is the server's own: Host's name and port (80 when Host has none). The scheme does not matter:
        nothing else can serve pages on the server's port."""
        return auth is not None and p[2] is not None and p[1] == auth[0] and p[2] == (auth[1] or 80)


class GatedHandler(BaseHTTPRequestHandler):
    """the handlers' base: the server's Gate decides on each request before it is handled (and before a 100 Continue);
    a refused one gets a 403 and the connection closes (its body is not read). An admitted browser origin gets
    Access-Control-Allow-Origin + Vary: Origin on every response (errors and event streams too), and its CORS preflights
    (OPTIONS) are answered. A request with a body its handler does not read (any method but POST: a GET, an OPTIONS)
    is answered with Connection: close and the connection closes, so that the body is not taken for the next request
    (behind a proxy: a smuggled one)."""
    protocol_version = "HTTP/1.1"
    cors = None  # the request's admitted Origin: CORS headers on the response
    admitted = None  # the gate's verdict on the request (None: not asked yet)
    BODY_METHODS = ("POST",)  # the methods whose handlers read the request body
    unread = False  # the request has a body its handler does not read: the response closes the connection

    def log_message(self, *a):
        pass

    def handle_one_request(self):
        self.cors = self.admitted = None
        self.unread = False
        super().handle_one_request()

    def parse_request(self):
        if not (super().parse_request() and self.admit()):
            return False
        cl = [v.strip() for v in self.headers.get_all("Content-Length") or []]  # (each one: "0" then "80" is a body)
        self.unread = self.command not in self.BODY_METHODS and bool(self.headers.get_all("Transfer-Encoding")
                                                                      or any(v != "0" for v in cl))
        return True

    def admit(self):
        """True if the gate lets the request through (asked once per request); else it is answered with a 403 (host,
        origin) or a 401 (API key)"""
        if self.admitted is None:
            gate = getattr(self.server, "gate", None)
            why, self.cors = gate.check(self.headers) if gate else (None, None)
            code, kind = 403, "permission_error"
            if why is None and gate and not gate.authorized(self.command, self.path, self.headers):
                why = "invalid or missing API key (the server was started with --api-key / MK_API_KEY)"
                code, kind = 401, "authentication_error"
            self.admitted = why is None
            if why:
                path = self.path.split("?")[0]
                log(f"{self.command} {path[:200]} -> {code}: {why}")
                err = ({"type": "error", "error": {"type": kind, "message": why}} if path.startswith("/v1/messages")
                       else {"error": {"code": code, "message": why, "type": kind}})
                data = json.dumps(err).encode()
                self.send_response(code)
                if code == 401:
                    self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                self.wfile.write(data)
        return self.admitted

    def end_headers(self):
        if self.cors:
            self.send_header("Access-Control-Allow-Origin", self.cors)
            self.send_header("Vary", "Origin")
        if self.unread and not self.close_connection:  # (send_header sets close_connection: once)
            self.send_header("Connection", "close")
        super().end_headers()

    def do_OPTIONS(self):
        """a CORS preflight of an admitted origin (the others got the gate's 403): the methods, the headers it asked for
        and how long it may keep the answer; any other OPTIONS: the methods"""
        headers = [("Allow", CORS_METHODS)]
        if self.cors and self.headers.get("Access-Control-Request-Method"):
            headers += [("Access-Control-Allow-Methods", CORS_METHODS), ("Access-Control-Max-Age", str(CORS_MAX_AGE))]
            want = (self.headers.get("Access-Control-Request-Headers") or "").strip()
            if want and HEADER_LIST_RE.fullmatch(want):
                headers.append(("Access-Control-Allow-Headers", want))
            if (self.headers.get("Access-Control-Request-Private-Network") or "").strip().lower() == "true":
                headers.append(("Access-Control-Allow-Private-Network", "true"))  # (Chrome: a public page to this machine)
        self.send_response(200)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()


def make_handler(srv):
    class H(GatedHandler):
        def client_gone(self):
            """True once the client has closed its connection (end of file or reset). The request was read in full, so
            anything readable is the next request of a keep-alive client (still connected) or the close."""
            try:
                if not select.select([self.connection], [], [], 0)[0]:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def _json(self, code, obj, close=False):
            """close: then close the connection (a request body that was not read must not be taken for the next request)"""
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def _body_length(self):
            """(the request body's length, or "chunked", None), or (None, (status, message)) when it is not acceptable:
            no Content-Length (411), a malformed one (400), over MK_MAX_BODY_MB (413, refused before reading); a
            transfer coding other than chunked (501), chunked with a Content-Length too (400: two framings)"""
            te = [c.strip().lower() for v in self.headers.get_all("Transfer-Encoding") or [] for c in v.split(",")]
            if any(te):
                if [c for c in te if c] != ["chunked"]:
                    return None, (501, f"Transfer-Encoding {', '.join(c for c in te if c)!r}: only chunked is supported")
                if self.headers.get_all("Content-Length"):
                    return None, (400, "Content-Length and Transfer-Encoding: chunked together")
                return "chunked", None
            vals = [v.strip() for v in self.headers.get_all("Content-Length") or []]
            if not vals:
                return None, (411, "a Content-Length header is required")
            if len(set(vals)) > 1 or not re.fullmatch(r"[0-9]+", vals[0]):
                return None, (400, f"bad Content-Length {vals[0][:40]!r}")
            n = int(vals[0]) if len(vals[0]) <= 18 else 1 << 62
            if n > MAX_BODY_MB << 20:
                return None, (413, f"request body of {n} bytes: over the limit of {MAX_BODY_MB} MB (MK_MAX_BODY_MB)")
            return n, None

        CHUNK_LINE, TRAILER_LINES = 4096, 64  # (a chunk-size line with its extensions; trailer field lines)

        def _read_chunked(self):
            """a Transfer-Encoding: chunked body (RFC 9112 7.1), decoded into a buffer of at most MK_MAX_BODY_MB (413 as
            soon as a chunk would pass it); a malformed size line (not hex, over 16 digits, no CRLF), data without its
            CRLF or a malformed trailer section is a 400; trailers are read and dropped. Both close the connection (the
            lingering close drains what the client still sends). None: answered, or the client left mid-body"""
            cap, buf = MAX_BODY_MB << 20, bytearray()

            def refuse(code, msg):
                log(f"{self.command} {self.path.split('?')[0]} -> {code}: {msg}")
                self._json(code, {"error": {"code": code, "message": msg, "type": "invalid_request_error"}}, close=True)

            while True:
                line = self.rfile.readline(self.CHUNK_LINE + 1)
                if not line:  # (the client closed the connection mid-body)
                    self.close_connection = True
                    return None
                if not line.endswith(b"\r\n"):
                    return refuse(400, "chunked body: a chunk-size line without its CRLF (or over 4096 bytes)")
                size = line[:-2].split(b";", 1)[0].rstrip(b" \t")
                if not re.fullmatch(rb"[0-9A-Fa-f]{1,16}", size):
                    return refuse(400, f"chunked body: bad chunk size {size[:40].decode('latin-1')!r}")
                n = int(size, 16)
                if n == 0:
                    break
                if len(buf) + n > cap:
                    return refuse(413, f"chunked request body over the limit of {MAX_BODY_MB} MB (MK_MAX_BODY_MB)")
                data = self.rfile.read(n)
                if len(data) < n:
                    self.close_connection = True
                    return None
                buf += data
                end = self.rfile.read(2)
                if end != b"\r\n":
                    if len(end) < 2:
                        self.close_connection = True
                        return None
                    return refuse(400, "chunked body: chunk data without its CRLF")
            for _ in range(self.TRAILER_LINES + 1):  # (trailer fields: dropped)
                line = self.rfile.readline(self.CHUNK_LINE + 1)
                if line == b"\r\n":
                    return bytes(buf)
                if not line:
                    self.close_connection = True
                    return None
                if not line.endswith(b"\r\n"):
                    return refuse(400, "chunked body: a trailer line without its CRLF (or over 4096 bytes)")
            return refuse(400, f"chunked body: more than {self.TRAILER_LINES} trailer lines")

        def _refuse_body(self):
            """answers an unacceptable request body (_body_length) without reading it; True if it did"""
            refused = self._body_length()[1]
            if refused:
                log(f"{self.command} {self.path.split('?')[0]} -> {refused[0]}: {refused[1]}")
                self._json(refused[0], {"error": {"code": refused[0], "message": refused[1], "type": "invalid_request_error"}},
                           close=True)
            return bool(refused)

        def handle_expect_100(self):
            # (curl sends large bodies after a 100 Continue: a refused one is answered before it is sent)
            if not self.admit() or self.command == "POST" and self._refuse_body():
                return False
            return super().handle_expect_100()

        def _text(self, code, text, ctype="text/plain; charset=utf-8", headers=()):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, srv.models())
            elif path == "/props":
                self._json(200, srv.props())
            elif path == "/slots":
                if parse_qs(query).get("fail_on_no_slot", [""])[0] and srv.busy():
                    self._json(503, {"error": {"code": 503, "message": "no slot available", "type": "unavailable_error"}})
                else:
                    self._json(200, srv.slots())
            elif path == "/metrics":
                self._text(200, srv.metrics_text(), "text/plain; version=0.0.4",
                           [("Process-Start-Time-Unix", str(int(srv.t_start)))])
            else:
                self._error(404, "not found")

        def do_POST(self):
            path = self.path.split("?")[0]
            if self._refuse_body():
                return
            n = self._body_length()[0]
            if n == "chunked":
                raw = self._read_chunked()
                if raw is None:  # (refused, or the client left)
                    return
            else:
                raw = self.rfile.read(n) if n else b""
                if len(raw) < n:  # (the client closed the connection mid-body)
                    self.close_connection = True
                    return
            route = POST_ROUTES.get(path)
            # (errors in Anthropic's shape on /v1/messages, OpenAI's on /v1/responses: _error)
            anth = "responses" if path.startswith(("/v1/responses", "/responses")) else path.startswith("/v1/messages")
            if route is None:
                known = path in UNSUPPORTED or path.startswith(("/slots", "/props", "/models/"))
                self._error(501 if known else 404, f"{path} is not supported by the megakernel server", anth)
                return
            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):  # (RecursionError: nested too deep)
                body = None
            if not isinstance(body, dict):
                self._error(400, "the request body must be a JSON object", anth)
                return
            if route in ("tokenize", "detokenize", "apply_template", "input_tokens", "count_tokens", "responses_input_tokens"):
                try:  # (no generation: the tokenizer, from this thread; images take the engine's lock)
                    t0 = time.time()
                    out = getattr(self, "_" + route)(body)
                    if route.endswith("_tokens"):  # (a client counting its prompt: one console line each)
                        msgs = body.get("messages") if route != "responses_input_tokens" else body.get("input")
                        what = [plural(len(msgs), "message" if route != "responses_input_tokens" else "item")
                                if isinstance(msgs, list) else "a text input"]
                        if body.get("tools"):
                            what.append(plural(len(body["tools"]), "tool"))
                        ms = (time.time() - t0) * 1e3
                        log(f"{path}: {out['input_tokens']:,} tok ({', '.join(what)}) in {ms:.0f} ms")
                    self._json(200, out)
                except BadRequest as e:
                    self._bad(e, anth)
                except DISCONNECTS:
                    self.close_connection = True
                except Exception as e:  # (answered, not a dropped connection)
                    log(f"{path} error:", repr(e))
                    try:
                        self._error(500, str(e), anth)
                    except Exception:
                        pass
                return
            rid = next(srv.rids)
            if srv.sched is None and srv.busy():
                log(f"#{rid} {route}: waiting for #{srv.rid} (the engine runs one request at a time)")
            srv.count(requests=1, deferred=1)  # (deferred: waiting for the engine)
            with srv.lock:
                srv.count(deferred=-1)
                srv.rid, srv.req = rid, (route, body.get("model"), time.time())
                try:
                    if route == "messages":
                        self._anthropic(body)
                    elif route == "responses":
                        self._responses(body)
                    elif body.get("stream"):
                        self._stream(body)
                    else:
                        self._complete(body)
                except BadRequest as e:
                    srv.count(requests_bad=1)
                    self._bad(e, anth, rid)
                except (*DISCONNECTS, ClientGone) as e:
                    srv.count(requests_cancelled=1)
                    self.close_connection = True
                    if not isinstance(e, ClientGone):  # (generate reports the disconnects it notices itself)
                        log(f"#{rid} the client disconnected ({type(e).__name__})")
                except Exception as e:  # engine failure: report it, keep serving
                    srv.count(requests_failed=1)
                    log(f"#{rid} error:", repr(e))
                    try:
                        self._error(500, str(e), anth)
                    except Exception:
                        pass
                finally:
                    srv.request_done()

        def _error(self, code, message, anthropic=False, rid=None, kind=None, extra=None):
            """answers with an error (OpenAI / llama-server's shape with its code, Anthropic's, or with anthropic
            "responses" the OpenAI API's: code a string or null); a 4xx is also logged (a 5xx was logged with its
            exception). kind: the type instead of the code's; extra: more fields"""
            if code < 500:
                log(f"#{rid} {code}: {message}" if rid else f"{self.command} {self.path.split('?')[0]} -> {code}: {message}")
            if anthropic == "responses":
                kind = kind or {400: "invalid_request_error", 404: "not_found_error", 501: "not_supported"}.get(code, "server_error")
                self._json(code, {"error": {"message": message, "type": kind, "param": None, "code": None, **(extra or {})}})
            elif anthropic:
                err = {"type": kind or ANTHROPIC_ERRORS.get(code, "api_error"), "message": message, **(extra or {})}
                self._json(code, {"type": "error", "error": err})
            else:
                kind = kind or {400: "invalid_request_error", 404: "not_found_error", 501: "not_supported"}.get(code, "server_error")
                self._json(code, {"error": {"code": code, "message": message, "type": kind, **(extra or {})}})

        def _bad(self, e, anthropic=False, rid=None):
            """answers a BadRequest (400); a prompt longer than the context in llama-server's / Anthropic's shape"""
            if not isinstance(e, ContextOverflow):
                return self._error(400, str(e), anthropic, rid)
            extra = {"n_prompt_tokens": e.n_prompt, "n_ctx": e.n_ctx}
            if anthropic == "responses":  # (OpenAI's error code for it)
                return self._error(400, str(e), anthropic, rid, extra={**extra, "code": "context_length_exceeded"})
            if anthropic:
                return self._error(400, e.anthropic, True, rid, extra=extra)
            self._error(400, str(e), False, rid, "exceed_context_size_error", extra)

        # ---- routes without generation (llama-server's) ----
        def _tokenize(self, body):
            content = body.get("content")
            # llama-server's flags: parse_special [true], add_special [false; for the first element only, if a string]
            special = body.get("parse_special") is not False
            add = body.get("add_special") is True
            if isinstance(content, str):
                content = [content]
            if not isinstance(content, list) or not all(isinstance(c, (str, int)) and not isinstance(c, bool) for c in content):
                raise BadRequest("content: a string, or an array of strings and token ids")

            def tok(i, c):  # (ids of the tokenizer's own, or ids given in the content: those only in the vocabulary get a piece)
                if not isinstance(c, str):
                    return [c]
                first = add and i == 0
                return srv.eng.tokenize(c) if special and not first else srv.eng.tokenize(c, special, first)
            try:
                toks = [(t, isinstance(c, str)) for i, c in enumerate(content) for t in tok(i, c)]
            except ValueError as e:  # (a library without mk_tokenize2)
                raise BadRequest(str(e))
            if not body.get("with_pieces"):
                return {"tokens": [t for t, _ in toks]}
            out = []
            for t, own in toks:
                p = srv.eng.piece(t) if own or 0 <= t < srv.n_vocab() else b""
                try:
                    out.append({"id": t, "piece": p.decode()})
                except UnicodeDecodeError:
                    out.append({"id": t, "piece": list(p)})
            return {"tokens": out}

        def _detokenize(self, body):
            toks = body.get("tokens") or []
            nv = srv.n_vocab()
            if not nv:
                raise BadRequest("/detokenize needs a newer libqwen38mk (mk_n_vocab)")
            if not isinstance(toks, list) or not all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < nv for t in toks):
                raise BadRequest(f"tokens: an array of token ids in [0, {nv})")
            return {"content": b"".join(srv.eng.piece(t) for t in toks).decode(errors="replace")}

        def _apply_template(self, body):
            check_params(body)
            return {"prompt": srv.render(body)[0]}

        def _input_tokens(self, body):
            check_params(body)
            return {"input_tokens": srv.prompt_tokens(body), "object": "response.input_tokens"}

        def _count_tokens(self, body):
            return {"input_tokens": srv.prompt_tokens(self._anthropic_chat(body))}

        # ---- Anthropic Messages ----
        def _anthropic_chat(self, body):
            """anthropic_to_chat; what it skipped or converted is a one-time console warning per kind"""
            notes = []
            if getattr(srv, "mid_system", None) is None:  # (the template's answer, once)
                srv.mid_system = takes_mid_system(srv.template)
            chat = anthropic_to_chat(body, notes, srv.mid_system)
            for key, msg in notes:
                srv.warn_once(key, msg)
            return chat

        def _anthropic(self, body):
            chat = self._anthropic_chat(body)
            model = body.get("model") if isinstance(body.get("model"), str) and body.get("model") else srv.model_id
            if chat["stream"]:
                return self._anthropic_stream(chat, model)
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, _ = srv.generate(chat, lambda k, t: parts[k].append(t), self.client_gone)
            blocks = []
            reasoning = "".join(parts["reasoning"]).strip()
            if reasoning:
                blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
            if parts["content"]:
                blocks.append({"type": "text", "text": "".join(parts["content"])})
            blocks += [anthropic_tool_use(c) for c in calls]
            self._json(200, {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model,
                             "content": blocks, "stop_reason": anthropic_stop_reason(finish, srv.stop_word),
                             "stop_sequence": srv.stop_word, "usage": anthropic_usage(usage)})

        def _stream_head(self):
            """ends an event stream's headers: chunked, or for HTTP/1.0 (no chunked) a body that ends with the connection"""
            if self.request_version == "HTTP/1.0":
                self.send_header("Connection", "close")
                self.close_connection = True
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _stream_send(self, data, last=False):
            """one piece of an event stream's body (_stream_head's framing); last: then the end of the body"""
            if self.request_version == "HTTP/1.0":
                self.wfile.write(data)
            else:
                self.wfile.write(b"%x\r\n" % len(data) + data + (b"\r\n0\r\n\r\n" if last else b"\r\n"))
            self.wfile.flush()

        def _anthropic_stream(self, chat, model):
            """Anthropic's SSE: message_start, ping, content blocks (thinking / text deltas as they come, tool_use blocks
            with their input as input_json_delta pieces as it is generated), message_delta, message_stop"""
            mid = "msg_" + uuid.uuid4().hex
            st = {"started": False, "index": -1, "open": None, "ws": ""}  # ws: the thinking's held trailing whitespace
            send = self._stream_send

            def event(name, obj, last=False):
                send(f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode(), last)

            def start():
                if st["started"]:
                    return
                st["started"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._stream_head()
                n, reused = srv.last_prompt
                event("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": n - reused, "cache_creation_input_tokens": 0,
                                                     "cache_read_input_tokens": reused, "output_tokens": 0}}})
                event("ping", {"type": "ping"})

            def close():
                if st["open"]:
                    event("content_block_stop", {"type": "content_block_stop", "index": st["index"]})
                    st["open"] = None

            def block(cb):
                close()
                st["index"] += 1
                st["open"] = cb["type"]
                event("content_block_start", {"type": "content_block_start", "index": st["index"], "content_block": cb})

            def delta(d):
                event("content_block_delta", {"type": "content_block_delta", "index": st["index"], "delta": d})

            def emit(kind, text):
                try:
                    start()
                    if kind == "reasoning":  # (trimmed at both ends, as the non-streamed reply's thinking block)
                        t = st["ws"] + text
                        if st["open"] != "thinking":
                            t = t.lstrip()
                            if not t:
                                return True
                            block({"type": "thinking", "thinking": "", "signature": ""})
                        body = t.rstrip()
                        st["ws"] = t[len(body):]
                        if body:
                            delta({"type": "thinking_delta", "thinking": body})
                    elif kind == "tool_call":  # (ToolCallParser's events: a call's start, then its input's JSON text)
                        if "id" in text:
                            block({"type": "tool_use", "id": "toolu_" + text["id"].removeprefix("call_"),
                                   "name": text["function"]["name"], "input": {}})
                        if text["function"].get("arguments"):
                            delta({"type": "input_json_delta", "partial_json": text["function"]["arguments"]})
                    else:
                        if st["open"] != "text":
                            block({"type": "text", "text": ""})
                        delta({"type": "text_delta", "text": text})
                except DISCONNECTS:
                    return False
                return True

            check_params(chat)  # (parameter / render errors: a 400 before the stream starts)
            srv.render(chat)
            try:
                finish, calls, usage, _ = srv.generate(chat, emit, self.client_gone)
            except Exception as e:
                if not st["started"] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {e!r}")
                event("error", {"type": "error", "error": {"type": "invalid_request_error" if bad else "api_error",
                                                         "message": str(e)}}, last=True)
                return
            start()
            close()
            event("message_delta", {"type": "message_delta", "delta": {
                "stop_reason": anthropic_stop_reason(finish, srv.stop_word), "stop_sequence": srv.stop_word},
                "usage": anthropic_usage(usage)})
            event("message_stop", {"type": "message_stop"}, last=True)

        # ---- OpenAI Responses ----
        def _responses_chat(self, body):
            """responses_to_chat; what it dropped or converted is a one-time console warning per kind"""
            notes = []
            if getattr(srv, "mid_system", None) is None:  # (the template's answer, once)
                srv.mid_system = takes_mid_system(srv.template)
            chat, info = responses_to_chat(body, notes, srv.mid_system)
            for key, msg in notes:
                srv.warn_once(key, msg)
            return chat, info

        def _responses_input_tokens(self, body):
            return {"object": "response.input_tokens", "input_tokens": srv.prompt_tokens(self._responses_chat(body)[0])}

        def _responses(self, body):
            chat, info = self._responses_chat(body)
            rid, created = "resp_" + uuid.uuid4().hex, int(time.time())
            sealed = "reasoning.encrypted_content" in (body.get("include") or [])
            if chat["stream"]:
                return self._responses_stream(body, chat, info, rid, created, sealed)
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, _ = srv.generate(chat, lambda k, t: parts[k].append(t), self.client_gone)
            output = []
            reasoning = "".join(parts["reasoning"]).strip()
            if reasoning:
                output.append(responses_reasoning(reasoning, "rs_" + uuid.uuid4().hex, sealed))
            if parts["content"]:
                output.append(responses_message("".join(parts["content"]), "msg_" + uuid.uuid4().hex))
            output += [responses_call(c["id"], c["function"]["name"], c["function"]["arguments"], info) for c in calls]
            done = finish != "length"
            self._json(200, response_object(
                rid, created, body, srv.model_id, "completed" if done else "incomplete", output,
                responses_usage(usage, srv.reasoning_tokens), incomplete=None if done else {"reason": "max_output_tokens"}))

        def _responses_stream(self, body, chat, info, rid, created, sealed):
            """the Responses API's SSE: response.created and response.in_progress when the prefill starts (a request
            refused before that is an HTTP error), response.in_progress every RESP_KEEPALIVE_S during a long prefill (a
            client times out a silent stream: Codex after 300 s), then the output items as they are generated, each
            opened with output_item.added and closed with its done events: reasoning (reasoning_text deltas, trimmed at
            both ends as the non-streamed item), message (output_text deltas), function_call (its arguments' JSON
            text as it is generated) or custom_tool_call (its input whole at the end); then response.completed
            (response.incomplete after max_output_tokens) with the usage. An error once it has started is
            response.failed (code server_error, invalid_prompt for a bad request). Every event has its sequence_number."""
            st = {"started": False, "seq": 0, "items": [], "open": None, "ws": "", "t": 0.0}
            send = self._stream_send

            def event(obj, last=False):
                obj["sequence_number"] = st["seq"]
                st["seq"] += 1
                st["t"] = time.monotonic()
                send(f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode(), last)

            def response(status, **kw):
                return response_object(rid, created, body, srv.model_id, status, list(st["items"]), **kw)

            def start():
                if st["started"]:
                    return
                st["started"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._stream_head()
                event({"type": "response.created", "response": response("in_progress")})
                event({"type": "response.in_progress", "response": response("in_progress")})

            def progress(cached, processed, total):  # (Server.prompt_progress: the prefill's start, every 0.5 s, its end)
                try:
                    if not st["started"]:
                        start()
                    elif time.monotonic() - st["t"] >= RESP_KEEPALIVE_S:
                        event({"type": "response.in_progress", "response": response("in_progress")})
                except DISCONNECTS:  # (generate notices the disconnect)
                    pass

            def where():
                o = st["open"]
                return {"item_id": o["item"]["id"], "output_index": o["index"]}

            def open_item(item):
                close()
                st["open"] = {"index": len(st["items"]), "item": item, "text": ""}
                st["items"].append(item)
                event({"type": "response.output_item.added", "output_index": st["open"]["index"], "item": item})

            def close():
                o = st["open"]
                if o is None:
                    return
                item, text, at = o["item"], o["text"], where()
                if item["type"] == "reasoning":
                    event({"type": "response.reasoning_text.done", **at, "content_index": 0, "text": text})
                    event({"type": "response.content_part.done", **at, "content_index": 0,
                           "part": {"type": "reasoning_text", "text": text}})
                    done = responses_reasoning(text, item["id"], sealed)
                elif item["type"] == "message":
                    done = responses_message(text, item["id"])
                    event({"type": "response.output_text.done", **at, "content_index": 0, "text": text, "logprobs": []})
                    event({"type": "response.content_part.done", **at, "content_index": 0, "part": done["content"][0]})
                else:  # (a call's arguments are valid JSON only when it is complete: ToolCallParser)
                    try:
                        json.loads(text)
                        status = "completed"
                    except ValueError:
                        status = "incomplete"
                    done = responses_call(item["call_id"], item["name"], text, info, status)
                    if done["type"] == "custom_tool_call":
                        if done["input"]:
                            event({"type": "response.custom_tool_call_input.delta", **at, "delta": done["input"]})
                        event({"type": "response.custom_tool_call_input.done", **at, "input": done["input"]})
                    else:
                        event({"type": "response.function_call_arguments.done", **at, "name": item["name"],
                               "arguments": text})
                st["items"][o["index"]] = done
                st["open"] = None
                event({"type": "response.output_item.done", "output_index": o["index"], "item": done})

            def emit(kind, x):
                try:
                    start()
                    o = st["open"]
                    if kind == "reasoning":
                        t = st["ws"] + x
                        if not o or o["item"]["type"] != "reasoning":
                            t = t.lstrip()
                            if not t:
                                return True
                            open_item({"type": "reasoning", "id": "rs_" + uuid.uuid4().hex, "summary": [], "content": [],
                                       "encrypted_content": None, "status": "in_progress"})
                            event({"type": "response.content_part.added", **where(), "content_index": 0,
                                   "part": {"type": "reasoning_text", "text": ""}})
                        piece = t.rstrip()
                        st["ws"] = t[len(piece):]
                        if piece:
                            st["open"]["text"] += piece
                            event({"type": "response.reasoning_text.delta", **where(), "content_index": 0, "delta": piece})
                    elif kind == "tool_call":  # (ToolCallParser's events: a call's start, then its arguments' JSON text)
                        if "id" in x:
                            item = responses_call(x["id"], x["function"]["name"], "", info, "in_progress")
                            if item["type"] == "custom_tool_call":
                                item["input"] = ""
                            open_item(item)
                        frag = x["function"].get("arguments")
                        if frag:
                            st["open"]["text"] += frag
                            if st["open"]["item"]["type"] == "function_call":  # (a custom tool's input: at its end)
                                event({"type": "response.function_call_arguments.delta", **where(), "delta": frag})
                    else:
                        if not o or o["item"]["type"] != "message":
                            open_item(responses_message("", "msg_" + uuid.uuid4().hex, "in_progress") | {"content": []})
                            event({"type": "response.content_part.added", **where(), "content_index": 0,
                                   "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})
                        st["open"]["text"] += x
                        event({"type": "response.output_text.delta", **where(), "content_index": 0, "delta": x,
                               "logprobs": []})
                except DISCONNECTS:
                    return False
                return True

            check_params(chat)  # (parameter / render errors: a 400 before the stream starts)
            srv.render(chat)
            srv.on_progress = progress
            try:
                finish, calls, usage, _ = srv.generate(chat, emit, self.client_gone)
                start()
                close()
            except Exception as e:
                if not st["started"] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise  # nothing sent yet: do_POST answers with a status (400 / 500)
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {e!r}")
                event({"type": "response.failed", "response": response("failed", error={
                    "code": "invalid_prompt" if bad else "server_error", "message": str(e)})}, last=True)
                return
            finally:
                srv.on_progress = None
            usage = responses_usage(usage, srv.reasoning_tokens)
            if finish == "length":
                event({"type": "response.incomplete", "response": response(
                    "incomplete", usage=usage, incomplete={"reason": "max_output_tokens"})}, last=True)
            else:
                event({"type": "response.completed", "response": response("completed", usage=usage)}, last=True)

        def _complete(self, body):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(body, lambda k, t: parts[k].append(t), self.client_gone)
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})

        def _stream(self, body):
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            started = [False]
            send = self._stream_send

            def chunk(delta, finish=None, usage=None, timings=None, progress=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                if progress is not None:
                    obj["prompt_progress"] = progress
                send(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self._stream_head()
                    chunk({"role": "assistant", "content": None})

            per_token = bool(body.get("timings_per_token"))  # (llama-server's: the timings so far in every chunk)

            def emit(kind, text):
                head()
                try:
                    chunk({"tool_calls": [text]} if kind == "tool_call" else
                          {"reasoning_content": text} if kind == "reasoning" else {"content": text},
                          timings=srv.timings_now() if per_token else None)
                except DISCONNECTS:
                    return False
                return True

            t_p = []

            def progress(cached, processed, total):  # (return_progress: llama-server's prompt_progress chunks)
                t_p[:] = t_p or [time.monotonic()]
                try:
                    head()
                    chunk({"role": "assistant", "content": None}, progress={
                        "total": total, "cache": cached, "processed": processed,
                        "time_ms": int((time.monotonic() - t_p[0]) * 1e3)})
                except DISCONNECTS:  # (generate notices the disconnect)
                    pass

            # parameter / render errors are reported as a 400 before the stream starts
            check_params(body)
            srv.render(body)
            srv.on_progress = progress if body.get("return_progress") else None
            try:
                finish, calls, usage, timings = srv.generate(body, emit, self.client_gone)
            except Exception as e:
                if not started[0] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise  # nothing sent yet: do_POST answers with a status (400 / 500)
                # the status (200) and part of the reply are out: the error is the stream's last event, OpenAI style
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {e!r}")
                err = {"message": str(e), "type": "invalid_request_error" if bad else "server_error", "code": 400 if bad else 500}
                send(b"data: " + json.dumps({"error": err}, ensure_ascii=False).encode() + b"\n\n")
                send(b"data: [DONE]\n\n", last=True)
                return
            finally:
                srv.on_progress = None
            head()  # (the tool calls went out as they were generated)
            chunk({}, finish=finish, timings=timings)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            send(b"data: [DONE]\n\n", last=True)

    return H


def make_loading_handler():
    """the handler while the model loads (the port opens first, as llama-server's does): every request gets llama-server's
    503 "Loading model" and the connection closes (a request body is not read: the lingering close drains it); the gate
    and CORS preflights as in the real handler"""
    class Loading(GatedHandler):
        def _loading(self):
            data = json.dumps({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def handle_expect_100(self):
            if self.admit():
                self._loading()
            return False

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _loading

    return Loading


def linger(sock, seconds=None, idle=None, cap=None):
    """after shutdown(SHUT_WR): reads and drops what the client still sends until it closes, at most `cap` bytes and
    `seconds` (`idle` seconds without data). Closing a socket with unread data makes the kernel send a reset, which can
    reach the client before it has read the answer."""
    end = time.monotonic() + (LINGER_S if seconds is None else seconds)
    left = LINGER_BYTES if cap is None else cap
    idle = LINGER_IDLE_S if idle is None else idle
    buf = bytearray(1 << 16)
    try:
        while left > 0:
            t = end - time.monotonic()
            if t <= 0:
                break
            sock.settimeout(min(t, idle))
            n = sock.recv_into(buf, min(left, len(buf)))
            if not n:
                break
            left -= n
    except OSError:  # (timeout, reset)
        pass


class HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a lingering close: when the server ends a connection (an early 400 / 411 / 413 for a
    request body it does not read, the base class's errors, Connection: close), it half-closes it (the client reads the
    whole answer and the end of the stream), then drains what the client still sends (linger) before closing, so the
    client is not reset in the middle of its upload and gets the answer (the reason) instead of a connection error."""
    daemon_threads = True
    gate = None  # the request Gate (main sets it; None: every request is answered)
    # Windows: SO_REUSEADDR (the base class's allow_reuse_address) lets a second server bind a port that is in use with
    # no error, and requests then reach either one; an exclusive bind makes the second one fail at startup instead.
    # (Linux: SO_REUSEADDR only allows a restart over connections in TIME_WAIT; a port in use still fails to bind.)
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        """a client that resets its connection (an idle keep-alive one at the end of its session) is not an error; other
        exceptions are reported as the base class does, on the console"""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        log(f"error on the connection from {client_address}:\n{traceback.format_exc().rstrip()}")

    def shutdown_request(self, request):
        try:
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        else:
            linger(request)
        self.close_request(request)


def listen(host, port, handler):
    """the HTTPServer on host:port; a port that is taken (another server on it) ends the process with a message"""
    try:
        return HTTPServer((host, port), handler)
    except OSError as e:
        taken = (e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", None))
                 or getattr(e, "winerror", 0) in (10048, 10013))  # (Windows: in use / held exclusively)
        sys.exit(f"cannot listen on {host}:{port}: {e.strerror or e}" + (
            " (another server is using the port: choose another with --port, or the scripts' first argument)"
            if taken else ""))


def env_int(name, default, minimum=1):
    """an integer setting >= minimum from the environment (default when unset or empty)"""
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    if not v.isdecimal() or int(v) < minimum:
        sys.exit(f"{name}: expected {'a positive integer' if minimum == 1 else f'an integer >= {minimum}'}, not {v!r}")
    return int(v)


def gguf_text_limits(path):
    """(the GGUF's <arch>.context_length, the byte length of its longest token string), None where unreadable"""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # (GGUF scalar value types)
    ints = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    arch, ctxs, longest = None, {}, None
    try:
        with open(path, "rb") as f:
            def rd(fmt):
                b = f.read(struct.calcsize(fmt))
                if len(b) != struct.calcsize(fmt):
                    raise EOFError
                return struct.unpack(fmt, b)[0]

            def skip(vt, n=1):
                if vt == 8:  # (strings: u64 length + bytes)
                    for _ in range(n):
                        f.seek(rd("<Q"), 1)
                elif vt == 9:  # (arrays: element type, count, elements)
                    for _ in range(n):
                        et, m = rd("<I"), rd("<Q")
                        skip(et, m)
                else:
                    f.seek(sizes[vt] * n, 1)
            if f.read(4) != b"GGUF" or rd("<I") < 2:
                return None, None
            rd("<Q")  # (tensor count)
            for _ in range(rd("<Q")):
                key = f.read(rd("<Q")).decode("utf-8", "replace")
                vt = rd("<I")
                if key == "general.architecture" and vt == 8:
                    arch = f.read(rd("<Q")).decode("utf-8", "replace")
                elif key.endswith(".context_length") and vt in ints:
                    ctxs[key] = rd(ints[vt])
                elif key == "tokenizer.ggml.tokens" and vt == 9:
                    et, m = rd("<I"), rd("<Q")
                    if et != 8:
                        skip(et, m)
                        continue
                    longest = 0
                    for _ in range(m):  # (a token string's UTF-8 length >= its decoded piece's bytes)
                        n = rd("<Q")
                        f.seek(n, 1)
                        longest = max(longest, n)
                else:
                    skip(vt)
                if longest is not None and arch and f"{arch}.context_length" in ctxs:
                    break
    except (OSError, EOFError, KeyError, struct.error):
        pass
    return (ctxs.get(f"{arch}.context_length") if arch else None), longest


def main():
    global MAX_BODY_MB, MAX_IMAGES, MAX_VIDEO_FRAMES, IMAGE_MAX_PIXELS, REASONING_MAX_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="a host name clients may reach the server under, besides localhost, IP addresses and --host's "
                         "(repeatable or comma-separated; '*': any name, e.g. behind a reverse proxy, which turns the "
                         "DNS-rebinding check off; env MK_ALLOW_HOSTS); requests for other names get a 403")
    ap.add_argument("--cors-origin", "--cors-origins", dest="cors_origin", action="append", default=[], metavar="ORIGIN",
                    help="a web origin (scheme://host[:port]) whose pages may call the server: its requests get CORS "
                         "headers (repeatable or comma-separated; env MK_CORS_ORIGINS); requests from pages of other origins "
                         "get a 403 (default: none, only the server's own origin; clients that are not web pages send no "
                         "Origin and are not affected)")
    ap.add_argument("--api-key", default=os.environ.get("MK_API_KEY") or None, metavar="KEY",
                    help="require this key on every request but /health, as 'Authorization: Bearer KEY' or 'x-api-key: "
                         "KEY'; others get a 401 (env MK_API_KEY, which keeps the key out of the process list)")
    ap.add_argument("-c", "--ctx", type=int, default=None,
                    help="context positions (default 32768; with --agents: per slot, 0 = the most that fit, the default "
                         "there)")
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--mtp", default=None,
                    help="the MTP draft layer from another GGUF of the same model (default: the model file's; env MK_MTP)")
    ap.add_argument("--no-mtp", action="store_true", help="no MTP layer: plain decoding, depth 0 (env MK_NO_MTP=1)")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), LIB_MK))
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit; with --agents: q8_0)")
    ap.add_argument("--agents", type=int, default=env_int("MK_AGENTS", 0, 0), metavar="N",
                    help="serve up to N requests at once (e.g. several coding agents), each in a parallel slot of the "
                         "engine with its own context of --ctx positions, decoded together in one batched launch per "
                         "cycle; each reply is the one the request gets alone. Text only; no presence / frequency "
                         "penalties, side requests, parked conversations or pinned checkpoints (0: one request at a "
                         "time; env MK_AGENTS)")
    ap.add_argument("--parallel", "-np", type=int, default=0, metavar="N",
                    help="--agents N with -c the context of each slot, default 32768 (llama-server's and open-jet's option)")
    ap.add_argument("--ck-every", type=int, default=4096,
                    help="DeltaNet checkpoint every N prompt tokens (0: off); see MK_CK_SLOTS for the pool size")
    ap.add_argument("--ck-msgs", type=int, default=1, help="DeltaNet checkpoint at each message start (1/0)")
    ap.add_argument("--ck-pin", type=int, default=1,
                    help="pin the DeltaNet checkpoint at the end of a prompt's system message (and where a new session "
                         "left the cached one inside it) against the pool's eviction, and park it with the conversation: "
                         "new sessions of a client resume after its system prompt and tools (1/0)")
    ap.add_argument("--exact-reuse", action="store_true",
                    help="reused requests answer exactly as fresh ones (MK_EXACT=1): restore a checkpoint and prefill "
                         "instead of appending to decoded rows (costs prompt time when a long reply is reused)")
    ap.add_argument("--side-max", type=int, default=env_int("MK_SIDE_MAX", 8192, 0), metavar="N",
                    help="side requests: a first-turn prompt (no assistant reply in it) of at most N tokens that would discard "
                         "at least twice its length of the cached conversation (with parking on: one without tools, "
                         "such as a client's title or summary request, or one that asks for at most 1024 tokens; and "
                         "any request without tools beside a conversation whose last request had tools) runs without "
                         "evicting it: the KV rows it overwrites go to host memory and come back after its reply; "
                         "with parking on, its own conversation is parked too (evicted first), so a follow-up without "
                         "tools or with a capped reply continues it the same way (0: off; env MK_SIDE_MAX)")
    ap.add_argument("--park-mib", type=int, default=env_int("MK_PARK_MIB", 8192, 0), metavar="MIB",
                    help="host memory for parked conversations: a request for another conversation parks the cached one "
                         "(KV, recurrent state, checkpoints) and it comes back bit for bit when its next request arrives; "
                         "least recently parked ones go first (allocated on first use; 0: off; env MK_PARK_MIB)")
    ap.add_argument("--park-min", type=int, default=1024, metavar="N",
                    help="conversations of fewer tokens are not parked; a parked one comes back when its prompt reuses at "
                         "least N more tokens than the cached one's")
    ap.add_argument("--park-cks", type=int, default=2, metavar="N",
                    help="checkpoints kept with a parked conversation (the newest, besides its pinned ones: --ck-pin; with "
                         "exact reuse any checkpoint gives the same bits, fewer only mean more re-prefill after an edit)")
    ap.add_argument("--content-specials", choices=["text", "parse"],
                    default=os.environ.get("MK_CONTENT_SPECIALS") or "text",
                    help="special-token text (<|im_end|>, <|im_start|>, ...) inside messages, tool results, tool calls and "
                         "tool definitions: text (tokenized as plain text, so content such as a file an agent reads cannot "
                         "forge turns) or parse (as those special tokens, as llama-server does) (env MK_CONTENT_SPECIALS)")
    ap.add_argument("--chat-template-file", dest="chat_template", action=ChatTemplateArg, metavar="PATH",
                    help="Jinja chat template file to use instead of the GGUF's (as llama-server), e.g. another "
                         "template for a fine-tune of the same model")
    ap.add_argument("--chat-template", dest="chat_template", action=ChatTemplateArg, metavar="TEMPLATE",
                    help="the chat template itself (Jinja source; 'chatml': llama.cpp's ChatML) instead of the GGUF's; "
                         "the last of --chat-template / --chat-template-file wins")
    ap.add_argument("--reasoning-format", default=os.environ.get("MK_REASONING_FORMAT") or "auto", metavar="FORMAT",
                    help="llama-server's: auto / deepseek (the thinking as reasoning_content, streamed too: the default), "
                         "deepseek-legacy (also left in the content; a stream: only there), none (left in the content); "
                         "a request's reasoning_format wins; Anthropic replies keep their thinking blocks "
                         "(env MK_REASONING_FORMAT)")
    ap.add_argument("--chat-template-kwargs", default=os.environ.get("MK_CHAT_TEMPLATE_KWARGS") or None, metavar="JSON",
                    help="defaults for every request's chat_template_kwargs (a JSON object of "
                         + ", ".join(TEMPLATE_KWARGS) + "; the request's own win), e.g. '{\"enable_thinking\": false}' "
                         "(env MK_CHAT_TEMPLATE_KWARGS)")
    ap.add_argument("--mmproj", default=None,
                    help="vision projector GGUF (e.g. mmproj-*.gguf): image_url inputs, encoded on the CPU (no VRAM)")
    ap.add_argument("--vision-lib", default=os.path.join(os.path.dirname(HERE), LIB_VISION))
    ap.add_argument("--vision-threads", type=int, default=0,
                    help="CPU threads of the image encoder (0: every logical CPU but two)")
    ap.add_argument("--vision-backend", choices=["auto", "cpu", "mtmd"], default="auto",
                    help="image encoder: cpu (AVX2 / FMA fp32 kernels), mtmd (llama.cpp's), auto (cpu if the CPU has AVX2 + FMA)")
    ap.add_argument("--image-min-tokens", type=int, default=0, help="minimum tokens per image (0: the mmproj's default)")
    ap.add_argument("--image-max-tokens", type=int, default=0,
                    help="maximum tokens per image (0: the mmproj's default, 4096 for Qwen3.x; fewer encode faster)")
    ap.add_argument("--image-cache", type=int, default=8,
                    help="images kept decoded / encoded for reuse (a request holds all of its own while it runs)")
    ap.add_argument("--video-fps", type=float, default=4.0,
                    help="video input (input_video / video_url parts; --mmproj and ffmpeg): frames sampled per second "
                         "(llama.cpp's default 4); every two frames are one grid of tokens")
    ap.add_argument("--video-max-tokens", type=int, default=0,
                    help="maximum tokens per video frame (pair) (0: --image-max-tokens, as llama.cpp; fewer encode faster "
                         "and fit longer clips; cpu vision backend)")
    ap.add_argument("--video-cache", type=int, default=2,
                    help="videos kept decoded / encoded for reuse by the next requests of a conversation")
    ap.add_argument("--ffmpeg", default=None,
                    help="ffmpeg executable (or its directory) for video input (env MK_FFMPEG; default: ffmpeg on PATH, "
                         "else the imageio-ffmpeg package's)")
    args = ap.parse_args()
    if args.parallel < 0 or args.agents < 0:
        ap.error("--agents / --parallel: expected a number of slots >= 0")
    if args.ctx is None:  # (--agents: the most that fits in each slot; --parallel, as open-jet's: 32768 each)
        args.ctx = 0 if args.agents > 0 and not args.parallel else 32768
    if args.parallel:
        args.agents = args.parallel
    if args.agents > 0:
        if args.kv == "f16":
            ap.error("--agents: the parallel slots need the q8_0 KV cache (--kv q8 or auto)")
        if args.ctx < 0:
            ap.error("--ctx: expected positions per slot >= 0 (0: the most that fit)")
        args.kv = "q8"
        # prefill chunks of 1024 (half the scratch of the default: more context for the slots, same prefill speed), and
        # prefill rows independent of how a prompt is cut into chunks (MK_EXACT): a prompt fed beside other slots'
        # cycles computes what it computes alone
        os.environ.setdefault("MK_PF_CAP", "1024")
        os.environ.setdefault("MK_EXACT", "1")
        for k in ("MK_PF_MIX", "MK_PF_CHUNK", "MK_PF_SOLO"):  # (the slot scheduler's: checked before the model loads)
            env_int(k, 0, 0)
    if not 0 < args.video_fps <= 30:
        ap.error("--video-fps: expected a rate in (0, 30]")
    if args.video_max_tokens and args.video_max_tokens < max(8, args.image_min_tokens):
        ap.error("--video-max-tokens: expected 0 or at least 8 (and --image-min-tokens)")
    for k in ("side_max", "park_mib", "park_min", "park_cks"):
        if getattr(args, k) < 0:
            ap.error(f"--{k.replace('_', '-')}: expected a number >= 0")
    if args.content_specials not in ("text", "parse"):  # (an MK_CONTENT_SPECIALS default is not checked by choices)
        ap.error(f"MK_CONTENT_SPECIALS: expected text or parse, not {args.content_specials!r}")
    if args.reasoning_format not in REASONING_FORMATS:
        ap.error(f"--reasoning-format: one of {', '.join(REASONING_FORMATS)}, not {args.reasoning_format!r}")
    if args.chat_template_kwargs is not None:
        try:
            kw = json.loads(args.chat_template_kwargs)
        except ValueError as e:
            ap.error(f"--chat-template-kwargs: not JSON ({e})")
        if not isinstance(kw, dict) or set(kw) - set(TEMPLATE_KWARGS):
            ap.error(f"--chat-template-kwargs: a JSON object of {', '.join(TEMPLATE_KWARGS)} (the keys the server hands to "
                     "the template)")
        args.chat_template_kwargs = kw
        log(f"chat template kwargs: {json.dumps(kw)} (under each request's)")
    if args.reasoning_format != "auto":
        log(f"reasoning format: {args.reasoning_format}")
    args.chat_template_src = None  # (before the port opens: a bad template fails at once, not after the model load)
    if args.chat_template:
        try:
            args.chat_template_src = chat_template_source(*args.chat_template)
        except ValueError as e:
            ap.error(str(e))
        kind, value = args.chat_template
        log(f"chat template: {value if kind == 'file' else 'from --chat-template'} instead of the GGUF's")

    def listed(env, values):  # (the environment's comma-separated values, then the options')
        return [v.strip() for s in [os.environ.get(env, "")] + values for v in s.split(",") if v.strip()]
    try:
        gate = Gate(args.host, listed("MK_ALLOW_HOSTS", args.allow_host), listed("MK_CORS_ORIGINS", args.cors_origin),
                    args.api_key)
    except ValueError as e:
        ap.error(str(e))
    GatedHandler.timeout = env_int("MK_TIMEOUT_S", TIMEOUT_S, 0) or None  # (StreamRequestHandler applies it)
    MAX_BODY_MB = env_int("MK_MAX_BODY_MB", MAX_BODY_MB)
    MAX_IMAGES = env_int("MK_MAX_IMAGES", MAX_IMAGES)
    MAX_VIDEO_FRAMES = env_int("MK_MAX_VIDEO_FRAMES", MAX_VIDEO_FRAMES)
    n_train, longest = gguf_text_limits(args.model)
    REASONING_MAX_BYTES = max(n_train or 0, args.ctx) * (longest or 256)
    if args.mmproj:  # (the vision library checks images against it; the video decoder, frames)
        IMAGE_MAX_PIXELS = env_int("MK_IMAGE_MAX_PIXELS", IMAGE_MAX_PIXELS)
    if args.exact_reuse:
        os.environ["MK_EXACT"] = "1"  # read by the engine at construction
    # the port opens first (a busy port fails before the model loads); until the load is done every request gets a 503
    # (llama-server's "Loading model"), then the handler is swapped for the real one (new connections take it)
    httpd = listen(args.host, args.port, make_loading_handler())
    httpd.gate = gate
    log(f"listening on http://{args.host}:{args.port} (503 until the model is loaded)")
    if gate.any_host or gate.hosts != {"localhost"}:
        log("host names: any (--allow-host *)" if gate.any_host else f"host names: {', '.join(sorted(gate.hosts))} and IP addresses")
    if gate.origins:
        log(f"CORS: pages of {', '.join(sorted(gate.origins))} may call the server")
    if gate.api_key:
        log("API key: required on every request but /health")
    box = {}

    def load():
        try:
            srv = (SlotServer if args.agents > 0 else Server)(args)
            if srv.eng.exact and not args.agents:
                log("exact reuse: on (MK_EXACT)")
            box["srv"] = srv
            httpd.RequestHandlerClass = make_handler(srv)
        except BaseException as e:  # (the server exits: serve_forever returns)
            box["error"] = e
            traceback.print_exc()
            httpd.shutdown()

    threading.Thread(target=load, name="mk-load", daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CON.status("")
        if "srv" in box:
            box["srv"].eng.keep_warm(False)
        httpd.server_close()
    if "error" in box:
        sys.exit(f"[mk-server] startup failed: {box['error']}")


if __name__ == "__main__":
    main()
