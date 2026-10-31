#!/usr/bin/env python3
"""Checks the README's "Exact output" claims on this build, and with --compare that another build's results are identical."""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
LEGS = ("depth", "mtp", "chunks", "server")
PROMPTS = {"code": "Write a C++ function that parses a GGUF header.",
           "story": "Tell me a short story about a lighthouse keeper."}
SAMPLED = ["--temp", "1", "--top-p", "0.95", "--top-k", "20", "--seed", "7"]
CHUNKS = (2048, 1280, 768)  # MK_PF_CAP values: multiples of 128, all >= 640 (smaller caps take another end-snapshot path)
U1 = "Name three uses of a lighthouse, one line each."
U2 = "Now write a four-line poem about the second one."
U2_EDIT = "Now write a four-line poem about the third one."

DESCRIPTION = """\
Checks the exactness claims of README.md ("Exact output") on this build:
  depth   greedy token ids are identical at every draft depth of --depths (two prompts)
  mtp     greedy ids without the MTP layer (--no-mtp, depth 0) equal the drafted ones
  chunks  with exact reuse (MK_EXACT=1) the state digest and greedy ids after a ~5k-token prompt do not depend on the
          prefill chunk size (MK_PF_CAP)
  server  with --exact-reuse, a follow-up turn, an edited turn and a sampled follow-up (fixed seed) on a warm server
          equal the same request sent to a fresh server
Every result goes to <out>/exactness.json. --compare DIR then checks that another build's results (Windows against Linux,
or the previous release) are identical. Needs the GPU to itself: one CLI process per depth, one without the MTP layer,
one per chunk size, and four server starts."""


class LegFailed(Exception):
    def __init__(self, what, detail=""):
        super().__init__(what)
        self.what, self.detail = what, detail


def long_prompt():
    lines = ["Explain what each of these functions computes.", "", "```python"]
    for i in range(1, 241):
        lines += [f"def step_{i}(x, y):", f"    return (x * {i} + y) % {997 + i}", ""]
    return "\n".join(lines + ["```"])


def quoted(s):
    """one argument of a --batch line: '...' is taken literally there (main refuses an --out with a quote)"""
    assert "'" not in s, s
    return f"'{s}'"


def tail(path, n=8):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).rstrip() or "(empty)"
    except OSError:
        return "(no output)"


def first_difference(a, b):
    if a is None or b is None:
        return "no ids"
    x, y = a.split()[1:], b.split()[1:]  # (after "IDS:")
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return f"first difference at output token {i}"
    return "same" if len(x) == len(y) else f"lengths {len(x)} / {len(y)}"


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_info():
    info = os.path.join(HERE, "BUILD_INFO.txt")  # (release packages)
    if os.path.isfile(info):
        with open(info, encoding="utf-8", errors="replace") as f:
            return f.readline().strip()
    try:
        r = subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def user(text):
    return {"role": "user", "content": text}


class Server:
    """server/mk_server.py with exact reuse on --port; stops it (and waits for its exit) when the block ends"""

    def __init__(self, chk, name):
        self.chk, self.name = chk, name
        self.url = f"http://127.0.0.1:{chk.a.port}"
        self.log = chk.path(name, "log")

    def __enter__(self):
        a = self.chk.a
        if port_open(a.port):
            raise LegFailed(f"{self.name}: port {a.port} is in use (--port)")
        cmd = [a.python, os.path.join(HERE, "server", "mk_server.py"), "-m", a.model, "--port", str(a.port), "-c", "32768",
               "--kv", "q8", "-d", str(max(a.depths)), "--exact-reuse"]
        if a.draft_vocab:
            cmd += ["--draft-vocab", a.draft_vocab]
        env = {k: v for k, v in os.environ.items() if k != "MK_API_KEY"}  # (its own server: the requests carry no key)
        self.logf = open(self.log, "wb")
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                                      env=dict(env, MK_KEEP_WARM_S="0"))
        except OSError as e:
            self.logf.close()
            raise LegFailed(f"{self.name}: cannot start {a.python} ({e})")
        end = time.monotonic() + a.timeout
        while time.monotonic() < end:
            if self.p.poll() is not None:
                self.logf.close()
                raise LegFailed(f"{self.name}: the server exited with code {self.p.returncode}", tail(self.log))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except OSError:  # (refused while it starts; 503 while the model loads)
                pass
            time.sleep(0.5)
        self.stop()
        raise LegFailed(f"{self.name}: not ready after {a.timeout} s", tail(self.log))

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.logf.close()

    def chat(self, messages, opts):
        body = dict(messages=messages, max_tokens=128, stream=False, chat_template_kwargs={"enable_thinking": False}, **opts)
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.chk.a.timeout) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            raise LegFailed(f"{self.name}: HTTP {e.code}", e.read()[:500].decode(errors="replace"))
        except (OSError, ValueError) as e:
            raise LegFailed(f"{self.name}: the request failed ({e})", tail(self.log))
        c = d["choices"][0]
        return {"content": c["message"].get("content"), "reasoning": c["message"].get("reasoning_content"),
                "finish": c.get("finish_reason"), "tokens": (d.get("usage") or {}).get("completion_tokens")}


class Check:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        self.rec = {}
        self.passed = self.failed = 0
        os.makedirs(self.out, exist_ok=True)

    def path(self, name, ext):
        return os.path.join(self.out, f"{name}.{ext}")

    def verdict(self, ok, what, detail=""):
        print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok and detail:
            print("     " + detail.replace("\n", "\n     "), flush=True)
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def cli(self, name, args, env=None):
        """one qwen38_mk process, stdout / stderr to <name>.out / .err"""
        a = self.a
        cmd = [a.bin, "-m", a.model, "--kv", "q8"] + (["--draft-vocab", a.draft_vocab] if a.draft_vocab else []) + args
        with open(self.path(name, "out"), "wb") as o, open(self.path(name, "err"), "wb") as e:
            try:
                rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=o, stderr=e, timeout=a.timeout,
                                    env=dict(os.environ, **(env or {}))).returncode
            except subprocess.TimeoutExpired:
                rc = "a timeout"
        if rc != 0:
            raise LegFailed(f"{name}: qwen38_mk ended with {rc}", tail(self.path(name, "err")))

    def batch(self, name, jobs, extra=()):
        """jobs [(output name, job arguments)] in one process (one model load)"""
        lines = [" ".join(quoted(x) for x in args + ["-n", "256", "--ids", "--quiet", "-o", os.path.join(self.out, job)])
                 for job, args in jobs]
        with open(self.path(name, "jobs"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        self.cli(name, ["-c", "8192", "--batch", self.path(name, "jobs")] + list(extra))

    def ids(self, name):
        try:
            with open(self.path(name, "out"), encoding="utf-8", errors="replace") as f:
                return next((ln.strip() for ln in f if ln.startswith("IDS:")), None)
        except OSError:
            return None

    def err_match(self, name, pattern):
        m = re.search(pattern, tail(self.path(name, "err"), 200))
        return m.group(1) if m else None

    def same(self, what, got):
        """a verdict on whether the labelled results in got are all present and identical"""
        vals = list(got.values())
        ok = None not in vals and len(set(vals)) == 1
        ref = next(iter(got))
        if all(v is None or v.startswith("IDS:") for v in vals):
            detail = "; ".join(f"{k} vs {ref}: {first_difference(got[ref], v)}" for k, v in got.items() if k != ref)
        else:
            detail = "; ".join(f"{k}: {v}" for k, v in got.items())
        self.verdict(ok, what, detail)

    def depth(self):
        d = self.a.depths
        for n in d:  # (a process per depth, as a user runs it: a --batch engine is built for its deepest job)
            jobs = [(f"depth_{p}_d{n}", ["-p", t, "-d", str(n)]) for p, t in PROMPTS.items()]
            if n == max(d):
                jobs.append(("sampled_story", ["-p", PROMPTS["story"], "-d", str(n)] + SAMPLED))
            self.batch(f"depth_d{n}", jobs)
        for p in PROMPTS:
            got = {f"d{n}": self.ids(f"depth_{p}_d{n}") for n in d}
            for k, v in got.items():
                self.rec[f"cli/{p}/{k}"] = v
            self.same(f"depth, {p} prompt: greedy ids identical at {', '.join(map(str, d))} drafts", got)
        self.rec["cli/story/sampled"] = self.ids("sampled_story")  # (--compare only: across depths only the distribution is kept)

    def mtp(self):
        top = f"d{max(self.a.depths)}"
        self.batch("nomtp", [(f"nomtp_{p}", ["-p", t, "-d", "0"]) for p, t in PROMPTS.items()], ["--no-mtp"])
        for p in PROMPTS:
            got = {"no MTP layer": self.ids(f"nomtp_{p}"), top: self.rec.get(f"cli/{p}/{top}")}
            self.rec[f"cli/{p}/nomtp"] = got["no MTP layer"]
            self.same(f"mtp, {p} prompt: greedy ids without the MTP layer equal those at {top[1:]} drafts", got)

    def chunks(self):
        prompt = self.path("chunks_prompt", "txt")
        with open(prompt, "w", encoding="utf-8", newline="\n") as f:
            f.write(long_prompt())
        args = ["-c", "16384", "-f", prompt, "-n", "64", "-d", str(max(self.a.depths)), "--digest", "--ids", "--quiet"]
        for cap in CHUNKS:
            self.cli(f"chunks_{cap}", args, {"MK_EXACT": "1", "MK_PF_CAP": str(cap)})
            self.rec[f"chunks/{cap}/digest"] = self.err_match(f"chunks_{cap}", r"state digest after the prefill: ([0-9a-f]+)")
            self.rec[f"chunks/{cap}/ids"] = self.ids(f"chunks_{cap}")
        n = self.err_match(f"chunks_{CHUNKS[0]}", r"prompt (\d+) tok") or "?"
        caps = ", ".join(map(str, CHUNKS))
        self.same(f"chunks: MK_EXACT state digest after a {n}-token prompt identical at prefill chunks of {caps}",
                  {f"chunk {c}": self.rec[f"chunks/{c}/digest"] for c in CHUNKS})
        self.same(f"chunks: greedy ids identical at prefill chunks of {caps}", {f"chunk {c}": self.rec[f"chunks/{c}/ids"] for c in CHUNKS})

    def server(self):
        greedy, sampled = {"temperature": 0}, {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}
        with Server(self, "server_warm") as s:
            r1 = s.chat([user(U1)], greedy)
            hist = [user(U1), {"role": "assistant", "content": r1["content"] or ""}]
            cases = {"follow-up": (hist + [user(U2)], greedy), "edit": (hist + [user(U2_EDIT)], greedy),
                     "sampled follow-up": (hist + [user(U2)], sampled)}
            warm = {k: s.chat(m, o) for k, (m, o) in cases.items()}
        self.rec["server/first turn"] = r1
        for k, (m, o) in cases.items():
            with Server(self, "server_fresh_" + k.replace(" ", "_")) as s:
                fresh = s.chat(m, o)
            self.rec[f"server/{k}"] = fresh
            self.verdict(fresh == warm[k], f"server, {k}: the warm server's reply (exact reuse) equals a fresh server's",
                         f"warm:  {warm[k]!r}\nfresh: {fresh!r}")

    def save(self):
        a = self.a
        meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "build": build_info(), "bin_sha256": sha256(a.bin),
                "model": os.path.basename(a.model), "model_bytes": os.path.getsize(a.model), "depths": a.depths,
                "legs": a.legs, "platform": sys.platform,
                "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MK_") and k != "MK_API_KEY"}}
        with open(self.path("exactness", "json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"meta": meta, "results": self.rec}, f, indent=1)
        return meta

    def compare(self, meta, other):
        path = os.path.join(other, "exactness.json") if os.path.isdir(other) else other
        with open(path, encoding="utf-8") as f:
            o = json.load(f)
        om, theirs = o["meta"], o["results"]
        if (om["model"], om["model_bytes"]) != (meta["model"], meta["model_bytes"]):
            self.verdict(False, f"compare: {path} ran another model file ({om['model']}, {om['model_bytes']} bytes)")
            return
        common = sorted(k for k in set(self.rec) & set(theirs) if self.rec[k] is not None and theirs[k] is not None)
        differ = [k for k in common if self.rec[k] != theirs[k]]
        self.verdict(bool(common) and not differ,
                     f"compare: {len(common) - len(differ)} of {len(common)} results identical to {path} "
                     f"({om['build']}, {om['platform']}, {om['time']})", "differ: " + ", ".join(differ))
        only = sorted(set(self.rec) ^ set(theirs))
        if only:
            print(f"     (in one run only, not compared: {', '.join(only)})")
        if om.get("env") != meta["env"]:
            print(f"     (the runs' MK_* settings differ: {om.get('env')} there, {meta['env']} here)")


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    dv = os.path.join(HERE, "draft_vocab_32k.bin")
    ap.add_argument("-m", "--model", default=os.environ.get("MK_MODEL"), help="the GGUF (default: MK_MODEL)")
    ap.add_argument("--bin", default=os.path.join(HERE, "qwen38_mk.exe" if WIN else "qwen38_mk"), help="the CLI")
    ap.add_argument("--draft-vocab", default=dv if os.path.isfile(dv) else "", help="(default: draft_vocab_32k.bin)")
    ap.add_argument("--depths", default="0,4,7", help="draft depths (default 0,4,7; a `make MKFLAGS=` build: 0,4)")
    ap.add_argument("--legs", default=",".join(LEGS), help=f"the checks to run (default {','.join(LEGS)})")
    ap.add_argument("--out", default="exactness-out", help="output directory (default exactness-out)")
    ap.add_argument("--compare", metavar="DIR", help="another run's output directory (or its exactness.json)")
    ap.add_argument("--port", type=int, default=18190, help="the server checks' port (default 18190)")
    ap.add_argument("--python", default=sys.executable, help="Python for the server, with jinja2 (default: this one)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per CLI run, server start or request")
    a = ap.parse_args()
    if not a.model or not os.path.isfile(a.model):
        ap.error(f"no model file {a.model!r} (-m or MK_MODEL)")
    if not os.path.isfile(a.bin):
        ap.error(f"no {a.bin}: build it (make / build.ps1) or pass --bin")
    if "'" in os.path.abspath(a.out):
        ap.error("--out: a path without quotes (the CLI's --batch lines cannot hold one)")
    a.model, a.bin = os.path.abspath(a.model), os.path.abspath(a.bin)
    a.draft_vocab = os.path.abspath(a.draft_vocab) if a.draft_vocab else ""
    try:
        a.depths = sorted({int(x) for x in a.depths.split(",")})
    except ValueError:
        ap.error("--depths: comma-separated integers")
    a.legs = [x for x in a.legs.split(",") if x]
    if set(a.legs) - set(LEGS):
        ap.error(f"--legs: some of {', '.join(LEGS)}")
    if "mtp" in a.legs and "depth" not in a.legs:
        ap.error("--legs: mtp compares against the depth check's ids, so it needs depth too")
    a.legs = [x for x in LEGS if x in a.legs]
    chk = Check(a)
    t0 = time.monotonic()
    for leg in a.legs:
        print(f"== {leg} ({time.monotonic() - t0:.0f} s)", flush=True)
        try:
            getattr(chk, leg)()
        except LegFailed as e:
            chk.verdict(False, e.what, e.detail)
    meta = chk.save()
    if a.compare:
        chk.compare(meta, a.compare)
    print(f"exactness: {chk.passed} passed, {chk.failed} failed in {time.monotonic() - t0:.0f} s "
          f"({meta['build']}; {chk.path('exactness', 'json')})")
    return 1 if chk.failed else 0


if __name__ == "__main__":
    sys.exit(main())
