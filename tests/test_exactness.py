"""Tests of scripts/exactness.py without a GPU: stub programs stand in for qwen38_mk and for the Python that runs
server/mk_server.py, and STUB_* variables make them break one claim at a time."""
import os
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "exactness.py")

# qwen38_mk: ids and a digest from the prompt (+ STUB_SALT); STUB_BREAK=d<n> / cap<n> changes them at that depth / chunk
STUB_CLI = r'''
import hashlib, os, re, shlex, sys

def run(args, out, err):
    prompt, depth, sampled, dig = "default", "2", False, False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-p": prompt = args[i + 1]; i += 1
        elif a == "-f": prompt = open(args[i + 1]).read(); i += 1
        elif a == "-d": depth = args[i + 1]; i += 1
        elif a == "--temp": sampled = True; i += 1
        elif a == "--digest": dig = True
        elif a in ("-n", "-o", "--top-p", "--top-k", "--seed"): i += 1
        i += 1
    brk = os.environ.get("STUB_BREAK", "") in ("d" + depth, "cap" + os.environ.get("MK_PF_CAP", "-"))
    h = hashlib.sha256((os.environ.get("STUB_SALT", "") + prompt + ("s" if sampled else "")).encode()).digest()
    ids = [str(b) for b in h[:12]]
    if brk: ids[5] = "99999"
    out.write("IN_IDS: 1 2 3\nIDS: " + " ".join(ids) + "\n")
    err.write("[mk] prompt %d tok in 0.1 s\n" % len(prompt.split()))
    if dig: err.write("[mk] state digest after the prefill: %s (9 words)\n" % (h[12:20].hex() if not brk else "0" * 16))

if os.environ.get("STUB_FAIL"):
    sys.stderr.write("[mk] error: stub failure\n")
    sys.exit(1)
argv = sys.argv[1:]
if "--batch" in argv:
    for line in open(argv[argv.index("--batch") + 1]):
        job = shlex.split(line)
        prefix = job[job.index("-o") + 1]
        with open(prefix + ".out", "w") as o, open(prefix + ".err", "w") as e:
            run(job, o, e)
else:
    run(argv, sys.stdout, sys.stderr)
'''

# the server's Python: answers /health and chat requests; STUB_BREAK=warm changes every reply after a server's first
STUB_SERVER = r'''
import hashlib, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
served = [0]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.send({"status": "ok"})
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        served[0] += 1
        key = os.environ.get("STUB_SALT", "") + json.dumps(json.loads(body), sort_keys=True)
        text = hashlib.sha256(key.encode()).hexdigest()[:24]
        if os.environ.get("STUB_BREAK") == "warm" and served[0] > 1: text += "!"
        self.send({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 7}})

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ExactnessScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.name == "nt":
            raise unittest.SkipTest("the stubs run as #! scripts")
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        cls.cli, cls.py, cls.model = (os.path.join(d, n) for n in ("qwen38_mk", "python", "model.gguf"))
        for path, src in ((cls.cli, STUB_CLI), (cls.py, STUB_SERVER)):
            with open(path, "w") as f:
                f.write(f"#!{sys.executable}\n{src}")
            os.chmod(path, 0o755)
        with open(cls.model, "wb") as f:
            f.write(b"GGUF stub")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_script(self, out, *args, **env):
        cmd = [sys.executable, SCRIPT, "-m", self.model, "--bin", self.cli, "--python", self.py, "--port", str(free_port()),
               "--out", os.path.join(self.tmp.name, out), "--timeout", "60", *args]
        env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")} | env
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        return r.returncode, r.stdout + r.stderr

    def test_every_check_passes_on_an_exact_build(self) -> None:
        rc, out = self.run_script("pass")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count("PASS "), 9, out)  # depth 2, mtp 2, chunks 2, server 3
        self.assertIn("9 passed, 0 failed", out)

    def test_a_depth_that_changes_the_ids_fails(self) -> None:
        rc, out = self.run_script("d4", "--legs", "depth,mtp", STUB_BREAK="d4")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL depth, code prompt: greedy ids identical at 0, 4, 7 drafts", out)
        self.assertIn("d4 vs d0: first difference at output token 5", out)
        self.assertIn("PASS mtp, code prompt", out)

    def test_a_chunk_size_that_changes_the_state_fails(self) -> None:
        rc, out = self.run_script("cap", "--legs", "chunks", STUB_BREAK="cap1280")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL chunks: MK_EXACT state digest", out)
        self.assertIn("chunk 1280: 0000000000000000", out)
        self.assertIn("FAIL chunks: greedy ids identical", out)

    def test_a_warm_server_that_answers_differently_fails(self) -> None:
        rc, out = self.run_script("warm", "--legs", "server", STUB_BREAK="warm")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL server, "), 3, out)

    def test_a_failing_cli_is_reported_with_its_error(self) -> None:
        rc, out = self.run_script("fail", "--legs", "depth", STUB_FAIL="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL depth_d0: qwen38_mk ended with 1", out)
        self.assertIn("[mk] error: stub failure", out)

    def test_compare_finds_another_build_that_differs(self) -> None:
        self.assertEqual(self.run_script("ref")[0], 0)
        ref = os.path.join(self.tmp.name, "ref")
        rc, out = self.run_script("same", "--compare", ref)
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS compare: 19 of 19 results identical", out)
        rc, out = self.run_script("other", "--compare", ref, STUB_SALT="x")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL compare: 0 of 19 results identical", out)

    def test_mtp_needs_the_depth_check(self) -> None:
        rc, out = self.run_script("bad", "--legs", "mtp")
        self.assertEqual(rc, 2, out)
        self.assertIn("mtp compares against the depth check's ids", out)


if __name__ == "__main__":
    unittest.main()
