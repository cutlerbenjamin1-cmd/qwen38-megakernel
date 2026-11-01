#!/usr/bin/env bash
# The GPU checks of a build in one command (Linux / WSL2; the GPU to itself, ~30 min):
#   scripts/gate.sh -m <gguf> --record DIR   run every check, keep the outputs in DIR as a reference
#   scripts/gate.sh -m <gguf> --ref DIR      run every check, and every output must equal DIR's
#   scripts/gate.sh -m <gguf>                run every check against the build itself only
# Each check prints ok / FAIL / skip; the exit status is 1 if any failed (2: bad arguments or a stale build). A reference
# is kept only if all its checks passed (its MANIFEST says PASS), and it holds for the GPU, driver and model files it was
# recorded with: the token ids of a sampled run or a KL report are not portable to another GPU.
# Options: --out DIR (default gate-out; --record writes to its DIR), --legs LIST (default: all, see LEGS), --model2 <gguf>
# (a second model: one more greedy run and the 4-slot checks), --kld-ref FILE (tools/llama_ref's reference: the kld leg),
# --capi FILE (default scripts/capi_checks.py) and --capi-cases "CASE ..." (default: the script's own), --python PY (with
# jinja2 for the server checks; default python3), --port N (the server checks, default 18190), --timeout S (per run).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGS_ALL="benches exactness pfcap cli tinv slots capi kld"
MODEL=${MK_MODEL:-}; MODEL2=; REF=; REC=; O=gate-out; LEGS=$LEGS_ALL; KLD_REF=; PY=python3; PORT=18190; TO=1800
CAPI=$REPO/scripts/capi_checks.py; CAPI_CASES=
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
while [ $# -gt 0 ]; do
    case $1 in
        -m|--model) MODEL=$2; shift 2 ;;
        --model2) MODEL2=$2; shift 2 ;;
        --record) REC=$2; shift 2 ;;
        --ref) REF=$2; shift 2 ;;
        --out) O=$2; shift 2 ;;
        --legs) LEGS=${2//,/ }; shift 2 ;;
        --kld-ref) KLD_REF=$2; shift 2 ;;
        --capi) CAPI=$2; shift 2 ;;
        --capi-cases) CAPI_CASES=$2; shift 2 ;;
        --python) PY=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --timeout) TO=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option $1"; usage ;;
    esac
done
die() { echo "gate: $*" >&2; exit 2; }
[ -n "$MODEL" ] && [ -f "$MODEL" ] || die "no model file (-m <gguf> or MK_MODEL)"
[ -z "$MODEL2" ] || [ -f "$MODEL2" ] || die "no file $MODEL2"
[ -z "$KLD_REF" ] || [ -f "$KLD_REF" ] || die "no file $KLD_REF"
[ -n "$REC" ] && [ -n "$REF" ] && die "--record and --ref exclude each other"
for l in $LEGS; do [[ " $LEGS_ALL " == *" $l "* ]] || die "unknown leg $l (legs: $LEGS_ALL)"; done
BIN=$REPO/qwen38_mk; LIB=$REPO/libqwen38mk.so; DV=$REPO/draft_vocab_32k.bin
[ -x "$BIN" ] && [ -f "$LIB" ] || die "build qwen38_mk and libqwen38mk.so first (make)"
stale=$(find "$REPO" -maxdepth 1 \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' \) \( -newer "$BIN" -o -newer "$LIB" \) | head -3)
[ -z "$stale" ] || die "the binaries are older than $(echo $stale | xargs -n1 basename | tr '\n' ' ')- run make"
if [ -n "$REC" ]; then
    O=$REC
    [ ! -e "$O" ] || [ -z "$(ls -A "$O")" ] || die "$O is not empty: a reference is recorded into a new directory"
fi
if [ -n "$REF" ]; then
    REF=$(cd "$REF" 2>/dev/null && pwd) || die "no reference directory"
    grep -q '^status: PASS' "$REF/MANIFEST" 2>/dev/null || die "$REF/MANIFEST is missing or not PASS: not a usable reference"
fi
mkdir -p "$O" && O=$(cd "$O" && pwd)
[ "$O" != "$REF" ] || die "--out is the reference directory"
for l in $LEGS_ALL; do rm -rf "${O:?}/$l"; done
rm -f "$O/gate.log" "$O/MANIFEST.run"
exec > >(tee "$O/gate.log") 2>&1

# a model file's identity without reading 17 GB: size and the SHA-256 of its first and last 16 MiB
fid() { echo "$(basename "$1") $(stat -c %s "$1") $({ head -c 16M "$1"; tail -c 16M "$1"; } | sha256sum | cut -c1-16)"; }
gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
{
    echo "head: $(git -C "$REPO" describe --always --dirty 2>/dev/null) ($(git -C "$REPO" branch --show-current 2>/dev/null))"
    echo "build: $(grep -ho '"[^"]*"' "$REPO/build/mk_build_info.h" 2>/dev/null | tr '\n' ' ')"
    echo "gpu: $gpu"
    echo "model: $(fid "$MODEL")"
    [ -z "$MODEL2" ] || echo "model2: $(fid "$MODEL2")"
    [ -z "$KLD_REF" ] || echo "kld_ref: $(fid "$KLD_REF")"
    echo "legs: $LEGS"
    echo "date: $(date -Is)"
} > "$O/MANIFEST.run"
echo "== gate $(date +%T): $REPO -> $O"
sed 's/^/   /' "$O/MANIFEST.run"
if [ -n "$REF" ]; then
    echo "== reference $REF"
    sed 's/^/   /' "$REF/MANIFEST"
    for k in model model2 kld_ref; do
        a=$(grep "^$k:" "$O/MANIFEST.run"); b=$(grep "^$k:" "$REF/MANIFEST")
        [ -z "$a" ] || [ -z "$b" ] || [ "$a" = "$b" ] || die "the reference was recorded with another $k (${b#*: })"
    done
    [ "$(grep '^gpu:' "$REF/MANIFEST")" = "gpu: $gpu" ] || echo "   NOTE: recorded on another GPU / driver; a difference there may be the machine's"
fi

NOK=0; NFAIL=0; NSKIP=0; FAILED=
ok() { echo "ok   $*"; NOK=$((NOK + 1)); }
fail() { echo "FAIL $*"; NFAIL=$((NFAIL + 1)); FAILED="$FAILED ${1%%:*}"; }
skip() { echo "skip $*"; NSKIP=$((NSKIP + 1)); }
# ref <name> <file>: with --ref, the file must equal the reference's file of the same path (if that leg was recorded)
ref() {
    [ -n "$REF" ] || return 0
    local rel=${2#"$O"/} leg=${2#"$O"/}; leg=${leg%%/*}
    grep -q "^legs:.*\b$leg\b" "$REF/MANIFEST" || { skip "$1 reference: leg $leg not recorded"; return 0; }
    if [ ! -f "$REF/$rel" ]; then fail "$1: no $rel in the reference"
    elif cmp -s "$2" "$REF/$rel"; then ok "$1 == reference"
    else fail "$1: differs from the reference ($rel)"; fi
}
run() { timeout -s KILL "$TO" "$@" < /dev/null; }

# kernel microbenchmarks against the engine's code, no model (Makefile `check`)
leg_benches() {
    make -C "$REPO" check > "$O/benches/make_check.log" 2>&1
    local rc=$?
    grep -E 'differ|FAIL|T-invariance|bad' "$O/benches/make_check.log" | sed 's/^/   /' | head -20
    [ $rc = 0 ] && ok "benches: make check" || fail "benches: make check exit $rc (benches/make_check.log)"
}

# README's exact-output claims (scripts/exactness.py), and with --ref every result equal to the reference's run
leg_exactness() {
    local a=(-m "$MODEL" --out "$O/exactness/run" --port "$PORT")
    [ -z "$REF" ] || ! grep -q '^legs:.*\bexactness\b' "$REF/MANIFEST" || a+=(--compare "$REF/exactness/run")
    "$PY" "$REPO/scripts/exactness.py" "${a[@]}" > "$O/exactness/exactness.log" 2>&1 < /dev/null
    local rc=$?
    grep -E '^(PASS|FAIL)' "$O/exactness/exactness.log" | sed 's/^/   /'
    [ $rc = 0 ] && ok "exactness: $(grep -c '^PASS' "$O/exactness/exactness.log") checks" || fail "exactness: exit $rc (exactness/exactness.log)"
}

# MK_EXACT=1 with a prefill cap below two staging slots (256) against the default cap: the same state digest and ids
leg_pfcap() {
    local d=$O/pfcap c
    head -c 9000 "$REPO/bench/suite/prompts/pf38k.txt" | head -n -1 > "$d/prompt.txt" # (whole lines)
    for c in default 256; do
        if [ $c = default ]; then e=(); else e=(MK_PF_CAP=$c); fi
        env "${e[@]}" MK_EXACT=1 timeout -s KILL "$TO" "$BIN" -m "$MODEL" -c 8192 -f "$d/prompt.txt" --raw -n 32 --ids --digest \
            > "$d/cap_$c.out" 2> "$d/cap_$c.err" < /dev/null || { fail "pfcap: cap $c exit $? (pfcap/cap_$c.err)"; return; }
        grep -h 'state digest' "$d/cap_$c.err" > "$d/cap_$c.digest"
    done
    if [ -s "$d/cap_default.digest" ] && cmp -s "$d/cap_default.digest" "$d/cap_256.digest" && cmp -s "$d/cap_default.out" "$d/cap_256.out"; then
        ok "pfcap: MK_EXACT state digest and ids at prefill cap 256 == the default cap ($(grep -o '[0-9a-f]\{16\}' "$d/cap_default.digest"))"
    else
        fail "pfcap: cap 256 differs from the default cap (or no digest)"
    fi
    ref "pfcap ids" "$d/cap_default.out"
    ref "pfcap digest" "$d/cap_default.digest"
}

# the CLI: sampled runs at 0 / 7 drafts, presence penalty, a draft-chain policy, greedy on --model2 (reference checks),
# and a context past the model's refused
leg_cli() {
    local d=$O/cli k
    local X=("$BIN" --draft-vocab "$DV" --ids --quiet --kv q8 -n 256)
    local S=(--temp 1 --top-p 0.95 --top-k 20 --seed 7 -p "Tell me a short story about a lighthouse keeper.")
    one() { # name, args...
        local k=$1; shift
        run "${X[@]}" "$@" > "$d/$k.out" 2> "$d/$k.err"
        local rc=$?
        if [ $rc != 0 ] || ! grep -q '^IDS:' "$d/$k.out"; then fail "cli $k: exit $rc / no ids (cli/$k.err)"; return; fi
        ok "cli $k: $(grep -ho 'decode .*' "$d/$k.err" | tail -1)"
        ref "cli $k" "$d/$k.out"
    }
    one smp_d0 -m "$MODEL" -d 0 "${S[@]}"
    one smp_d7 -m "$MODEL" -d 7 "${S[@]}"
    one pres_d7 -m "$MODEL" -d 7 "${S[@]}" --presence-penalty 1.0
    one chain2_d7 -m "$MODEL" -d 7 "${S[@]}" --policy chain=2
    if [ -n "$MODEL2" ]; then one m2_d7 -m "$MODEL2" -d 7; else skip "cli m2_d7: no --model2"; fi
    run "${X[@]}" -m "$MODEL" -c 262145 -n 8 -p hi > "$d/cap.out" 2> "$d/cap.err"
    if grep -q 'context length' "$d/cap.err" && ! [ -s "$d/cap.out" ]; then ok "cli: -c 262145 refused"; else fail "cli: -c 262145 not refused (cli/cap.err)"; fi
}

# greedy ids of the four bench prompts at 0, 4 and 7 drafts on the q8 cache (one process, --batch): identical per prompt
leg_tinv() {
    local d=$O/tinv i k
    local -a P
    mapfile -t P < <(grep -v '^\s*$' "$REPO/bench/prompts.txt")
    : > "$d/jobs.txt"
    for i in 0 1 2 3; do
        printf '%s' "${P[$i]}" > "$d/prompt_p$i.txt"
        for k in 0 4 7; do echo "-f '$d/prompt_p$i.txt' -d $k -o '$d/p${i}_d$k'" >> "$d/jobs.txt"; done
    done
    run "$BIN" -m "$MODEL" -n 512 --kv q8 -c 8192 --draft-vocab "$DV" --ids --quiet --batch "$d/jobs.txt" > "$d/batch.out" 2> "$d/batch.err" \
        || { fail "tinv: exit $? (tinv/batch.err)"; return; }
    for i in 0 1 2 3; do
        for k in 0 4 7; do grep '^IDS:' "$d/p${i}_d$k.out" > "$d/p${i}_d$k.ids" 2>/dev/null; done
        if [ -s "$d/p${i}_d0.ids" ] && cmp -s "$d/p${i}_d0.ids" "$d/p${i}_d4.ids" && cmp -s "$d/p${i}_d0.ids" "$d/p${i}_d7.ids"; then
            ok "tinv p$i: greedy ids identical at 0, 4, 7 drafts ($(grep -ho 'decode.*' "$d/p${i}_d7.err" | tail -1))"
        else
            fail "tinv p$i: ids differ between draft depths (or none)"
        fi
        ref "tinv p$i" "$d/p${i}_d0.ids"
    done
}

# parallel slots over the C API (bench/slots_bench.py): batched == solo and prefix reuse at 4 and 2 slots, with the MTP
# window, a slot prefilled between the others' steps; sampled 4 / 2-slot outputs (reference checks)
leg_slots() {
    local d=$O/slots m
    sb() { # name, "VAR=value ..." or "", then slots_bench.py's arguments
        local k=$1 e=$2
        shift 2
        # shellcheck disable=SC2086
        env $e timeout -s KILL "$TO" "$PY" "$REPO/bench/slots_bench.py" "$@" --lib "$LIB" > "$d/$k.out" 2>&1 < /dev/null
    }
    same_as_solo() { # name, env, model, slot count or "stagger"
        local mode=exact n=$4
        [ "$4" = stagger ] && mode=stagger && n=4
        if sb "$1" "$2" "$3" $mode --n $n --draft 7; then ok "slots $1: $(grep -c IDENTICAL "$d/$1.out") IDENTICAL"
        else fail "slots $1: $(grep -h RESULT "$d/$1.out") (slots/$1.out)"; fi
    }
    same_as_solo exact4 "" "$MODEL" 4
    same_as_solo exact2 "" "$MODEL" 2
    same_as_solo exact4_win MK_MTP_WIN=4096 "$MODEL" 4
    same_as_solo stagger MK_EXACT=1 "$MODEL" stagger
    for m in 4 2; do
        if sb "speed$m" "" "$MODEL" speed --n $m --nmin $m --draft 7 --temp 0.6; then
            grep -ho '\[outputs [0-9a-f]*\]' "$d/speed$m.out" > "$d/speed$m.hash"
            ok "slots speed$m: $(grep -ho '[0-9.]* tok/s total' "$d/speed$m.out") $(cat "$d/speed$m.hash")"
            ref "slots speed$m outputs" "$d/speed$m.hash"
        else
            fail "slots speed$m: exit $? (slots/speed$m.out)"
        fi
    done
    if [ -n "$MODEL2" ]; then same_as_solo m2_exact4 "" "$MODEL2" 4; else skip "slots m2_exact4: no --model2"; fi
}

# the C API checks (engine errors, roll-backs, penalties, policies): every case must PASS
leg_capi() {
    [ -f "$CAPI" ] || { skip "capi: no $CAPI"; return; }
    local l=$O/capi/capi.log
    # shellcheck disable=SC2086
    run "$PY" "$CAPI" "$LIB" "$MODEL" "$DV" $CAPI_CASES > "$l" 2>&1
    local rc=$? nf np
    nf=$(grep -c '^FAIL' "$l"); np=$(grep -c '^PASS' "$l")
    grep -E '^FAIL' "$l" | sed 's/^/   /'
    if [ $rc = 0 ] && [ "$nf" = 0 ] && [ "$np" -gt 0 ]; then ok "capi: $np PASS"; else fail "capi: exit $rc, $nf FAIL, $np PASS (capi/capi.log)"; fi
}

# accuracy against llama.cpp (qwen38_mk --kld): the report itself is checked against the reference's
leg_kld() {
    [ -n "$KLD_REF" ] || { skip "kld: no --kld-ref"; return; }
    local d=$O/kld
    run "$BIN" -m "$MODEL" --kld "$KLD_REF" --score-path mixed --kv q8 > "$d/kld.out" 2> "$d/kld.err" || { fail "kld: exit $? (kld/kld.err)"; return; }
    ok "kld: $(grep -h 'top token agrees' "$d/kld.out" | tr -s ' ') / $(grep -h 'KL(ref' "$d/kld.out" | grep -o 'mean [0-9.]*' | head -1)"
    ref "kld report" "$d/kld.out"
}

T0=$(date +%s)
for l in $LEGS; do
    mkdir -p "$O/$l"
    echo "== $l $(date +%T)"
    "leg_$l"
done
status=PASS
[ $NFAIL = 0 ] || status=FAIL
echo "== $(date +%T), $(( ($(date +%s) - T0) / 60 )) min: $NOK ok, $NFAIL FAIL, $NSKIP skipped"
if [ -n "$REC" ]; then
    { cat "$O/MANIFEST.run"; echo "status: $status"; } > "$O/MANIFEST"
    [ $status = PASS ] && echo "reference recorded: $O" || echo "NOT a usable reference (a check failed): $O/MANIFEST says FAIL"
fi
if [ $status = PASS ]; then echo "GATE PASS"; exit 0; fi
echo "GATE FAIL:$FAILED"
exit 1
