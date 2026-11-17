"""Run a matrix of writer runs across Claude profiles with a per-profile concurrency cap. A released LM of 1B
parameters or more counts twice against the cap (spec.slot_weight).

Runs that already have result.json are skipped. A prepared run without result.json is left alone (delete its
directory to rerun it) but holds its slot until its result.json appears, so a restarted dispatcher does not
over-launch next to runs that are still going. At most one run starts per slot and cycle; with --mem-ceiling-mb,
none while the user's unreclaimable memory (its cgroup's anon + shmem + kernel) is above that ceiling. A run this
dispatcher started that ends invalid on an API failure (a timeout or an exhausted quota, read from the writer's last
result) is moved to <root>_retry/ and queued again, at most MAX_RETRIES times.
"""
import argparse
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness.spec import PROFILES, WRITERS, RunSpec, slot_weight, token_hours_left, token_ok

REPO = Path(__file__).resolve().parents[1]
POLL_S = 30
MAX_RETRIES = 2
API_FAILURES = ("Request timed out", "API Error", "rate limit", "overloaded", "Connection error", "usage limit",
                "hit your")
# "You've hit your 5-hour limit · resets 12pm (<time zone>)": the profile pauses until then
LIMIT_RESET = re.compile(r"hit your .*limit.*resets (\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)
CGROUP = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/memory.stat")


def unreclaimable_mb() -> float:
    stats = dict(line.split() for line in CGROUP.read_text().splitlines())
    return sum(int(stats[k]) for k in ("anon", "shmem", "kernel")) / 2**20


def alive(s: RunSpec) -> bool:
    """Whether a harness.run process of this spec is running on this host."""
    want = ["-m", "harness.run", "--arch", s.arch, "--data-access", s.data_access,
            "--forward-budget", str(s.forward_budget), "--rep", str(s.rep), "--writer", s.writer]
    for f in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = f.read_bytes().decode().split("\0")
        except OSError:  # the process exited while we listed /proc
            continue
        if argv[1:1 + len(want)] == want:
            return True
    return False


def api_failure(s: RunSpec) -> str | None:
    """The API failure a finished run ended on, from the last result event of its transcript; None otherwise. A run
    cut this way counts even when what it had written was scored as valid: the writer did not decide to stop."""
    if not (s.run_dir / "result.json").exists():
        return None
    event = last_result(s)
    if not event.get("is_error"):
        return None
    text = str(event.get("result", ""))
    return next((m for m in API_FAILURES if m.lower() in text.lower()), None)


def pause_profile(paused: dict, s: RunSpec) -> None:
    reset = limit_reset(str(last_result(s).get("result", "")))
    if reset is not None:
        profile = adopted_profile(s)
        paused[profile] = max(paused.get(profile, 0), reset)
        print(f"profile {profile}: usage limit, paused until {time.strftime('%H:%M', time.localtime(reset))}",
              flush=True)


def last_result(s: RunSpec) -> dict:
    out = {}
    transcript = s.run_dir / "transcript.jsonl"
    if transcript.exists():
        for line in transcript.read_text(errors="replace").splitlines():
            if '"type":"result"' in line.replace(" ", ""):
                out = json.loads(line)
    return out


def limit_reset(text: str) -> float | None:
    """Epoch time (local clock) at which a profile's usage limit named in text resets, None if text names none."""
    m = LIMIT_RESET.search(text)
    if not m:
        return None
    hour = int(m[1]) % 12 + (12 if m[3].lower() == "pm" else 0)
    now = time.localtime()
    reset = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, int(m[2] or 0), 0, 0, 0, -1))
    return reset + 86400 if reset < time.time() else reset


def attempts(s: RunSpec) -> int:
    """Earlier attempts of this run already set aside by set_aside."""
    return len(list(s.run_dir.parent.with_name(s.run_dir.parent.name + "_retry").glob(f"{s.run_id}_try*")))


def set_aside(s: RunSpec, attempt: int) -> Path:
    dest = s.run_dir.parent.with_name(s.run_dir.parent.name + "_retry") / f"{s.run_id}_try{attempt}"
    dest.parent.mkdir(exist_ok=True)
    shutil.move(s.run_dir, dest)
    return dest


def adopted_profile(s: RunSpec) -> str | None:
    launch = s.run_dir / "launch.json"
    return json.loads(launch.read_text())["profile"] if launch.exists() else None


def matrix(arch_list: list[str], conditions: list[str], reps: int, writer: str) -> list[RunSpec]:
    specs = []
    for arch, cond, rep in itertools.product(arch_list, conditions, range(reps)):
        access, k = cond.split("-K")
        specs.append(RunSpec(arch, access, int(k), rep, writer))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["roles", "basis", "rule"])
    ap.add_argument("--conditions", nargs="+", default=["blind-K0", "open-K0", "open-K50", "view-K0", "view-K50"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES))
    ap.add_argument("--per-profile", type=int, default=2)
    ap.add_argument("--profile-cap", nargs="*", default=[], metavar="PROFILE=N",
                    help="per-profile override of --per-profile, e.g. b=1")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--writer", default="opus", choices=list(WRITERS))
    ap.add_argument("--mem-ceiling-mb", type=float,
                    help="start no run while the user's unreclaimable memory (MB) is above this")
    a = ap.parse_args()
    todo, adopted, paused = [], [], {}
    for s in matrix(a.archs, a.conditions, a.reps, a.writer):
        d = s.run_dir
        if (d / "result.json").exists():
            failure = api_failure(s)
            if failure and attempts(s) < MAX_RETRIES:
                pause_profile(paused, s)
                print(f"retry {s.run_id}: ended on {failure!r}; moved to {set_aside(s, attempts(s) + 1)}", flush=True)
                todo.append(s)
            continue
        if d.exists():
            print(f"skip {s.run_id}: prepared but unfinished, delete {d} to rerun; it holds a slot", flush=True)
            adopted.append(s)
            continue
        todo.append(s)
    random.Random(a.order_seed).shuffle(todo)  # spread archs and conditions across profiles
    print(f"{len(todo)} runs to launch: " + " ".join(s.run_id for s in todo), flush=True)
    slots = a.profiles
    caps = {p: a.per_profile for p in slots}
    caps |= {k: int(v) for k, v in (c.split("=") for c in a.profile_cap)}
    running: dict[str, list[tuple[RunSpec, subprocess.Popen, object]]] = {p: [] for p in slots}
    waiting: set[str] = set()
    while todo or adopted or any(running.values()):
        for s in [s for s in adopted if (s.run_dir / "result.json").exists()]:
            failure = api_failure(s)
            if failure and attempts(s) < MAX_RETRIES:
                pause_profile(paused, s)
                print(f"retry {s.run_id}: ended on {failure!r}; moved to {set_aside(s, attempts(s) + 1)}", flush=True)
                todo.append(s)
        adopted = [s for s in adopted if s.run_dir.exists() and not (s.run_dir / "result.json").exists()
                   and alive(s)]
        for p in slots:
            for item in list(running[p]):
                s, proc, log = item
                if proc.poll() is not None:
                    running[p].remove(item)
                    log.close()
                    print(f"done {s.run_id} profile={p} exit={proc.returncode}", flush=True)
                    failure = api_failure(s)
                    if failure and attempts(s) < MAX_RETRIES:
                        pause_profile(paused, s)
                        dest = set_aside(s, attempts(s) + 1)
                        todo.append(s)
                        print(f"retry {s.run_id}: ended on {failure!r}; moved to {dest}", flush=True)
            if todo and time.time() < paused.get(p, 0):
                continue
            if todo and not token_ok(p):
                if p not in waiting:
                    print(f"profile {p}: token has {token_hours_left(p):.2f} h left, waiting for a refresh", flush=True)
                    waiting.add(p)
                continue
            waiting.discard(p)
            if todo and (a.mem_ceiling_mb is None or unreclaimable_mb() < a.mem_ceiling_mb):
                used = sum(slot_weight(r.arch) for r, _, _ in running[p]) + sum(
                    slot_weight(s.arch) for s in adopted if adopted_profile(s) == p)
                limit = caps[p]
                s = next((t for t in todo if used + min(slot_weight(t.arch), limit) <= limit), None)
                if s is None:
                    continue
                todo.remove(s)
                tag = "" if s.writer == "opus" else f"{s.writer}_"
                log_path = REPO / "logs" / f"run_{tag}{s.run_id}.log"
                cmd = [sys.executable, "-m", "harness.run", "--arch", s.arch, "--data-access", s.data_access,
                       "--forward-budget", str(s.forward_budget), "--rep", str(s.rep), "--writer", s.writer,
                       "--profile", p]
                log = log_path.open("w")
                running[p].append((s, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log))
                print(f"start {s.run_id} profile={p}", flush=True)
        time.sleep(POLL_S)
    print(json.dumps({"finished": time.time()}))


if __name__ == "__main__":
    main()
