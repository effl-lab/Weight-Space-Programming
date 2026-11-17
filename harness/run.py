"""One writer run: prepare the sandbox, launch the headless writer model inside bwrap, serve forward calls,
then build and score the result on the host.

Run layout under RUNS_DIR/<run_id>/:
  work/      mounted at /srv/work (TASK.md, arch_spec.json, tools/, build_weights.py, data/, .forward/)
  config/    mounted at /srv/config: a fresh CLAUDE_CONFIG_DIR holding only an access token
  forward/   host-only: every brokered forward request (req_NNN.py) and the meter (log.jsonl)
  proxy.sock, proxy_log.jsonl   host proxy; the sandbox has no network of its own
  selftest.json, transcript.jsonl, stderr.log, launch.json, weights.npz, result.json
"""
import argparse
import hashlib
import inspect
import json
import os
import shlex
import shutil
import stat
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from harness import audit, buildcheck, lm_score, olmo_score, task, tokens, views
from harness.netproxy import AllowlistProxy
from harness.spec import (PROFILES, TOKEN_MARGIN_S, TOKEN_NEED_S, WALL_LIMIT_S, WRITER_EFFORT, WRITER_ENV, WRITERS,
                          EFFORT, RunSpec)
from wsp import (arc, archs, distill, evaluate, free_model, given_archs, lineage, lm_start, memory, olmo2_1b,
                 olmo_start, paths)

REPO = Path(__file__).resolve().parents[1]
CLAUDE_BIN_ENV = "WSP_CLAUDE_BIN"  # the Claude Code executable the writer runs (the paper's runs used 2.1.319)
VENV = buildcheck.VENV
PROMPT = "Read /srv/work/TASK.md and complete the task described there."
TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
ALLOWED_HOSTS = {"api.anthropic.com"}  # CONNECT targets of Claude-profile writers
PLATEAU_CALLS = task.PLATEAU_CALLS  # stop when the best practice accuracy has not risen in this many calls
MAX_CONTINUATIONS = 10
K0_OUTPUT_TOKEN_TARGET = 120_000
RELAY_PORT = 3128
MAX_REQUEST_BYTES = 4_000_000
PRACTICE = "practice.jsonl"  # labelled ARC training questions in work/data of an fb-arc run (wsp.arc.practice)
TRAINING = "arc_train.jsonl"  # every labelled ARC training question in work/data of a fit-arc run (wsp.arc.training)
VARIANT_DATA = {"fb": PRACTICE, "fit": TRAINING}
LABEL_FREE_KEYS = {"x", "words"}  # npz arrays a view run may hold: inputs and patch words, never y
LIB_FILES = {"wsp/__init__.py": "wsp/__init__.py", "wsp/paths.py": "wsp/paths.py", "wsp/archs.py": "wsp/archs.py",
             "wsp/text_archs.py": "wsp/text_archs.py", "wsp/vision_words.py": "wsp/vision_words.py",
             "wsp/vision_tokens.py": "wsp/vision_tokens.py", "wsp/memory.py": "wsp/memory.py",
             "wsp/free_model.py": "wsp/free_model.py", "wsp/lm_start.py": "wsp/lm_start.py",
             "wsp/lineage.py": "wsp/lineage.py", "wsp/distill.py": "wsp/distill.py", "wsp/olmo_start.py": "wsp/olmo_start.py",
             "wsp/olmo2_1b.py": "wsp/olmo2_1b.py",
             "wsp/given_archs/__init__.py": "wsp/given_archs/__init__.py",
             "wsp/given_archs/transformer.py": "wsp/given_archs/transformer.py",
             "wsp/given_archs/bow.py": "wsp/given_archs/bow.py",
             "wsp/given_archs/span_transformer.py": "wsp/given_archs/span_transformer.py",
             "wsp/given_archs/gen_transformer.py": "wsp/given_archs/gen_transformer.py",
             "harness/buildcheck.py": "harness/buildcheck.py", "harness/netproxy.py": "netproxy.py",
             "harness/nestpid.sh": "nestpid.sh"}
RELAY = f"WSP_RELAY_FORK=1 python -I /srv/lib/netproxy.py /srv/proxy.sock {RELAY_PORT}; sleep 1; "
SELFTEST = rf"""
set -u
# /usr, /lib*, /bin, /sbin and the venv hold system and library code only, so they are not searched.
find /etc /srv /tmp /home /opt -path {VENV} -prune -o \
  \( -iname '*cifar*' -o -iname '*agnews*' -o -iname 'test.npz' -o -iname 'practice.npz' -o -iname '*.parquet' \) -print 2>/dev/null \
  | sed 's/^/hit=/'
env | cut -d= -f1 | grep -E 'TOKEN|KEY|SECRET|PASSWORD' | grep -v '^CLAUDE_CODE_' | sed 's/^/secret_env=/'
for p in "{paths.DATA_ROOT}" "{Path.home()}" "{paths.HF_HOME}" /run/user; do
  [ -e "$p" ] && echo "visible=$p"
done
""" + r"""python -I -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)" >/dev/null 2>&1 \
  && echo "net=direct_open"
python -I -c "import urllib.request as u; u.urlopen('https://huggingface.co', timeout=10)" >/dev/null 2>&1 \
  && echo "net=proxy_open"
python -I - <<'PY'
import glob, json, os, numpy as np
for root in ("/srv/work/data", "/srv/work/views"):
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            print(f"file={p}")
            if n.endswith(".jsonl"):
                with open(p) as f:
                    if "label" in json.loads(f.readline() or "{}"):
                        print(f"jsonl_label={p}")
for p in sorted(set(glob.glob("/srv/**/*.npz", recursive=True) + glob.glob("/tmp/**/*.npz", recursive=True))):
    with np.load(p) as z:
        print(f"npz={p} keys={','.join(sorted(z.files))}")
PY
echo done
"""


class CredentialError(RuntimeError):
    pass


def access_token_credentials(profile_key: str) -> dict:
    """The profile's access token without its refresh token, so the run cannot rotate it."""
    oauth = json.loads((PROFILES[profile_key] / ".credentials.json").read_text())["claudeAiOauth"]
    remaining = oauth["expiresAt"] / 1000 - time.time()
    if remaining < TOKEN_NEED_S:
        raise CredentialError(f"profile {profile_key}: access token expires in {remaining / 3600:.2f} h, "
                              f"run needs {TOKEN_NEED_S / 3600:.2f} h")
    return {"claudeAiOauth": {k: v for k, v in oauth.items()
                              if k not in ("refreshToken", "refreshTokenExpiresAt")}}


def mounted_data(spec: RunSpec) -> set[str]:
    t = archs.task_of(spec.arch)
    files = {"blind": (), "open": t.open_files, "view": t.view_files}[spec.data_access]
    return {f"/srv/work/data/{f}" for f in files}


def allowed_data(spec: RunSpec) -> set[str]:
    """Every data file a run may see: the task's files mounted for its data access, and the file that prepare() writes
    for fb-arc (the practice questions) or fit-arc (the training questions)."""
    name = VARIANT_DATA.get(lineage.arc_variant(spec.arch))
    return mounted_data(spec) | ({f"/srv/work/data/{name}"} if name else set())


def fixed_arc(arch: str) -> str | None:
    """Name of the given architecture of a fixed ARC, MathBench or SQuAD arch (<name>-arc, <name>-mb,
    transformer-sq), None for any other."""
    if arch.endswith(("-sq", "-sq1")):
        return None if arch.startswith("free-") else given_archs.SPAN
    if arch.endswith("-gen"):
        return None if arch == "free-gen" else given_archs.GEN
    if arch.endswith(("-all", "-text", "-mm")):
        return None
    if arch.endswith("-mb"):
        return None if arch == "free-mb" else arch.removesuffix("-mb")
    if not arch.endswith("-arc") or lineage.parse(arch) or lineage.arc_variant(arch) or arch in distill.ARCHS:
        return None
    if (given := lineage.given_lineage(arch)) is not None:
        return given[0]
    return arch.removesuffix("-arc")


def _best_valid(root: Path, arch: str) -> Path | None:
    """The valid run of `arch` under `root` with the highest validation accuracy (the lower rep on a tie)."""
    results = [(json.loads((d / "result.json").read_text()), d) for d in sorted(root.glob(f"{arch}_blind_K0_r*"))
               if (d / "result.json").exists()]
    valid = [(r["validation"]["accuracy"], -r["rep"], d) for r, d in results if r["valid"]]
    return max(valid)[2] if valid else None


def lineage_parent(spec: RunSpec) -> Path | None:
    """Run directory of a lineage run's parent: the memory for MMLU generation 1, else the valid run of the
    previous generation with the highest validation accuracy (the lower rep on a tie), for a free design and for a
    given-architecture lineage (wsp.lineage.GIVEN_LINEAGES) alike. The conv-arc variant's parent
    is that run of the writer's last ARC generation whose runs all have a result, the conv1-arc variant's that run of
    generation 1."""
    variant = lineage.arc_variant(spec.arch)
    if variant in lineage.CONVERT_VARIANTS:
        root = RunSpec("free-arc", "blind", 0, 0, spec.writer).run_dir.parent
        runs = {g: list(root.glob(f"{lineage.arch(g, 'arc')}_blind_K0_r*"))
                for g in range(1, lineage.MAX_GENERATION + 1)}
        done = [g for g, dirs in runs.items() if dirs and all((d / "result.json").exists() for d in dirs)]
        generation = 1 if variant == "conv1" else max(done, default=None)
        best = _best_valid(root, lineage.arch(generation, "arc")) if generation in done else None
        if best is None:
            raise FileNotFoundError(f"{spec.run_id}: no finished generation with a valid run in {root}")
        return best
    if (vl := lineage.variant_lineage(spec.arch)) and vl[1] > 1:
        arch = lineage.variant_arch(vl[0], vl[1] - 1)
    elif (given := lineage.given_lineage(spec.arch)) and given[1] > 1:
        arch = lineage.given_arch(given[0], given[1] - 1)
    else:
        arch = lineage.parent_arch(spec.arch) if lineage.parse(spec.arch) else None
    if arch is None:
        return None
    if arch.startswith("mem-"):
        return RunSpec(arch, "blind", 0, 0, spec.writer).run_dir
    root = RunSpec(arch, "blind", 0, 0, spec.writer).run_dir.parent
    best = _best_valid(root, arch)
    if best is None:
        raise FileNotFoundError(f"{spec.run_id}: no valid run of the parent generation {arch} in {root}")
    return best


def prepare(spec: RunSpec, profile_key: str | None) -> Path:
    creds = access_token_credentials(profile_key)
    run_dir = spec.run_dir
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; delete it to restart this run")
    work = run_dir / "work"
    (work / "tools").mkdir(parents=True)
    (run_dir / "config").mkdir()
    (run_dir / "forward").mkdir()
    cred_path = run_dir / "config" / ".credentials.json"
    cred_path.write_text(json.dumps(creds))
    cred_path.chmod(0o600)
    lm, olmo = lm_start.is_lm(spec.arch), olmo_start.is_olmo(spec.arch)
    parent_dir = lineage_parent(spec)
    parent = json.loads((parent_dir / "result.json").read_text()) if parent_dir else None
    (work / "TASK.md").write_text(task.render(spec.arch, spec.data_access, spec.forward_budget, parent))
    shapes = (lm_start.manifest(spec.arch)["tensors"] if lm else olmo_start.manifest(spec.arch)["tensors"] if olmo
              else archs.shapes(spec.arch))
    (work / "arch_spec.json").write_text(json.dumps(
        {"arch": spec.arch, "parameters": {k: list(s) for k, s in shapes.items()}}, indent=1))
    task_def = archs.task_of(spec.arch)
    for f in task_def.arch_files:
        shutil.copy(task_def.data_dir / f, work / f)
    deliverable = archs.ARCHS[spec.arch].deliverable
    checks = {memory.DELIVERABLE: "check_facts.py", free_model.DELIVERABLE: "check_model.py",
              distill.DELIVERABLE: "check_data.py"}
    tools = ["check_lm.py" if lm else "check_olmo.py" if olmo else checks.get(deliverable, "check.py")]
    fixed = fixed_arc(spec.arch)
    if fixed:
        (work / free_model.DELIVERABLE).mkdir()
        (work / free_model.DELIVERABLE / "arch.py").write_text(given_archs.source(fixed))
    if deliverable == distill.DELIVERABLE:
        (work / distill.DELIVERABLE).mkdir()
    if lm:
        (work / free_model.DELIVERABLE).mkdir()
        (work / free_model.DELIVERABLE / "arch.py").write_text(lm_start.arch_source(lm_start.parse(spec.arch)[0]))
    if olmo:
        (work / free_model.DELIVERABLE).mkdir()
        (work / free_model.DELIVERABLE / "arch.py").write_text(inspect.getsource(olmo2_1b))
    if parent_dir:
        (work / "parent").mkdir()
        if parent["arch"].startswith("mem-"):
            shutil.copy(parent_dir / memory.DELIVERABLE, work / "parent" / memory.DELIVERABLE)
        else:
            shutil.copytree(parent_dir / free_model.DELIVERABLE, work / "parent" / free_model.DELIVERABLE)
    if spec.data_access == "open" and task_def.name == "cifar":
        tools.append("view.py")
    if spec.data_access != "blind":
        (work / "data").mkdir()
        for f in mounted_data(spec):
            (work / "data" / Path(f).name).touch()
    if (variant := lineage.arc_variant(spec.arch)) in VARIANT_DATA:
        (work / "data").mkdir()
        rows = arc.practice() if variant == "fb" else arc.training()
        (work / "data" / VARIANT_DATA[variant]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if spec.data_access == "view":
        views.CLASS_VIEWS[task_def.name](work / "views")
    if spec.forward_budget > 0:
        tools.append("forward.py")
    for t in tools:
        shutil.copy(REPO / "harness" / "tools" / t, work / "tools" / t)
    return run_dir


def sandbox_argv(spec: RunSpec, run_dir: Path) -> list[str]:
    argv = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
            "--clearenv"]
    for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
        argv += ["--ro-bind", p, p]
    argv += ["--tmpfs", "/etc/profile.d", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--ro-bind", str(VENV), str(VENV), "--ro-bind", str(paths.env_path(CLAUDE_BIN_ENV)), "/opt/claude",
             "--bind", str(run_dir / "config"), "/srv/config",
             "--bind", str(run_dir / "work"), "/srv/work",
             "--ro-bind", str(run_dir / "proxy.sock"), "/srv/proxy.sock"]
    for src, dst in LIB_FILES.items():
        argv += ["--ro-bind", str(REPO / src), f"/srv/lib/{dst}"]
    for f in sorted(mounted_data(spec)):
        argv += ["--ro-bind", str(archs.task_of(spec.arch).data_dir / Path(f).name), f]
    lm = lm_start.is_lm(spec.arch) or olmo_start.is_olmo(spec.arch)
    if lm_start.is_lm(spec.arch):
        argv += ["--ro-bind", str(lm_start.start_dir(spec.arch)), "/srv/start",
                 *lm_score.lib_binds(lm_start.parse(spec.arch)[0])]
    if olmo_start.is_olmo(spec.arch):
        argv += ["--ro-bind", str(olmo_start.start_dir(spec.arch)), "/srv/start", *olmo_score.lib_binds()]
    proxy = f"http://127.0.0.1:{RELAY_PORT}"
    env = {"HOME": "/srv/home", "CLAUDE_CONFIG_DIR": "/srv/config", "SHELL": "/bin/bash",
           "PATH": f"{VENV}/bin:/usr/bin:/bin", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
           "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
           # tool commands get their own pid namespace, so they cannot write into claude's stdout
           "CLAUDE_CODE_SHELL_PREFIX": "/srv/lib/nestpid.sh", **WRITER_ENV.get(spec.writer, {})}
    if lm:  # the tokenizer is local; a remote-code tokenizer module is cached under HF_HOME
        env |= {"HF_HUB_OFFLINE": "1", "HF_HOME": "/tmp/hf"}
    for k, v in env.items():
        argv += ["--setenv", k, v]
    return argv + ["--chdir", "/srv/work"]


def self_test(spec: RunSpec, run_dir: Path) -> list[str]:
    proc = subprocess.run(sandbox_argv(spec, run_dir) + ["/bin/bash", "-c", RELAY + SELFTEST],
                          capture_output=True, text=True, timeout=600)
    out = proc.stdout
    (run_dir / "selftest.json").write_text(json.dumps({"stdout": out, "stderr": proc.stderr[-2000:]}))
    allowed = allowed_data(spec)
    problems = [f"data file visible: {line[4:]}" for line in out.splitlines()
                if line.startswith("hit=") and line[4:] not in allowed]
    data_files = {line[5:] for line in out.splitlines() if line.startswith("file=/srv/work/data/")}
    if data_files != allowed:
        problems.append(f"data files {sorted(data_files)} != allowed {sorted(allowed)}")
    if spec.data_access != "open":
        problems += [f"labelled text in a {spec.data_access} run: {line[12:]}" for line in out.splitlines()
                     if line.startswith("jsonl_label=")]
    problems += [line for line in out.splitlines()
                 if line.startswith(("secret_env=", "visible=", "net="))]
    for line in out.splitlines():
        if not line.startswith("npz="):
            continue
        path, keys = line[4:].split(" keys=")
        if path not in allowed:
            problems.append(f"unexpected npz: {path}")
        elif spec.data_access != "open" and not set(keys.split(",")) <= LABEL_FREE_KEYS:
            problems.append(f"labelled data in a {spec.data_access} run: {path} ({keys})")
    if "done" not in out.splitlines():
        problems.append(f"self-test did not finish: {proc.stderr[-500:]}")
    return problems


def claude_command(prompt: str, writer: str, resume: str | None = None) -> str:
    args = ["/opt/claude", "-p", "--model", WRITERS[writer], "--effort", WRITER_EFFORT.get(writer, EFFORT),
            "--setting-sources", "project",
            "--strict-mcp-config", "--permission-mode", "dontAsk", "--tools", ",".join(TOOLS),
            "--allowedTools", *TOOLS, "--output-format", "stream-json", "--verbose"]
    if resume:
        args += ["--resume", resume]
    args += ["--", prompt]
    return RELAY + "exec " + " ".join(shlex.quote(a) for a in args)


def forward_report(res: dict, used: int, budget: int, task) -> str:
    """Overall accuracy and prediction counts only: per-class accuracy and the confusion matrix are
    label-conditioned statistics, which the label-free condition must not receive, and both rules
    get the same feedback so their difference isolates label access."""
    lines = [f"forward call {used}/{budget}"]
    if res["accuracy"] is None:
        return "\n".join(lines + ["logits are not finite"])
    return "\n".join(lines + [
        f"practice accuracy: {res['accuracy']:.4f} on {res['n']} {task.item}",
        "prediction counts: " + ", ".join(f"{n} {c}" for n, c in zip(task.classes, res["pred_hist"]))])


def _read_request(qfd: int, name: str) -> str:
    """Reads a writer-supplied request without following links or blocking on special files."""
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=qfd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("request is not a regular file")
        if st.st_size > MAX_REQUEST_BYTES:
            raise ValueError(f"request larger than {MAX_REQUEST_BYTES} bytes")
        return os.read(fd, MAX_REQUEST_BYTES + 1).decode("utf-8")
    finally:
        os.close(fd)


def _remove_writer_entry(qfd: int, name: str) -> None:
    """Removes an entry of the writer-controlled queue; never follows links."""
    try:
        st = os.stat(name, dir_fd=qfd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(name, dir_fd=qfd)
    else:
        os.unlink(name, dir_fd=qfd)


def _write_response(qfd: int, stem: str, text: str) -> None:
    name = f"{stem}.resp"
    _remove_writer_entry(qfd, name)
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=qfd)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"text": text}))


def _accuracies(meter: Path) -> list[float]:
    if not meter.exists():
        return []
    return [json.loads(line)["accuracy"] or 0.0 for line in meter.read_text().splitlines()]


def plateaued(meter: Path) -> bool:
    acc = _accuracies(meter)
    return len(acc) > PLATEAU_CALLS and max(acc[-PLATEAU_CALLS:]) <= max(acc[:-PLATEAU_CALLS])


def _answer(spec: RunSpec, store: Path, source: str, practice) -> tuple[str, dict | None]:
    meter = store / "log.jsonl"
    used = sum(1 for _ in meter.open()) if meter.exists() else 0
    script = store / f"req_{len(list(store.glob('req_*.py'))):03d}.py"
    script.write_text(source)
    if used >= spec.forward_budget:
        return f"REFUSED: forward budget of {spec.forward_budget} calls is used up.", None
    if plateaued(meter):
        return (f"REFUSED: practice accuracy has not improved in the last {PLATEAU_CALLS} calls, so the "
                "run is ending. Leave /srv/work/build_weights.py in its final state."), None
    weights, problems = buildcheck.build(script, isolate=True)
    problems = problems or archs.validate(spec.arch, weights)
    if problems:
        return "BUILD FAILED (no budget used)\n" + "\n".join(problems), None
    w = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in weights.items()}
    res = evaluate.evaluate(spec.arch, w, *practice)
    return forward_report(res, used + 1, spec.forward_budget, archs.task_of(spec.arch)), {
        "t": time.time(), "request": script.name, "accuracy": res["accuracy"],
        **tokens.progress(store.parent / "transcript.jsonl")}


def _log_rejection(store: Path, name: str, what: str) -> None:
    with (store / "rejected.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.time(), "request": name, "stage": what,
                            "error": traceback.format_exc()}) + "\n")


def serve_forward(spec: RunSpec, run_dir: Path, practice) -> None:
    """Answers pending forward requests one at a time; the budget lives only on the host.

    The queue directory is opened once without following links and every later operation is
    relative to that handle, so swapping .forward for a symlink cannot redirect host I/O.
    """
    store = run_dir / "forward"
    try:
        qfd = os.open(run_dir / "work" / ".forward", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return
    try:
        for name in sorted(n for n in os.listdir(qfd) if n.endswith(".req")):
            try:
                text, entry = _answer(spec, store, _read_request(qfd, name), practice)
            except Exception as e:  # malformed request: the writer gets the reason, the run continues
                reason = str(e) if isinstance(e, ValueError) else type(e).__name__
                text, entry = f"REQUEST REJECTED (no budget used): {reason}", None
                _log_rejection(store, name, "answer")
            if entry:
                with (store / "log.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")
            try:
                _remove_writer_entry(qfd, name)
                _write_response(qfd, name[:-4], text)
            except OSError:
                _log_rejection(store, name, "respond")
    finally:
        os.close(qfd)


def _init_tools(transcript: Path, offset: int) -> list[str] | None:
    """Tool list of the first init event at or after byte `offset` (the current segment)."""
    with transcript.open("rb") as f:
        f.seek(offset)
        for raw in f:
            line = raw.decode("utf-8", "replace")
            if '"subtype":"init"' in line.replace(" ", ""):
                try:
                    return json.loads(line).get("tools")
                except json.JSONDecodeError:  # line still being written; read again next tick
                    return None
    return None


def _session_id(transcript: Path) -> str | None:
    """Session id of the latest segment (a resumed session may get a new id)."""
    ids = [e.get("session_id") for e in tokens.events(transcript)
           if e.get("type") == "system" and e.get("subtype") == "init"]
    return ids[-1] if ids else None


def _continuation(spec: RunSpec, run_dir: Path, meta: dict) -> str | None:
    """The next nudge, or None when the run should end (reason stored in meta['ended'])."""
    meter = run_dir / "forward" / "log.jsonl"
    segs = meta["segments"]
    if len(segs) > distill.MAX_CONTINUATIONS.get(spec.arch, MAX_CONTINUATIONS):
        meta["ended"] = "max continuations"
        return None
    if spec.forward_budget:
        acc = _accuracies(meter)
        if len(acc) >= spec.forward_budget:
            meta["ended"] = "forward budget used"
            return None
        if plateaued(meter):
            meta["ended"] = f"no improvement in {PLATEAU_CALLS} calls"
            return None
        if len(segs) > 1 and segs[-1]["forward_calls_after"] == segs[-2]["forward_calls_after"]:
            meta["ended"] = "continuation made no forward call"
            return None
        return (f"You have {spec.forward_budget - len(acc)} forward calls left; the best practice accuracy "
                f"so far is {max(acc, default=0.0):.4f}. Keep improving /srv/work/build_weights.py under the "
                "same rules, then reply when done.")
    used = tokens.totals(run_dir / "transcript.jsonl")
    cap = distill.COST_CAP_USD.get(spec.arch)
    if cap is not None and used["cost_usd_equiv"] >= cap:
        meta["ended"] = "cost cap reached"
        return None
    if cap is None and used["output_tokens"] >= K0_OUTPUT_TOKEN_TARGET:
        meta["ended"] = "output token target reached"
        return None
    if len(segs) > 1 and segs[-1]["script_sha"] == segs[-2]["script_sha"]:
        meta["ended"] = "continuation left the script unchanged"
        return None
    deliverable = archs.ARCHS[spec.arch].deliverable
    noun = {memory.DELIVERABLE: "memory", free_model.DELIVERABLE: "model", distill.DELIVERABLE: "data"}.get(
        deliverable, "network")
    return (f"Review your {noun} against the task and improve /srv/work/{deliverable} under the same "
            "rules if you can, then reply when done.")


def _script_sha(spec: RunSpec, run_dir: Path) -> str | None:
    p = run_dir / "work" / archs.ARCHS[spec.arch].deliverable
    if p.is_symlink():
        return None
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(q for q in p.rglob("*") if q.is_file() and not q.is_symlink()):
            h.update(str(f.relative_to(p)).encode() + f.read_bytes())
        return h.hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def snapshot_data(run_dir: Path, meta: dict, force: bool = False) -> None:
    """Every distill.SNAPSHOT_S seconds, a copy of the writer's train.jsonl when it changed, named by the seconds
    since the run started and its digest, so two snapshots in the same 10 ms keep both (the cost spent by then comes
    from the transcript's timestamps)."""
    now = time.time()
    if not force and now - meta.get("last_snapshot", meta["started"]) < distill.SNAPSHOT_S:
        return
    meta["last_snapshot"] = now
    src = run_dir / "work" / distill.DELIVERABLE / distill.TRAIN_FILE
    if src.is_symlink() or not src.is_file():
        return
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == meta.get("last_snapshot_sha"):
        return
    meta["last_snapshot_sha"] = digest
    (run_dir / "snapshots").mkdir(exist_ok=True)
    (run_dir / "snapshots" / f"{now - meta['started']:09.2f}_{digest[:8]}.jsonl").write_bytes(data)


def _run_segment(spec: RunSpec, run_dir: Path, meta: dict, command: str, practice) -> int:
    argv = sandbox_argv(spec, run_dir) + ["/bin/bash", "-c", command]
    transcript = run_dir / "transcript.jsonl"
    offset = transcript.stat().st_size if transcript.exists() else 0
    tools_checked = False
    with transcript.open("a") as out, (run_dir / "stderr.log").open("a") as err:
        proc = subprocess.Popen(argv, stdout=out, stderr=err)
        while proc.poll() is None:
            if time.time() > meta["deadline"]:
                meta["stopped"] = meta["deadline_reason"]
            if not tools_checked and (tools := _init_tools(transcript, offset)) is not None:
                tools_checked = True
                meta["init_tools"] = tools
                if sorted(tools) != sorted(TOOLS):
                    meta["stopped"] = f"unexpected tool set: {tools}"
            if "stopped" in meta:
                proc.kill()
                proc.wait()
                break
            if spec.forward_budget:
                serve_forward(spec, run_dir, practice)
            if archs.ARCHS[spec.arch].deliverable == distill.DELIVERABLE:
                snapshot_data(run_dir, meta)
            time.sleep(0.5)
    return proc.returncode


def launch(spec: RunSpec, run_dir: Path, profile_key: str | None) -> int:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"spec": asdict(spec), "profile": profile_key, "model": WRITERS[spec.writer],
            "effort": WRITER_EFFORT.get(spec.writer, EFFORT),
            "harness_commit": commit, "argv_prefix": sandbox_argv(spec, run_dir), "started": time.time(),
            "segments": []}
    parent = lineage_parent(spec)
    meta["lineage_parent"] = str(parent) if parent else None
    meta["deadline"] = meta["started"] + distill.WALL_LIMIT_S.get(spec.arch, WALL_LIMIT_S)
    meta["deadline_reason"] = "wall limit"
    # stop before the copied access token expires: the run holds no refresh token
    creds = run_dir / "config" / ".credentials.json"
    expiry = json.loads(creds.read_text())["claudeAiOauth"]["expiresAt"] / 1000 - TOKEN_MARGIN_S
    if expiry < meta["deadline"]:
        meta["deadline"], meta["deadline_reason"] = expiry, "token expiry"
    (run_dir / "launch.json").write_text(json.dumps(meta, indent=1))
    practice = archs.task_of(spec.arch).load("practice") if spec.forward_budget else None
    command, code = claude_command(PROMPT, spec.writer), None
    while command is not None:
        seg_start = time.time()
        code = _run_segment(spec, run_dir, meta, command, practice)
        meta["segments"].append({"started": seg_start, "finished": time.time(), "exit_code": code,
                                 "forward_calls_after": len(_accuracies(run_dir / "forward" / "log.jsonl")),
                                 "script_sha": _script_sha(spec, run_dir)})
        (run_dir / "launch.json").write_text(json.dumps(meta, indent=1))
        if "stopped" in meta or code != 0:
            meta["ended"] = meta.get("stopped") or f"exit code {code}"
            break
        nudge = _continuation(spec, run_dir, meta)
        session = _session_id(run_dir / "transcript.jsonl")
        command = claude_command(nudge, spec.writer, resume=session) if nudge and session else None
    if archs.ARCHS[spec.arch].deliverable == distill.DELIVERABLE:
        snapshot_data(run_dir, meta, force=True)
    meta.update({"finished": time.time(), "exit_code": code})
    (run_dir / "launch.json").write_text(json.dumps(meta, indent=1))
    return code


def score_facts(spec: RunSpec, run_dir: Path) -> dict:
    path = run_dir / "work" / memory.DELIVERABLE
    if path.is_symlink() or not path.is_file():
        return {"valid": False, "problems": [f"no regular {memory.DELIVERABLE}"]}
    text = path.read_text()
    units, groups, problems = memory.parse(text)
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copy(path, run_dir / memory.DELIVERABLE)
    w = memory.build(text)
    t = archs.task_of(spec.arch)
    test, val = memory.evaluate(w, *t.load("test")), memory.evaluate(w, *t.load("validation"))
    return {"valid": True, "problems": [], "test_accuracy": test["accuracy"], "practice_accuracy": None,
            "test": test, "validation": val, "units": len(units), "same_word_groups": len(groups),
            "words": len(w["vocab"]), "weights": memory.n_weights(w)}


def score_design_images(spec: RunSpec, run_dir: Path) -> dict:
    """design-cifar: test images, and the practice split as validation."""
    folder = run_dir / "work" / free_model.DELIVERABLE
    t = archs.task_of(spec.arch)
    parts = {"validation": t.load("practice"), "test": t.load("test")}
    logits, size, problems = free_model.run_images(folder, {s: x for s, (x, _) in parts.items()}, len(t.classes))
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    test, val = (free_model.accuracy(logits[s], parts[s][1]) for s in ("test", "validation"))
    return {"valid": True, "problems": [], "test_accuracy": test["accuracy"], "practice_accuracy": None,
            "test": test, "validation": val, **size}


def score_free(spec: RunSpec, run_dir: Path) -> dict:
    folder = run_dir / "work" / free_model.DELIVERABLE
    t = archs.task_of(spec.arch)
    fixed = fixed_arc(spec.arch)
    if fixed and (folder / "arch.py").read_text() != given_archs.source(fixed):
        return {"valid": False, "problems": ["the given arch.py was changed"]}
    parts = {s: t.load(s) for s in ("validation", "test")}
    batches = {s: (q, c) for s, (q, c, _) in parts.items()}
    if lineage.arc_variant(spec.arch) == "py":
        logits, size, problems = free_model.run_py(folder, batches)
    else:
        logits, size, problems = free_model.run(folder, batches, encoder=free_model.encode_of(spec.arch),
                                                max_bytes=free_model.max_bytes_of(spec.arch))
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    words = len(free_model.vocab_of(folder)) if (folder / "vocab.txt").is_file() else None
    test, val = (free_model.accuracy(logits[s], parts[s][2]) for s in ("test", "validation"))
    extra = {}
    if t.subsets:
        sub = t.subset("test")
        extra["test_by_subset"] = {s: free_model.accuracy(logits["test"][sub == s], parts["test"][2][sub == s])
                                   for s in t.subsets}
    return {"valid": True, "problems": [], "test_accuracy": test["accuracy"], "practice_accuracy": None,
            "test": test, "validation": val, "words": words, **size, **extra}


def score_spans(spec: RunSpec, run_dir: Path) -> dict:
    """SQuAD 2.0 (and 1.1, where the "no answer" slot is ignored): official exact match and F1; test_accuracy and
    the validation accuracy are F1 as a fraction."""
    folder = run_dir / "work" / free_model.DELIVERABLE
    t = archs.task_of(spec.arch)
    fixed = fixed_arc(spec.arch)
    if fixed and (folder / "arch.py").read_text() != given_archs.source(fixed):
        return {"valid": False, "problems": ["the given arch.py was changed"]}
    parts = {s: t.load(s) for s in ("validation", "test")}
    logits, size, problems = free_model.run_spans(folder, {s: (q, c) for s, (q, c, _) in parts.items()})
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    decode = free_model.span_decode_forced if spec.arch.endswith("-sq1") else free_model.span_decode
    answers = {s: decode(logits[s], parts[s][1]) for s in parts}
    (run_dir / "test_answers.json").write_text(json.dumps(answers["test"]))
    test, val = (free_model.span_scores(answers[s], parts[s][2]) for s in ("test", "validation"))
    return {"valid": True, "problems": [], "metric": "f1", "test_accuracy": test["accuracy"], "practice_accuracy": None,
            "test": test, "validation": val, "words": len(free_model.vocab_of(folder)), **size}


def score_gen(spec: RunSpec, run_dir: Path) -> dict:
    """SciQ generation: exact match and F1 of the generated words against the answer, as in the SQuAD evaluation;
    test_accuracy and the validation accuracy are F1 as a fraction."""
    folder = run_dir / "work" / free_model.DELIVERABLE
    t = archs.task_of(spec.arch)
    fixed = fixed_arc(spec.arch)
    if fixed and (folder / "arch.py").read_text() != given_archs.source(fixed):
        return {"valid": False, "problems": ["the given arch.py was changed"]}
    parts = {s: t.load(s) for s in ("validation", "test")}
    answers, size, problems = free_model.run_gen(folder, {s: q for s, (q, _) in parts.items()})
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    (run_dir / "test_answers.json").write_text(json.dumps(answers["test"]))
    test, val = (free_model.span_scores(answers[s], parts[s][1]) for s in ("test", "validation"))
    return {"valid": True, "problems": [], "metric": "f1", "test_accuracy": test["accuracy"], "practice_accuracy": None,
            "test": test, "validation": val, "words": len(free_model.vocab_of(folder)), **size}


def score_all(spec: RunSpec, run_dir: Path) -> dict:
    """The general task: accuracy, SQuAD F1 or ROUGE-L per part (free_model.all_scores); test_accuracy and the
    validation accuracy are the mean over its four groups."""
    folder = run_dir / "work" / free_model.DELIVERABLE
    t = archs.task_of(spec.arch)
    parts = {s: t.load(s) for s in ("validation", "test")}
    answers, size, problems = free_model.run_all(folder, parts)
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    vocab = free_model.vocab_of(folder)
    texts = {s: free_model.all_text(vocab, parts[s], answers[s]) for s in parts}
    (run_dir / "test_answers.json").write_text(json.dumps({"ids": answers["test"].tolist(), "texts": texts["test"]}))
    test, val = (free_model.all_scores(parts[s], answers[s], texts[s]) for s in ("test", "validation"))
    return {"valid": True, "problems": [], "metric": "mean of four groups", "test_accuracy": test["accuracy"],
            "practice_accuracy": None, "test": test, "validation": val, "words": len(vocab), **size}


def score_text(spec: RunSpec, run_dir: Path) -> dict:
    """The language-model task: the model continues the prompts of wsp.textlm; test_accuracy is the mean over the
    four groups of the general task. There is no validation split."""
    from wsp import textlm
    folder = run_dir / "work" / free_model.DELIVERABLE
    rows = textlm.load("test")
    out, size, problems = free_model.run_lm(folder, {"test": ([textlm.prompt(r) for r in rows], [textlm.cap(r) for r in rows])})
    if problems:
        return {"valid": False, "problems": problems}
    shutil.copytree(folder, run_dir / free_model.DELIVERABLE)
    pieces = free_model.piece_vocab(folder)
    texts = free_model.piece_text(pieces, out["test"])
    (run_dir / "test_answers.json").write_text(json.dumps({"letters": out["test_first"][:, list(free_model.LETTER_COLUMNS)].tolist(), "texts": texts}))
    test = textlm.scores(rows, out["test_first"], texts)
    return {"valid": True, "problems": [], "metric": "mean of four groups", "test_accuracy": test["accuracy"],
            "practice_accuracy": None, "test": test, "validation": None, "words": len(pieces), **size}


def score_distill(spec: RunSpec, run_dir: Path) -> dict:
    """The data is checked and kept here; its students are trained and scored by baselines/distill_students.py."""
    folder = run_dir / "work" / distill.DELIVERABLE
    path = folder / distill.TRAIN_FILE
    if folder.is_symlink() or path.is_symlink() or not path.is_file():
        return {"valid": False, "problems": [f"no regular {distill.DELIVERABLE}/{distill.TRAIN_FILE}"]}
    rows, problems = distill.read(path)
    if not rows:
        return {"valid": False, "problems": problems or ["no valid question"]}
    shutil.copytree(folder, run_dir / distill.DELIVERABLE, symlinks=True)
    return {"valid": True, "problems": problems, "test_accuracy": None, "data": distill.stats(rows)}


def score(spec: RunSpec, run_dir: Path) -> dict:
    if lm_start.is_lm(spec.arch):
        return lm_score.score(spec.arch, run_dir)
    if olmo_start.is_olmo(spec.arch):
        return olmo_score.score(spec.arch, run_dir)
    if archs.ARCHS[spec.arch].deliverable == memory.DELIVERABLE:
        return score_facts(spec, run_dir)
    if spec.arch == "design-cifar":
        return score_design_images(spec, run_dir)
    if spec.arch.endswith(("-sq", "-sq1")):
        return score_spans(spec, run_dir)
    if spec.arch.endswith("-gen"):
        return score_gen(spec, run_dir)
    if spec.arch.endswith("-all"):
        return score_all(spec, run_dir)
    if spec.arch.endswith("-text"):
        return score_text(spec, run_dir)
    if archs.ARCHS[spec.arch].deliverable == distill.DELIVERABLE:
        return score_distill(spec, run_dir)
    if archs.ARCHS[spec.arch].deliverable == free_model.DELIVERABLE:
        return score_free(spec, run_dir)
    script = run_dir / "work" / "build_weights.py"
    if script.is_symlink() or not script.is_file():
        return {"valid": False, "problems": ["no regular build_weights.py"]}
    weights, problems = buildcheck.build(script, isolate=True)
    problems = problems or archs.validate(spec.arch, weights)
    if problems:
        return {"valid": False, "problems": problems}
    np.savez(run_dir / "weights.npz", **weights)
    w = evaluate.load_weights(spec.arch, run_dir / "weights.npz")
    t = archs.task_of(spec.arch)
    test = evaluate.evaluate(spec.arch, w, *t.load("test"))
    practice = evaluate.evaluate(spec.arch, w, *t.load("practice"))
    ok = test["accuracy"] is not None
    return {"valid": ok, "problems": [] if ok else [test.get("error")], "test_accuracy": test["accuracy"],
            "practice_accuracy": practice["accuracy"], "test_per_class": test.get("per_class"),
            "test_pred_hist": test.get("pred_hist"), "script": buildcheck.literal_stats(script.read_text())}


def trajectory(spec: RunSpec, run_dir: Path) -> list[dict]:
    """Every counted forward request with its test accuracy and calibrated cumulative output tokens."""
    meter = run_dir / "forward" / "log.jsonl"
    if not meter.exists():
        return []
    entries = [json.loads(line) for line in meter.read_text().splitlines()]
    transcript = run_dir / "transcript.jsonl"
    entries = tokens.calibrate(entries, tokens.progress(transcript), tokens.totals(transcript))
    test = archs.task_of(spec.arch).load("test")
    for e in entries:
        weights, problems = buildcheck.build(run_dir / "forward" / e["request"], isolate=True)
        problems = problems or archs.validate(spec.arch, weights)
        if problems:
            e["test_accuracy"] = None
            continue
        w = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in weights.items()}
        e["test_accuracy"] = evaluate.evaluate(spec.arch, w, *test)["accuracy"]
    return entries


def finalize(spec: RunSpec, run_dir: Path) -> dict:
    meta = json.loads((run_dir / "launch.json").read_text())
    result = {"run_id": spec.run_id, **asdict(spec), "profile": meta["profile"],
              "exit_code": meta.get("exit_code"), "ended": meta.get("ended"),
              "segments": len(meta.get("segments", [])), "init_tools": meta.get("init_tools"),
              "wall_s": round(meta.get("finished", time.time()) - meta["started"], 1)}
    script = run_dir / "work" / archs.ARCHS[spec.arch].deliverable
    steps = (
        ("score", lambda: score(spec, run_dir)),
        ("tokens", lambda: {"tokens": tokens.totals(run_dir / "transcript.jsonl")}),
        ("trajectory", lambda: {"trajectory": trajectory(spec, run_dir)}),
        ("audit", lambda: {"audit": audit.audit_run(run_dir, script.read_text() if script.is_file() else "")}),
    )
    for key, fn in steps:
        try:
            result.update(fn())
        except Exception:  # recorded in result.json so the run shows up as failed, not as unfinished
            result.setdefault("problems", []).append(f"{key} crashed")
            result["valid"] = False
            result[f"{key}_traceback"] = traceback.format_exc()
    traj = [e for e in result.get("trajectory", []) if e.get("accuracy") is not None]
    if traj:
        best = max(traj, key=lambda e: e["accuracy"])
        result["best_practice_request"] = {k: best.get(k) for k in
                                           ("request", "accuracy", "test_accuracy", "output_tokens_est")}
        result["forward_calls"] = len(result["trajectory"])
    proxy_log = run_dir / "proxy_log.jsonl"
    if proxy_log.exists():
        entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
        result["proxy_denied"] = sorted({e.get("host") or e.get("request", "") for e in entries if not e["allowed"]})
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def execute(spec: RunSpec, profile_key: str | None) -> dict:
    run_dir = prepare(spec, profile_key)
    proxy = AllowlistProxy(run_dir / "proxy.sock", ALLOWED_HOSTS, run_dir / "proxy_log.jsonl")
    try:
        problems = self_test(spec, run_dir)
        if (run_dir / "proxy_log.jsonl").exists():
            (run_dir / "proxy_log.jsonl").rename(run_dir / "proxy_log_selftest.jsonl")
        if problems:
            result = {"run_id": spec.run_id, **asdict(spec), "profile": profile_key, "valid": False,
                      "problems": ["self-test failed"] + problems}
            (run_dir / "result.json").write_text(json.dumps(result, indent=1))
            return result
        launch(spec, run_dir, profile_key)
    finally:
        proxy.close()
    return finalize(spec, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(archs.ARCHS))
    ap.add_argument("--data-access", required=True, choices=["blind", "open", "view"])
    ap.add_argument("--forward-budget", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--profile", choices=list(PROFILES), help="Claude profile")
    ap.add_argument("--writer", default="opus", choices=list(WRITERS))
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args()
    if a.data_access == "blind" and a.forward_budget:
        ap.error("blind runs have no data and no forward budget")
    spec = RunSpec(a.arch, a.data_access, a.forward_budget, a.rep, a.writer)
    if not a.profile and not a.finalize_only:
        ap.error(f"--profile is required for writer {a.writer}")
    torch.set_num_threads(int(os.environ.get("WSP_THREADS", "2")))
    res = finalize(spec, spec.run_dir) if a.finalize_only else execute(spec, a.profile)
    print(json.dumps({k: res.get(k) for k in ("run_id", "valid", "test_accuracy", "practice_accuracy",
                                               "exit_code", "wall_s", "problems")}))


if __name__ == "__main__":
    main()
