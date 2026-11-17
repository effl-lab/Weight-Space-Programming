"""Summarize a writer transcript (claude -p stream-json) and flag rule-relevant patterns.

Flags are leads for a transcript review, not verdicts: e.g. `svd` may appear in an analysis
that never touches the weights.
"""
import json
import re
from collections import Counter
from pathlib import Path

from wsp import paths

PATTERNS = {
    "fitting": r"lstsq|pinv|linalg\.solve|linalg\.inv|\.backward\(|torch\.optim|sklearn|\.fit\(|kmeans|"
               r"np\.linalg\.svd|linalg\.eig|\bPCA\b|polyfit|scipy\.optimize",
    "search_loop": r"random\.(?:randn|normal|uniform).*\bbest\b|hill.?climb|evolution|mutate|perturb",
    "forward_reimpl": r"conv2d|F\.linear|archs\.forward|\.forward\(|max_pool|softmax\(|evaluate\.evaluate|"
                      r"build_weights.*train\.npz|weights\.npz",
    "outside_paths": "|".join(re.escape(str(p)) for p in (paths.DATA_ROOT, Path.home())) +
                     r"|huggingface|test\.npz|/srv/lib|"
                     r"/proc/|/run/|\.forward/",
    "own_api": r"\.credentials|api\.anthropic\.com|import anthropic|from anthropic|x-api-key|Authorization",
    "network": r"urllib|requests\.|http://|https://|wget|curl|torchvision\.datasets|load_dataset",
}


def _iter_events(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Checked only in code the writer ran or wrote (Bash, Write, Edit), not in Read calls: viewing the
# class grids with Read is allowed, touching them or labels from code is not.
CODE_PATTERNS = {
    "label_stats": r"\by\s*==|\[y\b|\by\]|bincount|\[.y.\]|views\b|zlib|\.png\b|imread|Image\.open|"
                   r"PIL|imageio|cv2|matplotlib\.image|read_bytes|frombuffer",
}
# Grep/Glob on the class views counts or filters labelled examples without Read; reading them is the only allowed use.
SEARCH_PATTERNS = {"views_search": r"views"}
LITERAL = re.compile(r"-?\d*\.\d{2,}")
NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


ROW_MIN = 4


def _numbers(text: str) -> list[str]:
    """Number tokens in a canonical form, so `.70`, `0.70` and `0.7` compare equal."""
    out = []
    for tok in NUMBER.findall(text):
        try:
            out.append(f"{float(tok):.6g}")
        except ValueError:
            continue
    return out


def literal_provenance(script: str, outputs: str) -> dict:
    """Rows of the build script holding 4+ numbers whose number sequence appears, in order and
    contiguously, in earlier tool output (a printed statistics table typed back in).

    Single literals are not counted: short decimals collide by chance (a random 2-decimal value is
    found in a long transcript most of the time), while a run of 4+ numbers essentially never does.
    """
    out_nums = _numbers(outputs)
    grams = {tuple(out_nums[i:i + ROW_MIN]) for i in range(len(out_nums) - ROW_MIN + 1)}
    rows = [nums for line in script.splitlines()
            if len(nums := _numbers(line.split("#", 1)[0])) >= ROW_MIN and LITERAL.search(line)]
    copied = [r for r in rows if all(tuple(r[i:i + ROW_MIN]) in grams for i in range(len(r) - ROW_MIN + 1))]
    return {"numeric_rows": len(rows), "rows_copied_from_tool_output": len(copied),
            "examples": [" ".join(r) for r in copied[:5]]}


def audit_run(run_dir: Path, script: str) -> dict:
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        return {"error": "no transcript"}
    tools, bash_cmds, code_blobs, code_only, searches, outputs, errors = Counter(), [], [], [], [], [], 0
    final = {}
    for ev in _iter_events(path):
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                name, inp = block.get("name"), block.get("input", {})
                tools[name] += 1
                if name == "Bash":
                    bash_cmds.append(inp.get("command", ""))
                    code_blobs.append(inp.get("command", ""))
                    code_only.append(inp.get("command", ""))
                elif name in ("Write", "Edit"):
                    code_blobs.append(inp.get("content", "") + inp.get("new_string", ""))
                    code_only.append(inp.get("content", "") + inp.get("new_string", ""))
                elif name in ("Read", "Glob", "Grep"):
                    code_blobs.append(json.dumps(inp))
                    if name != "Read":
                        searches.append(json.dumps(inp))
        elif ev.get("type") == "user":
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = block.get("content")
                    outputs.append(c if isinstance(c, str) else json.dumps(c))
                    errors += bool(block.get("is_error"))
        elif ev.get("type") == "result":
            final = ev
    text = "\n".join(code_blobs)
    flags = {k: sorted(set(m.group(0) for m in re.finditer(p, text)))[:20] for k, p in PATTERNS.items()}
    code = "\n".join(code_only)
    flags |= {k: sorted(set(m.group(0) for m in re.finditer(p, code)))[:20] for k, p in CODE_PATTERNS.items()}
    search = "\n".join(searches)
    flags |= {k: sorted(set(m.group(0) for m in re.finditer(p, search)))[:20] for k, p in SEARCH_PATTERNS.items()}
    forward_cmds = sum(bool(re.search(r"python3?\s+\S*tools/forward\.py", c)) for c in bash_cmds)
    meter = run_dir / "forward" / "log.jsonl"
    return {
        "tool_calls": dict(tools), "tool_errors": errors, "bash_commands": len(bash_cmds),
        "forward_commands_transcript": forward_cmds,
        "forward_calls_counted": sum(1 for _ in meter.open()) if meter.exists() else 0,
        "literal_provenance": literal_provenance(script, "\n".join(outputs)),
        "flags": {k: v for k, v in flags.items() if v},
        "is_error": final.get("is_error"),
        "final_message": (final.get("result") or "")[:3000],
    }
