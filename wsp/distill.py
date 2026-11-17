"""distill-arc: the writer writes training data for ARC instead of a model; standard small networks are then trained on
it by gradient descent (baselines/distill_students.py).

The writer delivers the folder DELIVERABLE with TRAIN_FILE, one JSON object per line:
  {"question": str, "options": [str, str, str, str], "answer": 0-3}
plus anything it used to make that file. While the writer works, the harness copies TRAIN_FILE every SNAPSHOT_S
seconds (when it changed) to run_dir/snapshots/<elapsed seconds>_<digest>.jsonl, so that students can be trained on the data
the writer had written by a given cost.
"""
import json
from collections import Counter
from pathlib import Path

from wsp import free_model

DELIVERABLE = "data"
TRAIN_FILE = "train.jsonl"
VOCAB = 8000  # the students' vocabulary: the most frequent words of the questions and options
SNAPSHOT_S = 600
# distill100-arc: the same task as distill-arc, ended by the LLM cost (API-equivalent USD) instead of the output-token
# target, with a longer wall limit, so that students can be read against cost up to the LLM-designed lineages' cost
ARCHS = ("distill-arc", "distill100-arc")
COST_CAP_USD = {"distill100-arc": 100.0}
WALL_LIMIT_S = {"distill100-arc": 10 * 3600}
MAX_CONTINUATIONS = {"distill100-arc": 400}  # 40 ended r0 at $29: its continuations are short


def read(path: Path) -> tuple[list[dict], list[str]]:
    """The valid rows of a train.jsonl and the problems found (line numbers 1-based, at most 20 listed)."""
    rows, problems = [], []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            ok = (isinstance(r, dict) and isinstance(r.get("question"), str) and isinstance(r.get("options"), list)
                  and len(r["options"]) == 4 and all(isinstance(o, str) for o in r["options"])
                  and r.get("answer") in (0, 1, 2, 3))
        except json.JSONDecodeError:
            ok = False
        if ok:
            rows.append({"question": r["question"], "options": r["options"], "answer": r["answer"]})
        elif len(problems) < 20:
            problems.append(f"line {i}: not a question object with four string options and an answer 0-3")
    return rows, problems


def vocab(rows: list[dict]) -> list[str]:
    counts = Counter(w for r in rows for t in [r["question"], *r["options"]] for w in free_model.words(t))
    return [w for w, _ in counts.most_common(VOCAB)]


def stats(rows: list[dict]) -> dict:
    v = set(vocab(rows))
    words = [w for r in rows for t in [r["question"], *r["options"]] for w in free_model.words(t)]
    keys = Counter(json.dumps([r["question"], r["options"]]) for r in rows)
    return {"questions": len(rows), "exact_duplicates": sum(c - 1 for c in keys.values()),
            "answer_position": [sum(r["answer"] == k for r in rows) for k in range(4)],
            "words": len(words), "word_types": len(set(words)),
            "words_outside_vocab": sum(w not in v for w in words)}
