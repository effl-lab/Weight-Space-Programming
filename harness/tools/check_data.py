"""Check /srv/work/data/train.jsonl: the format of every line, the number of questions, exact duplicates, how the
answer positions are spread, and how many words fall outside the students' 8,000-word vocabulary. Nothing is trained
and no accuracy is reported.
Usage: python /srv/work/tools/check_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/srv/lib")
from wsp import distill  # noqa: E402

PATH = Path("/srv/work") / distill.DELIVERABLE / distill.TRAIN_FILE


def main() -> None:
    if not PATH.is_file():
        sys.exit(f"FAIL: no {PATH}")
    rows, problems = distill.read(PATH)
    for p in problems:
        print(p)
    if not rows:
        sys.exit("FAIL: no valid question")
    print(json.dumps(distill.stats(rows), indent=1))
    print("OK" if not problems else f"OK for the {len(rows)} valid lines; fix the lines listed above")


if __name__ == "__main__":
    main()
