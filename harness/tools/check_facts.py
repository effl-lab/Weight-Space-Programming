"""Check /srv/work/facts.txt against the line format and limits. Does not run the network.

Usage: python /srv/work/tools/check_facts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/srv/lib")
from wsp import memory  # noqa: E402

path = Path("/srv/work/facts.txt")
if not path.is_file():
    print("CHECK FAILED\nno /srv/work/facts.txt")
    sys.exit(1)
units, groups, problems = memory.parse(path.read_text())
if problems:
    print("CHECK FAILED")
    print("\n".join(problems))
    sys.exit(1)
print(f"CHECK OK\n{len(units)} memory units, {len(groups)} same-word groups")
