"""Run the network built by /srv/work/build_weights.py on the 512 practice images.

The harness builds and evaluates the script outside this sandbox and reports accuracy,
per-class accuracy and the confusion matrix. Each successful call uses one unit of the forward
budget; calls beyond the budget are refused. Usage: python /srv/work/tools/forward.py
"""
import json
import sys
import time
import uuid
from pathlib import Path

QUEUE = Path("/srv/work/.forward")
TIMEOUT_S = 1800

QUEUE.mkdir(exist_ok=True)
rid = f"{time.time():.6f}_{uuid.uuid4().hex[:8]}"
tmp = QUEUE / f"{rid}.part"
tmp.write_text(Path("/srv/work/build_weights.py").read_text())
tmp.rename(QUEUE / f"{rid}.req")
resp = QUEUE / f"{rid}.resp"
deadline = time.time() + TIMEOUT_S
while not resp.exists():
    if time.time() > deadline:
        print("no response from the harness"); sys.exit(1)
    time.sleep(0.5)
print(json.loads(resp.read_text())["text"])
