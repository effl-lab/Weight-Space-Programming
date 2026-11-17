"""Run identity and profile paths, importable without torch (the dispatcher uses this)."""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from wsp import lineage, lm_start, olmo_start, paths

RUNS_DIR = paths.DATA_ROOT / "runs"
WALL_LIMIT_S = 3 * 3600
TOKEN_MARGIN_S = 15 * 60
# A run starts only on a profile whose access token outlives it. Finished Opus runs took at most 2.35 h of wall time,
# scoring included (2026-11-12), so 2.5 h is asked for rather than the full WALL_LIMIT_S.
TOKEN_NEED_S = int(2.5 * 3600)
WRITERS = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-2"}
# Reasoning-effort variants of each writer (<writer>_<effort>): the same model at another --effort; the plain names
# run at EFFORT.
EFFORT = "high"
EFFORT_VARIANTS = ("low", "medium", "xhigh")
WRITER_EFFORT = {f"opus_{e}": e for e in EFFORT_VARIANTS}
WRITERS |= {v: WRITERS[v.rsplit("_", 1)[0]] for v in WRITER_EFFORT}
# Extra sandbox environment per writer. Fable 5.2 at high effort thinks past the CLI's default 32,000-token response
# limit before its first tool call (three blind CIFAR runs ended that way, kept in runs_fable_cap32k).
WRITER_ENV = {"fable": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}}
# Claude Code config directories that hold a logged-in profile: WSP_CLAUDE_PROFILES=<key>=<dir>[,<key>=<dir>...]
PROFILES = {k: Path(v) for k, _, v in (p.partition("=") for p in os.environ.get("WSP_CLAUDE_PROFILES", "").split(",")
                                        if p)}


@dataclass(frozen=True)
class RunSpec:
    arch: str
    data_access: str  # "blind" | "open" (labels, class statistics allowed) | "view" (no numeric labels)
    forward_budget: int
    rep: int
    writer: str = "opus"  # key in WRITERS

    @property
    def run_id(self) -> str:
        return f"{self.arch}_{self.data_access}_K{self.forward_budget}_r{self.rep}"

    @property
    def run_dir(self) -> Path:
        """Opus runs of the weight-file architectures live in RUNS_DIR. Another writer, and the MMLU, ARC, MathBench
        and released-LM architectures (other deliverables and result fields), get their own roots, so analyses of
        RUNS_DIR stay one-writer and one result format."""
        kind = ("_arc" if self.arch.endswith("-arc") else "_mb" if self.arch.endswith("-mb")
                else "_mm" if self.arch.endswith("-mm")
                else "_sq" if self.arch.endswith("-sq") else "_sq1" if self.arch.endswith("-sq1")
                else "_gen" if self.arch.endswith("-gen")
                else "_all" if self.arch.endswith("-all")
                else "_text" if self.arch.endswith("-text")
                else "_memory" if self.arch.startswith("mem-") or lineage.parse(self.arch)
                else "_lm" if lm_start.is_lm(self.arch) else "_olmo" if olmo_start.is_olmo(self.arch) else "")
        root = "runs" + kind + ("" if self.writer == "opus" else f"_{self.writer}")
        return RUNS_DIR.with_name(root) / self.run_id


def slot_weight(arch: str) -> int:
    """Dispatcher slots a run takes: two for a released LM of lm_start.BIG_PARAMS or more, whose check tool holds the
    whole model in memory on the host."""
    big = (lm_start.is_lm(arch) and lm_start.manifest(arch)["parameters"] >= lm_start.BIG_PARAMS
           or olmo_start.is_olmo(arch))
    return 2 if big else 1


def token_hours_left(profile_key: str) -> float:
    oauth = json.loads((PROFILES[profile_key] / ".credentials.json").read_text())["claudeAiOauth"]
    return (oauth["expiresAt"] / 1000 - time.time()) / 3600


def token_ok(profile_key: str) -> bool:
    return token_hours_left(profile_key) * 3600 >= TOKEN_NEED_S
