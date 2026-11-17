import json
import time

from harness import dispatch, spec


def test_limit_reset_is_the_next_occurrence():
    r = dispatch.limit_reset("You've hit your 5-hour limit · resets 12pm (UTC)")
    assert r is not None and time.localtime(r).tm_hour == 12 and 0 < r - time.time() <= 86400
    assert dispatch.limit_reset("Request timed out") is None


def _run(tmp_path, monkeypatch, valid: bool, result: str, is_error: bool) -> spec.RunSpec:
    monkeypatch.setattr(spec, "RUNS_DIR", tmp_path / "runs")
    s = spec.RunSpec("free-arc", "blind", 0, 0, "opus")
    s.run_dir.mkdir(parents=True)
    (s.run_dir / "result.json").write_text(json.dumps({"valid": valid}))
    event = {"type": "result", "is_error": is_error, "result": result}
    (s.run_dir / "transcript.jsonl").write_text(json.dumps(event) + "\n")
    return s


def test_cut_by_usage_limit_is_retried_even_when_scored(tmp_path, monkeypatch):
    s = _run(tmp_path, monkeypatch, True, "You've hit your 5-hour limit · resets 12pm (UTC)", True)
    assert dispatch.api_failure(s) == "hit your"


def test_cut_by_timeout_is_retried_even_when_scored(tmp_path, monkeypatch):
    assert dispatch.api_failure(_run(tmp_path, monkeypatch, True, "Request timed out", True)) == "Request timed out"


def test_finished_run_is_not_retried(tmp_path, monkeypatch):
    assert dispatch.api_failure(_run(tmp_path, monkeypatch, True, "Done.", False)) is None
    assert dispatch.api_failure(_run(tmp_path / "b", monkeypatch, False, "Done.", False)) is None
    assert dispatch.api_failure(_run(tmp_path / "c", monkeypatch, True, "No rate limit was hit.", False)) is None
