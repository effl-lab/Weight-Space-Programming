"""Token and progress accounting from a claude -p stream-json transcript.

Exact usage exists only in `result` events (one per session segment). For progress at an
arbitrary moment (e.g. each forward call) the transcript gives two running counts:
- thinking tokens: the sum of `thinking_tokens` event deltas (matched the exact total within
  0.2% on the first smoke run: 79,924 vs 79,836);
- visible output characters (text and tool inputs), converted to tokens after the run with the
  run's own tokens-per-character ratio from the exact totals.
"""
import json
from pathlib import Path


def events(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # last line may still be being written
                    return


def progress(transcript: Path) -> dict:
    think, chars, msgs, segments = 0, 0, set(), 0
    for e in events(transcript):
        t = e.get("type")
        if t == "system" and e.get("subtype") == "thinking_tokens":
            think += e.get("estimated_tokens_delta", 0)
        elif t == "system" and e.get("subtype") == "init":
            segments += 1
        elif t == "assistant":
            msgs.add(e["message"].get("id"))
            for b in e["message"].get("content", []):
                if b.get("type") == "text":
                    chars += len(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    chars += len(json.dumps(b.get("input", {})))
    return {"thinking_tokens_est": think, "visible_chars": chars, "assistant_messages": len(msgs),
            "segment": segments}


def _since_last(e: dict) -> dict:
    """Per-segment usage summed over models, from model_usage_since_last_result."""
    out = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
           "cache_creation_input_tokens": 0, "thinking_tokens": 0, "cost_usd_equiv": 0.0}
    names = {"inputTokens": "input_tokens", "outputTokens": "output_tokens",
             "cacheReadInputTokens": "cache_read_input_tokens",
             "cacheCreationInputTokens": "cache_creation_input_tokens",
             "thinkingTokens": "thinking_tokens", "costUSD": "cost_usd_equiv"}
    for usage in (e.get("model_usage_since_last_result") or {}).values():
        for src, dst in names.items():
            out[dst] += max(usage.get(src) or 0, 0)
    return out


def totals(transcript: Path) -> dict:
    """Exact usage over all segments, summing each result event's since-last-result usage.

    `semantics` records how the event's own `usage.output_tokens` relates to it: "per_segment" when
    they match, "cumulative" when usage equals the previous usage plus this segment, else "unverified".
    Do not sum `total_cost_usd` or `modelUsage`: both are cumulative over a resumed session.
    """
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
            "thinking_tokens", "cost_usd_equiv")
    out = {k: 0 for k in keys} | {"num_turns": 0, "duration_ms": 0, "segments": 0, "semantics": []}
    prev_usage_out = 0
    for e in events(transcript):
        if e.get("type") != "result":
            continue
        seg = _since_last(e)
        for k in keys:
            out[k] += seg[k]
        usage_out = (e.get("usage") or {}).get("output_tokens") or 0
        if usage_out == seg["output_tokens"]:
            out["semantics"].append("per_segment")
        elif usage_out == prev_usage_out + seg["output_tokens"]:
            out["semantics"].append("cumulative")
        else:
            out["semantics"].append("unverified")
        prev_usage_out = usage_out
        out["num_turns"] += max(e.get("num_turns") or 0, 0)
        out["duration_ms"] += max(e.get("duration_ms") or 0, 0)
        out["segments"] += 1
        cumulative = e.get("modelUsage") or {}
        last_cum = {"output_tokens": sum(max(u.get("outputTokens") or 0, 0) for u in cumulative.values()),
                    "cost_usd_equiv": sum(max(u.get("costUSD") or 0.0, 0.0) for u in cumulative.values())}
    # modelUsage and total_cost_usd are cumulative over a resumed session; the per-segment sums must
    # reproduce the last cumulative value (checked on a 4-segment run: 37,714 tokens, $1.259).
    if out["segments"]:
        out["matches_cumulative"] = (out["output_tokens"] == last_cum["output_tokens"]
                                     and abs(out["cost_usd_equiv"] - last_cum["cost_usd_equiv"]) < 1e-6)
    out["cost_basis"] = "cli"
    return out


def calibrate(entries: list[dict], final_progress: dict, exact: dict) -> list[dict]:
    """Adds `output_tokens_est` to each progress snapshot using this run's chars-to-tokens ratio."""
    visible_tokens = max(exact["output_tokens"] - exact["thinking_tokens"], 0)
    ratio = visible_tokens / final_progress["visible_chars"] if final_progress["visible_chars"] else 0.0
    return [e | {"output_tokens_est": round(e["thinking_tokens_est"] + ratio * e["visible_chars"])}
            for e in entries]
