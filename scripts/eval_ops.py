#!/usr/bin/env python3
"""Score the operators themselves against injected ground truth.

Reads ``annotated.jsonl`` (every sample, kept or dropped, with its scores and
drop reason) and reports:

  1. Overall pipeline precision / recall / F1 -- "dropped" is the positive
     prediction, ``meta.is_noise`` is the label.
  2. Per-operator attribution: what each operator caught, and how many clean
     samples it destroyed (false positives).
  3. Per-noise-type recall: which noise classes leak through, and via which
     operator they are caught.
  4. A threshold sweep for any score, so a cutoff can be justified from a P/R
     curve rather than picked by feel.

Usage:
    python scripts/eval_ops.py --input outputs/dev/annotated.jsonl
    python scripts/eval_ops.py --input outputs/dev/annotated.jsonl \
        --sweep clip_score --sweep-target mismatch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core.sample import Sample, read_jsonl  # noqa: E402


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def op_of(reason: str) -> str:
    return (reason or "").split(":")[0] or "kept"


def evaluate(samples: List[Sample]) -> dict:
    is_noise = {s.id: bool(s.meta.get("is_noise")) for s in samples}
    dropped = [s for s in samples if not s.keep]

    tp = sum(1 for s in dropped if is_noise[s.id])
    fp = len(dropped) - tp
    fn = sum(1 for s in samples if s.keep and is_noise[s.id])
    overall = prf(tp, fp, fn)

    # Per operator: precision is local (of what this op dropped, how much was
    # genuinely noise); recall is attributable (share of all noise it accounted for).
    total_noise = sum(1 for s in samples if is_noise[s.id])
    per_op_tp: Counter = Counter()
    per_op_fp: Counter = Counter()
    per_op_types: Dict[str, Counter] = defaultdict(Counter)
    for s in dropped:
        op = op_of(s.drop_reason)
        if is_noise[s.id]:
            per_op_tp[op] += 1
            per_op_types[op][s.meta.get("noise_type", "unknown")] += 1
        else:
            per_op_fp[op] += 1

    per_op = {}
    for op in sorted(set(per_op_tp) | set(per_op_fp)):
        t, f = per_op_tp[op], per_op_fp[op]
        per_op[op] = {
            "dropped": t + f,
            "true_noise": t,
            "false_positives": f,
            "precision": round(t / (t + f), 4) if t + f else 0.0,
            "share_of_all_noise": round(t / total_noise, 4) if total_noise else 0.0,
            "noise_types": dict(per_op_types[op]),
        }

    # Per noise type: recall and the operator that actually caught it.
    type_total: Counter = Counter()
    type_caught: Counter = Counter()
    type_by_op: Dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        nt = s.meta.get("noise_type")
        if not nt:
            continue
        type_total[nt] += 1
        if not s.keep:
            type_caught[nt] += 1
            type_by_op[nt][op_of(s.drop_reason)] += 1

    per_type = {
        nt: {
            "total": type_total[nt],
            "caught": type_caught[nt],
            "recall": round(type_caught[nt] / type_total[nt], 4),
            "caught_by": dict(type_by_op[nt]),
        }
        for nt in sorted(type_total)
    }

    return {
        "n_samples": len(samples),
        "n_noise": total_noise,
        "n_dropped": len(dropped),
        "overall": overall,
        "per_operator": per_op,
        "per_noise_type": per_type,
        "dedup_groups": dedup_groups(samples),
    }


def sweep(samples: List[Sample], score_key: str, target: str, steps: int = 20) -> List[dict]:
    """P/R at each candidate threshold for one score (keep if score >= t).

    Candidates are quantiles of the observed values, not evenly spaced points.
    Score distributions here are heavily skewed -- blur variance spans 0.3 to
    587 with most mass at the low end -- and uniform steps put nearly every
    candidate in the empty upper range, hiding the real decision boundary.
    """
    scored = [s for s in samples if score_key in s.scores]
    if not scored:
        return []
    if target == "any":
        labelled = [(s.scores[score_key], bool(s.meta.get("is_noise"))) for s in scored]
    else:
        # Restrict to the class this score is meant to catch, plus all clean
        # samples -- otherwise other noise classes distort the curve.
        labelled = [
            (s.scores[score_key], s.meta.get("noise_type") == target)
            for s in scored
            if not s.meta.get("is_noise") or s.meta.get("noise_type") == target
        ]
    if not labelled:
        return []
    vals = sorted(v for v, _ in labelled)
    n = len(vals)
    candidates = sorted({vals[min(n - 1, int(i / steps * n))] for i in range(steps + 1)}
                        | {vals[-1] + abs(vals[-1]) * 0.01 + 1e-6})
    rows = []
    for t in candidates:
        tp = sum(1 for v, y in labelled if v < t and y)
        fp = sum(1 for v, y in labelled if v < t and not y)
        fn = sum(1 for v, y in labelled if v >= t and y)
        rows.append({"threshold": round(t, 4), **prf(tp, fp, fn)})
    return rows


def dedup_groups(samples: List[Sample]) -> dict:
    """Group-level dedup scoring.

    Sample-level precision/recall systematically understates deduplication. When
    a sample and its copy both exist, collapsing the pair to one survivor is
    correct no matter which copy survives -- but the ground truth only labels the
    *copy* as noise, so keeping the copy and dropping the original scores a miss
    plus a false positive for an entirely correct action. With shuffled input
    that happens about half the time, capping apparent recall near 50%.

    The meaningful question is per group: did exactly one member survive?
    """
    by_id = {s.id: s for s in samples}
    groups: Dict[str, set] = defaultdict(set)
    for s in samples:
        src = s.meta.get("duplicate_source")
        if src and src in by_id:
            groups[src].update({src, s.id})

    collapsed = under = over = 0
    for gid, members in groups.items():
        survivors = sum(1 for m in members if by_id[m].keep)
        if survivors == 1:
            collapsed += 1
        elif survivors > 1:
            under += 1
        else:
            over += 1

    n = len(groups)
    return {
        "n_groups": n,
        "collapsed_to_one": collapsed,
        "under_deduplicated": under,
        "over_deleted": over,
        "collapse_rate": round(collapsed / n, 4) if n else 0.0,
    }


def to_markdown(res: dict, sweep_rows: List[dict], score_key: str, target: str) -> str:
    o = res["overall"]
    lines = [
        "# Operator evaluation",
        "",
        f"- samples: {res['n_samples']} ({res['n_noise']} noisy, "
        f"{res['n_dropped']} dropped)",
        f"- **overall precision {o['precision']:.1%} / recall {o['recall']:.1%} / "
        f"F1 {o['f1']:.3f}**  (tp={o['tp']} fp={o['fp']} fn={o['fn']})",
        "",
        "## Per operator",
        "",
        "| operator | dropped | true noise | false pos | precision | share of all noise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for op, d in res["per_operator"].items():
        lines.append(
            f"| `{op}` | {d['dropped']} | {d['true_noise']} | {d['false_positives']} | "
            f"{d['precision']:.1%} | {d['share_of_all_noise']:.1%} |"
        )
    lines += ["", "## Per noise type", "",
              "| noise type | total | caught | recall | caught by |",
              "|---|---:|---:|---:|---|"]
    for nt, d in res["per_noise_type"].items():
        by = ", ".join(f"{k}={v}" for k, v in sorted(d["caught_by"].items(),
                                                     key=lambda kv: -kv[1]))
        lines.append(
            f"| {nt} | {d['total']} | {d['caught']} | {d['recall']:.1%} | {by} |"
        )

    dg = res.get("dedup_groups") or {}
    if dg.get("n_groups"):
        lines += [
            "", "## Deduplication (group-level)", "",
            "Collapsing a duplicate group to one survivor is correct regardless of "
            "*which* member survives; the sample-level recall above penalises half "
            "of those correct actions and should not be read as dedup quality.",
            "",
            f"- duplicate groups: **{dg['n_groups']}**",
            f"- collapsed to exactly one survivor: **{dg['collapsed_to_one']}** "
            f"({dg['collapse_rate']:.1%})",
            f"- still holding >1 survivor (missed): {dg['under_deduplicated']}",
            f"- all members deleted (over-deletion): {dg['over_deleted']}",
        ]

    if sweep_rows:
        best = max(sweep_rows, key=lambda r: r["f1"])
        lines += ["", f"## Threshold sweep — `{score_key}` vs `{target}`", "",
                  f"Best F1 {best['f1']:.3f} at threshold **{best['threshold']}** "
                  f"(P {best['precision']:.1%} / R {best['recall']:.1%})", "",
                  "| threshold | precision | recall | f1 |", "|---:|---:|---:|---:|"]
        for r in sweep_rows:
            lines.append(
                f"| {r['threshold']} | {r['precision']:.1%} | {r['recall']:.1%} | "
                f"{r['f1']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="annotated.jsonl from a run")
    p.add_argument("--out-dir", default=None,
                   help="defaults to the directory of --input")
    p.add_argument("--sweep", default=None, help="score key to sweep, e.g. clip_score")
    p.add_argument("--sweep-target", default="any",
                   help="noise type the score targets, e.g. mismatch")
    p.add_argument("--sweep-steps", type=int, default=20)
    args = p.parse_args()

    samples = read_jsonl(args.input)
    if not any(s.meta.get("noise_type") for s in samples):
        print("[warn] no noise_type labels found -- run inject_noise.py first; "
              "precision/recall will be meaningless")
    res = evaluate(samples)
    rows = sweep(samples, args.sweep, args.sweep_target, args.sweep_steps) \
        if args.sweep else []

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "op_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": res, "sweep": rows}, f, ensure_ascii=False, indent=2)
    md = to_markdown(res, rows, args.sweep, args.sweep_target)
    with open(os.path.join(out_dir, "op_eval.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[write] {out_dir}/op_eval.md")


if __name__ == "__main__":
    main()
