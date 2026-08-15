"""Per-operator statistics and Markdown/JSON report generation.

Every operator must produce three numbers (drop rate, score distribution,
elapsed time) -- that requirement is what turns the pipeline from a script into
something observable. When ground-truth ``noise_type`` labels are present, the
report also breaks each operator's drops down by noise type for free.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.sample import Sample


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)

    def q(p: float) -> float:
        return vs[min(n - 1, int(p * n))]

    return {
        "min": round(vs[0], 4),
        "p10": round(q(0.10), 4),
        "p50": round(q(0.50), 4),
        "p90": round(q(0.90), 4),
        "max": round(vs[-1], 4),
        "mean": round(sum(vs) / n, 4),
    }


@dataclass
class OpStat:
    name: str
    stage: str
    params: dict = field(default_factory=dict)
    n_in: int = 0
    n_dropped: int = 0
    elapsed_s: float = 0.0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    dropped_by_noise: Dict[str, int] = field(default_factory=dict)
    score_dist: Dict[str, float] = field(default_factory=dict)

    @property
    def n_out(self) -> int:
        return self.n_in - self.n_dropped

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / self.n_in if self.n_in else 0.0

    @property
    def throughput(self) -> float:
        return self.n_in / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "stage": self.stage,
            "params": self.params,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "n_dropped": self.n_dropped,
            "drop_rate": round(self.drop_rate, 4),
            "elapsed_s": round(self.elapsed_s, 3),
            "throughput_sps": round(self.throughput, 2),
            "drop_reasons": self.drop_reasons,
            "score_dist": self.score_dist,
        }
        if self.dropped_by_noise:
            d["dropped_by_noise"] = self.dropped_by_noise
        return d


class RunReport:
    def __init__(self, config_name: str, device: str):
        self.config_name = config_name
        self.device = device
        self.stats: List[OpStat] = []
        self.n_input = 0
        self.n_output = 0
        self.total_elapsed = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def record(
        self,
        op,
        samples_before_alive: int,
        samples: List[Sample],
        elapsed: float,
    ) -> OpStat:
        """Snapshot an operator's effect. Called by Pipeline after each stage."""
        just_dropped = [
            s for s in samples if not s.keep and (s.drop_reason or "").startswith(op.name)
        ]
        stat = OpStat(
            name=op.name,
            stage=op.stage,
            params=op.describe(),
            n_in=samples_before_alive,
            n_dropped=len(just_dropped),
            elapsed_s=elapsed,
            drop_reasons=dict(Counter(s.drop_reason for s in just_dropped)),
            dropped_by_noise=dict(
                Counter(s.noise_type or "clean" for s in just_dropped)
            ),
        )
        key = getattr(op, "score_key", None)
        if key:
            stat.score_dist = _percentiles(
                [s.scores[key] for s in samples if key in s.scores]
            )
        self.stats.append(stat)
        return stat

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "device": self.device,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "overall_drop_rate": round(
                1 - self.n_output / self.n_input, 4
            )
            if self.n_input
            else 0.0,
            "total_elapsed_s": round(self.total_elapsed, 2),
            "throughput_sps": round(
                self.n_input / self.total_elapsed, 2
            )
            if self.total_elapsed
            else 0.0,
            "embedding_cache": {"hits": self.cache_hits, "misses": self.cache_misses},
            "operators": [s.to_dict() for s in self.stats],
        }

    def to_markdown(self) -> str:
        d = self.to_dict()
        lines = [
            f"# Pipeline report — `{self.config_name}`",
            "",
            f"- device: `{self.device}`",
            f"- input / output: **{d['n_input']} → {d['n_output']}** "
            f"(overall drop rate {d['overall_drop_rate']:.1%})",
            f"- total time: {d['total_elapsed_s']}s "
            f"({d['throughput_sps']} samples/s)",
            f"- embedding cache: {self.cache_hits} hits / {self.cache_misses} misses",
            "",
            "## Per-operator",
            "",
            "| operator | stage | in | out | drop rate | time (s) | samples/s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for s in d["operators"]:
            lines.append(
                f"| `{s['name']}` | {s['stage']} | {s['n_in']} | {s['n_out']} | "
                f"{s['drop_rate']:.1%} | {s['elapsed_s']} | {s['throughput_sps']} |"
            )

        dists = [s for s in d["operators"] if s["score_dist"]]
        if dists:
            lines += ["", "## Score distributions", "",
                      "| operator | min | p10 | p50 | p90 | max | mean |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for s in dists:
                sd = s["score_dist"]
                lines.append(
                    f"| `{s['name']}` | {sd['min']} | {sd['p10']} | {sd['p50']} | "
                    f"{sd['p90']} | {sd['max']} | {sd['mean']} |"
                )

        noisy = [s for s in d["operators"] if s.get("dropped_by_noise")]
        if noisy:
            lines += ["", "## Drops by ground-truth noise type", "",
                      "(`clean` = false positives, i.e. good samples discarded)", "",
                      "| operator | breakdown |", "|---|---|"]
            for s in noisy:
                bits = ", ".join(
                    f"{k}: {v}" for k, v in sorted(
                        s["dropped_by_noise"].items(), key=lambda kv: -kv[1]
                    )
                )
                lines.append(f"| `{s['name']}` | {bits} |")
        return "\n".join(lines) + "\n"

    def save(self, work_dir: str) -> None:
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(work_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
