#!/usr/bin/env python3
"""Evaluate base + A/B/C on public benchmarks and write the comparison table.

    python scripts/run_eval.py --vlmevalkit ~/VLMEvalKit \\
        --data MMBench_DEV_EN_V11 MME --out docs/results.md

Wraps VLMEvalKit rather than reimplementing it: MMBench's circular-evaluation
protocol and MME's yes/no scoring both have details that are easy to get subtly
wrong, and a benchmark number nobody else can reproduce is worth nothing in a
write-up.

What this script owns is the part VLMEvalKit does not do:

  * generating one config that evaluates all four models identically, so a
    difference in the table cannot come from a difference in decoding settings
  * harvesting results whose file layout varies between VLMEvalKit versions
  * reporting **B vs C** as the headline, not B vs base

That last point is the whole experiment. B beating the base model only proves
fine-tuning does something. B beating C -- same size, same hyperparameters,
same seed, differing only in whether the pipeline chose the samples -- is the
only comparison that isolates data quality, and it is the one a reviewer will
look for. A run where B does *not* beat C is a real result and gets reported as
one; the honest negative is more defensible than a quietly dropped column.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Headline metric per benchmark. MME reports perception/cognition sub-scores on
# a different scale from MMBench accuracy, so they are never mixed in one column.
METRIC_COLUMN = {
    "MMBench": ("Overall", "acc"),
    "MMStar": ("Overall", "acc"),
    "MME": ("perception", "score"),
    "SEEDBench": ("Overall", "acc"),
    "AI2D": ("Overall", "acc"),
    "TextVQA": ("Overall", "acc"),
}

GROUP_LABEL = {
    "base": "base（未微调）",
    "a_dirty": "A 脏数据全量",
    "b_cleaned": "B 清洗后",
    "c_random": "C 同规模随机子集",
}


def metric_for(dataset: str) -> tuple:
    for key, val in METRIC_COLUMN.items():
        if dataset.upper().startswith(key.upper()):
            return val
    return ("Overall", "acc")


def build_config(models: Dict[str, str], datasets: List[str],
                 model_class: str, max_pixels: int) -> Dict[str, Any]:
    """One config, four models, identical decoding kwargs.

    Passing the same max_pixels used at training time matters: evaluating B at a
    higher resolution than it was trained on would flatter it for a reason that
    has nothing to do with data quality.
    """
    return {
        "model": {
            name: {
                "class": model_class,
                "model_path": path,
                "min_pixels": 3136,
                "max_pixels": max_pixels,
            }
            for name, path in models.items()
        },
        "data": {d: {"class": "auto", "dataset": d} for d in datasets},
    }


def harvest(work_dir: str, model: str, dataset: str) -> Optional[float]:
    """Pull the headline number out of whatever VLMEvalKit wrote.

    Layout has moved between versions (model/ subdir or not, _acc.csv or
    _score.csv or a json), so this globs rather than assuming a path. Returns
    None when nothing matches, which the report renders as a gap instead of
    silently dropping the row.
    """
    column, _ = metric_for(dataset)
    patterns = [
        os.path.join(work_dir, model, f"*{dataset}*.csv"),
        os.path.join(work_dir, f"{model}*{dataset}*.csv"),
        os.path.join(work_dir, "**", f"*{model}*{dataset}*.csv"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # Prefer the scored summary over raw per-item predictions.
    files.sort(key=lambda f: (0 if ("acc" in f or "score" in f) else 1, len(f)))

    for f in files:
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for row in rows:
            for k, v in row.items():
                if k and k.strip().lower() == column.lower():
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
    return None


def render(results: Dict[str, Dict[str, Optional[float]]], datasets: List[str],
           manifest: Optional[dict]) -> str:
    order = [g for g in ("base", "a_dirty", "b_cleaned", "c_random") if g in results]
    out = ["# 微调对照实验结果", ""]

    if manifest:
        g = manifest.get("groups", {})
        out += ["## 三组数据", "",
                "| 组 | 样本数 | 优化步数 | 说明 |", "|---|---|---|---|"]
        desc = {
            "a_dirty": "含噪声的全量池",
            "b_cleaned": "流水线保留的样本",
            "c_random": "从 A 中随机抽取，规模与 B 相同",
        }
        for name in ("a_dirty", "b_cleaned", "c_random"):
            if name in g:
                out.append(f"| {GROUP_LABEL[name]} | {g[name]['n']} | "
                           f"{g[name]['optimizer_steps']} | {desc[name]} |")
        out += ["", "超参完全一致（同一份 `train/qwen2_5vl_lora_sft.yaml`，"
                    "同一个 seed），三组之间只有数据不同。", ""]

    out += ["## 基准得分", "",
            "| 模型 | " + " | ".join(datasets) + " |",
            "|---|" + "---|" * len(datasets)]
    for gname in order:
        cells = []
        for d in datasets:
            v = results[gname].get(d)
            cells.append("—" if v is None else f"{v:.2f}")
        out.append(f"| {GROUP_LABEL.get(gname, gname)} | " + " | ".join(cells) + " |")
    out.append("")

    b, c, a = results.get("b_cleaned"), results.get("c_random"), results.get("a_dirty")
    if b and c:
        out += ["## 结论：B vs C（本实验的核心对比）", "",
                "两组样本数相同、超参相同、seed 相同，**唯一的差别是样本由算子选出还是随机选出**。",
                "", "| 基准 | B 清洗后 | C 随机 | B − C |", "|---|---|---|---|"]
        deltas = []
        for d in datasets:
            bv, cv = b.get(d), c.get(d)
            if bv is None or cv is None:
                out.append(f"| {d} | {'—' if bv is None else f'{bv:.2f}'} | "
                           f"{'—' if cv is None else f'{cv:.2f}'} | — |")
                continue
            delta = bv - cv
            deltas.append(delta)
            out.append(f"| {d} | {bv:.2f} | {cv:.2f} | **{delta:+.2f}** |")
        out.append("")
        if deltas:
            wins = sum(1 for d in deltas if d > 0)
            if wins == len(deltas):
                out.append(f"清洗后的数据在全部 {len(deltas)} 个基准上都优于同规模随机子集，"
                           f"说明收益来自**样本选择质量**而不是数据量变化。")
            elif wins == 0:
                out.append("**清洗后的数据没有优于同规模随机子集。** 这是一个负结果，"
                           "如实记录。可能的原因：噪声比例太低导致清洗掉的样本"
                           "本来就不影响训练；或阈值过严误杀了有效样本——"
                           "可以对照 `outputs/*/report.md` 的丢弃原因分布来判断。")
            else:
                out.append(f"{len(deltas)} 个基准中 {wins} 个 B 优于 C，结果不一致，"
                           f"不足以支撑「清洗必然有效」的结论。需要更大的评测集"
                           f"或更高的噪声比例来降低噪音。")
        out.append("")

    if a and b:
        out += ["## 参考：A vs B", "",
                "A 的数据更多、优化步数也更多，因此 A 与 B 的差异同时包含了"
                "数据量和数据质量两个变量，**不能单独用来论证清洗有效**——"
                "这正是设置 C 组的原因。", "",
                "| 基准 | A 脏数据 | B 清洗后 | B − A |", "|---|---|---|---|"]
        for d in datasets:
            av, bv = a.get(d), b.get(d)
            if av is None or bv is None:
                out.append(f"| {d} | {'—' if av is None else f'{av:.2f}'} | "
                           f"{'—' if bv is None else f'{bv:.2f}'} | — |")
            else:
                out.append(f"| {d} | {av:.2f} | {bv:.2f} | {bv - av:+.2f} |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vlmevalkit", default=os.environ.get("VLMEVALKIT_HOME", ""),
                   help="path to a VLMEvalKit checkout; omit to only re-harvest "
                        "and re-render from an existing --work-dir")
    p.add_argument("--sft-root", default="outputs/sft")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--model-class", default="Qwen2VLChat",
                   help="VLMEvalKit class name for the architecture")
    p.add_argument("--data", nargs="+", default=["MMBench_DEV_EN_V11", "MME"])
    p.add_argument("--groups", nargs="+",
                   default=["a_dirty", "b_cleaned", "c_random"])
    p.add_argument("--max-pixels", type=int, default=262144,
                   help="must match image_max_pixels used during training")
    p.add_argument("--work-dir", default="outputs/eval")
    p.add_argument("--manifest", default="data/sft/manifest.json")
    p.add_argument("--out", default="docs/results.md")
    p.add_argument("--skip-run", action="store_true",
                   help="do not invoke VLMEvalKit, just harvest and render")
    args = p.parse_args()

    models: Dict[str, str] = {"base": args.base_model}
    for g in args.groups:
        merged = os.path.join(args.sft_root, f"{g}_merged")
        if os.path.isdir(merged):
            models[g] = os.path.abspath(merged)
        else:
            print(f"[warn] {merged} not found; skipping {g}")

    os.makedirs(args.work_dir, exist_ok=True)
    cfg = build_config(models, args.data, args.model_class, args.max_pixels)
    cfg_path = os.path.join(args.work_dir, "vlmeval_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[eval] config -> {cfg_path} ({len(models)} models x {len(args.data)} datasets)")

    if not args.skip_run:
        if not args.vlmevalkit or not os.path.isdir(args.vlmevalkit):
            print("[error] --vlmevalkit points nowhere. Clone it first:\n"
                  "    git clone https://github.com/open-compass/VLMEvalKit\n"
                  "    pip install -e VLMEvalKit\n"
                  "Or pass --skip-run to render from existing results.")
            return 1
        cmd = [sys.executable, "run.py", "--config", os.path.abspath(cfg_path),
               "--work-dir", os.path.abspath(args.work_dir)]
        print(f"[eval] $ {' '.join(cmd)}  (cwd={args.vlmevalkit})")
        r = subprocess.run(cmd, cwd=args.vlmevalkit)
        if r.returncode != 0:
            # Harvest anyway: a crash on the last model should not throw away
            # the hours already spent on the earlier ones.
            print(f"[warn] VLMEvalKit exited {r.returncode}; harvesting what exists")

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name in models:
        results[name] = {d: harvest(args.work_dir, name, d) for d in args.data}
        got = {d: v for d, v in results[name].items() if v is not None}
        print(f"[eval] {name}: " + (", ".join(f"{d}={v:.2f}" for d, v in got.items())
                                    or "no results found"))

    manifest = None
    if os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest, encoding="utf-8"))

    md = render(results, args.data, manifest)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] -> {args.out}")

    missing = sum(1 for m in results.values() for v in m.values() if v is None)
    if missing:
        print(f"[warn] {missing} cells have no number; check {args.work_dir} "
              f"for what VLMEvalKit actually wrote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
