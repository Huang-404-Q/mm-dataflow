#!/usr/bin/env python3
"""Build the A/B/C fine-tuning sets that make the cleaning measurable.

The experiment answers one question: **does the cleaning actually help, or does
it just make the dataset smaller?** Three groups, trained with byte-identical
hyperparameters:

    A  dirty     every sample in the noisy pool
    B  cleaned   the samples the pipeline kept
    C  random    |B| samples drawn uniformly at random from A

C is the group that carries the argument. B beating A proves very little on its
own -- B is smaller, and smaller datasets change the number of optimizer steps,
the effective learning-rate schedule, and how much the model overfits. C holds
size constant and varies only *how* the samples were chosen, so:

    B vs C   same size, different selection  ->  the quality effect (the claim)
    A vs C   same selection, different size  ->  the quantity effect
    A vs B   both differ                     ->  the practical question

C must therefore be drawn from **A**, not from B. Sampling from B would just be
a smaller clean set and would measure nothing.

Fairness constraint: a sample whose image is missing locally is excluded from
all three groups, not just the one that happens to reference it. Otherwise the
groups differ by download luck as well as by selection.

    python scripts/make_sft_sets.py \\
        --annotated outputs/full/annotated.jsonl \\
        --image-root data/images --out-dir data/sft

Emits LLaMA-Factory sharegpt JSON plus a dataset_info.json and a manifest.json
recording sizes, step counts and noise composition, so the numbers in
docs/results.md can be traced back to an exact input.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core.sample import Sample, read_jsonl  # noqa: E402

# LLaVA marks the image slot with this token inside the first human turn.
IMAGE_TOKEN = "<image>"
ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def to_sharegpt(s: Sample, image_root: str) -> Optional[Dict[str, Any]]:
    """Convert one Sample to a LLaMA-Factory sharegpt record.

    Returns None when the record cannot train: no conversation, no image, or a
    turn order the collator would silently mis-pair. Dropping here rather than
    letting the trainer choke keeps a 3-run experiment from dying at hour two.
    """
    convs = s.conversations or []
    if not convs or not s.image_path:
        return None

    messages: List[Dict[str, str]] = []
    for c in convs:
        role = ROLE_MAP.get(c.get("from", ""))
        content = (c.get("value") or "").strip()
        if role is None or not content:
            return None
        messages.append({"role": role, "content": content})

    # Must alternate user/assistant and start with user, or the chat template
    # produces a mangled sequence instead of raising.
    if messages[0]["role"] != "user" or len(messages) < 2:
        return None
    for i, m in enumerate(messages):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None

    # Exactly one <image>, in the first user turn. LLaVA usually satisfies this,
    # but a stray token elsewhere makes the image-token count disagree with the
    # single image we pass, which fails deep inside the processor.
    total = sum(m["content"].count(IMAGE_TOKEN) for m in messages)
    if total == 0:
        messages[0]["content"] = f"{IMAGE_TOKEN}\n{messages[0]['content']}"
    elif total > 1 or IMAGE_TOKEN not in messages[0]["content"]:
        return None

    return {"messages": messages, "images": [os.path.join(image_root, s.image_path)]}


def compose(samples: List[Sample]) -> Dict[str, int]:
    """Noise-type histogram, for the manifest."""
    return dict(Counter(s.meta.get("noise_type", "clean") for s in samples))


def steps_for(n: int, batch: int, accum: int, epochs: float) -> int:
    return max(1, int(n * epochs / (batch * accum)))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annotated", required=True,
                   help="annotated.jsonl from a pipeline run over the noisy pool")
    p.add_argument("--image-root", default="data/images",
                   help="prepended to each image_path; LLaMA-Factory resolves "
                        "the result relative to its own cwd, so pass an "
                        "absolute path when training from another directory")
    p.add_argument("--out-dir", default="data/sft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--require-image", dest="require_image", action="store_true",
                   default=True,
                   help="drop records whose image is missing, from every group "
                        "(default; keeps the groups comparable)")
    p.add_argument("--no-require-image", dest="require_image", action="store_false")
    args = p.parse_args()

    samples = read_jsonl(args.annotated)
    print(f"[load] {len(samples)} annotated samples")

    if args.require_image:
        before = len(samples)
        samples = [
            s for s in samples
            if s.image_path and os.path.exists(os.path.join(args.image_root, s.image_path))
        ]
        if before != len(samples):
            # Applied before the split so all three groups lose the same records.
            print(f"[filter] dropped {before - len(samples)} records with a "
                  f"missing image (applied to all groups)")

    pool_a = samples
    pool_b = [s for s in samples if s.keep]
    if not pool_b:
        print("[error] the pipeline kept nothing -- check the thresholds first")
        return 1
    if len(pool_b) > len(pool_a):  # defensive; cannot happen, but the split
        print("[error] cleaned set larger than the pool")  # would be meaningless
        return 1

    rng = random.Random(args.seed)
    pool_c = rng.sample(pool_a, len(pool_b))

    os.makedirs(args.out_dir, exist_ok=True)
    groups = [("a_dirty", pool_a), ("b_cleaned", pool_b), ("c_random", pool_c)]
    manifest: Dict[str, Any] = {
        "source": args.annotated,
        "seed": args.seed,
        "image_root": args.image_root,
        "hparams": {
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
        },
        "groups": {},
    }

    info: Dict[str, Any] = {}
    for name, pool in groups:
        records, skipped = [], 0
        for s in pool:
            r = to_sharegpt(s, args.image_root)
            if r is None:
                skipped += 1
                continue
            records.append(r)

        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

        n = len(records)
        manifest["groups"][name] = {
            "file": f"{name}.json",
            "n": n,
            "unconvertible": skipped,
            "optimizer_steps": steps_for(n, args.per_device_batch, args.grad_accum,
                                         args.epochs),
            "noise_composition": compose(pool),
        }
        info[f"mmdf_{name}"] = {
            "file_name": f"{name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
        print(f"[write] {path}: {n} records"
              + (f" ({skipped} unconvertible)" if skipped else ""))

    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    g = manifest["groups"]
    print("\n[summary]")
    print(f"  {'group':<12}{'n':>8}{'steps':>8}   noise composition")
    for name, _ in groups:
        comp = ", ".join(f"{k}={v}" for k, v in sorted(g[name]['noise_composition'].items()))
        print(f"  {name:<12}{g[name]['n']:>8}{g[name]['optimizer_steps']:>8}   {comp}")

    # B and C are the load-bearing comparison, so make it loud if they drifted
    # apart -- unconvertible records can desynchronise sizes that were built equal.
    if g["b_cleaned"]["n"] != g["c_random"]["n"]:
        print(f"\n[warn] B={g['b_cleaned']['n']} != C={g['c_random']['n']}: the "
              f"size-controlled comparison is no longer exactly size-controlled. "
              f"Report the delta rather than hiding it.")
    else:
        print(f"\n  B and C are both {g['b_cleaned']['n']} records / "
              f"{g['b_cleaned']['optimizer_steps']} steps -- size is controlled, "
              f"only selection differs.")
    print(f"  A is {g['a_dirty']['n']} records / {g['a_dirty']['optimizer_steps']} "
          f"steps: more data AND more steps than B, so A-vs-B alone cannot "
          f"separate quality from quantity. That is what C is for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
