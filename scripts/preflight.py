#!/usr/bin/env python3
"""Check everything the A/B/C run needs, before the GPU meter starts.

Renting a GPU and then spending two hours on a template mismatch is the most
expensive kind of bug in this project. Every check here is one that has a real
chance of failing and would otherwise fail *inside* the trainer, minutes into a
run, with a stack trace from deep in the processor.

Runs in two places and degrades gracefully in both:

    # on the Mac, before renting anything -- data and template checks
    python scripts/preflight.py --sft-dir data/sft

    # on the GPU box, after pip install -- adds the CUDA and memory checks
    python scripts/preflight.py --sft-dir data/sft --model Qwen/Qwen2.5-VL-3B-Instruct

Exit code is the number of hard failures, so it composes with `&&` in a runner.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

OK, WARN, FAIL, SKIP = "  ok ", " warn", " FAIL", " skip"
_results: List[str] = []


def report(status: str, title: str, detail: str = "") -> None:
    _results.append(status)
    line = f"[{status}] {title}"
    print(line if not detail else f"{line}\n         {detail}")


def check_sft_dir(sft_dir: str, groups: List[str]) -> Dict[str, Any]:
    """dataset_info.json must register every group under the name the runner uses."""
    info_path = os.path.join(sft_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        report(FAIL, "dataset_info.json", f"missing at {info_path}; "
                                          "run scripts/make_sft_sets.py")
        return {}
    info = json.load(open(info_path, encoding="utf-8"))
    missing = [g for g in groups if f"mmdf_{g}" not in info]
    if missing:
        report(FAIL, "dataset registration",
               f"dataset_info.json has no entry for: {', '.join('mmdf_' + m for m in missing)}")
    else:
        report(OK, "dataset registration", f"mmdf_{{{','.join(groups)}}} registered")
    return info


def check_group_files(sft_dir: str, groups: List[str]) -> Dict[str, List[dict]]:
    loaded: Dict[str, List[dict]] = {}
    for g in groups:
        p = os.path.join(sft_dir, f"{g}.json")
        if not os.path.exists(p):
            report(FAIL, f"{g}.json", "missing")
            continue
        try:
            recs = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            report(FAIL, f"{g}.json", f"unparseable: {e}")
            continue
        if not recs:
            report(FAIL, f"{g}.json", "empty")
            continue
        loaded[g] = recs
        report(OK, f"{g}.json", f"{len(recs)} records")
    return loaded


def check_size_control(sft_dir: str, loaded: Dict[str, List[dict]]) -> None:
    """B and C must be the same size, or the central claim loses its control."""
    if "b_cleaned" not in loaded or "c_random" not in loaded:
        report(SKIP, "size control (B vs C)", "a group is missing")
        return
    nb, nc = len(loaded["b_cleaned"]), len(loaded["c_random"])
    if nb == nc:
        report(OK, "size control (B vs C)", f"both {nb} records")
    else:
        report(WARN, "size control (B vs C)",
               f"B={nb}, C={nc}. The quality-vs-quantity separation is weakened; "
               f"report the delta in results.md rather than omitting it.")


def check_images(loaded: Dict[str, List[dict]], probe: int, seed: int) -> None:
    """Sample real paths off disk. A path that resolves on the Mac but not on the
    GPU box (absolute vs relative) is a classic cross-machine failure."""
    rng = random.Random(seed)
    for g, recs in loaded.items():
        sample = rng.sample(recs, min(probe, len(recs)))
        bad = [r["images"][0] for r in sample
               if not r.get("images") or not os.path.exists(r["images"][0])]
        if bad:
            report(FAIL, f"images resolve ({g})",
                   f"{len(bad)}/{len(sample)} probed paths do not exist, e.g. {bad[0]}")
        else:
            report(OK, f"images resolve ({g})", f"{len(sample)} probed")


def check_image_token(loaded: Dict[str, List[dict]]) -> None:
    """Exactly one <image> per record, in the first user turn, matching the one
    image passed. A mismatch surfaces as an opaque shape error in the processor."""
    for g, recs in loaded.items():
        bad = 0
        for r in recs:
            n_tok = sum(m["content"].count("<image>") for m in r["messages"])
            if n_tok != len(r.get("images", [])):
                bad += 1
        if bad:
            report(FAIL, f"<image> token count ({g})",
                   f"{bad} records where the token count != the image count")
        else:
            report(OK, f"<image> token count ({g})", "all records consistent")


def check_llamafactory() -> None:
    """The runner passes overrides as `key=value`, which needs the OmegaConf
    merge path in read_args(). Older versions silently ignore them -- and
    silently training three identical runs on the same dataset is the worst
    possible failure, because it produces plausible numbers."""
    try:
        import llamafactory
    except ImportError:
        report(SKIP, "llama-factory", "not installed (expected on the Mac)")
        return
    ver = getattr(llamafactory, "__version__", "unknown")
    try:
        import inspect
        from llamafactory.hparams import parser as lf_parser
        src = inspect.getsource(lf_parser.read_args)
        if "from_cli" in src or "OmegaConf" in src:
            report(OK, "llama-factory", f"v{ver}, supports key=value overrides")
        else:
            report(FAIL, "llama-factory",
                   f"v{ver} does not merge CLI overrides into the YAML. "
                   f"run_experiment.sh would train all three groups on the same "
                   f"dataset. Upgrade, or split the config per group.")
    except Exception as e:
        report(WARN, "llama-factory", f"v{ver}, could not verify override support: {e}")


def check_template(model: str, loaded: Dict[str, List[dict]]) -> None:
    """Push one real record through the actual processor. This is the check that
    catches template/token problems, and it only needs the ~1MB processor
    config, not the weights -- so it runs on the Mac."""
    if not model:
        report(SKIP, "chat template", "pass --model to enable")
        return
    if not loaded:
        report(SKIP, "chat template", "no records loaded")
        return
    try:
        from transformers import AutoProcessor
    except ImportError:
        report(SKIP, "chat template", "transformers not installed")
        return
    try:
        proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        report(WARN, "chat template", f"could not load processor for {model}: {e}")
        return

    rec = next(iter(loaded.values()))[0]
    try:
        from PIL import Image
        img = Image.open(rec["images"][0]).convert("RGB")
        msgs = [{"role": m["role"],
                 "content": ([{"type": "image"}] if "<image>" in m["content"] else [])
                            + [{"type": "text",
                                "text": m["content"].replace("<image>", "").strip()}]}
                for m in rec["messages"]]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        batch = proc(text=[text], images=[img], return_tensors="pt")
        n_tok = batch["input_ids"].shape[1]
        report(OK, "chat template", f"1 record -> {n_tok} tokens "
                                    f"(cutoff_len is 2048; raise it if this is close)")
        if n_tok > 2048:
            report(WARN, "cutoff_len",
                   f"{n_tok} tokens exceeds cutoff_len=2048, so this record would "
                   f"be truncated mid-answer")
    except Exception as e:
        report(FAIL, "chat template",
               f"a real record failed to process: {type(e).__name__}: {e}")


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        report(SKIP, "gpu", "torch not installed")
        return
    if not torch.cuda.is_available():
        backend = "mps" if torch.backends.mps.is_available() else "cpu"
        report(SKIP, "gpu", f"no CUDA device (running on {backend})")
        return
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 3.75B params in bf16 = 7.5GB, plus LoRA state, activations and allocator
    # slack. Below ~20GB the config needs image_max_pixels or batch size cut.
    if gb >= 20:
        report(OK, "gpu", f"{name}, {gb:.0f}GB -- config fits as written")
    else:
        report(WARN, "gpu", f"{name}, {gb:.0f}GB is tight for 3B bf16 + activations; "
                            f"drop per_device_train_batch_size to 1 or "
                            f"image_max_pixels to 200704")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sft-dir", default="data/sft")
    p.add_argument("--model", default="",
                   help="enables the chat-template check; only the processor "
                        "config is downloaded, not the weights")
    p.add_argument("--groups", nargs="+",
                   default=["a_dirty", "b_cleaned", "c_random"])
    p.add_argument("--probe", type=int, default=200,
                   help="how many image paths to stat per group")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"preflight: {args.sft_dir}\n")
    check_sft_dir(args.sft_dir, args.groups)
    loaded = check_group_files(args.sft_dir, args.groups)
    check_size_control(args.sft_dir, loaded)
    check_images(loaded, args.probe, args.seed)
    check_image_token(loaded)
    check_llamafactory()
    check_template(args.model, loaded)
    check_gpu()

    fails = _results.count(FAIL)
    warns = _results.count(WARN)
    skips = _results.count(SKIP)
    print(f"\n{len(_results) - fails - warns - skips} ok, {warns} warn, "
          f"{skips} skip, {fails} FAIL")
    if fails:
        print("Fix the failures before starting a paid GPU.")
    return fails


if __name__ == "__main__":
    sys.exit(main())
