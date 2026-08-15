"""CLI: python -m mmdataflow run <config.yaml> [--limit N] [--resume]"""
from __future__ import annotations

import argparse
import sys

from . import ops  # noqa: F401  (import registers all operators)
from .core.pipeline import Pipeline
from .core.registry import list_ops


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mmdataflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a pipeline config")
    run.add_argument("config")
    run.add_argument("--limit", type=int, default=None,
                     help="only process the first N samples (smoke tests)")
    run.add_argument("--resume", action="store_true",
                     help="continue from the newest stage checkpoint")
    run.add_argument("--output-dir", default=None,
                     help="override output_dir from the config")
    run.add_argument("--workers", default=None,
                     help="process-pool size for parallel_safe operators "
                          "('auto' = cpu_count - 1); overrides num_workers")

    sub.add_parser("list-ops", help="list registered operators")

    args = parser.parse_args(argv)

    if args.cmd == "list-ops":
        for name, cls in sorted(list_ops().items()):
            doc = (cls.__doc__ or "").strip().splitlines()
            print(f"{name:28s} stage={cls.stage:11s} {doc[0] if doc else ''}")
        return 0

    Pipeline.from_yaml(args.config).run(
        limit=args.limit,
        resume=args.resume,
        work_dir=args.output_dir,
        num_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
