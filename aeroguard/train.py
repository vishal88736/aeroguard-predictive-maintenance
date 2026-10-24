"""Explicit command-line baseline training/evaluation; no automatic downloads."""

import argparse
import json

from fleetpm.config import ARTIFACT_DIR, DATA_DIR, SUBSETS
from .baseline import train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run official endpoint evaluation after fitting; never used for promotion")
    args = parser.parse_args()
    for subset in args.subsets:
        config = {"subsets": args.subsets, "subset": subset, "iterations": args.iterations,
                  "seed": args.seed, "split_seed": args.split_seed, "stride": args.stride,
                  "track_mlflow": not args.no_mlflow}
        metadata = train_model(config, DATA_DIR, ARTIFACT_DIR,
                               lambda name, records, detail: print(f"{subset} {name}: {records} · {detail}", flush=True))
        print(json.dumps(metadata, indent=2), flush=True)
        if args.evaluate:
            print(json.dumps(evaluate_model(metadata["run_id"], DATA_DIR, ARTIFACT_DIR), indent=2), flush=True)


if __name__ == "__main__":
    main()
