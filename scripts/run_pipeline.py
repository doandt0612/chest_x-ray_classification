"""Re-run the whole pipeline, stage 1 to 7, with one command.

Calls the existing stage modules in order, each as `python -m cxr.<module>` with the same
configs, and stops at the first failing stage:
  download -> validate -> dedup -> split -> reference_stats -> train -> register -> evaluate

Training uses configs/train.yaml (the main model; the baseline only served to fix D6) and
writes runs/pipeline_<version>/. The trained run is registered as models/<version>/ and
evaluated once on test. Before any stage runs, the script checks the version format and
refuses to start if the run, the model version or its test results already exist.

The data stages do rewrite their outputs, including committed files (data/reference/stats.json,
reports/data_quality/*). On a re-run, `git diff` on them shows whether the data is reproduced.
If training crashes, its run folder stays and blocks a retry under the same version: delete
runs/pipeline_<version>/ first.

On a fresh clone, reports/evaluation/v1.0.0/ is already committed. To reproduce v1.0.0
there, write the new test results elsewhere and compare (D6, "Tái lập được"). Only
test_metrics.json and test_predictions.csv are produced; val and error-analysis reports are not:
  uv run python scripts/run_pipeline.py --version v1.0.0 --eval-out-dir reports/rerun/evaluation

Other usage:
  uv run python scripts/run_pipeline.py --version v1.1.0                    # all stages
  uv run python scripts/run_pipeline.py --version v1.1.0 --from train       # data already prepared
  uv run python scripts/run_pipeline.py --version v1.1.0 --to reference_stats
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from cxr.models.registry import VERSION_PATTERN
from cxr.utils.config import load_config

STAGES = ("download", "validate", "dedup", "split", "reference_stats", "train", "register", "evaluate")
DATA_STAGES = ("download", "validate", "dedup", "split", "reference_stats")


def select_stages(start: str, end: str) -> list[str]:
    i, j = STAGES.index(start), STAGES.index(end)
    if i > j:
        raise ValueError(f"--from {start} comes after --to {end}")
    return list(STAGES[i : j + 1])


def run_name(version: str) -> str:
    return f"pipeline_{version}"


def runs_dir(args: argparse.Namespace) -> Path:
    """Where train.py writes runs: train.output_dir of the train config."""
    return Path(load_config(args.train_config)["train"]["output_dir"])


def stage_command(stage: str, args: argparse.Namespace) -> list[str]:
    """The `python -m cxr...` command of one stage."""
    py = [sys.executable, "-m"]
    if stage in DATA_STAGES:
        return py + [f"cxr.data.{stage}", "--config", args.data_config]
    if stage == "train":
        cmd = py + [
            "cxr.models.train",
            "--data-config",
            args.data_config,
            "--train-config",
            args.train_config,
        ]
        cmd += ["--eval-config", args.eval_config, "--run-name", run_name(args.version)]
        return cmd + (["--seed", str(args.seed)] if args.seed is not None else [])
    if stage == "register":
        run_dir = runs_dir(args) / run_name(args.version)
        return py + [
            "cxr.models.registry",
            "--run",
            str(run_dir),
            "--version",
            args.version,
            "--models-dir",
            args.models_dir,
        ]
    if stage == "evaluate":
        return py + [
            "cxr.evaluation.evaluate",
            "--version", args.version,
            "--models-dir", args.models_dir,
            "--data-config", args.data_config,
            "--eval-config", args.eval_config,
            "--out-dir", args.eval_out_dir,
        ]  # fmt: skip
    raise ValueError(f"Unknown stage {stage!r}")


def check_no_overwrite(stages: list[str], args: argparse.Namespace) -> None:
    """Fail before any stage runs, not after hours of training: bad version or an existing output."""
    if "register" in stages and not VERSION_PATTERN.match(args.version):
        raise ValueError(f"Version must look like v1.0.0, got {args.version!r}")
    targets = []
    if "train" in stages:  # reads the train config only when training runs
        targets.append(runs_dir(args) / run_name(args.version))
    if "register" in stages:
        targets.append(Path(args.models_dir) / args.version)
    if "evaluate" in stages:
        targets.append(Path(args.eval_out_dir) / args.version / "test_metrics.json")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Would overwrite {existing}; pick another --version or --eval-out-dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True, help="model version to register, e.g. v1.1.0")
    parser.add_argument("--from", dest="start", default=STAGES[0], choices=STAGES)
    parser.add_argument("--to", dest="end", default=STAGES[-1], choices=STAGES)
    parser.add_argument("--seed", type=int, help="override train.seed")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--eval-out-dir", default="reports/evaluation")
    args = parser.parse_args()

    stages = select_stages(args.start, args.end)
    check_no_overwrite(stages, args)
    for n, stage in enumerate(stages, 1):
        cmd = stage_command(stage, args)
        print(f"\n=== [{n}/{len(stages)}] {stage}: {' '.join(cmd[1:])}", flush=True)
        start = time.perf_counter()
        result = subprocess.run(cmd, check=False)  # report the failing stage ourselves
        if result.returncode != 0:
            sys.exit(f"Stage {stage} failed (exit code {result.returncode}); later stages not run")
        print(f"=== {stage} done in {time.perf_counter() - start:.0f} s", flush=True)
    print(f"\nPipeline finished: {', '.join(stages)}")


if __name__ == "__main__":
    main()
