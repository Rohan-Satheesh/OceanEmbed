"""Lightweight experiment-tracking fallback (PRD Section 22.1b step 7, Section 30).

Uses Weights & Biases if WANDB_API_KEY is set; otherwise appends run metrics
to a CSV/JSON log under training/runs/. No signup required for the fallback.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).parent / "runs"


class RunLogger:
    def __init__(self, run_name: str, config: dict[str, Any]):
        self.run_name = run_name
        self.config = config
        self.use_wandb = bool(os.environ.get("WANDB_API_KEY"))
        self._wandb = None

        if self.use_wandb:
            import wandb

            self._wandb = wandb
            self._wandb.init(
                project=os.environ.get("WANDB_PROJECT", "oceanembed"),
                name=run_name,
                config=config,
            )
        else:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = RUNS_DIR / f"{run_name}.jsonl"
            with self.log_path.open("a") as f:
                f.write(json.dumps({"event": "run_start", "time": time.time(), "config": config}) + "\n")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.use_wandb:
            self._wandb.log(metrics, step=step)
        else:
            record = {"event": "metrics", "time": time.time(), "step": step, **metrics}
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def finish(self) -> None:
        if self.use_wandb:
            self._wandb.finish()


if __name__ == "__main__":
    # Dummy run to confirm logging works end to end (bootstrap step 7).
    logger = RunLogger("bootstrap_dummy_run", {"note": "Chunk 1 smoke test"})
    logger.log({"dummy_metric": 1.0}, step=0)
    logger.finish()
    print("Dummy run logged successfully.")
    if not os.environ.get("WANDB_API_KEY"):
        print(f"See {RUNS_DIR / 'bootstrap_dummy_run.jsonl'}")
