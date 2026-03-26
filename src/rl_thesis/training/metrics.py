"""CSV-based metrics logging for training runs."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any


_EVAL_FIELDS = [
    "step",
    "episode",
    "eval_reward",
    "eval_survival",
    "eval_food_eaten",
    "eval_damage_taken",
    "eval_death_rate",
    "epsilon",
    "loss",
    "wall_time",
]

_EPISODE_FIELDS = [
    "step",
    "episode",
    "reward",
    "survival",
    "food_eaten",
    "damage_taken",
    "terminated",
]


class MetricsLogger:
    """Append-only CSV logger for eval and episode metrics."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._eval_path = log_dir / "eval.csv"
        self._episode_path = log_dir / "episodes.csv"
        self._start_time = time.monotonic()

        # Count existing rows before opening (for resume support).
        self._episode_count = self._count_rows(self._episode_path)

        self._eval_writer = self._open_csv(self._eval_path, _EVAL_FIELDS)
        self._episode_writer = self._open_csv(self._episode_path, _EPISODE_FIELDS)

        self._best_eval_reward = float("-inf")
        self._best_survival = 0
        self._latest_eval: Dict[str, Any] = {}

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @staticmethod
    def _count_rows(path: Path) -> int:
        """Count data rows in an existing CSV (0 if missing or empty)."""
        if not path.exists() or path.stat().st_size == 0:
            return 0
        with open(path) as f:
            return max(0, sum(1 for _ in f) - 1)  # subtract header

    def _open_csv(self, path: Path, fields: list[str]) -> csv.DictWriter:
        handle = open(path, "a", newline="")
        writer = csv.DictWriter(handle, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
            handle.flush()
        self._handles = getattr(self, "_handles", [])
        self._handles.append(handle)
        return writer

    def log_eval(self, step: int, episode: int, eval_results: Dict[str, float],
                 epsilon: float, loss: float) -> None:
        row = {
            "step": step,
            "episode": episode,
            "eval_reward": f"{eval_results['reward']:.2f}",
            "eval_survival": f"{eval_results['survival']:.1f}",
            "eval_food_eaten": f"{eval_results['food_eaten']:.2f}",
            "eval_damage_taken": f"{eval_results['damage_taken']:.2f}",
            "eval_death_rate": f"{eval_results['death_rate']:.2f}",
            "epsilon": f"{epsilon:.4f}",
            "loss": f"{loss:.6f}" if loss > 0 else "",
            "wall_time": f"{time.monotonic() - self._start_time:.0f}",
        }
        self._eval_writer.writerow(row)
        self._flush_eval()

        self._best_eval_reward = max(self._best_eval_reward, eval_results["reward"])
        self._best_survival = max(self._best_survival, eval_results["survival"])
        self._latest_eval = eval_results

    def log_episode(self, step: int, stats: Dict[str, Any]) -> None:
        self._episode_count += 1
        row = {
            "step": step,
            "episode": self._episode_count,
            "reward": f"{stats['episode_reward']:.2f}",
            "survival": stats["ticks_survived"],
            "food_eaten": stats["food_eaten"],
            "damage_taken": f"{stats['damage_taken']:.2f}",
            "terminated": int(stats["terminated_by_death"]),
        }
        self._episode_writer.writerow(row)
        # Flush every 50 episodes to balance I/O and data safety.
        if self._episode_count % 50 == 0:
            self._flush_episodes()

    def save_run_config(self, config_name: str, seed: int,
                        world_config: Any, dqn_config: Any) -> None:
        """Dump full config to JSON for reproducibility."""
        from dataclasses import asdict
        payload = {
            "config_name": config_name,
            "seed": seed,
            "world": {k: _serialize(v) for k, v in asdict(world_config).items()},
            "dqn": {k: _serialize(v) for k, v in asdict(dqn_config).items()},
        }
        path = self.log_dir / "config.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "episodes": self._episode_count,
            "best_reward": self._best_eval_reward,
            "best_survival": self._best_survival,
            "avg_reward": self._latest_eval.get("reward", 0.0),
            "avg_survival": self._latest_eval.get("survival", 0.0),
            "training_time": time.monotonic() - self._start_time,
        }

    def close(self) -> None:
        for h in getattr(self, "_handles", []):
            h.flush()
            h.close()

    def _flush_eval(self) -> None:
        self._handles[0].flush()

    def _flush_episodes(self) -> None:
        self._handles[1].flush()


def _serialize(v: Any) -> Any:
    """Make dataclass values JSON-serializable."""
    if isinstance(v, Path):
        return str(v)
    return v
