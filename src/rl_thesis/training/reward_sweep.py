from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rl_thesis.config.reward_configs import get_config_names


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKERS_PER_GPU = 2


@dataclass(frozen=True)
class SweepTask:
    config: str
    seed: int


def run_reward_sweep(
    steps: int | None,
    seeds: int,
    start_seed: int,
    configs: Sequence[str] | None,
    workers: int | None,
    gpu_slots: int | None,
    log_dir: Path,
) -> int:
    from rl_thesis.config.config import DQNConfig

    config_names = list(configs) if configs else get_config_names()
    tasks = _build_tasks(config_names=config_names, seeds=seeds, start_seed=start_seed)
    if not tasks:
        raise ValueError("No tasks available for the reward sweep.")

    resolved_steps = steps if steps is not None else DQNConfig().total_timesteps
    visible_gpu_ids = _resolve_gpu_ids(gpu_slots)
    worker_count = _resolve_worker_count(
        requested_workers=workers,
        task_count=len(tasks),
        gpu_count=len(visible_gpu_ids),
    )

    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _write_plan(
        log_dir=log_dir,
        tasks=tasks,
        worker_count=worker_count,
        visible_gpu_ids=visible_gpu_ids,
        steps=resolved_steps,
    )

    print(f"Starting reward sweep with {worker_count} worker(s) across {len(tasks)} task(s).")
    print(f"Visible GPU slots: {visible_gpu_ids or ['cpu']}")
    print(f"Steps per run: {resolved_steps:,}")
    print(f"Plan written to {plan_path}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)

    processes = _start_workers(
        ctx=ctx,
        worker_count=worker_count,
        task_queue=task_queue,
        log_dir=log_dir,
        visible_gpu_ids=visible_gpu_ids,
        steps=resolved_steps,
    )

    exit_code = 0
    try:
        remaining = set(range(worker_count))
        while remaining:
            for worker_index, process in enumerate(processes):
                if worker_index not in remaining:
                    continue

                process_exit_code = process.exitcode
                if process_exit_code is None:
                    continue

                remaining.remove(worker_index)
                if process_exit_code != 0 and exit_code == 0:
                    exit_code = process_exit_code
                    print(
                        f"Worker {worker_index} failed with exit code {process_exit_code}. "
                        "Terminating remaining workers."
                    )
                    _terminate_processes(processes)
                    remaining.clear()
                    break

            if remaining:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted. Terminating workers...")
        exit_code = 130
        _terminate_processes(processes)
    finally:
        for process in processes:
            process.join(timeout=5)

    return exit_code


def _build_tasks(config_names: Sequence[str], seeds: int, start_seed: int) -> list[SweepTask]:
    return [
        SweepTask(config=config_name, seed=seed)
        for config_name in config_names
        for seed in range(start_seed, start_seed + seeds)
    ]


def _resolve_worker_count(requested_workers: int | None, task_count: int, gpu_count: int) -> int:
    default_workers = max(gpu_count * DEFAULT_WORKERS_PER_GPU, 1)
    worker_count = requested_workers or default_workers
    return min(worker_count, task_count)


def _resolve_gpu_ids(gpu_slots: int | None) -> list[str]:
    visible_gpu_ids = _visible_gpu_ids_from_env()
    if not visible_gpu_ids:
        import torch

        visible_gpu_ids = [str(index) for index in range(torch.cuda.device_count())]

    if gpu_slots is None:
        return visible_gpu_ids
    if gpu_slots == 0:
        return []
    if len(visible_gpu_ids) < gpu_slots:
        raise ValueError(
            f"Requested {gpu_slots} GPU slots, but only {len(visible_gpu_ids)} visible GPUs are available."
        )
    return visible_gpu_ids[:gpu_slots]


def _visible_gpu_ids_from_env() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return []
    return [device.strip() for device in visible_devices.split(",") if device.strip()]


def _write_plan(
    log_dir: Path,
    tasks: Sequence[SweepTask],
    worker_count: int,
    visible_gpu_ids: Sequence[str],
    steps: int,
) -> Path:
    plan = {
        "steps_per_run": steps,
        "workers": worker_count,
        "visible_gpu_ids": list(visible_gpu_ids),
        "worker_gpu_ids": [
            visible_gpu_ids[worker_index % len(visible_gpu_ids)] if visible_gpu_ids else None
            for worker_index in range(worker_count)
        ],
        "tasks": [asdict(task) for task in tasks],
    }
    plan_path = log_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path


def _start_workers(
    ctx: mp.context.BaseContext,
    worker_count: int,
    task_queue: mp.Queue,
    log_dir: Path,
    visible_gpu_ids: Sequence[str],
    steps: int,
) -> list[mp.Process]:
    processes: list[mp.Process] = []
    for worker_index in range(worker_count):
        process = ctx.Process(
            target=_worker_loop,
            args=(worker_index, task_queue, log_dir, list(visible_gpu_ids), steps),
            name=f"reward-sweep-worker-{worker_index}",
        )
        process.start()
        processes.append(process)
        print(f"Worker {worker_index} started with pid {process.pid}; log={log_dir / f'worker_{worker_index}.log'}")
    return processes


def _worker_loop(
    worker_index: int,
    task_queue: mp.Queue,
    log_dir: Path,
    visible_gpu_ids: Sequence[str],
    steps: int,
) -> None:
    gpu_id = visible_gpu_ids[worker_index % len(visible_gpu_ids)] if visible_gpu_ids else None
    worker_log_path = log_dir / f"worker_{worker_index}.log"
    shutdown_requested = False

    def request_shutdown(_signum, _frame) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        raise KeyboardInterrupt

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    with worker_log_path.open("w", buffering=1) as worker_log:
        with contextlib.redirect_stdout(worker_log), contextlib.redirect_stderr(worker_log):
            os.chdir(REPO_ROOT)
            signal.signal(signal.SIGINT, request_shutdown)

            from rl_thesis.config.config import DQNConfig
            from rl_thesis.training.train import run_single

            gpu_label = gpu_id if gpu_id is not None else "auto"
            print(f"Worker {worker_index} starting on {gpu_label}.", flush=True)

            try:
                while True:
                    task = task_queue.get()
                    if task is None:
                        break

                    print(
                        f"Worker {worker_index} running config={task.config} seed={task.seed}",
                        flush=True,
                    )
                    try:
                        run_single(
                            config_name=task.config,
                            seed=task.seed,
                            dqn_config=DQNConfig(total_timesteps=steps),
                        )
                    except Exception:
                        print(
                            f"Worker {worker_index} failed on config={task.config} seed={task.seed}",
                            flush=True,
                        )
                        traceback.print_exc()
                        raise SystemExit(1)

                    if shutdown_requested:
                        raise KeyboardInterrupt

                print(f"Worker {worker_index} finished.", flush=True)
            except KeyboardInterrupt:
                print(f"Worker {worker_index} interrupted.", flush=True)
                raise SystemExit(130)


def _terminate_processes(processes: Sequence[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            try:
                os.kill(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.time() + 10
    for process in processes:
        if not process.is_alive():
            continue
        remaining = max(0.0, deadline - time.time())
        process.join(timeout=remaining)

    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.time() + 10
    for process in processes:
        if not process.is_alive():
            continue
        remaining = max(0.0, deadline - time.time())
        process.join(timeout=remaining)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
