"""Evaluate a local motion-tracking checkpoint over clean full-motion episodes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the evaluation parser without importing the Isaac runtime."""
    parser = argparse.ArgumentParser(description="Evaluate a local RSL-RL motion-tracking checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_id", choices=("walk_run", "jump", "kick"), required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    return parser


def _disable_manager_terms(config):
    """Disable configured manager terms without removing their config container."""
    for term_name in tuple(vars(config)):
        setattr(config, term_name, None)
    return config


def _snapshot_file(path: Path, directory: Path) -> tuple[Path, str]:
    """Copy and hash one immutable input stream used by the evaluator."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, snapshot_name = tempfile.mkstemp(dir=directory, prefix=f"{path.stem}-", suffix=path.suffix)
    snapshot = Path(snapshot_name)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, digest.hexdigest()


def _install_terminal_metric_capture(termination_manager, command):
    """Capture the post-physics body error before Isaac automatically resets environments."""
    original_compute = termination_manager.compute
    captured = None

    def compute_and_capture():
        nonlocal captured
        result = original_compute()
        command._update_metrics()
        body_error = command.metrics["error_body_pos"]
        captured = body_error.clone() if hasattr(body_error, "clone") else body_error.copy()
        return result

    def get_captured():
        if captured is None:
            raise RuntimeError("terminal metric capture has not run")
        return captured

    termination_manager.compute = compute_and_capture
    return get_captured


def _require_no_unexpected_timeouts(timeout_mask, failed_mask, completed_mask) -> None:
    import torch

    unexpected = (
        torch.as_tensor(timeout_mask, dtype=torch.bool)
        & ~torch.as_tensor(failed_mask, dtype=torch.bool)
        & ~torch.as_tensor(completed_mask, dtype=torch.bool)
    )
    unexpected_ids = torch.where(unexpected)[0]
    if unexpected_ids.numel() > 0:
        raise RuntimeError(
            f"unexpected timeout before full motion completion in environments {unexpected_ids.tolist()}"
        )


def _reset_all_to_start(base_env, command):
    import torch

    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    command.reset_to_start(env_ids)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.observation_manager.reset(env_ids)
    return base_env.observation_manager.compute(update_history=True)


def _reset_terminal_environments(base_env, command, failed_ids, completed_ids, timeout_ids=()):
    import torch

    completed = torch.sort(torch.as_tensor(completed_ids, dtype=torch.long, device=base_env.device)).values
    if completed.numel() > 0:
        base_env._reset_idx(completed)
    terminal = torch.cat(
        [
            torch.as_tensor(failed_ids, dtype=torch.long, device=base_env.device),
            completed,
            torch.as_tensor(timeout_ids, dtype=torch.long, device=base_env.device),
        ]
    )
    if terminal.numel() > 0:
        terminal = torch.unique(terminal, sorted=True)
        command.reset_to_start(terminal)
        base_env.scene.write_data_to_sim()
        base_env.sim.forward()
        base_env.observation_manager.reset(terminal)
    return base_env.observation_manager.compute(update_history=True)


def _run_evaluation(args_cli, simulation_app) -> None:
    from importlib import metadata

    import gymnasium as gym
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from whole_body_tracking.utils.evaluation import EpisodeAccumulator, atomic_write_evaluation_json

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    motion_file = Path(args_cli.motion_file).expanduser().resolve()
    output_file = Path(args_cli.output_file).expanduser().resolve()
    for label, path in (("checkpoint", checkpoint), ("motion file", motion_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args_cli.episodes < 1 or args_cli.num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    if output_file.exists():
        raise FileExistsError(output_file)

    started_at = datetime.now(UTC)
    snapshot_directory = tempfile.TemporaryDirectory(prefix="whole-body-evaluation-")
    try:
        snapshot_root = Path(snapshot_directory.name)
        checkpoint_input, checkpoint_sha256 = _snapshot_file(checkpoint, snapshot_root)
        motion_input, motion_sha256 = _snapshot_file(motion_file, snapshot_root)

        @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
        def evaluate(env_cfg, agent_cfg):
            env_cfg.scene.num_envs = args_cli.num_envs
            env_cfg.seed = args_cli.seed
            _disable_manager_terms(env_cfg.events)
            _disable_manager_terms(env_cfg.curriculum)
            env_cfg.commands.motion.motion_file = str(motion_input)
            agent_cfg.seed = args_cli.seed

            env = None
            try:
                env = gym.make(args_cli.task, cfg=env_cfg)
                wrapped_env = RslRlVecEnvWrapper(env)
                base_env = wrapped_env.unwrapped
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
                runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(str(checkpoint_input))
                policy = runner.get_inference_policy(device=base_env.device)
                command = base_env.command_manager.get_term("motion")
                captured_body_error = _install_terminal_metric_capture(base_env.termination_manager, command)
                accumulator = EpisodeAccumulator(args_cli.num_envs, args_cli.episodes)

                observations = TensorDict(
                    _reset_all_to_start(base_env, command), batch_size=[args_cli.num_envs]
                )
                while simulation_app.is_running() and not accumulator.done:
                    if not all(torch.all(torch.isfinite(value)) for value in observations.values()):
                        raise RuntimeError("non-finite policy observation during evaluation")
                    last_frame = command.time_steps >= command.motion.time_step_total - 1
                    with torch.no_grad():
                        actions = policy(observations)
                        observations, _, _, _ = wrapped_env.step(actions)

                    body_error = captured_body_error()
                    if not torch.all(torch.isfinite(body_error)) or torch.any(body_error < 0):
                        raise RuntimeError("invalid error_body_pos during evaluation")
                    failed = base_env.termination_manager.terminated.clone()
                    completed = last_frame & ~failed
                    _require_no_unexpected_timeouts(base_env.termination_manager.time_outs, failed, completed)
                    accumulator.update(
                        body_error.detach().cpu().numpy(),
                        failed.detach().cpu().numpy(),
                        completed.detach().cpu().numpy(),
                    )

                    failed_ids = torch.where(failed)[0]
                    completed_ids = torch.where(completed)[0]
                    if failed_ids.numel() + completed_ids.numel() > 0:
                        raw_observations = _reset_terminal_environments(base_env, command, failed_ids, completed_ids)
                        accumulator.reset(torch.cat([failed_ids, completed_ids]).cpu().tolist())
                        observations = TensorDict(raw_observations, batch_size=[args_cli.num_envs])

                if not accumulator.done:
                    raise RuntimeError("simulation stopped before evaluation completed")
                summary = accumulator.result()
                payload = {
                    "schema_version": 1,
                    "motion_id": args_cli.motion_id,
                    "seed": args_cli.seed,
                    "episodes": summary.episodes,
                    "completed": summary.completed,
                    "falls": summary.falls,
                    "tracking_errors": list(summary.tracking_errors),
                    "control_hz": 50,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "motion_file": str(motion_file),
                    "motion_npz_sha256": motion_sha256,
                    "started_at_utc": started_at.isoformat(),
                    "ended_at_utc": datetime.now(UTC).isoformat(),
                }
                atomic_write_evaluation_json(output_file, payload)
            finally:
                if env is not None:
                    env.close()

        evaluate()
    finally:
        snapshot_directory.cleanup()


def main(argv=None) -> None:
    """Launch Isaac Sim and run clean evaluation."""
    from isaaclab.app import AppLauncher

    parser = create_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0], *hydra_args]
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run_evaluation(args_cli, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
