"""Evaluate a local motion-tracking checkpoint over clean full-motion episodes."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the evaluation parser without importing the Isaac runtime."""
    parser = argparse.ArgumentParser(description="Evaluate a local RSL-RL motion-tracking checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_all_to_start(base_env, command):
    import torch

    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    command.reset_to_start(env_ids)
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
        base_env.observation_manager.reset(terminal)
    return base_env.observation_manager.compute(update_history=True)


def _run_evaluation(args_cli, simulation_app) -> None:
    import gymnasium as gym
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def evaluate(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.events = None
        env_cfg.curriculum = None
        env_cfg.commands.motion.motion_file = str(motion_file)
        agent_cfg.seed = args_cli.seed

        env = None
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            wrapped_env = RslRlVecEnvWrapper(env)
            base_env = wrapped_env.unwrapped
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=base_env.device)
            command = base_env.command_manager.get_term("motion")
            accumulator = EpisodeAccumulator(args_cli.num_envs, args_cli.episodes)

            observations = TensorDict(
                _reset_all_to_start(base_env, command), batch_size=[args_cli.num_envs]
            )
            while simulation_app.is_running() and not accumulator.done:
                if not all(torch.all(torch.isfinite(value)) for value in observations.values()):
                    raise RuntimeError("non-finite policy observation during evaluation")
                last_frame = command.time_steps >= command.motion.time_step_total - 1
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, _, _ = wrapped_env.step(actions)

                body_error = command.metrics["error_body_pos"]
                if not torch.all(torch.isfinite(body_error)) or torch.any(body_error < 0):
                    raise RuntimeError("invalid error_body_pos during evaluation")
                failed = base_env.termination_manager.terminated.clone()
                completed = last_frame & ~failed
                accumulator.update(
                    body_error.detach().cpu().numpy(),
                    failed.detach().cpu().numpy(),
                    completed.detach().cpu().numpy(),
                )

                failed_ids = torch.where(failed)[0]
                completed_ids = torch.where(completed)[0]
                timeout_ids = torch.where(base_env.termination_manager.time_outs & ~failed)[0]
                if failed_ids.numel() + completed_ids.numel() + timeout_ids.numel() > 0:
                    raw_observations = _reset_terminal_environments(
                        base_env, command, failed_ids, completed_ids, timeout_ids
                    )
                    accumulator.reset(torch.cat([failed_ids, completed_ids, timeout_ids]).cpu().tolist())
                    observations = TensorDict(raw_observations, batch_size=[args_cli.num_envs])

            if not accumulator.done:
                raise RuntimeError("simulation stopped before evaluation completed")
            summary = accumulator.result()
            payload = {
                "schema_version": 1,
                "motion_id": motion_file.parent.name,
                "seed": args_cli.seed,
                "episodes": summary.episodes,
                "completed": summary.completed,
                "falls": summary.falls,
                "tracking_errors": list(summary.tracking_errors),
                "control_hz": 50,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "motion_file": str(motion_file),
                "motion_npz_sha256": _sha256_file(motion_file),
                "started_at_utc": started_at.isoformat(),
                "ended_at_utc": datetime.now(UTC).isoformat(),
            }
            atomic_write_evaluation_json(output_file, payload)
        finally:
            if env is not None:
                env.close()

    evaluate()


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
