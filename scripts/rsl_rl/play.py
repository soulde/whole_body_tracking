"""Play an RSL-RL motion-tracking checkpoint."""

from __future__ import annotations

import argparse
import sys

# local imports
import cli_args  # isort: skip


class _PlaybackArgumentParser(argparse.ArgumentParser):
    """Validate the local pair or the explicit WandB compatibility mode."""

    def parse_known_args(self, args=None, namespace=None):
        parsed, remaining = super().parse_known_args(args, namespace)
        local_values = (parsed.checkpoint, parsed.motion_file)
        if parsed.wandb_path is not None:
            if any(value is not None for value in local_values):
                self.error("--wandb_path cannot be combined with --checkpoint or --motion_file")
        elif any(value is None for value in local_values):
            self.error("local playback requires both --checkpoint and --motion_file")
        return parsed, remaining


def create_parser() -> argparse.ArgumentParser:
    """Create the playback CLI parser without importing or launching Isaac Sim."""
    parser = _PlaybackArgumentParser(description="Play an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--motion_file", type=str, default=None, help="Path to a local motion NPZ file.")
    cli_args.add_rsl_rl_args(parser)
    return parser


def _run_playback(args_cli, simulation_app):
    """Import the Isaac runtime and execute playback after the app is launched."""
    import os
    from pathlib import Path

    import gymnasium as gym
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from whole_body_tracking.utils.motion_source import resolve_motion_source

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def play(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlOnPolicyRunnerCfg,
    ):
        """Play from an exact local pair or an explicitly selected WandB run."""
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        if args_cli.wandb_path is not None:
            import wandb

            run_path = args_cli.wandb_path
            api = wandb.Api()
            if "model" in args_cli.wandb_path:
                run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
            wandb_run = api.run(run_path)
            files = [file.name for file in wandb_run.files() if "model" in file.name]
            if "model" in args_cli.wandb_path:
                file_name = args_cli.wandb_path.split("/")[-1]
            else:
                file_name = max(files, key=lambda name: int(name.split("_")[1].split(".")[0]))

            wandb_file = wandb_run.file(str(file_name))
            wandb_file.download("./logs/rsl_rl/temp", replace=True)
            print(f"[INFO]: Loading model checkpoint from: {run_path}/{file_name}")
            resume_path = f"./logs/rsl_rl/temp/{file_name}"

            artifact = next((item for item in wandb_run.used_artifacts() if item.type == "motions"), None)
            if artifact is None:
                print("[WARN] No motion artifact found in the run.")
            else:
                env_cfg.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
            resume_path = str(checkpoint)
            motion_path = resolve_motion_source(args_cli.motion_file, None)
            env_cfg.commands.motion.motion_file = str(motion_path)
            print(f"[INFO]: Loading local model checkpoint from: {resume_path}")
            print(f"[INFO]: Using local motion file: {motion_path}")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        log_dir = os.path.dirname(resume_path)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording video during playback.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)

        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_onnx_metadata(
            env.unwrapped,
            args_cli.wandb_path if args_cli.wandb_path is not None else "local",
            export_model_dir,
        )

        obs, _ = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
        env.close()

    play()


def main(argv=None):
    """Launch Isaac Sim and execute the playback workflow."""
    from isaaclab.app import AppLauncher

    parser = create_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args(argv)
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0], *hydra_args]
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run_playback(args_cli, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
