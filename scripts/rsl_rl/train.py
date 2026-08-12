# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL motion-tracking policy."""

from __future__ import annotations

import argparse
import sys

# local imports
import cli_args  # isort: skip


def create_parser() -> argparse.ArgumentParser:
    """Create the training CLI parser without importing or launching Isaac Sim."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument("--log_dir", type=str, default=None, help="Exact directory for local training outputs.")
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None, help="Exact local checkpoint path to resume from."
    )
    motion_source = parser.add_mutually_exclusive_group(required=True)
    motion_source.add_argument("--motion_file", type=str, help="Path to a local motion NPZ file.")
    motion_source.add_argument("--registry_name", type=str, help="Name of a WandB motion registry artifact.")
    cli_args.add_rsl_rl_args(parser)
    return parser


def _resolve_log_dir(args_cli, agent_cfg) -> str:
    """Return an explicit output directory or the legacy timestamped layout."""
    if args_cli.log_dir is not None:
        return args_cli.log_dir

    import os
    from datetime import datetime

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    return os.path.join(log_root_path, log_dir)


def _run_training(args_cli, simulation_app):
    """Import the Isaac runtime and execute training after the app is launched."""
    import os
    from importlib import metadata

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
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from whole_body_tracking.utils.motion_source import resolve_motion_source
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

    del simulation_app  # The application lifetime is owned by main().

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlOnPolicyRunnerCfg,
    ):
        """Train with the selected local or registry motion source."""
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        motion_path = resolve_motion_source(args_cli.motion_file, args_cli.registry_name)
        env_cfg.commands.motion.motion_file = str(motion_path)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        log_dir = _resolve_log_dir(args_cli, agent_cfg)
        print(f"[INFO] Logging experiment in directory: {log_dir}")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            registry_name=args_cli.registry_name,
        )
        runner.add_git_repo_to_log(__file__)
        if args_cli.resume_checkpoint is not None:
            resume_path = os.path.abspath(args_cli.resume_checkpoint)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        elif agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()

    train()


def _run_application(args_cli, simulation_app) -> None:
    """Train successfully before beginning graceful simulator shutdown."""
    _run_training(args_cli, simulation_app)
    simulation_app.close()


def main(argv=None):
    """Launch Isaac Sim and execute the training workflow."""
    from isaaclab.app import AppLauncher

    parser = create_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args(argv)
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0], *hydra_args]
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    _run_application(args_cli, simulation_app)


if __name__ == "__main__":
    main()
