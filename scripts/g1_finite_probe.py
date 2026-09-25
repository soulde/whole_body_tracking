"""Finite G1 task reset/step probe for the Isaac Lab 3 migration branch."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import whole_body_tracking  # noqa: F401  # registers Tracking-Flat-G1 tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("Tracking-Flat-G1-v0", device=args.device, num_envs=args.num_envs)
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    for step in range(2):
        env.step(torch.zeros_like(env.action_manager.action))
        print(f"G1_PROBE_STEP={step + 1}", flush=True)
    env.close()
    app.close()
    print("G1_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
