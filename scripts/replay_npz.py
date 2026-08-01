"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create the replay command-line parser without loading Isaac Sim."""
    parser = argparse.ArgumentParser(description="Replay converted motions.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion_file", type=str, help="Path to a local motion npz file.")
    source.add_argument("--registry_name", type=str, help="The name of the WandB registry artifact.")
    return parser


def _run_simulator(args_cli, simulation_app):
    """Load the simulator runtime and replay the requested motion."""
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
    from whole_body_tracking.tasks.tracking.mdp import MotionLoader
    from whole_body_tracking.utils.motion_source import resolve_motion_source

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                    "kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
        robot: Articulation = scene["robot"]
        sim_dt = sim.get_physics_dt()

        motion_file = resolve_motion_source(args_cli.motion_file, args_cli.registry_name)
        motion = MotionLoader(
            str(motion_file),
            torch.tensor([0], dtype=torch.long, device=sim.device),
            sim.device,
        )
        time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

        while simulation_app.is_running():
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            scene.write_data_to_sim()
            sim.render()  # We don't want physic (sim.step())
            scene.update(sim_dt)

            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


def main(argv=None):
    """Launch Isaac Sim and run the replay workflow."""
    from isaaclab.app import AppLauncher

    parser = create_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run_simulator(args_cli, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
