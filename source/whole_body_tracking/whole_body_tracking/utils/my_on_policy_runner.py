import os
from pathlib import Path

from isaaclab_rl.rsl_rl import export_policy_as_onnx
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _export_location(checkpoint_path: str) -> tuple[str, str]:
    """Return the checkpoint's run directory and its stable ONNX filename."""
    policy_path = str(Path(checkpoint_path).parent)
    filename = f"{Path(policy_path).name}.onnx"
    return policy_path, filename


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in {"tensorboard", "wandb"}:
            policy_path, filename = _export_location(path)
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            run_name = "local"
            if self.logger_type == "wandb":
                import wandb

                if wandb.run is not None:
                    run_name = wandb.run.name
            attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)

        if self.logger_type == "wandb":
            import wandb

            if wandb.run is not None:
                wandb.save(os.path.join(policy_path, filename), base_path=policy_path)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str | None = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in {"tensorboard", "wandb"}:
            policy_path, filename = _export_location(path)
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            run_name = "local"
            if self.logger_type == "wandb":
                import wandb

                if wandb.run is not None:
                    run_name = wandb.run.name
            attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)

        if self.logger_type == "wandb":
            import wandb

            if wandb.run is not None:
                wandb.save(os.path.join(policy_path, filename), base_path=policy_path)

                # link the artifact registry to this run
                if self.registry_name is not None:
                    wandb.run.use_artifact(self.registry_name)
                    self.registry_name = None
