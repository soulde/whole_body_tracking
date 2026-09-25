from pathlib import Path


def test_active_migration_targets_isaac_lab_30_docker_image() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/isaac-lab:3.0.0-rc1" in text


def test_teacher_source_does_not_define_downstream_model_packages() -> None:
    root = Path(__file__).parents[1]
    assert not (root / "vae_distillation").exists()
    assert not (root / "diffusion_training").exists()
