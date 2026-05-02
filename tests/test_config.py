from mirage.config import load_config


def test_smoke_config_loads() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.run_name == "smoke"
    assert config.steps_per_cycle > 0
    assert config.backend == "numpy"
