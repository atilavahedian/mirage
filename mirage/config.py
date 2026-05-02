from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MirageConfig:
    seed: int = 7
    backend: str = "auto"
    device: str = "auto"
    run_name: str = "local"
    steps_per_cycle: int = 600
    episodes_per_eval: int = 2
    world_model_steps: int = 80
    actor_critic_steps: int = 80
    batch_size: int = 32
    sequence_length: int = 24
    imagination_horizon: int = 16
    replay_capacity: int = 100000
    warmup_steps: int = 1000
    max_episode_steps: int = 600
    render_size: int = 128
    video_fps: int = 20
    d_model: int = 128
    latent_dim: int = 64
    hidden_dim: int = 192
    learning_rate: float = 3e-4
    discount: float = 0.99
    entropy_scale: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"unsupported config line in {path}: {raw_line!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value)
    return data


def load_config(path: str | Path | None = None, **overrides: Any) -> MirageConfig:
    data: dict[str, Any] = {}
    if path is not None:
        data.update(_read_simple_yaml(Path(path)))
    data.update({k: v for k, v in overrides.items() if v is not None})
    valid = MirageConfig.__dataclass_fields__
    unknown = sorted(set(data) - set(valid))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return MirageConfig(**data)
