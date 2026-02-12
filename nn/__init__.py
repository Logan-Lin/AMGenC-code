from .velocity_net import EgnnVelocityNet

MODEL_REGISTRY: dict[str, type] = {
    "egnn_velocity_net": EgnnVelocityNet,
}


def create_model(name: str, kwargs: dict | None = None):
    """Create model from name and kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**(kwargs or {}))
