def create_model(name: str, kwargs: dict | None = None):
    """Create model from name and kwargs."""
    from .velocity_net import EgnnVelocityNet

    registry: dict[str, type] = {
        "egnn_velocity_net": EgnnVelocityNet,
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(registry.keys())}")
    return registry[name](**(kwargs or {}))
