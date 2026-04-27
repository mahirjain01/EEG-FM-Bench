from .eegpt   import EEGPTAdapter
from .labram   import LaBraMAdapter
from .cbramod  import CBraMoDAdapter
from .biot     import BIOTAdapter
from .csbrain  import CSBrainAdapter
from .reve     import REVEAdapter

__all__ = [
    "EEGPTAdapter", "LaBraMAdapter", "CBraMoDAdapter",
    "BIOTAdapter", "CSBrainAdapter", "REVEAdapter",
]

# Convenience mapping: model name string → adapter class
ADAPTER_REGISTRY: dict[str, type] = {
    "eegpt":   EEGPTAdapter,
    "labram":  LaBraMAdapter,
    "cbramod": CBraMoDAdapter,
    "biot":    BIOTAdapter,
    "csbrain": CSBrainAdapter,
    "reve":    REVEAdapter,
}


def get_adapter_class(model_type: str) -> type:
    cls = ADAPTER_REGISTRY.get(model_type)
    if cls is None:
        raise KeyError(
            f"Unknown model '{model_type}'. "
            f"Available: {list(ADAPTER_REGISTRY)}"
        )
    return cls


def build_adapter(model_type: str, run_json_path, **kwargs):
    """Convenience: look up adapter class and call from_run_json()."""
    return get_adapter_class(model_type).from_run_json(run_json_path, **kwargs)
