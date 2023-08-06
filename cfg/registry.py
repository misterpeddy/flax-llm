import fiddle as fdl
import importlib
import trainer

_CFG_REGISTRY = {
    "gpt2-117m": "cfg.gpt2_117m",
    "bigram": "cfg.bigram",
}


def get_cfg(name: str) -> fdl.Config[trainer.Trainer]:
    """Returns config, only doing necessary imports to save import time."""
    if not name in _CFG_REGISTRY:
        raise ValueError(
            f"Unknown config {name}. Available configs: {_CFG_REGISTRY.keys()}")
    module_path = _CFG_REGISTRY[name]
    cfg_module = importlib.import_module(module_path)
    assert hasattr(cfg_module, "get_cfg")
    return cfg_module.get_cfg()
