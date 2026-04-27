from uniformevalbench.models.abstract.config import AbstractConfig
from uniformevalbench.models.abstract.adapter import AbstractDataLoaderFactory
from uniformevalbench.models.abstract.trainer import AbstractTrainer


class ModelRegistry:
    """Registry for FM models — maps model_type to (config, adapter, trainer) classes."""

    configs:  dict = {}
    adapters: dict = {}
    trainers: dict = {}

    @classmethod
    def register_model(cls, model_type, config_class, adapter_class, trainer_class):
        cls.configs[model_type]  = config_class
        cls.adapters[model_type] = adapter_class
        cls.trainers[model_type] = trainer_class

    @classmethod
    def get_config_class(cls, model_type):
        if model_type not in cls.configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.configs)}")
        return cls.configs[model_type]

    @classmethod
    def create_trainer(cls, config) -> AbstractTrainer:
        trainer_class = cls.trainers.get(config.model_type)
        if trainer_class is None:
            raise ValueError(f"Unknown model type: {config.model_type}")
        return trainer_class(config)

    @classmethod
    def list_models(cls) -> list:
        return list(cls.configs.keys())
