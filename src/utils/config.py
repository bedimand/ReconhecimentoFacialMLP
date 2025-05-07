import yaml
import os
from typing import Any

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_path(self, key: str) -> str:
        """Get a path from the configuration, ensuring it exists"""
        path = self.get(f"paths.{key}")
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def get_color(self, key: str) -> tuple:
        """Get a color tuple from the configuration"""
        color = self.get(key)
        if color:
            return tuple(color)
        return (0, 0, 0)  # Default black

    def get_device(self) -> str:
        """Get the current device (cuda/cpu)"""
        return self.get('system.device', 'cuda')

    def get_model_dir(self) -> str:
        """Get the models directory path"""
        return self.get_path('models_dir')

    def get_dataset_dir(self) -> str:
        """Get the dataset directory path"""
        return self.get_path('dataset_dir')

    def get_detected_faces_dir(self) -> str:
        """Get the detected faces directory path"""
        return self.get_path('detected_faces_dir')

    def get_model_save_path(self) -> str:
        """Get the model save path"""
        return self.get('paths.model_save_path')

# Create a global config instance
config = Config() 