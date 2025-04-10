"""
Configuration settings for the machine learning optimization project.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    
    # General optimization settings
    individuals_count: int = 100
    sampling_method: str = 'LHS'
    samples_count: int = 20
    max_iterations: int = 50
    
    # Learning rate settings
    learning_rate_range: Tuple[float, float] = (0.0001, 0.01)
    
    # PSO specific settings
    pso_settings: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pso_settings is None:
            self.pso_settings = {
                'cognitive_coefficient': 0.5,
                'social_coefficient': 0.5,
                'inertia_weight': 0.4
            }

@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    
    # Input shapes for different models
    input_shape_cnn: Tuple[int, int, int] = (28, 28, 1)
    input_shape_vgg: Tuple[int, int, int] = (32, 32, 3)
    
    # Model architecture settings
    num_classes: int = 10
    default_activation: str = 'relu'
    output_activation: str = 'softmax'
    
    # Training settings
    batch_size_range: Tuple[int, int] = (32, 256)
    epochs: int = 50
    validation_split: float = 0.2

@dataclass
class PathConfig:
    """Configuration for project paths."""
    
    # Base directories
    base_dir: Path = Path('src')
    data_dir: Path = base_dir / 'data'
    models_dir: Path = base_dir / 'models'
    results_dir: Path = base_dir / 'results'
    
    # Data directories
    raw_data_dir: Path = data_dir / 'raw'
    processed_data_dir: Path = data_dir / 'processed'
    training_data_dir: Path = data_dir / 'trainings'
    
    # Results directories
    model_checkpoints_dir: Path = results_dir / 'checkpoints'
    optimization_results_dir: Path = results_dir / 'optimization'
    logs_dir: Path = results_dir / 'logs'
    
    def __post_init__(self):
        """Create all directories if they don't exist."""
        for directory in [
            self.data_dir, self.models_dir, self.results_dir,
            self.raw_data_dir, self.processed_data_dir, self.training_data_dir,
            self.model_checkpoints_dir, self.optimization_results_dir, self.logs_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    file_name: str = 'experiment.log'

class Config:
    """Main configuration class that holds all config instances."""
    
    def __init__(self):
        self.optimization = OptimizationConfig()
        self.model = ModelConfig()
        self.paths = PathConfig()
        self.logging = LoggingConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        instance = cls()
        
        if 'optimization' in config_dict:
            instance.optimization = OptimizationConfig(**config_dict['optimization'])
        if 'model' in config_dict:
            instance.model = ModelConfig(**config_dict['model'])
        if 'paths' in config_dict:
            instance.paths = PathConfig(**config_dict['paths'])
        if 'logging' in config_dict:
            instance.logging = LoggingConfig(**config_dict['logging'])
        
        return instance

# Create default configuration instance
config = Config() 