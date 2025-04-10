"""
Base model class for neural networks.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History

from src.config.settings import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all neural network models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration instance
        """
        self.config = config
        self._model: Optional[Sequential] = None
    
    @property
    def model(self) -> Sequential:
        """Get the Keras model instance."""
        if self._model is None:
            self._model = self.create_model()
        return self._model
    
    @abstractmethod
    def create_model(self) -> Sequential:
        """
        Create and return a compiled Keras model.
        
        Returns:
            Compiled Keras Sequential model
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data before training/inference.
        
        Args:
            data: Input data array
            
        Returns:
            Preprocessed data array
        """
        pass
    
    @abstractmethod
    def postprocess_output(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            output: Model output array
            
        Returns:
            Postprocessed output array
        """
        pass
    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> History:
        """
        Train the model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            validation_data: Optional tuple of validation data and labels
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            Training history
        """
        # Preprocess data
        x_train = self.preprocess_data(x_train)
        if validation_data is not None:
            x_val, y_val = validation_data
            validation_data = (self.preprocess_data(x_val), y_val)
        
        # Set default training parameters
        training_params = {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size_range[0],
            'validation_split': self.config.validation_split
        }
        training_params.update(kwargs)
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            **training_params
        )
        
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        x = self.preprocess_data(x)
        predictions = self.model.predict(x)
        return self.postprocess_output(predictions)
    
    def save(self, filepath: str) -> None:
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, config: ModelConfig) -> 'BaseModel':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            config: Model configuration
            
        Returns:
            Loaded model instance
        """
        instance = cls(config)
        instance._model = Sequential.load(filepath)
        return instance
    
    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'config': self.config.__dict__,
            'model_config': self.model.get_config()
        } 