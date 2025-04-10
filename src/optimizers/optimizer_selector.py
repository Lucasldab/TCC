"""
Optimizer selector module for neural network training.
Provides functions to select and configure different optimization algorithms.
"""
from typing import Tuple, Union
from tensorflow import keras
from keras import optimizers
import random

# Constants for optimizer types
OPTIMIZER_TYPES = {
    'SGD': optimizers.SGD,
    'RMSprop': optimizers.RMSprop,
    'Adam': optimizers.Adam,
    'AdamW': optimizers.AdamW,
    'Adadelta': optimizers.Adadelta,
    'Adagrad': optimizers.Adagrad,
    'Adamax': optimizers.Adamax,
    'Nadam': optimizers.Nadam,
    'Ftrl': optimizers.Ftrl
}

OPTIMIZER_NAMES = list(OPTIMIZER_TYPES.keys())

def get_random_optimizer() -> Tuple[keras.optimizers.Optimizer, float]:
    """
    Generate a random optimizer with a random learning rate.
    
    Returns:
        Tuple containing:
            - Optimizer instance
            - Learning rate used
    """
    learning_rate = random.uniform(0.0001, 0.01)
    optimizer_idx = random.randint(0, len(OPTIMIZER_TYPES) - 1)
    optimizer_name = OPTIMIZER_NAMES[optimizer_idx]
    optimizer = OPTIMIZER_TYPES[optimizer_name](learning_rate=learning_rate)
    return optimizer, learning_rate

def get_optimizer_by_number(optimizer_number: int, learning_rate: float) -> keras.optimizers.Optimizer:
    """
    Get optimizer by its numerical identifier.
    
    Args:
        optimizer_number: Integer identifier for the optimizer (1-9)
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer_number is out of range
    """
    if not 1 <= optimizer_number <= len(OPTIMIZER_TYPES):
        raise ValueError(f"Optimizer number must be between 1 and {len(OPTIMIZER_TYPES)}")
    
    optimizer_name = OPTIMIZER_NAMES[optimizer_number - 1]
    return OPTIMIZER_TYPES[optimizer_name](learning_rate=learning_rate)

def define_optimizer(optimizer_name: str, learning_rate: float) -> Tuple[keras.optimizers.Optimizer, str]:
    """
    Define an optimizer by its name.
    
    Args:
        optimizer_name: Name of the optimizer (e.g., 'Adam', 'SGD')
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Tuple containing:
            - Configured optimizer instance
            - Name of the optimizer
            
    Raises:
        ValueError: If optimizer_name is not recognized
    """
    if optimizer_name not in OPTIMIZER_TYPES:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Available optimizers: {', '.join(OPTIMIZER_TYPES.keys())}")
    
    optimizer = OPTIMIZER_TYPES[optimizer_name](learning_rate=learning_rate)
    return optimizer, optimizer_name

def number_to_name(optimizer_number: int) -> str:
    """
    Convert optimizer number to its name.
    
    Args:
        optimizer_number: Integer identifier for the optimizer (0-8)
        
    Returns:
        Name of the optimizer
        
    Raises:
        ValueError: If optimizer_number is out of range
    """
    if not 0 <= optimizer_number < len(OPTIMIZER_NAMES):
        raise ValueError(f"Optimizer number must be between 0 and {len(OPTIMIZER_NAMES) - 1}")
    
    return OPTIMIZER_NAMES[optimizer_number] 