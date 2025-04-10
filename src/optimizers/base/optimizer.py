"""
Base optimizer class for hyperparameter optimization.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass

from src.config.settings import OptimizationConfig

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_solution: np.ndarray
    best_fitness: float
    history: Dict[str, list]
    parameters: Dict[str, Any]

class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the optimizer.
        
        Args:
            config: Optimization configuration instance
        """
        self.config = config
        self.history: Dict[str, list] = {
            'fitness': [],
            'best_fitness': [],
            'mean_fitness': []
        }
    
    @abstractmethod
    def optimize(self, 
                fitness_function: Callable[[np.ndarray], float],
                initial_population: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Run the optimization process.
        
        Args:
            fitness_function: Function that evaluates solutions
            initial_population: Optional initial population
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult containing the best solution and history
        """
        pass
    
    @abstractmethod
    def _initialize_population(self, 
                             dimension: int,
                             bounds: Tuple[float, float],
                             size: int) -> np.ndarray:
        """
        Initialize the population.
        
        Args:
            dimension: Number of dimensions
            bounds: Tuple of (lower_bound, upper_bound)
            size: Population size
            
        Returns:
            Initial population array
        """
        pass
    
    def _update_history(self, 
                       generation: int,
                       fitness_values: np.ndarray,
                       best_fitness: float) -> None:
        """
        Update optimization history.
        
        Args:
            generation: Current generation number
            fitness_values: Array of fitness values
            best_fitness: Best fitness value found
        """
        self.history['fitness'].append(fitness_values.tolist())
        self.history['best_fitness'].append(best_fitness)
        self.history['mean_fitness'].append(float(np.mean(fitness_values)))
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of optimizer parameters
        """
        return {
            'config': self.config.__dict__,
            'history': self.history
        }
    
    @classmethod
    def from_dict(cls, parameters: Dict[str, Any]) -> 'BaseOptimizer':
        """
        Create optimizer instance from parameters dictionary.
        
        Args:
            parameters: Dictionary of optimizer parameters
            
        Returns:
            Optimizer instance
        """
        config = OptimizationConfig(**parameters['config'])
        instance = cls(config)
        instance.history = parameters['history']
        return instance 