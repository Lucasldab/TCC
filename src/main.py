"""
Main script for neural network hyperparameter optimization.
"""
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config.settings import Config
from src.models.base.model import BaseModel
from src.models.factory import ModelFactory
from src.optimizers.base.optimizer import BaseOptimizer
from src.optimizers.dde_optimizer import DDEOptimizer
from src.optimizers.pso_optimizer import PSOOptimizer
from src.data.pipeline.base import Pipeline, DatasetMetadata
from src.utils.results_manager import ResultsManager, ExperimentResult

# Initialize configuration
config = Config()

# Set up logging
logging.basicConfig(
    level=config.logging.level,
    format=config.logging.format,
    datefmt=config.logging.date_format,
    filename=str(config.paths.logs_dir / config.logging.file_name)
)
logger = logging.getLogger(__name__)

def setup_experiment(model_type: str, optimizer_type: str) -> tuple[BaseModel, BaseOptimizer]:
    """
    Set up model and optimizer for an experiment.
    
    Args:
        model_type: Type of model to create
        optimizer_type: Type of optimizer to use
        
    Returns:
        Tuple of (model, optimizer)
    """
    # Create model
    model = ModelFactory.create(model_type, config=config.model)
    
    # Create optimizer
    if optimizer_type == 'dde':
        optimizer = DDEOptimizer(config=config.optimization)
    elif optimizer_type == 'pso':
        optimizer = PSOOptimizer(config=config.optimization)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return model, optimizer

def load_and_process_data(training_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and process training data.
    
    Args:
        training_number: Training iteration number
        
    Returns:
        Tuple of (features, targets)
    """
    # Load data
    data_file = config.paths.training_data_dir / f'training_{training_number}.csv'
    data = pd.read_csv(data_file)
    
    # Create data pipeline
    pipeline = Pipeline([
        # Add your data transformers here
    ])
    
    # Set metadata
    metadata = DatasetMetadata(
        name=f'training_{training_number}',
        shape=data.shape,
        features=list(data.columns[:-1]),
        target=data.columns[-1]
    )
    pipeline.set_metadata(metadata)
    
    # Process data
    processed_data = pipeline.fit_transform(data)
    features = processed_data[:, :-1]
    targets = processed_data[:, -1]
    
    return features, targets

def run_experiment(model_type: str, optimizer_type: str, training_number: int) -> ExperimentResult:
    """
    Run a single experiment.
    
    Args:
        model_type: Type of model to use
        optimizer_type: Type of optimizer to use
        training_number: Training iteration number
        
    Returns:
        Experiment results
    """
    logger.info(f"Starting experiment: {model_type} with {optimizer_type}")
    
    # Setup
    model, optimizer = setup_experiment(model_type, optimizer_type)
    features, targets = load_and_process_data(training_number)
    
    # Define fitness function
    def fitness_function(hyperparameters: np.ndarray) -> float:
        """Evaluate model performance with given hyperparameters."""
        try:
            # Configure model with hyperparameters
            model.configure(hyperparameters)
            
            # Train and evaluate
            history = model.train(features, targets)
            return min(history.history['val_loss'])
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {str(e)}")
            return float('inf')
    
    # Run optimization
    result = optimizer.optimize(
        fitness_function=fitness_function,
        initial_population=None
    )
    
    # Create experiment result
    experiment_result = ExperimentResult(
        model_type=model_type,
        optimizer_type=optimizer_type,
        parameters={
            'model_config': model.get_config(),
            'optimizer_config': optimizer.get_parameters(),
            'best_solution': result.best_solution.tolist(),
            'best_fitness': result.best_fitness
        },
        metrics={
            'best_fitness': result.best_fitness,
            'history': result.history
        }
    )
    
    return experiment_result

def main():
    """Main execution function."""
    # Initialize results manager
    results_manager = ResultsManager(config.paths.results_dir)
    
    # Define experiment configurations
    model_types = ['mlp', 'cnn', 'vgg16']
    optimizer_types = ['dde', 'pso']
    
    # Run experiments
    for model_type in model_types:
        for optimizer_type in optimizer_types:
            for training_num in range(1, config.optimization.samples_count + 1):
                try:
                    # Run experiment
                    result = run_experiment(
                        model_type=model_type,
                        optimizer_type=optimizer_type,
                        training_number=training_num
                    )
                    
                    # Save results
                    results_manager.save_result(result)
                    
                    logger.info(f"Completed experiment: {model_type} with {optimizer_type}")
                except Exception as e:
                    logger.error(f"Error in experiment: {str(e)}")
                    continue

if __name__ == '__main__':
    main()