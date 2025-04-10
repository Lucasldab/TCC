"""
Base classes for data processing pipelines.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    shape: tuple
    features: List[str]
    target: str
    statistics: Dict[str, Any] = field(default_factory=dict)

class DataTransformer(ABC):
    """Abstract base class for data transformers."""
    
    @abstractmethod
    def fit(self, data: Any) -> 'DataTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            data: Input data
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform the data.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)

class Pipeline:
    """Data processing pipeline that chains multiple transformers."""
    
    def __init__(self, transformers: List[DataTransformer]):
        """
        Initialize the pipeline.
        
        Args:
            transformers: List of transformer instances
        """
        self.transformers = transformers
        self.metadata: Optional[DatasetMetadata] = None
    
    def fit(self, data: Any) -> 'Pipeline':
        """
        Fit all transformers to the data in sequence.
        
        Args:
            data: Input data
            
        Returns:
            Self for chaining
        """
        transformed_data = data
        for transformer in self.transformers:
            transformed_data = transformer.fit_transform(transformed_data)
        return self
    
    def transform(self, data: Any) -> Any:
        """
        Transform data through all transformers in sequence.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        transformed_data = data
        for transformer in self.transformers:
            transformed_data = transformer.transform(transformed_data)
        return transformed_data
    
    def fit_transform(self, data: Any) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def add_transformer(self, transformer: DataTransformer) -> 'Pipeline':
        """
        Add a transformer to the pipeline.
        
        Args:
            transformer: Transformer instance to add
            
        Returns:
            Self for chaining
        """
        self.transformers.append(transformer)
        return self
    
    def set_metadata(self, metadata: DatasetMetadata) -> 'Pipeline':
        """
        Set dataset metadata.
        
        Args:
            metadata: Dataset metadata instance
            
        Returns:
            Self for chaining
        """
        self.metadata = metadata
        return self
    
    def get_metadata(self) -> Optional[DatasetMetadata]:
        """
        Get dataset metadata.
        
        Returns:
            Dataset metadata if set, None otherwise
        """
        return self.metadata 