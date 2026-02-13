"""Model management system for screenshot2chat.

This module provides model registration, versioning, and loading capabilities.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelMetadata:
    """Model metadata data class.
    
    Stores comprehensive information about a model including its name, version,
    type, framework, training parameters, and performance metrics.
    
    Attributes:
        name: Model name (e.g., "text_detector", "bubble_detector")
        version: Model version string (e.g., "1.0.0", "2.1.3")
        model_type: Type of model (e.g., "detector", "extractor", "classifier")
        framework: Framework used (e.g., "paddleocr", "pytorch", "tensorflow")
        created_at: Timestamp when model was created
        metrics: Performance metrics dictionary (e.g., {"accuracy": 0.95, "mAP": 0.87})
        tags: List of tags for categorization (e.g., ["production", "experimental"])
        description: Human-readable description of the model
        training_params: Training parameters used (e.g., learning rate, epochs)
        model_path: Path to the model file (relative to model directory)
    """
    
    name: str
    version: str
    model_type: str
    framework: str
    created_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    training_params: Dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata with datetime converted to ISO format.
        """
        data = asdict(self)
        # Convert datetime to ISO format string for JSON serialization
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Deserialize metadata from dictionary.
        
        Args:
            data: Dictionary containing metadata fields
            
        Returns:
            ModelMetadata instance
        """
        # Convert ISO format string back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize metadata to JSON string.
        
        Returns:
            JSON string representation of metadata
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Deserialize metadata from JSON string.
        
        Args:
            json_str: JSON string containing metadata
            
        Returns:
            ModelMetadata instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class ModelManager:
    """Model management system.
    
    Provides centralized management of models including registration, versioning,
    loading, and caching. Supports multiple model versions and tags.
    
    Attributes:
        model_dir: Directory where models are stored
        registry: Dictionary mapping model IDs to metadata
        cache: Dictionary caching loaded model instances
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize ModelManager.
        
        Args:
            model_dir: Directory path for storing models (default: "models")
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, ModelMetadata] = {}
        self.cache: Dict[str, Any] = {}
        self._load_registry()
    
    def register(self, metadata: ModelMetadata, model_path: Optional[str] = None) -> None:
        """Register a model with metadata.
        
        Args:
            metadata: ModelMetadata instance containing model information
            model_path: Optional path to model file (if different from metadata.model_path)
            
        Raises:
            ValueError: If model with same name and version already exists
        """
        model_id = f"{metadata.name}:{metadata.version}"
        
        # Check if model already exists
        if model_id in self.registry:
            raise ValueError(
                f"Model {model_id} already registered. "
                f"Use a different version or unregister the existing model first."
            )
        
        # Update model path if provided
        if model_path:
            metadata.model_path = model_path
        
        # Register the model
        self.registry[model_id] = metadata
        self._save_registry()
    
    def load(self, name: str, version: str = "latest", tag: Optional[str] = None) -> Any:
        """Load a model by name and version or tag.
        
        Args:
            name: Model name
            version: Model version (default: "latest" - loads most recent version)
            tag: Optional tag to filter models (e.g., "production")
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model not found or multiple models match criteria
        """
        # Resolve version
        if version == "latest":
            resolved_version = self._get_latest_version(name, tag)
            if not resolved_version:
                raise ValueError(
                    f"No model found with name '{name}'"
                    + (f" and tag '{tag}'" if tag else "")
                )
            version = resolved_version
        
        model_id = f"{name}:{version}"
        
        # Check cache first
        if model_id in self.cache:
            return self.cache[model_id]
        
        # Get metadata
        metadata = self.registry.get(model_id)
        if metadata is None:
            raise ValueError(f"Model not found: {model_id}")
        
        # Verify tag if specified
        if tag and tag not in metadata.tags:
            raise ValueError(
                f"Model {model_id} does not have tag '{tag}'. "
                f"Available tags: {metadata.tags}"
            )
        
        # Load model from disk
        model = self._load_model_by_framework(metadata)
        
        # Cache the loaded model
        self.cache[model_id] = model
        
        return model
    
    def list_versions(self, name: str, tag: Optional[str] = None) -> List[str]:
        """List all versions of a model.
        
        Args:
            name: Model name
            tag: Optional tag to filter versions
            
        Returns:
            Sorted list of version strings
        """
        versions = []
        for model_id, metadata in self.registry.items():
            if metadata.name == name:
                # Filter by tag if specified
                if tag is None or tag in metadata.tags:
                    versions.append(metadata.version)
        
        # Sort versions (simple string sort, could be improved with semantic versioning)
        return sorted(versions)
    
    def list_models(self, model_type: Optional[str] = None, tag: Optional[str] = None) -> List[ModelMetadata]:
        """List all registered models with optional filtering.
        
        Args:
            model_type: Optional model type filter (e.g., "detector")
            tag: Optional tag filter (e.g., "production")
            
        Returns:
            List of ModelMetadata instances matching filters
        """
        models = []
        for metadata in self.registry.values():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if tag and tag not in metadata.tags:
                continue
            models.append(metadata)
        
        return models
    
    def unregister(self, name: str, version: str) -> None:
        """Unregister a model.
        
        Args:
            name: Model name
            version: Model version
            
        Raises:
            ValueError: If model not found
        """
        model_id = f"{name}:{version}"
        
        if model_id not in self.registry:
            raise ValueError(f"Model not found: {model_id}")
        
        # Remove from registry
        del self.registry[model_id]
        
        # Remove from cache if present
        if model_id in self.cache:
            del self.cache[model_id]
        
        self._save_registry()
    
    def get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            ModelMetadata instance or None if not found
        """
        model_id = f"{name}:{version}"
        return self.registry.get(model_id)
    
    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        self.cache.clear()
    
    def _get_latest_version(self, name: str, tag: Optional[str] = None) -> Optional[str]:
        """Get the latest version of a model.
        
        Args:
            name: Model name
            tag: Optional tag filter
            
        Returns:
            Latest version string or None if no versions found
        """
        versions = self.list_versions(name, tag)
        if not versions:
            return None
        
        # Get the model with the most recent created_at timestamp
        latest_metadata = None
        latest_time = None
        
        for version in versions:
            model_id = f"{name}:{version}"
            metadata = self.registry[model_id]
            
            if latest_time is None or metadata.created_at > latest_time:
                latest_time = metadata.created_at
                latest_metadata = metadata
        
        return latest_metadata.version if latest_metadata else None
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_path = self.model_dir / "registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Deserialize each model metadata
            for model_id, metadata_dict in data.items():
                self.registry[model_id] = ModelMetadata.from_dict(metadata_dict)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log warning but don't fail - start with empty registry
            print(f"Warning: Failed to load model registry: {e}")
            self.registry = {}
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.model_dir / "registry.json"
        
        # Serialize registry
        data = {}
        for model_id, metadata in self.registry.items():
            data[model_id] = metadata.to_dict()
        
        # Write to file
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_model_by_framework(self, metadata: ModelMetadata) -> Any:
        """Load model based on framework.
        
        Args:
            metadata: Model metadata containing framework information
            
        Returns:
            Loaded model instance
            
        Raises:
            NotImplementedError: If framework is not supported
            FileNotFoundError: If model file not found
        """
        model_path = self.model_dir / metadata.model_path
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Expected path: {metadata.model_path}"
            )
        
        # Framework-specific loading logic
        if metadata.framework == "paddleocr":
            # PaddleOCR models are typically loaded by the detector itself
            # Return the path for the detector to use
            return str(model_path)
        
        elif metadata.framework == "pytorch":
            try:
                import torch
                return torch.load(model_path)
            except ImportError:
                raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        elif metadata.framework == "tensorflow":
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            except ImportError:
                raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        elif metadata.framework == "onnx":
            try:
                import onnxruntime as ort
                return ort.InferenceSession(str(model_path))
            except ImportError:
                raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
        
        else:
            raise NotImplementedError(
                f"Framework '{metadata.framework}' is not supported. "
                f"Supported frameworks: paddleocr, pytorch, tensorflow, onnx"
            )
