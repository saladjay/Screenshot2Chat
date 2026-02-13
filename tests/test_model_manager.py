"""Unit tests for ModelManager and ModelMetadata."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from screenshot2chat import ModelManager, ModelMetadata


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_manager(temp_model_dir):
    """Create a ModelManager instance with temporary directory."""
    return ModelManager(model_dir=temp_model_dir)


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        name="test_model",
        version="1.0.0",
        model_type="detector",
        framework="paddleocr",
        description="Test model",
        metrics={"accuracy": 0.95},
        tags=["test", "production"],
        training_params={"lr": 0.001}
    )


class TestModelMetadata:
    """Tests for ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            name="test",
            version="1.0.0",
            model_type="detector",
            framework="pytorch"
        )
        
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "detector"
        assert metadata.framework == "pytorch"
        assert isinstance(metadata.created_at, datetime)
        assert metadata.metrics == {}
        assert metadata.tags == []
        assert metadata.description == ""
    
    def test_metadata_with_all_fields(self, sample_metadata):
        """Test creating metadata with all fields."""
        assert sample_metadata.name == "test_model"
        assert sample_metadata.metrics["accuracy"] == 0.95
        assert "test" in sample_metadata.tags
        assert sample_metadata.training_params["lr"] == 0.001
    
    def test_metadata_serialization_to_dict(self, sample_metadata):
        """Test serializing metadata to dictionary."""
        data = sample_metadata.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert isinstance(data["created_at"], str)  # Should be ISO format
        assert data["metrics"]["accuracy"] == 0.95
    
    def test_metadata_deserialization_from_dict(self, sample_metadata):
        """Test deserializing metadata from dictionary."""
        data = sample_metadata.to_dict()
        restored = ModelMetadata.from_dict(data)
        
        assert restored.name == sample_metadata.name
        assert restored.version == sample_metadata.version
        assert restored.model_type == sample_metadata.model_type
        assert restored.metrics == sample_metadata.metrics
    
    def test_metadata_json_serialization(self, sample_metadata):
        """Test JSON serialization."""
        json_str = sample_metadata.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["name"] == "test_model"
    
    def test_metadata_json_deserialization(self, sample_metadata):
        """Test JSON deserialization."""
        json_str = sample_metadata.to_json()
        restored = ModelMetadata.from_json(json_str)
        
        assert restored.name == sample_metadata.name
        assert restored.version == sample_metadata.version
    
    def test_metadata_round_trip(self, sample_metadata):
        """Test serialization round-trip preserves data."""
        json_str = sample_metadata.to_json()
        restored = ModelMetadata.from_json(json_str)
        
        assert sample_metadata.to_dict() == restored.to_dict()


class TestModelManager:
    """Tests for ModelManager class."""
    
    def test_manager_initialization(self, temp_model_dir):
        """Test ModelManager initialization."""
        manager = ModelManager(model_dir=temp_model_dir)
        
        assert manager.model_dir == Path(temp_model_dir)
        assert manager.model_dir.exists()
        assert manager.registry == {}
        assert manager.cache == {}
    
    def test_register_model(self, model_manager, sample_metadata):
        """Test registering a model."""
        model_manager.register(sample_metadata)
        
        model_id = f"{sample_metadata.name}:{sample_metadata.version}"
        assert model_id in model_manager.registry
        assert model_manager.registry[model_id] == sample_metadata
    
    def test_register_duplicate_model_raises_error(self, model_manager, sample_metadata):
        """Test registering duplicate model raises ValueError."""
        model_manager.register(sample_metadata)
        
        with pytest.raises(ValueError, match="already registered"):
            model_manager.register(sample_metadata)
    
    def test_register_with_custom_path(self, model_manager, sample_metadata):
        """Test registering model with custom path."""
        custom_path = "custom/path/model.pt"
        model_manager.register(sample_metadata, model_path=custom_path)
        
        model_id = f"{sample_metadata.name}:{sample_metadata.version}"
        assert model_manager.registry[model_id].model_path == custom_path
    
    def test_get_metadata(self, model_manager, sample_metadata):
        """Test retrieving model metadata."""
        model_manager.register(sample_metadata)
        
        retrieved = model_manager.get_metadata(
            sample_metadata.name,
            sample_metadata.version
        )
        
        assert retrieved == sample_metadata
    
    def test_get_metadata_nonexistent(self, model_manager):
        """Test retrieving non-existent metadata returns None."""
        result = model_manager.get_metadata("nonexistent", "1.0.0")
        assert result is None
    
    def test_list_versions(self, model_manager):
        """Test listing model versions."""
        # Register multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            metadata = ModelMetadata(
                name="test_model",
                version=version,
                model_type="detector",
                framework="pytorch"
            )
            model_manager.register(metadata)
        
        versions = model_manager.list_versions("test_model")
        assert versions == ["1.0.0", "1.1.0", "2.0.0"]
    
    def test_list_versions_with_tag_filter(self, model_manager):
        """Test listing versions with tag filter."""
        # Register versions with different tags
        for version, tags in [("1.0.0", ["prod"]), ("1.1.0", ["prod"]), ("2.0.0", ["dev"])]:
            metadata = ModelMetadata(
                name="test_model",
                version=version,
                model_type="detector",
                framework="pytorch",
                tags=tags
            )
            model_manager.register(metadata)
        
        prod_versions = model_manager.list_versions("test_model", tag="prod")
        assert prod_versions == ["1.0.0", "1.1.0"]
        
        dev_versions = model_manager.list_versions("test_model", tag="dev")
        assert dev_versions == ["2.0.0"]
    
    def test_list_models(self, model_manager):
        """Test listing all models."""
        # Register multiple models
        for name in ["model1", "model2"]:
            metadata = ModelMetadata(
                name=name,
                version="1.0.0",
                model_type="detector",
                framework="pytorch"
            )
            model_manager.register(metadata)
        
        models = model_manager.list_models()
        assert len(models) == 2
        assert any(m.name == "model1" for m in models)
        assert any(m.name == "model2" for m in models)
    
    def test_list_models_with_type_filter(self, model_manager):
        """Test listing models with type filter."""
        # Register models of different types
        for model_type in ["detector", "extractor"]:
            metadata = ModelMetadata(
                name=f"{model_type}_model",
                version="1.0.0",
                model_type=model_type,
                framework="pytorch"
            )
            model_manager.register(metadata)
        
        detectors = model_manager.list_models(model_type="detector")
        assert len(detectors) == 1
        assert detectors[0].model_type == "detector"
    
    def test_list_models_with_tag_filter(self, model_manager):
        """Test listing models with tag filter."""
        # Register models with different tags
        for tags in [["prod"], ["dev"]]:
            metadata = ModelMetadata(
                name=f"model_{tags[0]}",
                version="1.0.0",
                model_type="detector",
                framework="pytorch",
                tags=tags
            )
            model_manager.register(metadata)
        
        prod_models = model_manager.list_models(tag="prod")
        assert len(prod_models) == 1
        assert "prod" in prod_models[0].tags
    
    def test_unregister_model(self, model_manager, sample_metadata):
        """Test unregistering a model."""
        model_manager.register(sample_metadata)
        
        model_manager.unregister(sample_metadata.name, sample_metadata.version)
        
        model_id = f"{sample_metadata.name}:{sample_metadata.version}"
        assert model_id not in model_manager.registry
    
    def test_unregister_nonexistent_raises_error(self, model_manager):
        """Test unregistering non-existent model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            model_manager.unregister("nonexistent", "1.0.0")
    
    def test_unregister_clears_cache(self, model_manager, sample_metadata, temp_model_dir):
        """Test unregistering removes model from cache."""
        # Create dummy model file
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        
        model_manager.register(sample_metadata)
        
        # Load model to cache it
        try:
            model_manager.load(sample_metadata.name, sample_metadata.version)
        except:
            pass  # May fail due to framework, but should be in cache
        
        # Unregister
        model_manager.unregister(sample_metadata.name, sample_metadata.version)
        
        model_id = f"{sample_metadata.name}:{sample_metadata.version}"
        assert model_id not in model_manager.cache
    
    def test_load_model_by_version(self, model_manager, sample_metadata, temp_model_dir):
        """Test loading model by specific version."""
        # Create dummy model file
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        
        model_manager.register(sample_metadata)
        
        model = model_manager.load(sample_metadata.name, sample_metadata.version)
        assert model is not None
    
    def test_load_nonexistent_model_raises_error(self, model_manager):
        """Test loading non-existent model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            model_manager.load("nonexistent", "1.0.0")
    
    def test_load_with_wrong_tag_raises_error(self, model_manager, sample_metadata, temp_model_dir):
        """Test loading with non-matching tag raises ValueError."""
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        
        model_manager.register(sample_metadata)
        
        with pytest.raises(ValueError, match="does not have tag"):
            model_manager.load(sample_metadata.name, sample_metadata.version, tag="nonexistent")
    
    def test_load_latest_version(self, model_manager, temp_model_dir):
        """Test loading latest version."""
        # Register multiple versions with different timestamps
        import time
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            metadata = ModelMetadata(
                name="test_model",
                version=version,
                model_type="detector",
                framework="paddleocr",
                model_path=f"test_{version}.txt"
            )
            # Create dummy file
            model_path = Path(temp_model_dir) / metadata.model_path
            model_path.write_text("dummy")
            
            model_manager.register(metadata)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Load latest
        model = model_manager.load("test_model", version="latest")
        assert model is not None
    
    def test_model_caching(self, model_manager, sample_metadata, temp_model_dir):
        """Test that models are cached after loading."""
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        
        model_manager.register(sample_metadata)
        
        # Load model
        model1 = model_manager.load(sample_metadata.name, sample_metadata.version)
        
        # Load again - should use cache
        model2 = model_manager.load(sample_metadata.name, sample_metadata.version)
        
        # Should be same instance
        assert model1 is model2
    
    def test_clear_cache(self, model_manager, sample_metadata, temp_model_dir):
        """Test clearing model cache."""
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        
        model_manager.register(sample_metadata)
        model_manager.load(sample_metadata.name, sample_metadata.version)
        
        assert len(model_manager.cache) > 0
        
        model_manager.clear_cache()
        
        assert len(model_manager.cache) == 0
    
    def test_registry_persistence(self, temp_model_dir, sample_metadata):
        """Test that registry is persisted to disk."""
        # Create manager and register model
        manager1 = ModelManager(model_dir=temp_model_dir)
        manager1.register(sample_metadata)
        
        # Create new manager instance - should load registry
        manager2 = ModelManager(model_dir=temp_model_dir)
        
        model_id = f"{sample_metadata.name}:{sample_metadata.version}"
        assert model_id in manager2.registry
        assert manager2.registry[model_id].name == sample_metadata.name
    
    def test_load_missing_model_file_raises_error(self, model_manager, sample_metadata):
        """Test loading model with missing file raises FileNotFoundError."""
        sample_metadata.model_path = "nonexistent/model.txt"
        model_manager.register(sample_metadata)
        
        with pytest.raises(FileNotFoundError, match="not found"):
            model_manager.load(sample_metadata.name, sample_metadata.version)
    
    def test_unsupported_framework_raises_error(self, model_manager, sample_metadata, temp_model_dir):
        """Test loading model with unsupported framework raises NotImplementedError."""
        model_path = Path(temp_model_dir) / "test_model.txt"
        model_path.write_text("dummy")
        sample_metadata.model_path = "test_model.txt"
        sample_metadata.framework = "unsupported_framework"
        
        model_manager.register(sample_metadata)
        
        with pytest.raises(NotImplementedError, match="not supported"):
            model_manager.load(sample_metadata.name, sample_metadata.version)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
