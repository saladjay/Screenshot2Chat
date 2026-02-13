"""Model Manager Demo

This example demonstrates how to use the ModelManager for model registration,
versioning, and loading.
"""

from datetime import datetime
from pathlib import Path
from screenshot2chat import ModelManager, ModelMetadata


def demo_basic_registration():
    """Demonstrate basic model registration."""
    print("=" * 60)
    print("Demo 1: Basic Model Registration")
    print("=" * 60)
    
    # Create a ModelManager instance
    manager = ModelManager(model_dir="demo_models")
    
    # Create metadata for a text detection model
    metadata = ModelMetadata(
        name="text_detector",
        version="1.0.0",
        model_type="detector",
        framework="paddleocr",
        description="PaddleOCR-based text detection model",
        metrics={
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.905
        },
        tags=["production", "stable"],
        training_params={
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        },
        model_path="text_detector/v1.0.0/model.pdiparams"
    )
    
    # Register the model
    try:
        manager.register(metadata)
        print(f"✓ Registered model: {metadata.name}:{metadata.version}")
        print(f"  Type: {metadata.model_type}")
        print(f"  Framework: {metadata.framework}")
        print(f"  Tags: {metadata.tags}")
        print(f"  Metrics: {metadata.metrics}")
    except ValueError as e:
        print(f"✗ Registration failed: {e}")
    
    print()


def demo_version_management():
    """Demonstrate model version management."""
    print("=" * 60)
    print("Demo 2: Model Version Management")
    print("=" * 60)
    
    manager = ModelManager(model_dir="demo_models")
    
    # Register multiple versions of the same model
    versions = ["1.0.0", "1.1.0", "2.0.0"]
    
    for version in versions:
        metadata = ModelMetadata(
            name="bubble_detector",
            version=version,
            model_type="detector",
            framework="pytorch",
            description=f"Bubble detector version {version}",
            metrics={"accuracy": 0.85 + float(version.split('.')[0]) * 0.05},
            tags=["experimental"] if version == "2.0.0" else ["production"],
            model_path=f"bubble_detector/v{version}/model.pt"
        )
        
        try:
            manager.register(metadata)
            print(f"✓ Registered: bubble_detector:{version}")
        except ValueError:
            print(f"  (Already registered: bubble_detector:{version})")
    
    # List all versions
    print("\nAll versions of bubble_detector:")
    versions = manager.list_versions("bubble_detector")
    for v in versions:
        metadata = manager.get_metadata("bubble_detector", v)
        print(f"  - {v}: {metadata.tags}, accuracy={metadata.metrics.get('accuracy', 'N/A')}")
    
    # List only production versions
    print("\nProduction versions only:")
    prod_versions = manager.list_versions("bubble_detector", tag="production")
    for v in prod_versions:
        print(f"  - {v}")
    
    print()


def demo_model_loading():
    """Demonstrate model loading with version and tag selection."""
    print("=" * 60)
    print("Demo 3: Model Loading")
    print("=" * 60)
    
    manager = ModelManager(model_dir="demo_models")
    
    # Register a test model
    metadata = ModelMetadata(
        name="test_model",
        version="1.0.0",
        model_type="classifier",
        framework="paddleocr",
        description="Test model for loading demo",
        tags=["test"],
        model_path="test_model/v1.0.0/model.txt"
    )
    
    # Create a dummy model file for testing
    model_file = Path("demo_models") / metadata.model_path
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("dummy model content")
    
    try:
        manager.register(metadata)
        print(f"✓ Registered test model: {metadata.name}:{metadata.version}")
    except ValueError:
        print(f"  (Already registered)")
    
    # Load by specific version
    print("\nLoading by specific version:")
    try:
        model = manager.load("test_model", version="1.0.0")
        print(f"✓ Loaded model: {model}")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
    
    # Load latest version
    print("\nLoading latest version:")
    try:
        model = manager.load("test_model", version="latest")
        print(f"✓ Loaded latest model: {model}")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
    
    # Load with tag filter
    print("\nLoading with tag filter:")
    try:
        model = manager.load("test_model", version="latest", tag="test")
        print(f"✓ Loaded model with tag 'test': {model}")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
    
    print()


def demo_metadata_serialization():
    """Demonstrate metadata serialization and deserialization."""
    print("=" * 60)
    print("Demo 4: Metadata Serialization")
    print("=" * 60)
    
    # Create metadata
    original = ModelMetadata(
        name="serialization_test",
        version="1.0.0",
        model_type="extractor",
        framework="custom",
        description="Testing serialization",
        metrics={"score": 0.95},
        tags=["test", "demo"],
        training_params={"param1": "value1"}
    )
    
    print("Original metadata:")
    print(f"  Name: {original.name}")
    print(f"  Version: {original.version}")
    print(f"  Created: {original.created_at}")
    print(f"  Metrics: {original.metrics}")
    
    # Serialize to JSON
    json_str = original.to_json()
    print("\nSerialized to JSON:")
    print(json_str)
    
    # Deserialize from JSON
    restored = ModelMetadata.from_json(json_str)
    print("\nRestored from JSON:")
    print(f"  Name: {restored.name}")
    print(f"  Version: {restored.version}")
    print(f"  Created: {restored.created_at}")
    print(f"  Metrics: {restored.metrics}")
    
    # Verify round-trip
    print("\nRound-trip verification:")
    if original.to_dict() == restored.to_dict():
        print("✓ Serialization round-trip successful!")
    else:
        print("✗ Serialization round-trip failed!")
    
    print()


def demo_model_listing():
    """Demonstrate listing and filtering models."""
    print("=" * 60)
    print("Demo 5: Model Listing and Filtering")
    print("=" * 60)
    
    manager = ModelManager(model_dir="demo_models")
    
    # List all models
    print("All registered models:")
    all_models = manager.list_models()
    for metadata in all_models:
        print(f"  - {metadata.name}:{metadata.version} ({metadata.model_type}, {metadata.framework})")
    
    # Filter by model type
    print("\nDetector models only:")
    detectors = manager.list_models(model_type="detector")
    for metadata in detectors:
        print(f"  - {metadata.name}:{metadata.version}")
    
    # Filter by tag
    print("\nProduction models only:")
    production = manager.list_models(tag="production")
    for metadata in production:
        print(f"  - {metadata.name}:{metadata.version}")
    
    # Combined filters
    print("\nProduction detectors:")
    prod_detectors = manager.list_models(model_type="detector", tag="production")
    for metadata in prod_detectors:
        print(f"  - {metadata.name}:{metadata.version}")
    
    print()


def demo_cache_management():
    """Demonstrate model caching."""
    print("=" * 60)
    print("Demo 6: Model Cache Management")
    print("=" * 60)
    
    manager = ModelManager(model_dir="demo_models")
    
    print(f"Cache size before loading: {len(manager.cache)}")
    
    # Load a model (will be cached)
    try:
        model1 = manager.load("test_model", version="1.0.0")
        print(f"✓ Loaded model (first time)")
        print(f"Cache size after first load: {len(manager.cache)}")
        
        # Load same model again (should use cache)
        model2 = manager.load("test_model", version="1.0.0")
        print(f"✓ Loaded model (second time - from cache)")
        print(f"Cache size after second load: {len(manager.cache)}")
        
        # Verify it's the same instance
        if model1 is model2:
            print("✓ Same model instance returned (cache working)")
        else:
            print("✗ Different instances (cache not working)")
        
        # Clear cache
        manager.clear_cache()
        print(f"\nCache cleared. Size: {len(manager.cache)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def demo_error_handling():
    """Demonstrate error handling."""
    print("=" * 60)
    print("Demo 7: Error Handling")
    print("=" * 60)
    
    manager = ModelManager(model_dir="demo_models")
    
    # Try to load non-existent model
    print("Attempting to load non-existent model:")
    try:
        model = manager.load("nonexistent_model", version="1.0.0")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Try to register duplicate model
    print("\nAttempting to register duplicate model:")
    metadata = ModelMetadata(
        name="text_detector",
        version="1.0.0",
        model_type="detector",
        framework="paddleocr"
    )
    try:
        manager.register(metadata)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Try to load with wrong tag
    print("\nAttempting to load with non-matching tag:")
    try:
        model = manager.load("text_detector", version="1.0.0", tag="nonexistent_tag")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print()


def cleanup():
    """Clean up demo files."""
    import shutil
    demo_dir = Path("demo_models")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
        print("Cleaned up demo files")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODEL MANAGER DEMONSTRATION")
    print("=" * 60 + "\n")
    
    try:
        demo_basic_registration()
        demo_version_management()
        demo_model_loading()
        demo_metadata_serialization()
        demo_model_listing()
        demo_cache_management()
        demo_error_handling()
        
        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    finally:
        # Clean up
        print("\nCleaning up...")
        cleanup()
