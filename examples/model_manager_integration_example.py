"""ModelManager Integration Example

This example demonstrates how to use ModelManager with the screenshot2chat
pipeline for managing detector and extractor models.
"""

from pathlib import Path
from screenshot2chat import (
    ModelManager,
    ModelMetadata,
    Pipeline,
    PipelineStep,
    StepType,
    TextDetector,
    BubbleDetector,
    NicknameExtractor
)


def setup_model_registry():
    """Set up a model registry with detector and extractor models."""
    print("=" * 60)
    print("Setting up Model Registry")
    print("=" * 60)
    
    manager = ModelManager(model_dir="models")
    
    # Register text detection model
    text_detector_metadata = ModelMetadata(
        name="text_detector",
        version="1.0.0",
        model_type="detector",
        framework="paddleocr",
        description="PaddleOCR v5 server detection model",
        metrics={
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.905
        },
        tags=["production", "ocr"],
        model_path="PP-OCRv5_server_det/"
    )
    
    try:
        manager.register(text_detector_metadata)
        print(f"✓ Registered: {text_detector_metadata.name}:{text_detector_metadata.version}")
    except ValueError:
        print(f"  (Already registered: {text_detector_metadata.name}:{text_detector_metadata.version})")
    
    # Register bubble detection model
    bubble_detector_metadata = ModelMetadata(
        name="bubble_detector",
        version="1.0.0",
        model_type="detector",
        framework="custom",
        description="KMeans-based chat bubble detector",
        metrics={
            "accuracy": 0.87,
            "layout_detection_rate": 0.92
        },
        tags=["production", "layout"],
        training_params={
            "n_clusters": 2,
            "algorithm": "kmeans"
        }
    )
    
    try:
        manager.register(bubble_detector_metadata)
        print(f"✓ Registered: {bubble_detector_metadata.name}:{bubble_detector_metadata.version}")
    except ValueError:
        print(f"  (Already registered: {bubble_detector_metadata.name}:{bubble_detector_metadata.version})")
    
    # Register nickname extractor model
    nickname_extractor_metadata = ModelMetadata(
        name="nickname_extractor",
        version="1.0.0",
        model_type="extractor",
        framework="custom",
        description="Scoring-based nickname extraction",
        metrics={
            "top1_accuracy": 0.85,
            "top3_accuracy": 0.95
        },
        tags=["production", "extraction"],
        training_params={
            "top_k": 3,
            "min_top_margin_ratio": 0.05
        }
    )
    
    try:
        manager.register(nickname_extractor_metadata)
        print(f"✓ Registered: {nickname_extractor_metadata.name}:{nickname_extractor_metadata.version}")
    except ValueError:
        print(f"  (Already registered: {nickname_extractor_metadata.name}:{nickname_extractor_metadata.version})")
    
    print()
    return manager


def list_available_models(manager):
    """List all available models in the registry."""
    print("=" * 60)
    print("Available Models")
    print("=" * 60)
    
    # List all models
    all_models = manager.list_models()
    print(f"\nTotal models: {len(all_models)}")
    
    # Group by type
    detectors = manager.list_models(model_type="detector")
    extractors = manager.list_models(model_type="extractor")
    
    print(f"\nDetectors ({len(detectors)}):")
    for metadata in detectors:
        print(f"  - {metadata.name}:{metadata.version}")
        print(f"    Framework: {metadata.framework}")
        print(f"    Tags: {metadata.tags}")
        if metadata.metrics:
            print(f"    Metrics: {metadata.metrics}")
    
    print(f"\nExtractors ({len(extractors)}):")
    for metadata in extractors:
        print(f"  - {metadata.name}:{metadata.version}")
        print(f"    Framework: {metadata.framework}")
        print(f"    Tags: {metadata.tags}")
        if metadata.metrics:
            print(f"    Metrics: {metadata.metrics}")
    
    print()


def demonstrate_version_management(manager):
    """Demonstrate version management capabilities."""
    print("=" * 60)
    print("Version Management")
    print("=" * 60)
    
    # Register multiple versions of text detector
    versions = ["1.0.0", "1.1.0", "2.0.0-beta"]
    
    for version in versions[1:]:  # Skip 1.0.0 as it's already registered
        metadata = ModelMetadata(
            name="text_detector",
            version=version,
            model_type="detector",
            framework="paddleocr",
            description=f"PaddleOCR text detector version {version}",
            metrics={"f1_score": 0.90 + float(version.split('.')[0]) * 0.02},
            tags=["experimental"] if "beta" in version else ["production"],
            model_path=f"PP-OCRv5_server_det_v{version}/"
        )
        
        try:
            manager.register(metadata)
            print(f"✓ Registered: text_detector:{version}")
        except ValueError:
            print(f"  (Already registered: text_detector:{version})")
    
    # List all versions
    print("\nAll versions of text_detector:")
    versions = manager.list_versions("text_detector")
    for v in versions:
        metadata = manager.get_metadata("text_detector", v)
        print(f"  - {v}: {metadata.tags}, f1={metadata.metrics.get('f1_score', 'N/A')}")
    
    # List production versions only
    print("\nProduction versions:")
    prod_versions = manager.list_versions("text_detector", tag="production")
    for v in prod_versions:
        print(f"  - {v}")
    
    print()


def demonstrate_model_selection(manager):
    """Demonstrate model selection strategies."""
    print("=" * 60)
    print("Model Selection Strategies")
    print("=" * 60)
    
    # Strategy 1: Use latest production version
    print("\nStrategy 1: Latest Production Version")
    try:
        # Get latest production version
        prod_versions = manager.list_versions("text_detector", tag="production")
        if prod_versions:
            latest_prod = prod_versions[-1]
            metadata = manager.get_metadata("text_detector", latest_prod)
            print(f"  Selected: text_detector:{latest_prod}")
            print(f"  Metrics: {metadata.metrics}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Strategy 2: Use highest performing model
    print("\nStrategy 2: Highest Performing Model")
    try:
        detectors = manager.list_models(model_type="detector")
        text_detectors = [d for d in detectors if d.name == "text_detector"]
        
        if text_detectors:
            best = max(text_detectors, key=lambda m: m.metrics.get('f1_score', 0))
            print(f"  Selected: {best.name}:{best.version}")
            print(f"  F1 Score: {best.metrics.get('f1_score')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Strategy 3: Use specific version for reproducibility
    print("\nStrategy 3: Specific Version (Reproducibility)")
    try:
        metadata = manager.get_metadata("text_detector", "1.0.0")
        if metadata:
            print(f"  Selected: text_detector:1.0.0")
            print(f"  Ensures reproducible results")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


def demonstrate_metadata_tracking(manager):
    """Demonstrate metadata tracking and comparison."""
    print("=" * 60)
    print("Metadata Tracking and Comparison")
    print("=" * 60)
    
    # Compare different versions
    print("\nComparing text_detector versions:")
    versions = manager.list_versions("text_detector")
    
    print(f"\n{'Version':<15} {'Tags':<20} {'F1 Score':<10}")
    print("-" * 45)
    
    for version in versions:
        metadata = manager.get_metadata("text_detector", version)
        tags_str = ", ".join(metadata.tags)
        f1_score = metadata.metrics.get('f1_score', 'N/A')
        print(f"{version:<15} {tags_str:<20} {f1_score:<10}")
    
    # Show training parameters
    print("\nTraining Parameters:")
    for version in versions:
        metadata = manager.get_metadata("text_detector", version)
        if metadata.training_params:
            print(f"  {version}: {metadata.training_params}")
    
    print()


def demonstrate_pipeline_integration(manager):
    """Demonstrate using ModelManager with Pipeline."""
    print("=" * 60)
    print("Pipeline Integration")
    print("=" * 60)
    
    print("\nCreating pipeline with managed models...")
    
    # Get model metadata
    text_detector_meta = manager.get_metadata("text_detector", "1.0.0")
    bubble_detector_meta = manager.get_metadata("bubble_detector", "1.0.0")
    nickname_extractor_meta = manager.get_metadata("nickname_extractor", "1.0.0")
    
    if text_detector_meta and bubble_detector_meta and nickname_extractor_meta:
        print(f"✓ Using text_detector:{text_detector_meta.version}")
        print(f"  Metrics: {text_detector_meta.metrics}")
        
        print(f"✓ Using bubble_detector:{bubble_detector_meta.version}")
        print(f"  Metrics: {bubble_detector_meta.metrics}")
        
        print(f"✓ Using nickname_extractor:{nickname_extractor_meta.version}")
        print(f"  Metrics: {nickname_extractor_meta.metrics}")
        
        print("\nPipeline configuration:")
        print("  1. Text Detection (PaddleOCR)")
        print("  2. Bubble Detection (KMeans)")
        print("  3. Nickname Extraction (Scoring)")
        
        print("\nNote: Actual pipeline execution would use these model configurations")
    else:
        print("✗ Some models not found in registry")
    
    print()


def demonstrate_model_lifecycle(manager):
    """Demonstrate complete model lifecycle."""
    print("=" * 60)
    print("Model Lifecycle Management")
    print("=" * 60)
    
    # 1. Register experimental model
    print("\n1. Register Experimental Model")
    exp_metadata = ModelMetadata(
        name="experimental_detector",
        version="0.1.0",
        model_type="detector",
        framework="pytorch",
        description="Experimental deep learning detector",
        metrics={"accuracy": 0.75},
        tags=["experimental", "research"]
    )
    
    try:
        manager.register(exp_metadata)
        print(f"✓ Registered experimental model")
    except ValueError:
        print(f"  (Already registered)")
    
    # 2. Promote to production
    print("\n2. Promote to Production (new version)")
    prod_metadata = ModelMetadata(
        name="experimental_detector",
        version="1.0.0",
        model_type="detector",
        framework="pytorch",
        description="Production-ready deep learning detector",
        metrics={"accuracy": 0.92},
        tags=["production", "validated"]
    )
    
    try:
        manager.register(prod_metadata)
        print(f"✓ Promoted to production as v1.0.0")
    except ValueError:
        print(f"  (Already registered)")
    
    # 3. List all versions
    print("\n3. Version History")
    versions = manager.list_versions("experimental_detector")
    for v in versions:
        metadata = manager.get_metadata("experimental_detector", v)
        print(f"  - {v}: {metadata.tags}, accuracy={metadata.metrics.get('accuracy')}")
    
    # 4. Deprecate old version
    print("\n4. Deprecate Experimental Version")
    print("  (In practice, would update tags or unregister)")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODEL MANAGER INTEGRATION DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Set up registry
    manager = setup_model_registry()
    
    # Demonstrate features
    list_available_models(manager)
    demonstrate_version_management(manager)
    demonstrate_model_selection(manager)
    demonstrate_metadata_tracking(manager)
    demonstrate_pipeline_integration(manager)
    demonstrate_model_lifecycle(manager)
    
    print("=" * 60)
    print("Integration demonstration completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • ModelManager provides centralized model registry")
    print("  • Version management enables experimentation and rollback")
    print("  • Metadata tracking helps compare and select models")
    print("  • Seamless integration with Pipeline components")
    print("  • Complete lifecycle management from research to production")
