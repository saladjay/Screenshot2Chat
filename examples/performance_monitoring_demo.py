"""
Performance Monitoring Demo

This example demonstrates how to use the performance monitoring features
in the screenshot analysis pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors.text_detector import TextDetector
from screenshot2chat.extractors.nickname_extractor import NicknameExtractor


def main():
    """Demonstrate performance monitoring features."""
    
    print("=" * 70)
    print("Performance Monitoring Demo")
    print("=" * 70)
    print()
    
    # Create a pipeline with monitoring enabled
    print("1. Creating pipeline with performance monitoring enabled...")
    pipeline = Pipeline(name="monitored_pipeline", enable_monitoring=True)
    
    # Add steps
    text_detector = TextDetector(config={
        'backend': 'paddleocr',
        'model_dir': 'models/PP-OCRv5_server_det/'
    })
    
    nickname_extractor = NicknameExtractor(config={
        'source': 'text_detection',
        'top_k': 3
    })
    
    pipeline.add_step(PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector,
        config={'backend': 'paddleocr'}
    ))
    
    pipeline.add_step(PipelineStep(
        name="nickname_extraction",
        step_type=StepType.EXTRACTOR,
        component=nickname_extractor,
        config={'source': 'text_detection'},
        depends_on=['text_detection']
    ))
    
    print("✓ Pipeline created with 2 steps")
    print()
    
    # Execute pipeline multiple times to collect metrics
    print("2. Executing pipeline multiple times to collect metrics...")
    
    # Create test images
    test_images = [
        np.random.randint(0, 255, (1080, 720, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    
    for i, image in enumerate(test_images, 1):
        print(f"   Execution {i}...")
        try:
            results = pipeline.execute(image)
            
            # Get metrics from last execution
            metrics = pipeline.get_performance_metrics()
            print(f"   ✓ Completed in {sum(m['duration'] for m in metrics.values()):.3f}s")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print()
    
    # Display performance statistics
    print("3. Performance Statistics:")
    print("-" * 70)
    stats = pipeline.get_performance_stats()
    
    for step_name, step_stats in stats.items():
        print(f"\n{step_name}:")
        if 'duration_mean' in step_stats:
            print(f"  Average Duration: {step_stats['duration_mean']:.3f}s")
            print(f"  Min Duration:     {step_stats['duration_min']:.3f}s")
            print(f"  Max Duration:     {step_stats['duration_max']:.3f}s")
        if 'memory_delta_mean' in step_stats:
            print(f"  Avg Memory Delta: {step_stats['memory_delta_mean']:+.2f} MB")
    
    print()
    print("-" * 70)
    print()
    
    # Generate full report
    print("4. Full Performance Report:")
    print()
    report = pipeline.get_performance_report(detailed=False)
    print(report)
    
    # Export metrics to structured format
    print("\n5. Exporting metrics to structured format...")
    exported = pipeline.export_metrics()
    print(f"✓ Exported metrics for {len(exported['steps'])} steps")
    print(f"  Timestamp: {exported['timestamp']}")
    print()
    
    # Demonstrate enabling/disabling monitoring
    print("6. Demonstrating enable/disable monitoring:")
    print("   Disabling monitoring...")
    pipeline.disable_monitoring()
    
    print("   Executing with monitoring disabled...")
    try:
        results = pipeline.execute(test_images[0])
        print("   ✓ Execution completed (no metrics recorded)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    print("   Re-enabling monitoring...")
    pipeline.enable_monitoring()
    
    print("   Executing with monitoring enabled...")
    try:
        results = pipeline.execute(test_images[0])
        metrics = pipeline.get_performance_metrics()
        print(f"   ✓ Execution completed (metrics recorded)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Clear metrics
    print("7. Clearing metrics...")
    pipeline.clear_metrics()
    stats_after_clear = pipeline.get_performance_stats()
    print(f"✓ Metrics cleared (steps remaining: {len(stats_after_clear)})")
    print()
    
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
