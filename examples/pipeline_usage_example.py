"""
Example demonstrating Pipeline usage with actual detectors and extractors.

This example shows how to:
1. Create a pipeline programmatically
2. Add detection and extraction steps
3. Configure dependencies between steps
4. Execute the pipeline on an image
5. Save and load pipeline configurations
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.detectors.text_detector import TextDetector
from src.screenshot2chat.detectors.bubble_detector import BubbleDetector
from src.screenshot2chat.extractors.nickname_extractor import NicknameExtractor
from src.screenshot2chat.extractors.speaker_extractor import SpeakerExtractor
from src.screenshot2chat.extractors.layout_extractor import LayoutExtractor


def example_1_basic_pipeline():
    """Example 1: Create a basic pipeline programmatically."""
    print("=" * 60)
    print("Example 1: Basic Pipeline Creation")
    print("=" * 60)
    
    # Create pipeline
    pipeline = Pipeline(name="basic_chat_analysis")
    
    # Add text detection step
    text_detector = TextDetector(config={
        'backend': 'paddleocr',
        'model_dir': 'models/PP-OCRv5_server_det/'
    })
    
    text_step = PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector,
        config={'backend': 'paddleocr'}
    )
    pipeline.add_step(text_step)
    
    # Add layout extraction step
    layout_extractor = LayoutExtractor()
    layout_step = PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=['text_detection']
    )
    pipeline.add_step(layout_step)
    
    print(f"✓ Created pipeline: {pipeline}")
    print(f"  Steps: {[step.name for step in pipeline.steps]}")
    print()


def example_2_complex_pipeline():
    """Example 2: Create a complex pipeline with multiple dependencies."""
    print("=" * 60)
    print("Example 2: Complex Pipeline with Dependencies")
    print("=" * 60)
    
    pipeline = Pipeline(name="full_chat_analysis")
    
    # Step 1: Text detection
    text_detector = TextDetector()
    pipeline.add_step(PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector
    ))
    
    # Step 2: Bubble detection (depends on text detection)
    bubble_detector = BubbleDetector(config={'screen_width': 720})
    pipeline.add_step(PipelineStep(
        name="bubble_detection",
        step_type=StepType.DETECTOR,
        component=bubble_detector,
        depends_on=["text_detection"]
    ))
    
    # Step 3: Nickname extraction (depends on text detection)
    nickname_extractor = NicknameExtractor(config={'top_k': 3})
    pipeline.add_step(PipelineStep(
        name="nickname_extraction",
        step_type=StepType.EXTRACTOR,
        component=nickname_extractor,
        config={'source': 'text_detection'},
        depends_on=["text_detection"]
    ))
    
    # Step 4: Speaker extraction (depends on bubble detection)
    speaker_extractor = SpeakerExtractor()
    pipeline.add_step(PipelineStep(
        name="speaker_extraction",
        step_type=StepType.EXTRACTOR,
        component=speaker_extractor,
        config={'source': 'bubble_detection'},
        depends_on=["bubble_detection"]
    ))
    
    # Step 5: Layout extraction (depends on text detection)
    layout_extractor = LayoutExtractor()
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=["text_detection"]
    ))
    
    print(f"✓ Created complex pipeline: {pipeline}")
    print(f"  Steps: {[step.name for step in pipeline.steps]}")
    
    # Validate pipeline
    try:
        pipeline.validate()
        print("✓ Pipeline validation passed")
    except ValueError as e:
        print(f"✗ Pipeline validation failed: {e}")
    
    # Show execution order
    execution_order = pipeline._get_execution_order()
    print(f"  Execution order: {[step.name for step in execution_order]}")
    print()


def example_3_save_load_config():
    """Example 3: Save and load pipeline configuration."""
    print("=" * 60)
    print("Example 3: Save and Load Pipeline Configuration")
    print("=" * 60)
    
    # Create pipeline
    pipeline = Pipeline(name="saved_pipeline")
    
    text_detector = TextDetector()
    pipeline.add_step(PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector,
        config={'backend': 'paddleocr'}
    ))
    
    layout_extractor = LayoutExtractor()
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=['text_detection']
    ))
    
    # Save to YAML
    config_path = "pipeline_config_example.yaml"
    pipeline.save(config_path)
    print(f"✓ Saved pipeline configuration to: {config_path}")
    
    # Show saved config
    with open(config_path, 'r') as f:
        print("\nSaved configuration:")
        print("-" * 40)
        print(f.read())
        print("-" * 40)
    
    print()


def example_4_validation_errors():
    """Example 4: Demonstrate validation error handling."""
    print("=" * 60)
    print("Example 4: Validation Error Handling")
    print("=" * 60)
    
    # Test 1: Missing dependency
    print("\nTest 1: Missing dependency")
    pipeline = Pipeline(name="invalid_pipeline")
    
    extractor = LayoutExtractor()
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=extractor,
        depends_on=["nonexistent_step"]
    ))
    
    try:
        pipeline.validate()
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 2: Circular dependency
    print("\nTest 2: Circular dependency")
    pipeline = Pipeline(name="circular_pipeline")
    
    detector1 = TextDetector()
    pipeline.add_step(PipelineStep(
        name="step1",
        step_type=StepType.DETECTOR,
        component=detector1,
        depends_on=["step2"]
    ))
    
    detector2 = TextDetector()
    pipeline.add_step(PipelineStep(
        name="step2",
        step_type=StepType.DETECTOR,
        component=detector2,
        depends_on=["step1"]
    ))
    
    try:
        pipeline.validate()
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 3: Duplicate step names
    print("\nTest 3: Duplicate step names")
    pipeline = Pipeline(name="duplicate_pipeline")
    
    detector1 = TextDetector()
    pipeline.add_step(PipelineStep(
        name="detector",
        step_type=StepType.DETECTOR,
        component=detector1
    ))
    
    try:
        detector2 = TextDetector()
        pipeline.add_step(PipelineStep(
            name="detector",  # Same name
            step_type=StepType.DETECTOR,
            component=detector2
        ))
        print("✗ Should have failed when adding duplicate")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print()


def example_5_execution_order():
    """Example 5: Demonstrate execution order with dependencies."""
    print("=" * 60)
    print("Example 5: Execution Order with Dependencies")
    print("=" * 60)
    
    pipeline = Pipeline(name="ordered_pipeline")
    
    # Add steps in random order
    steps_to_add = [
        ("step_c", ["step_a", "step_b"]),
        ("step_a", []),
        ("step_d", ["step_c"]),
        ("step_b", ["step_a"]),
    ]
    
    for name, deps in steps_to_add:
        detector = TextDetector()
        pipeline.add_step(PipelineStep(
            name=name,
            step_type=StepType.DETECTOR,
            component=detector,
            depends_on=deps
        ))
    
    print("Steps added in order:", [name for name, _ in steps_to_add])
    
    # Get execution order
    execution_order = pipeline._get_execution_order()
    print("Execution order:", [step.name for step in execution_order])
    
    # Verify order is correct
    order_names = [step.name for step in execution_order]
    assert order_names.index("step_a") < order_names.index("step_b")
    assert order_names.index("step_a") < order_names.index("step_c")
    assert order_names.index("step_b") < order_names.index("step_c")
    assert order_names.index("step_c") < order_names.index("step_d")
    
    print("✓ Execution order respects all dependencies")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Pipeline Usage Examples")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_pipeline()
        example_2_complex_pipeline()
        example_3_save_load_config()
        example_4_validation_errors()
        example_5_execution_order()
        
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
