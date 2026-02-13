"""
Demonstration of conditional branching and parallel execution in Pipeline.

This example shows:
1. How to use conditional branching to execute steps based on intermediate results
2. How to configure parallel execution for independent steps
3. How to combine both features for complex workflows
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.core.base_detector import BaseDetector, DetectionResult
from screenshot2chat.core.base_extractor import BaseExtractor, ExtractionResult


# Simple mock components for demonstration
class SimpleDetector(BaseDetector):
    """A simple detector that returns a configurable number of results."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.count = config.get('count', 5) if config else 5
        self.name = config.get('name', 'detector') if config else 'detector'
    
    def load_model(self):
        pass
    
    def detect(self, image):
        print(f"  {self.name}: Detecting {self.count} items...")
        results = []
        for i in range(self.count):
            results.append(DetectionResult(
                bbox=[i*10, i*10, i*10+50, i*10+50],
                score=0.9,
                category='item',
                metadata={'index': i}
            ))
        return results


class SimpleExtractor(BaseExtractor):
    """A simple extractor for demonstration."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = config.get('name', 'extractor') if config else 'extractor'
    
    def extract(self, detection_results, image=None):
        print(f"  {self.name}: Processing {len(detection_results)} detections...")
        return ExtractionResult(
            data={
                'count': len(detection_results),
                'processor': self.name
            },
            confidence=0.9
        )


def demo_conditional_branching():
    """Demonstrate conditional branching."""
    print("\n" + "="*70)
    print("DEMO 1: Conditional Branching")
    print("="*70)
    print("\nConditional branching allows steps to execute only when conditions are met.")
    print("This example shows different processing paths based on detection count.\n")
    
    # Create pipeline
    pipeline = Pipeline(name="conditional_demo", enable_monitoring=True)
    
    # Step 1: Initial detection
    detector = PipelineStep(
        name="detector",
        step_type=StepType.DETECTOR,
        component=SimpleDetector(config={'count': 8, 'name': 'Initial Detector'}),
        config={'count': 8}
    )
    pipeline.add_step(detector)
    
    # Step 2: Process if many items detected (> 5)
    many_processor = PipelineStep(
        name="many_items_processor",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'Many Items Processor'}),
        config={'source': 'detector'},
        depends_on=['detector'],
        condition='len(result.detector) > 5'
    )
    pipeline.add_step(many_processor)
    
    # Step 3: Process if few items detected (<= 5)
    few_processor = PipelineStep(
        name="few_items_processor",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'Few Items Processor'}),
        config={'source': 'detector'},
        depends_on=['detector'],
        condition='len(result.detector) <= 5'
    )
    pipeline.add_step(few_processor)
    
    # Execute
    print("Executing pipeline with 8 detections...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(test_image)
    
    print("\nResults:")
    print(f"  detector: {len(results['detector'])} items detected")
    print(f"  many_items_processor: {'✓ EXECUTED' if 'many_items_processor' in results else '✗ SKIPPED'}")
    print(f"  few_items_processor: {'✓ EXECUTED' if 'few_items_processor' in results else '✗ SKIPPED'}")
    
    print("\nPerformance Metrics:")
    metrics = pipeline.get_performance_metrics()
    for step_name, metric in metrics.items():
        print(f"  {step_name}: {metric['duration']:.3f}s")


def demo_parallel_execution():
    """Demonstrate parallel execution."""
    print("\n" + "="*70)
    print("DEMO 2: Parallel Execution")
    print("="*70)
    print("\nParallel execution runs independent steps concurrently.")
    print("This example shows 3 detectors running in parallel.\n")
    
    import time
    
    # Create pipeline with parallel execution
    pipeline = Pipeline(
        name="parallel_demo",
        enable_monitoring=True,
        parallel_executor="thread",
        max_workers=3
    )
    
    # Add 3 detectors that will run in parallel
    for i in range(3):
        detector = PipelineStep(
            name=f"detector_{i}",
            step_type=StepType.DETECTOR,
            component=SimpleDetector(config={'count': 5, 'name': f'Detector {i}'}),
            config={'count': 5},
            parallel_group="parallel_detectors"
        )
        pipeline.add_step(detector)
    
    # Execute
    print("Executing 3 detectors in parallel...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    start_time = time.time()
    results = pipeline.execute(test_image)
    elapsed_time = time.time() - start_time
    
    print(f"\nExecution completed in {elapsed_time:.3f}s")
    print(f"Results: {list(results.keys())}")
    
    print("\nPerformance Metrics:")
    metrics = pipeline.get_performance_metrics()
    for step_name, metric in metrics.items():
        parallel_flag = "✓ Parallel" if metric.get('parallel', False) else "Sequential"
        print(f"  {step_name}: {metric['duration']:.3f}s ({parallel_flag})")


def demo_combined():
    """Demonstrate combined conditional and parallel execution."""
    print("\n" + "="*70)
    print("DEMO 3: Combined Conditional + Parallel")
    print("="*70)
    print("\nCombining both features enables sophisticated workflows.")
    print("This example shows parallel processing triggered by a condition.\n")
    
    # Create pipeline
    pipeline = Pipeline(
        name="combined_demo",
        enable_monitoring=True,
        parallel_executor="thread",
        max_workers=2
    )
    
    # Step 1: Initial detection
    initial = PipelineStep(
        name="initial_detector",
        step_type=StepType.DETECTOR,
        component=SimpleDetector(config={'count': 10, 'name': 'Initial Detector'}),
        config={'count': 10}
    )
    pipeline.add_step(initial)
    
    # Steps 2-3: Parallel processors (only if > 5 items detected)
    processor1 = PipelineStep(
        name="parallel_processor_1",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'Parallel Processor 1'}),
        config={'source': 'initial_detector'},
        depends_on=['initial_detector'],
        condition='len(result.initial_detector) > 5',
        parallel_group="conditional_parallel"
    )
    pipeline.add_step(processor1)
    
    processor2 = PipelineStep(
        name="parallel_processor_2",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'Parallel Processor 2'}),
        config={'source': 'initial_detector'},
        depends_on=['initial_detector'],
        condition='len(result.initial_detector) > 5',
        parallel_group="conditional_parallel"
    )
    pipeline.add_step(processor2)
    
    # Step 4: Final aggregation
    final = PipelineStep(
        name="final_aggregator",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'Final Aggregator'}),
        config={'source': 'initial_detector'},
        depends_on=['initial_detector', 'parallel_processor_1', 'parallel_processor_2']
    )
    pipeline.add_step(final)
    
    # Execute
    print("Executing combined pipeline...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(test_image)
    
    print("\nResults:")
    for step_name in results.keys():
        print(f"  {step_name}: ✓")
    
    print("\nPerformance Metrics:")
    metrics = pipeline.get_performance_metrics()
    total_time = sum(m['duration'] for m in metrics.values())
    print(f"  Total execution time: {total_time:.3f}s")
    for step_name, metric in metrics.items():
        parallel_flag = "✓ Parallel" if metric.get('parallel', False) else "Sequential"
        print(f"  {step_name}: {metric['duration']:.3f}s ({parallel_flag})")


def demo_yaml_config():
    """Demonstrate YAML configuration."""
    print("\n" + "="*70)
    print("DEMO 4: YAML Configuration")
    print("="*70)
    print("\nPipelines with conditional and parallel features can be saved to YAML.\n")
    
    # Create a pipeline
    pipeline = Pipeline(
        name="yaml_demo",
        parallel_executor="thread",
        max_workers=4
    )
    
    detector = PipelineStep(
        name="detector",
        step_type=StepType.DETECTOR,
        component=SimpleDetector(config={'count': 5}),
        config={'count': 5}
    )
    pipeline.add_step(detector)
    
    conditional = PipelineStep(
        name="conditional_processor",
        step_type=StepType.EXTRACTOR,
        component=SimpleExtractor(config={'name': 'conditional'}),
        config={'source': 'detector'},
        depends_on=['detector'],
        condition='len(result.detector) > 3'
    )
    pipeline.add_step(conditional)
    
    parallel1 = PipelineStep(
        name="parallel_processor_1",
        step_type=StepType.DETECTOR,
        component=SimpleDetector(config={'count': 3, 'name': 'parallel1'}),
        config={'count': 3},
        parallel_group="parallel_group"
    )
    pipeline.add_step(parallel1)
    
    parallel2 = PipelineStep(
        name="parallel_processor_2",
        step_type=StepType.DETECTOR,
        component=SimpleDetector(config={'count': 3, 'name': 'parallel2'}),
        config={'count': 3},
        parallel_group="parallel_group"
    )
    pipeline.add_step(parallel2)
    
    # Save to YAML
    config_path = "demo_pipeline_config.yaml"
    print(f"Saving pipeline to {config_path}...")
    pipeline.save(config_path)
    
    # Read and display
    print("\nGenerated YAML configuration:")
    print("-" * 70)
    with open(config_path, 'r') as f:
        print(f.read())
    print("-" * 70)
    
    # Cleanup
    import os
    os.remove(config_path)
    print(f"\nCleaned up {config_path}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("CONDITIONAL BRANCHING AND PARALLEL EXECUTION DEMO")
    print("="*70)
    
    demo_conditional_branching()
    demo_parallel_execution()
    demo_combined()
    demo_yaml_config()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFor more information, see:")
    print("  - docs/CONDITIONAL_PARALLEL_PIPELINE.md")
    print("  - .kiro/specs/screenshot-analysis-library-refactor/design.md")
    print()


if __name__ == "__main__":
    main()
