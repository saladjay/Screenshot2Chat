"""
Property-based tests for advanced pipeline features
Task 13.2, 13.4: Conditional branch and parallel execution property tests
Property 12: Pipeline Conditional Branch Correctness
Property 13: Pipeline Parallel Execution Completeness
Validates: Requirements 8.3, 8.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from src.screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.core.base_detector import BaseDetector
from src.screenshot2chat.core.base_extractor import BaseExtractor
from src.screenshot2chat.core.data_models import DetectionResult, ExtractionResult


class ConditionalDetector(BaseDetector):
    """Detector that returns different results based on condition"""
    
    def __init__(self, name="conditional", threshold=0.5, config=None):
        super().__init__(config)
        self.name = name
        self.threshold = threshold
    
    def load_model(self):
        self.model = "conditional_model"
    
    def detect(self, image):
        # Return different results based on image brightness
        brightness = np.mean(image)
        if brightness > self.threshold * 255:
            return [DetectionResult([0, 0, 10, 10], 0.9, "bright", {})]
        else:
            return [DetectionResult([0, 0, 10, 10], 0.9, "dark", {})]


class ParallelDetector(BaseDetector):
    """Detector for parallel execution testing"""
    
    def __init__(self, name="parallel", result_value=1, config=None):
        super().__init__(config)
        self.name = name
        self.result_value = result_value
    
    def load_model(self):
        self.model = f"model_{self.name}"
    
    def detect(self, image):
        return [DetectionResult([0, 0, 10, 10], 0.9, self.name, {"value": self.result_value})]


@settings(max_examples=50, deadline=None)
@given(
    brightness_threshold=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False),
    image_brightness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_property_12_pipeline_conditional_branch_correctness(brightness_threshold, image_brightness):
    """
    Feature: screenshot-analysis-library-refactor
    Property 12: Pipeline Conditional Branch Correctness
    
    For any pipeline with conditional branches, the branch taken should match 
    the condition evaluation based on intermediate results.
    """
    pipeline = Pipeline(name="conditional_pipeline")
    
    # Create detector with threshold
    detector = ConditionalDetector(threshold=brightness_threshold)
    detector.load_model()
    
    step = PipelineStep("conditional_step", StepType.DETECTOR, detector)
    pipeline.add_step(step)
    
    # Create image with specific brightness
    image_value = int(image_brightness * 255)
    image = np.full((100, 100, 3), image_value, dtype=np.uint8)
    
    # Execute pipeline
    results = pipeline.execute(image)
    
    # Verify correct branch was taken
    assert "conditional_step" in results
    detection_results = results["conditional_step"]
    
    if len(detection_results) > 0:
        result = detection_results[0]
        
        # Check that the category matches the expected branch
        if image_brightness > brightness_threshold:
            assert result.category == "bright", "Should take bright branch"
        else:
            assert result.category == "dark", "Should take dark branch"


@settings(max_examples=50, deadline=None)
@given(
    num_parallel_steps=st.integers(min_value=2, max_value=5)
)
def test_property_13_pipeline_parallel_execution_completeness(num_parallel_steps):
    """
    Feature: screenshot-analysis-library-refactor
    Property 13: Pipeline Parallel Execution Completeness
    
    For any pipeline with parallel steps, all parallel steps should execute 
    and their results should be correctly merged.
    """
    pipeline = Pipeline(name="parallel_pipeline")
    
    # Add parallel steps
    step_names = []
    for i in range(num_parallel_steps):
        detector = ParallelDetector(name=f"parallel_{i}", result_value=i)
        detector.load_model()
        
        step_name = f"parallel_step_{i}"
        step_names.append(step_name)
        
        step = PipelineStep(
            step_name,
            StepType.DETECTOR,
            detector,
            config={"parallel": True}
        )
        pipeline.add_step(step)
    
    # Execute pipeline
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(image)
    
    # Verify all parallel steps executed
    for step_name in step_names:
        assert step_name in results, f"Parallel step {step_name} should have executed"
        
        step_results = results[step_name]
        assert len(step_results) > 0, f"Step {step_name} should have results"
    
    # Verify results are correctly separated (not merged incorrectly)
    assert len(results) == num_parallel_steps, "Should have results for all parallel steps"


@settings(max_examples=50, deadline=None)
@given(
    num_parallel_steps=st.integers(min_value=2, max_value=4)
)
def test_parallel_execution_result_independence(num_parallel_steps):
    """
    Test that parallel steps produce independent results
    """
    pipeline = Pipeline(name="independent_parallel")
    
    # Add parallel steps with different result values
    for i in range(num_parallel_steps):
        detector = ParallelDetector(name=f"detector_{i}", result_value=i * 10)
        detector.load_model()
        
        step = PipelineStep(
            f"step_{i}",
            StepType.DETECTOR,
            detector,
            config={"parallel": True}
        )
        pipeline.add_step(step)
    
    # Execute
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(image)
    
    # Verify each step has its own unique results
    for i in range(num_parallel_steps):
        step_name = f"step_{i}"
        step_results = results[step_name]
        
        if len(step_results) > 0:
            result = step_results[0]
            expected_value = i * 10
            assert result.metadata.get("value") == expected_value, \
                f"Step {i} should have value {expected_value}"


@settings(max_examples=50, deadline=None)
@given(
    condition_value=st.integers(min_value=0, max_value=100)
)
def test_conditional_branch_determinism(condition_value):
    """
    Test that conditional branches are deterministic
    """
    pipeline = Pipeline(name="deterministic_conditional")
    
    # Create detector that branches based on condition
    class ValueDetector(BaseDetector):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold
        
        def load_model(self):
            self.model = "value_model"
        
        def detect(self, image):
            value = int(np.mean(image))
            category = "high" if value > self.threshold else "low"
            return [DetectionResult([0, 0, 10, 10], 0.9, category, {"value": value})]
    
    detector = ValueDetector(threshold=50)
    detector.load_model()
    
    step = PipelineStep("value_step", StepType.DETECTOR, detector)
    pipeline.add_step(step)
    
    # Create image with specific value
    image = np.full((100, 100, 3), condition_value, dtype=np.uint8)
    
    # Execute twice
    results1 = pipeline.execute(image)
    results2 = pipeline.execute(image)
    
    # Results should be identical
    assert results1["value_step"][0].category == results2["value_step"][0].category


@settings(max_examples=50, deadline=None)
@given(
    num_branches=st.integers(min_value=2, max_value=4)
)
def test_multiple_conditional_branches(num_branches):
    """
    Test pipeline with multiple conditional branches
    """
    pipeline = Pipeline(name="multi_branch")
    
    # Add multiple conditional steps
    for i in range(num_branches):
        threshold = (i + 1) / (num_branches + 1)  # Distribute thresholds
        detector = ConditionalDetector(name=f"branch_{i}", threshold=threshold)
        detector.load_model()
        
        step = PipelineStep(f"branch_step_{i}", StepType.DETECTOR, detector)
        pipeline.add_step(step)
    
    # Execute with different brightness levels
    for brightness in [0.2, 0.5, 0.8]:
        image_value = int(brightness * 255)
        image = np.full((100, 100, 3), image_value, dtype=np.uint8)
        
        results = pipeline.execute(image)
        
        # Verify all branches executed
        assert len(results) == num_branches


@settings(max_examples=50, deadline=None)
@given(
    num_parallel=st.integers(min_value=2, max_value=4),
    num_sequential=st.integers(min_value=1, max_value=3)
)
def test_mixed_parallel_sequential_execution(num_parallel, num_sequential):
    """
    Test pipeline with both parallel and sequential steps
    """
    pipeline = Pipeline(name="mixed_execution")
    
    # Add parallel steps
    for i in range(num_parallel):
        detector = ParallelDetector(name=f"parallel_{i}", result_value=i)
        detector.load_model()
        step = PipelineStep(f"parallel_{i}", StepType.DETECTOR, detector, {"parallel": True})
        pipeline.add_step(step)
    
    # Add sequential steps
    for i in range(num_sequential):
        detector = ParallelDetector(name=f"sequential_{i}", result_value=i + 100)
        detector.load_model()
        step = PipelineStep(f"sequential_{i}", StepType.DETECTOR, detector)
        pipeline.add_step(step)
    
    # Execute
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(image)
    
    # Verify all steps executed
    total_steps = num_parallel + num_sequential
    assert len(results) == total_steps, f"Should have {total_steps} results"


@settings(max_examples=50, deadline=None)
@given(
    branch_condition=st.booleans()
)
def test_conditional_branch_skip_logic(branch_condition):
    """
    Test that conditional branches can skip steps correctly
    """
    pipeline = Pipeline(name="skip_logic")
    
    # Create detector that may skip based on condition
    class SkipDetector(BaseDetector):
        def __init__(self, should_detect):
            super().__init__()
            self.should_detect = should_detect
        
        def load_model(self):
            self.model = "skip_model"
        
        def detect(self, image):
            if self.should_detect:
                return [DetectionResult([0, 0, 10, 10], 0.9, "detected", {})]
            else:
                return []  # Skip detection
    
    detector = SkipDetector(should_detect=branch_condition)
    detector.load_model()
    
    step = PipelineStep("skip_step", StepType.DETECTOR, detector)
    pipeline.add_step(step)
    
    # Execute
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(image)
    
    # Verify skip behavior
    if branch_condition:
        assert len(results["skip_step"]) > 0, "Should have results when condition is true"
    else:
        assert len(results["skip_step"]) == 0, "Should skip when condition is false"
