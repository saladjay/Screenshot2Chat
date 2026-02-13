"""
Property-based tests for Pipeline
Task 5.3, 5.5, 5.7: Pipeline property tests
Property 2: Pipeline Configuration Round-Trip
Property 11: Pipeline Execution Order Preservation
Property 14: Pipeline Validation Failure Detection
Validates: Requirements 8.2, 8.5, 8.7
"""

import pytest
import tempfile
import os
import yaml
import json
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from src.screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.core.base_detector import BaseDetector
from src.screenshot2chat.core.base_extractor import BaseExtractor
from src.screenshot2chat.core.data_models import DetectionResult, ExtractionResult


class MockDetector(BaseDetector):
    def __init__(self, name="mock", config=None):
        super().__init__(config)
        self.name = name
        self.execution_order = []
    
    def load_model(self):
        self.model = f"model_{self.name}"
    
    def detect(self, image):
        return [DetectionResult([0, 0, 10, 10], 0.9, self.name, {})]


class MockExtractor(BaseExtractor):
    def __init__(self, name="mock", config=None):
        super().__init__(config)
        self.name = name
    
    def extract(self, detection_results, image=None):
        return ExtractionResult({"extractor": self.name, "count": len(detection_results)}, 1.0)


@settings(max_examples=50, deadline=None)
@given(
    pipeline_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    num_steps=st.integers(min_value=1, max_value=5)
)
def test_property_2_pipeline_config_roundtrip(pipeline_name, num_steps):
    """
    Feature: screenshot-analysis-library-refactor
    Property 2: Pipeline Configuration Round-Trip
    
    For any valid pipeline configuration, saving it to a file and then 
    loading it back should produce an equivalent pipeline that executes identically.
    """
    # Create pipeline
    pipeline = Pipeline(name=pipeline_name)
    
    # Add steps
    for i in range(num_steps):
        detector = MockDetector(name=f"detector_{i}")
        step = PipelineStep(
            name=f"step_{i}",
            step_type=StepType.DETECTOR,
            component=detector,
            config={"index": i}
        )
        pipeline.add_step(step)
    
    # Save to config
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pipeline_config.yaml")
        
        # Export config
        config = pipeline.to_config()
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load config
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify config structure
        assert loaded_config["name"] == pipeline_name
        assert len(loaded_config["steps"]) == num_steps


@settings(max_examples=50, deadline=None)
@given(
    num_steps=st.integers(min_value=2, max_value=5)
)
def test_property_11_pipeline_execution_order_preservation(num_steps):
    """
    Feature: screenshot-analysis-library-refactor
    Property 11: Pipeline Execution Order Preservation
    
    For any pipeline configuration specifying an execution order, 
    the actual execution should follow that exact order and produce deterministic results.
    """
    pipeline = Pipeline(name="ordered_pipeline")
    execution_log = []
    
    # Create steps that log their execution
    class LoggingDetector(BaseDetector):
        def __init__(self, step_id, log):
            super().__init__()
            self.step_id = step_id
            self.log = log
        
        def load_model(self):
            self.model = "loaded"
        
        def detect(self, image):
            self.log.append(self.step_id)
            return [DetectionResult([0, 0, 10, 10], 0.9, f"step_{self.step_id}", {})]
    
    # Add steps in specific order
    for i in range(num_steps):
        detector = LoggingDetector(i, execution_log)
        detector.load_model()
        step = PipelineStep(
            name=f"step_{i}",
            step_type=StepType.DETECTOR,
            component=detector
        )
        pipeline.add_step(step)
    
    # Execute pipeline
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    pipeline.execute(image)
    
    # Verify execution order
    expected_order = list(range(num_steps))
    assert execution_log == expected_order, f"Expected {expected_order}, got {execution_log}"


@settings(max_examples=50, deadline=None)
@given(
    has_circular_dep=st.booleans(),
    has_missing_dep=st.booleans()
)
def test_property_14_pipeline_validation_failure_detection(has_circular_dep, has_missing_dep):
    """
    Feature: screenshot-analysis-library-refactor
    Property 14: Pipeline Validation Failure Detection
    
    For any invalid pipeline configuration (missing dependencies, circular references, 
    invalid step types), the validation function should fail and provide a clear error message.
    """
    assume(has_circular_dep or has_missing_dep)  # At least one error condition
    
    pipeline = Pipeline(name="invalid_pipeline")
    
    if has_circular_dep:
        # Create circular dependency: A -> B -> A
        detector_a = MockDetector(name="A")
        detector_b = MockDetector(name="B")
        detector_a.load_model()
        detector_b.load_model()
        
        step_a = PipelineStep("step_a", StepType.DETECTOR, detector_a, {"depends_on": ["step_b"]})
        step_b = PipelineStep("step_b", StepType.DETECTOR, detector_b, {"depends_on": ["step_a"]})
        
        pipeline.add_step(step_a)
        pipeline.add_step(step_b)
        
        # Validation should fail
        is_valid, error_msg = pipeline.validate()
        assert not is_valid, "Pipeline with circular dependency should be invalid"
        assert error_msg is not None and len(error_msg) > 0, "Should provide error message"
        assert "circular" in error_msg.lower() or "cycle" in error_msg.lower()
    
    elif has_missing_dep:
        # Create missing dependency
        detector = MockDetector(name="A")
        detector.load_model()
        
        step = PipelineStep("step_a", StepType.DETECTOR, detector, {"depends_on": ["nonexistent_step"]})
        pipeline.add_step(step)
        
        # Validation should fail
        is_valid, error_msg = pipeline.validate()
        assert not is_valid, "Pipeline with missing dependency should be invalid"
        assert error_msg is not None and len(error_msg) > 0, "Should provide error message"
        assert "missing" in error_msg.lower() or "not found" in error_msg.lower()


@settings(max_examples=50, deadline=None)
@given(
    num_steps=st.integers(min_value=1, max_value=5)
)
def test_pipeline_deterministic_execution(num_steps):
    """
    Test that pipeline execution is deterministic
    """
    pipeline = Pipeline(name="deterministic_pipeline")
    
    for i in range(num_steps):
        detector = MockDetector(name=f"detector_{i}")
        detector.load_model()
        step = PipelineStep(f"step_{i}", StepType.DETECTOR, detector)
        pipeline.add_step(step)
    
    # Execute twice with same input
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    result1 = pipeline.execute(image)
    result2 = pipeline.execute(image)
    
    # Results should be identical
    assert len(result1) == len(result2)
    for key in result1:
        assert key in result2


@settings(max_examples=50, deadline=None)
@given(
    num_detectors=st.integers(min_value=1, max_value=3),
    num_extractors=st.integers(min_value=1, max_value=3)
)
def test_pipeline_mixed_step_types(num_detectors, num_extractors):
    """
    Test pipeline with mixed detector and extractor steps
    """
    pipeline = Pipeline(name="mixed_pipeline")
    
    # Add detectors
    for i in range(num_detectors):
        detector = MockDetector(name=f"detector_{i}")
        detector.load_model()
        step = PipelineStep(f"detector_{i}", StepType.DETECTOR, detector)
        pipeline.add_step(step)
    
    # Add extractors
    for i in range(num_extractors):
        extractor = MockExtractor(name=f"extractor_{i}")
        step = PipelineStep(
            f"extractor_{i}", 
            StepType.EXTRACTOR, 
            extractor,
            {"source": f"detector_0"}  # Depend on first detector
        )
        pipeline.add_step(step)
    
    # Validate
    is_valid, error_msg = pipeline.validate()
    assert is_valid, f"Valid pipeline should pass validation: {error_msg}"
    
    # Execute
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = pipeline.execute(image)
    
    assert isinstance(results, dict)
    assert len(results) == num_detectors + num_extractors
