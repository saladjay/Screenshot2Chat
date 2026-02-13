"""
Property-based tests for BaseDetector
Task 1.5: Write property tests for BaseDetector
Property 7: Detector Interface Conformance
Validates: Requirements 3.5, 6.6
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from typing import List
from src.screenshot2chat.core.base_detector import BaseDetector, DetectionResult


class MockDetector(BaseDetector):
    """Mock detector for testing"""
    
    def load_model(self):
        self.model = "mock_model"
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        # Simple mock detection: return one detection per 100x100 region
        height, width = image.shape[:2]
        results = []
        for y in range(0, height, 100):
            for x in range(0, width, 100):
                bbox = [float(x), float(y), float(min(x+100, width)), float(min(y+100, height))]
                results.append(DetectionResult(bbox, 0.9, "mock", {}))
        return results


@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(min_value=50, max_value=500),
    height=st.integers(min_value=50, max_value=500),
    channels=st.sampled_from([1, 3])
)
def test_property_7_detector_interface_conformance(width, height, channels):
    """
    Feature: screenshot-analysis-library-refactor
    Property 7: Detector Interface Conformance
    
    For any detector implementation, it should conform to the BaseDetector 
    interface and return List[DetectionResult].
    """
    # Create random image
    if channels == 1:
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    else:
        image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    
    # Create detector
    detector = MockDetector()
    detector.load_model()
    
    # Execute detection
    results = detector.detect(image)
    
    # Verify interface conformance
    assert isinstance(results, list), "detect() must return a list"
    
    for result in results:
        assert isinstance(result, DetectionResult), "All results must be DetectionResult instances"
        assert isinstance(result.bbox, list), "bbox must be a list"
        assert len(result.bbox) == 4, "bbox must have 4 coordinates"
        assert all(isinstance(x, (int, float)) for x in result.bbox), "bbox coordinates must be numeric"
        assert isinstance(result.score, (int, float)), "score must be numeric"
        assert 0 <= result.score <= 1, "score must be between 0 and 1"
        assert isinstance(result.category, str), "category must be a string"
        assert isinstance(result.metadata, dict), "metadata must be a dict"


@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(min_value=100, max_value=400),
    height=st.integers(min_value=100, max_value=400)
)
def test_detector_preprocess_postprocess_pipeline(width, height):
    """
    Test that detector preprocess and postprocess methods work correctly
    """
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    detector = MockDetector()
    detector.load_model()
    
    # Test preprocess
    preprocessed = detector.preprocess(image)
    assert isinstance(preprocessed, np.ndarray), "preprocess must return numpy array"
    
    # Test detect
    results = detector.detect(preprocessed)
    
    # Test postprocess
    postprocessed = detector.postprocess(results)
    assert isinstance(postprocessed, list), "postprocess must return a list"


@settings(max_examples=100, deadline=None)
@given(
    config_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50))
    )
)
def test_detector_config_acceptance(config_dict):
    """
    Test that detectors accept arbitrary configuration dictionaries
    """
    detector = MockDetector(config=config_dict)
    assert detector.config == config_dict, "Detector must store provided config"
    
    # Should still be able to load model and detect
    detector.load_model()
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert isinstance(results, list)
