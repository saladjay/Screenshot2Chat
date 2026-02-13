"""
Property-based tests for BaseExtractor
Task 1.7: Write property tests for BaseExtractor
Property 9: Extractor JSON Output Validity
Validates: Requirements 7.6
"""

import pytest
import json
import numpy as np
from hypothesis import given, strategies as st, settings
from typing import List
from src.screenshot2chat.core.base_extractor import BaseExtractor, ExtractionResult
from src.screenshot2chat.core.data_models import DetectionResult


class MockExtractor(BaseExtractor):
    """Mock extractor for testing"""
    
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult:
        # Simple mock extraction: count detections
        count = len(detection_results)
        data = {
            "count": count,
            "categories": [r.category for r in detection_results]
        }
        confidence = 1.0 if count > 0 else 0.0
        return ExtractionResult(data, confidence)


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20),
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
)
def test_property_9_extractor_json_output_validity(num_detections, categories):
    """
    Feature: screenshot-analysis-library-refactor
    Property 9: Extractor JSON Output Validity
    
    For any extractor output, calling to_json() should produce valid, 
    parseable JSON that contains all extracted information.
    """
    # Create mock detection results
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        category = categories[i % len(categories)]
        detection_results.append(DetectionResult(bbox, 0.9, category, {}))
    
    # Create extractor and extract
    extractor = MockExtractor()
    result = extractor.extract(detection_results)
    
    # Convert to JSON
    json_data = result.to_json()
    
    # Verify JSON structure
    assert isinstance(json_data, dict), "to_json() must return a dict"
    assert "data" in json_data, "JSON must contain 'data' field"
    assert "confidence" in json_data, "JSON must contain 'confidence' field"
    
    # Verify JSON is serializable
    json_str = json.dumps(json_data)
    assert isinstance(json_str, str), "JSON must be serializable to string"
    
    # Verify JSON is parseable
    parsed = json.loads(json_str)
    assert parsed == json_data, "Parsed JSON must match original"
    
    # Verify data integrity
    assert parsed["data"]["count"] == num_detections
    assert len(parsed["data"]["categories"]) == num_detections


@settings(max_examples=100, deadline=None)
@given(
    data_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
            st.lists(st.integers(), max_size=10)
        ),
        min_size=1,
        max_size=10
    ),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_extraction_result_json_serialization(data_dict, confidence):
    """
    Test that ExtractionResult can serialize arbitrary data to valid JSON
    """
    result = ExtractionResult(data_dict, confidence)
    json_data = result.to_json()
    
    # Must be serializable
    json_str = json.dumps(json_data)
    parsed = json.loads(json_str)
    
    # Data must be preserved
    assert parsed["data"] == data_dict
    assert abs(parsed["confidence"] - confidence) < 1e-6


@settings(max_examples=100, deadline=None)
@given(
    config_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50))
    )
)
def test_extractor_config_acceptance(config_dict):
    """
    Test that extractors accept arbitrary configuration dictionaries
    """
    extractor = MockExtractor(config=config_dict)
    assert extractor.config == config_dict, "Extractor must store provided config"
    
    # Should still be able to extract
    detection_results = [DetectionResult([0, 0, 10, 10], 0.9, "test", {})]
    result = extractor.extract(detection_results)
    assert isinstance(result, ExtractionResult)


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20)
)
def test_extractor_validate_method(num_detections):
    """
    Test that extractor validate method works correctly
    """
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        detection_results.append(DetectionResult(bbox, 0.9, "test", {}))
    
    extractor = MockExtractor()
    result = extractor.extract(detection_results)
    
    # Validate should return boolean
    is_valid = extractor.validate(result)
    assert isinstance(is_valid, bool), "validate() must return a boolean"
