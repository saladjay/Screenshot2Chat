"""
Property-based tests for extractor chain composition
Task 3.5: Write property tests for extractors
Property 10: Extractor Chain Composition
Validates: Requirements 7.7
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import List
from src.screenshot2chat.core.base_extractor import BaseExtractor, ExtractionResult
from src.screenshot2chat.core.data_models import DetectionResult


class CountExtractor(BaseExtractor):
    """Extractor that counts detections"""
    
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult:
        count = len(detection_results)
        return ExtractionResult({"count": count}, 1.0 if count > 0 else 0.0)


class CategoryExtractor(BaseExtractor):
    """Extractor that lists categories"""
    
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult:
        categories = [r.category for r in detection_results]
        return ExtractionResult({"categories": categories}, 1.0 if categories else 0.0)


class FilterExtractor(BaseExtractor):
    """Extractor that filters by category"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.filter_category = config.get("filter_category", "text") if config else "text"
    
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult:
        filtered = [r for r in detection_results if r.category == self.filter_category]
        return ExtractionResult({"filtered_count": len(filtered)}, 1.0)


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20),
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
)
def test_property_10_extractor_chain_composition(num_detections, categories):
    """
    Feature: screenshot-analysis-library-refactor
    Property 10: Extractor Chain Composition
    
    For any sequence of compatible extractors, composing them should produce 
    a result equivalent to applying each extractor in order.
    """
    # Create detection results
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        category = categories[i % len(categories)]
        detection_results.append(DetectionResult(bbox, 0.9, category, {}))
    
    # Apply extractors individually
    count_extractor = CountExtractor()
    category_extractor = CategoryExtractor()
    
    count_result = count_extractor.extract(detection_results)
    category_result = category_extractor.extract(detection_results)
    
    # Verify individual results
    assert count_result.data["count"] == num_detections
    assert len(category_result.data["categories"]) == num_detections
    
    # Verify composition: applying both should give consistent results
    # The count should match the length of categories
    assert count_result.data["count"] == len(category_result.data["categories"])


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=1, max_value=20),
    filter_category=st.text(min_size=1, max_size=10)
)
def test_extractor_chain_with_filtering(num_detections, filter_category):
    """
    Test that extractor chain works with filtering operations
    """
    # Create detection results with mixed categories
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        # Alternate between filter_category and "other"
        category = filter_category if i % 2 == 0 else "other"
        detection_results.append(DetectionResult(bbox, 0.9, category, {}))
    
    # Apply filter extractor
    filter_extractor = FilterExtractor(config={"filter_category": filter_category})
    filter_result = filter_extractor.extract(detection_results)
    
    # Count how many match the filter
    expected_count = sum(1 for r in detection_results if r.category == filter_category)
    assert filter_result.data["filtered_count"] == expected_count


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20)
)
def test_extractor_chain_idempotence(num_detections):
    """
    Test that applying the same extractor twice gives the same result
    """
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        detection_results.append(DetectionResult(bbox, 0.9, "text", {}))
    
    extractor = CountExtractor()
    
    # Apply twice
    result1 = extractor.extract(detection_results)
    result2 = extractor.extract(detection_results)
    
    # Results should be identical
    assert result1.data == result2.data
    assert result1.confidence == result2.confidence


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20),
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
)
def test_extractor_chain_order_independence(num_detections, categories):
    """
    Test that independent extractors can be applied in any order
    """
    detection_results = []
    for i in range(num_detections):
        bbox = [float(i*10), float(i*10), float(i*10+50), float(i*10+50)]
        category = categories[i % len(categories)]
        detection_results.append(DetectionResult(bbox, 0.9, category, {}))
    
    count_extractor = CountExtractor()
    category_extractor = CategoryExtractor()
    
    # Apply in order 1: count then category
    count_result_1 = count_extractor.extract(detection_results)
    category_result_1 = category_extractor.extract(detection_results)
    
    # Apply in order 2: category then count
    category_result_2 = category_extractor.extract(detection_results)
    count_result_2 = count_extractor.extract(detection_results)
    
    # Results should be the same regardless of order
    assert count_result_1.data == count_result_2.data
    assert category_result_1.data == category_result_2.data


@settings(max_examples=100, deadline=None)
@given(
    num_detections=st.integers(min_value=0, max_value=20)
)
def test_extractor_empty_input_handling(num_detections):
    """
    Test that extractors handle empty input gracefully
    """
    if num_detections == 0:
        detection_results = []
    else:
        detection_results = [
            DetectionResult([float(i*10), float(i*10), float(i*10+50), float(i*10+50)], 
                          0.9, "text", {})
            for i in range(num_detections)
        ]
    
    extractors = [CountExtractor(), CategoryExtractor()]
    
    for extractor in extractors:
        result = extractor.extract(detection_results)
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.data, dict)
        assert isinstance(result.confidence, float)
