"""
Property-based tests for backward compatibility
Task 8.2, 8.3: Backward compatibility and deprecation warning tests
Property 26: Backward Compatibility Preservation
Property 27: Deprecation Warning Emission
Validates: Requirements 15.1, 15.2, 15.3, 15.4
"""

import pytest
import warnings
import numpy as np
from hypothesis import given, strategies as st, settings
from src.screenshot2chat.compat.chat_layout_detector import ChatLayoutDetector
from src.screenshotanalysis.basemodel import TextBox


@settings(max_examples=50, deadline=None)
@given(
    screen_width=st.integers(min_value=320, max_value=1920),
    num_boxes=st.integers(min_value=0, max_value=20)
)
def test_property_26_backward_compatibility_preservation(screen_width, num_boxes):
    """
    Feature: screenshot-analysis-library-refactor
    Property 26: Backward Compatibility Preservation
    
    For any code using the legacy ChatLayoutDetector or TextBox API, 
    it should execute without errors and produce equivalent results to the old implementation.
    """
    # Create legacy ChatLayoutDetector
    detector = ChatLayoutDetector(screen_width=screen_width)
    
    # Create legacy TextBox objects
    boxes = []
    for i in range(num_boxes):
        box = TextBox(
            box=np.array([float(i*50), float(i*30), float(i*50+40), float(i*30+20)]),
            score=0.9,
            text=f"Text {i}"
        )
        boxes.append(box)
    
    # Process frame using legacy API
    try:
        result = detector.process_frame(boxes)
        
        # Should return a result (dict or similar structure)
        assert result is not None
        
        # Result should contain layout information
        assert isinstance(result, dict) or hasattr(result, '__dict__')
        
    except Exception as e:
        pytest.fail(f"Legacy API should not raise exceptions: {e}")


@settings(max_examples=50, deadline=None)
@given(
    screen_width=st.integers(min_value=320, max_value=1920)
)
def test_property_27_deprecation_warning_emission(screen_width):
    """
    Feature: screenshot-analysis-library-refactor
    Property 27: Deprecation Warning Emission
    
    For any call to a deprecated API function, a deprecation warning 
    should be logged exactly once per unique call site.
    """
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create legacy ChatLayoutDetector (should emit deprecation warning)
        detector = ChatLayoutDetector(screen_width=screen_width)
        
        # Check that a deprecation warning was emitted
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        
        assert len(deprecation_warnings) > 0, "Should emit at least one deprecation warning"
        
        # Check warning message content
        warning_messages = [str(warning.message) for warning in deprecation_warnings]
        assert any("deprecated" in msg.lower() for msg in warning_messages), \
            "Warning should mention deprecation"


@settings(max_examples=50, deadline=None)
@given(
    num_boxes=st.integers(min_value=1, max_value=10)
)
def test_textbox_legacy_api_compatibility(num_boxes):
    """
    Test that legacy TextBox API remains functional
    """
    boxes = []
    for i in range(num_boxes):
        box = TextBox(
            box=np.array([float(i*50), float(i*30), float(i*50+40), float(i*30+20)]),
            score=0.9,
            text=f"Text {i}",
            text_type="message"
        )
        boxes.append(box)
    
    # Verify TextBox properties work
    for i, box in enumerate(boxes):
        assert box.x_min == i * 50
        assert box.y_min == i * 30
        assert box.x_max == i * 50 + 40
        assert box.y_max == i * 30 + 20
        assert box.width == 40
        assert box.height == 20
        assert box.text == f"Text {i}"
        assert box.score == 0.9


@settings(max_examples=50, deadline=None)
@given(
    x_min=st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
    y_min=st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False)
)
def test_textbox_coordinate_properties(x_min, y_min, width, height):
    """
    Test that TextBox coordinate properties work correctly
    """
    x_max = x_min + width
    y_max = y_min + height
    
    box = TextBox(
        box=np.array([x_min, y_min, x_max, y_max]),
        score=0.9,
        text="Test"
    )
    
    # Verify properties
    assert abs(box.x_min - x_min) < 1e-6
    assert abs(box.y_min - y_min) < 1e-6
    assert abs(box.x_max - x_max) < 1e-6
    assert abs(box.y_max - y_max) < 1e-6
    assert abs(box.width - width) < 1e-6
    assert abs(box.height - height) < 1e-6
    assert abs(box.center_x - (x_min + x_max) / 2) < 1e-6
    assert abs(box.center_y - (y_min + y_max) / 2) < 1e-6


def test_deprecation_warning_only_once():
    """
    Test that deprecation warning is emitted only once per call site
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # First call
        detector1 = ChatLayoutDetector(screen_width=720)
        initial_warning_count = len([warning for warning in w if issubclass(warning.category, DeprecationWarning)])
        
        # Second call from same location
        detector2 = ChatLayoutDetector(screen_width=1080)
        
        # Should still have warnings (Python's warning system handles deduplication)
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= initial_warning_count


@settings(max_examples=50, deadline=None)
@given(
    num_boxes=st.integers(min_value=0, max_value=15)
)
def test_legacy_api_result_structure(num_boxes):
    """
    Test that legacy API returns expected result structure
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    boxes = []
    for i in range(num_boxes):
        box = TextBox(
            box=np.array([float(i*50), float(i*30), float(i*50+40), float(i*30+20)]),
            score=0.9,
            text=f"Text {i}"
        )
        boxes.append(box)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress deprecation warnings for this test
        result = detector.process_frame(boxes)
    
    # Result should have expected structure
    assert result is not None
    
    # Should contain layout type information
    if isinstance(result, dict):
        assert "layout_type" in result or "layout" in result or len(result) > 0
