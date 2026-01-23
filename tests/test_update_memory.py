"""
Unit tests for update_memory method

This module contains unit tests to verify the memory management functionality
of the ChatLayoutDetector class.
"""

import sys
import tempfile
import json
from pathlib import Path
import numpy as np

# Add src directory to path
src_path = Path(__file__).parent.parent / "src" / "screenshotanalysis"
sys.path.insert(0, str(src_path))

# Import chat_layout_detector directly
import importlib.util
spec_detector = importlib.util.spec_from_file_location(
    "chat_layout_detector", 
    src_path / "chat_layout_detector.py"
)
chat_layout_detector = importlib.util.module_from_spec(spec_detector)
spec_detector.loader.exec_module(chat_layout_detector)
ChatLayoutDetector = chat_layout_detector.ChatLayoutDetector


# Define a minimal TextBox class for testing
class TextBox:
    """Minimal TextBox class for testing purposes"""
    def __init__(self, box, score, **kwargs):
        self.box = box
        self.score = score
        if isinstance(self.box, list):
            self.box = np.array(self.box)
        self.text_type = None
        self.source = None
        self.layout_det = None

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.x_min, self.y_min, self.x_max, self.y_max = self.box.tolist()

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2

    @property
    def width(self):
        return self.x_max - self.x_min


def test_update_memory_initialization():
    """
    Test that update_memory correctly initializes memory for first-time speakers
    
    Validates: Requirements 3.1, 3.2
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    # Create some test boxes for speaker A
    boxes_a = [
        TextBox(box=[100, 100, 200, 150], score=0.9),
        TextBox(box=[110, 200, 210, 250], score=0.9),
    ]
    
    # Create some test boxes for speaker B
    boxes_b = [
        TextBox(box=[500, 100, 600, 150], score=0.9),
        TextBox(box=[510, 200, 610, 250], score=0.9),
    ]
    
    assigned = {"A": boxes_a, "B": boxes_b}
    
    # Memory should be None initially
    assert detector.memory["A"] is None
    assert detector.memory["B"] is None
    
    # Update memory
    detector.update_memory(assigned)
    
    # Memory should now be initialized
    assert detector.memory["A"] is not None
    assert detector.memory["B"] is not None
    
    # Check that center and width are normalized and reasonable
    assert 0 <= detector.memory["A"]["center"] <= 1
    assert 0 <= detector.memory["A"]["width"] <= 1
    assert detector.memory["A"]["count"] == 2
    
    assert 0 <= detector.memory["B"]["center"] <= 1
    assert 0 <= detector.memory["B"]["width"] <= 1
    assert detector.memory["B"]["count"] == 2
    
    print("✓ Memory initialization test passed")


def test_update_memory_sliding_average():
    """
    Test that update_memory correctly applies sliding average
    
    Validates: Requirements 3.5
    """
    detector = ChatLayoutDetector(screen_width=720, memory_alpha=0.7)
    
    # Initialize memory with known values
    detector.memory["A"] = {
        "center": 0.2,  # normalized
        "width": 0.1,   # normalized
        "count": 10
    }
    
    # Create new boxes with different center and width
    boxes_a = [
        TextBox(box=[200, 100, 300, 150], score=0.9),  # center_x = 250, width = 100
    ]
    
    assigned = {"A": boxes_a, "B": []}
    
    # Calculate expected values
    new_center_normalized = 250 / 720
    new_width_normalized = 100 / 720
    
    expected_center = 0.7 * 0.2 + 0.3 * new_center_normalized
    expected_width = 0.7 * 0.1 + 0.3 * new_width_normalized
    expected_count = 11
    
    # Update memory
    detector.update_memory(assigned)
    
    # Check sliding average was applied correctly
    assert abs(detector.memory["A"]["center"] - expected_center) < 1e-6, \
        f"Expected center {expected_center}, got {detector.memory['A']['center']}"
    assert abs(detector.memory["A"]["width"] - expected_width) < 1e-6, \
        f"Expected width {expected_width}, got {detector.memory['A']['width']}"
    assert detector.memory["A"]["count"] == expected_count, \
        f"Expected count {expected_count}, got {detector.memory['A']['count']}"
    
    print("✓ Sliding average test passed")


def test_update_memory_count_increment():
    """
    Test that update_memory correctly increments count
    
    Validates: Requirements 3.5
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    # Initialize memory
    detector.memory["A"] = {
        "center": 0.2,
        "width": 0.1,
        "count": 5
    }
    
    # Create boxes
    boxes_a = [
        TextBox(box=[100, 100, 200, 150], score=0.9),
        TextBox(box=[110, 200, 210, 250], score=0.9),
        TextBox(box=[120, 300, 220, 350], score=0.9),
    ]
    
    assigned = {"A": boxes_a, "B": []}
    
    # Update memory
    detector.update_memory(assigned)
    
    # Count should be incremented by 3
    assert detector.memory["A"]["count"] == 8, \
        f"Expected count 8, got {detector.memory['A']['count']}"
    
    print("✓ Count increment test passed")


def test_update_memory_persistence():
    """
    Test that update_memory saves to disk when memory_path is provided
    
    Validates: Requirements 9.1, 9.2
    """
    # Create a temporary file for memory storage
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        detector = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        # Create boxes
        boxes_a = [TextBox(box=[100, 100, 200, 150], score=0.9)]
        boxes_b = [TextBox(box=[500, 100, 600, 150], score=0.9)]
        
        assigned = {"A": boxes_a, "B": boxes_b}
        
        # Update memory (should trigger save)
        detector.update_memory(assigned)
        
        # Check that file was created and contains valid JSON
        assert Path(temp_path).exists(), "Memory file was not created"
        
        with open(temp_path, 'r') as f:
            saved_data = json.load(f)
        
        # Verify structure
        assert "A" in saved_data
        assert "B" in saved_data
        assert "version" in saved_data
        assert "last_updated" in saved_data
        
        # Verify data
        assert saved_data["A"] is not None
        assert saved_data["B"] is not None
        assert saved_data["A"]["count"] == 1
        assert saved_data["B"]["count"] == 1
        
        print("✓ Memory persistence test passed")
        
    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)


def test_update_memory_skip_empty_speakers():
    """
    Test that update_memory skips speakers with no boxes
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    # Create boxes only for speaker A
    boxes_a = [TextBox(box=[100, 100, 200, 150], score=0.9)]
    
    assigned = {"A": boxes_a, "B": []}
    
    # Update memory
    detector.update_memory(assigned)
    
    # Only A should be initialized
    assert detector.memory["A"] is not None
    assert detector.memory["B"] is None
    
    print("✓ Skip empty speakers test passed")


if __name__ == "__main__":
    test_update_memory_initialization()
    test_update_memory_sliding_average()
    test_update_memory_count_increment()
    test_update_memory_persistence()
    test_update_memory_skip_empty_speakers()
    print("\n✅ All update_memory tests passed!")
