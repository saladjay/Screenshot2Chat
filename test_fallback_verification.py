"""Simple verification test for fallback mechanism"""
import sys
from pathlib import Path
import numpy as np

# Import chat_layout_detector directly to avoid paddleocr dependency
src_path = Path(__file__).parent / "src" / "screenshotanalysis"
sys.path.insert(0, str(src_path))

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
    def __init__(self, box, score):
        self.box = np.array(box)
        self.score = score
    
    @property
    def x_min(self): return self.box[0]
    @property
    def y_min(self): return self.box[1]
    @property
    def x_max(self): return self.box[2]
    @property
    def y_max(self): return self.box[3]
    @property
    def center_x(self): return (self.x_min + self.x_max) / 2
    @property
    def width(self): return self.x_max - self.x_min
    @property
    def height(self): return self.y_max - self.y_min

print("Testing Fallback Mechanism Implementation")
print("=" * 60)

# Test 1: should_use_fallback with empty memory
print("\n✓ Test 1: should_use_fallback with empty memory")
detector = ChatLayoutDetector(screen_width=720)
assert detector.should_use_fallback() == True
print("  Empty memory triggers fallback: PASS")

# Test 2: should_use_fallback with sufficient data
print("\n✓ Test 2: should_use_fallback with sufficient data")
detector.memory["A"] = {"center": 0.2, "width": 0.1, "count": 30}
detector.memory["B"] = {"center": 0.8, "width": 0.1, "count": 25}
assert detector.should_use_fallback() == False
print("  Sufficient data (55 total) does not trigger fallback: PASS")

# Test 3: split_columns_median_fallback returns correct metadata
print("\n✓ Test 3: split_columns_median_fallback returns metadata")
detector2 = ChatLayoutDetector(screen_width=720)
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([500, 100, 600, 150], 0.9),
    TextBox([510, 200, 610, 250], 0.9),
]
layout, left, right, metadata = detector2.split_columns_median_fallback(boxes)
assert metadata is not None
assert metadata["method"] == "median_fallback"
assert "reason" in metadata
print(f"  Metadata: {metadata}")
print("  Fallback metadata present: PASS")

# Test 4: Single-sided data returns single layout
print("\n✓ Test 4: Single-sided data returns single layout")
boxes_single = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([120, 300, 220, 350], 0.9),
    TextBox([130, 400, 230, 450], 0.9),
]
layout, left, right, metadata = detector2.split_columns_median_fallback(boxes_single)
print(f"  Layout: {layout}, Metadata: {metadata}")
assert layout == "single"
# The reason might be "single_sided_data" or "low_separation" depending on the data
assert metadata["reason"] in ["single_sided_data", "low_separation"]
print("  Single-sided data handled correctly: PASS")

# Test 5: split_columns uses fallback when needed
print("\n✓ Test 5: split_columns integrates fallback")
detector3 = ChatLayoutDetector(screen_width=720)
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([500, 100, 600, 150], 0.9),
    TextBox([510, 200, 610, 250], 0.9),
]
layout, left, right, metadata = detector3.split_columns(boxes)
assert metadata is not None
assert metadata["method"] == "median_fallback"
print("  split_columns uses fallback when memory empty: PASS")

# Test 6: split_columns uses KMeans when sufficient data
print("\n✓ Test 6: split_columns uses KMeans with sufficient data")
detector4 = ChatLayoutDetector(screen_width=720)
detector4.memory["A"] = {"center": 0.2, "width": 0.1, "count": 30}
detector4.memory["B"] = {"center": 0.8, "width": 0.1, "count": 25}
layout, left, right, metadata = detector4.split_columns(boxes)
assert metadata is None  # No fallback metadata when using KMeans
print("  split_columns uses KMeans with sufficient data: PASS")

# Test 7: process_frame includes fallback metadata
print("\n✓ Test 7: process_frame includes fallback metadata")
detector5 = ChatLayoutDetector(screen_width=720)
result = detector5.process_frame(boxes)
assert "method" in result["metadata"]
assert result["metadata"]["method"] == "median_fallback"
assert "reason" in result["metadata"]
print(f"  process_frame metadata: {result['metadata']}")
print("  Fallback info propagated to process_frame: PASS")

print("\n" + "=" * 60)
print("All fallback mechanism tests PASSED!")
print("=" * 60)
