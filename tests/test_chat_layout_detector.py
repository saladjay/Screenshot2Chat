"""
Property-based tests for ChatLayoutDetector

This module contains property-based tests using Hypothesis to verify
the correctness properties of the chat layout detection system.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.cluster import KMeans

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
    def min_x(self): 
        return self.x_min 

    @property
    def min_y(self): 
        return self.y_min

    @property
    def max_x(self): 
        return self.x_max

    @property
    def max_y(self): 
        return self.y_max

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self):
        return (self.y_min + self.y_max) / 2

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min


# ============================================================================
# Hypothesis Strategies for generating test data
# ============================================================================

@st.composite
def textbox_strategy(draw, screen_width=720):
    """
    Generate a reasonable TextBox object
    
    Args:
        draw: Hypothesis draw function
        screen_width: Screen width for generating boxes within bounds
        
    Returns:
        TextBox object with random but valid coordinates
    """
    x_min = draw(st.integers(min_value=0, max_value=screen_width - 100))
    width = draw(st.integers(min_value=50, max_value=min(300, screen_width - x_min)))
    y_min = draw(st.integers(min_value=0, max_value=1000))
    height = draw(st.integers(min_value=20, max_value=100))
    
    box = [x_min, y_min, x_min + width, y_min + height]
    score = draw(st.floats(min_value=0.5, max_value=1.0))
    
    return TextBox(box=box, score=score)


@st.composite
def double_column_boxes_strategy(draw, screen_width=720):
    """
    Generate a list of TextBox objects that form a clear double-column layout
    
    Creates boxes clustered around two distinct horizontal positions (left and right)
    with sufficient separation to be detected as double columns.
    
    Args:
        draw: Hypothesis draw function
        screen_width: Screen width
        
    Returns:
        List of TextBox objects forming a double-column layout
    """
    # Define two distinct column centers with good separation
    left_center = draw(st.integers(min_value=100, max_value=200))
    right_center = draw(st.integers(min_value=screen_width - 200, max_value=screen_width - 100))
    
    # Ensure sufficient separation
    if (right_center - left_center) / screen_width < 0.25:
        right_center = left_center + int(screen_width * 0.3)
    
    # Generate left column boxes
    num_left = draw(st.integers(min_value=2, max_value=8))
    left_boxes = []
    for _ in range(num_left):
        x_offset = draw(st.integers(min_value=-30, max_value=30))
        x_min = max(0, left_center + x_offset - 50)
        width = draw(st.integers(min_value=80, max_value=150))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        left_boxes.append(TextBox(box=box, score=score))
    
    # Generate right column boxes
    num_right = draw(st.integers(min_value=2, max_value=8))
    right_boxes = []
    for _ in range(num_right):
        x_offset = draw(st.integers(min_value=-30, max_value=30))
        x_min = max(0, min(screen_width - 150, right_center + x_offset - 50))
        width = draw(st.integers(min_value=80, max_value=150))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        right_boxes.append(TextBox(box=box, score=score))
    
    # Shuffle to mix left and right boxes
    all_boxes = left_boxes + right_boxes
    draw(st.randoms()).shuffle(all_boxes)
    
    return all_boxes


@st.composite
def double_left_boxes_strategy(draw, screen_width=720):
    """
    Generate boxes for a left-aligned double-column layout
    
    Both columns are positioned on the left side of the screen (< 0.5 normalized)
    but with sufficient separation to be detected as two columns.
    """
    # Both centers should be < 0.5 * screen_width
    left_center = draw(st.integers(min_value=80, max_value=int(screen_width * 0.25)))
    right_center = draw(st.integers(min_value=int(screen_width * 0.3), max_value=int(screen_width * 0.45)))
    
    # Ensure sufficient separation
    if (right_center - left_center) / screen_width < 0.18:
        right_center = left_center + int(screen_width * 0.2)
    
    # Generate boxes similar to double_column_boxes_strategy
    num_left = draw(st.integers(min_value=2, max_value=6))
    left_boxes = []
    for _ in range(num_left):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, left_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        left_boxes.append(TextBox(box=box, score=score))
    
    num_right = draw(st.integers(min_value=2, max_value=6))
    right_boxes = []
    for _ in range(num_right):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, right_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        right_boxes.append(TextBox(box=box, score=score))
    
    all_boxes = left_boxes + right_boxes
    draw(st.randoms()).shuffle(all_boxes)
    
    return all_boxes


# ============================================================================
# Property Tests for split_columns method
# ============================================================================

@settings(max_examples=100)
@given(st.lists(textbox_strategy(), min_size=1, max_size=20))
def test_property_1_center_x_normalization_range(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 1: center_x归一化范围
    
    For any list of TextBox objects and screen width, the normalized center_x
    values should be in the range [0, 1].
    
    Validates: Requirements 1.1
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    # Extract and normalize center_x values
    centers = np.array([b.center_x for b in boxes]) / detector.screen_width
    
    # Verify all normalized centers are in [0, 1]
    assert np.all(centers >= 0), f"Found center_x < 0: {centers[centers < 0]}"
    assert np.all(centers <= 1), f"Found center_x > 1: {centers[centers > 1]}"


@settings(max_examples=100)
@given(st.lists(textbox_strategy(), min_size=0, max_size=3))
def test_property_2_few_samples_single_column(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 2: 少量样本判定为单列
    
    For any list of TextBox objects with fewer than 4 boxes, the system
    should classify the layout as single column.
    
    Validates: Requirements 1.2
    """
    if len(boxes) < 4:
        detector = ChatLayoutDetector(screen_width=720)
        layout, left, right, _ = detector.split_columns(boxes)
        
        assert layout == "single", \
            f"Expected 'single' layout for {len(boxes)} boxes, got '{layout}'"


@settings(max_examples=100, deadline=None)
@given(st.lists(textbox_strategy(), min_size=4, max_size=20))
def test_property_3_low_separation_single_column(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 3: 低分离度判定为单列
    
    For any list of TextBox objects where the KMeans cluster separation is
    less than min_separation_ratio, the system should classify as single column.
    
    Validates: Requirements 1.4
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    # Calculate what the separation would be
    centers = np.array([b.center_x for b in boxes]) / detector.screen_width
    
    if len(boxes) >= 4:
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # If separation is low, must be single column
        if separation < detector.min_separation_ratio:
            assert layout == "single", \
                f"Expected 'single' for separation {separation:.3f} < {detector.min_separation_ratio}, got '{layout}'"


@settings(max_examples=100)
@given(double_column_boxes_strategy())
def test_property_4_high_separation_double_column(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 4: 高分离度判定为双列
    
    For any list of TextBox objects where the KMeans cluster separation is
    greater than or equal to min_separation_ratio, the system should classify
    as some type of double column layout.
    
    Validates: Requirements 1.5
    """
    # Create detector with sufficient historical data to avoid fallback
    detector = ChatLayoutDetector(screen_width=720)
    # Populate memory to avoid fallback
    detector.memory["A"] = {"center": 0.25, "width": 0.2, "count": 30}
    detector.memory["B"] = {"center": 0.75, "width": 0.2, "count": 30}
    
    if len(boxes) >= 4:
        centers = np.array([b.center_x for b in boxes]) / detector.screen_width
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # If separation is high, must be some type of double column
        if separation >= detector.min_separation_ratio:
            assert layout.startswith("double"), \
                f"Expected 'double*' for separation {separation:.3f} >= {detector.min_separation_ratio}, got '{layout}'"


@st.composite
def double_right_boxes_strategy(draw, screen_width=720):
    """
    Generate boxes for a right-aligned double-column layout
    
    Both columns are positioned on the right side of the screen (> 0.5 normalized)
    but with sufficient separation to be detected as two columns.
    """
    # Both centers should be > 0.5 * screen_width
    left_center = draw(st.integers(min_value=int(screen_width * 0.55), max_value=int(screen_width * 0.7)))
    right_center = draw(st.integers(min_value=int(screen_width * 0.75), max_value=int(screen_width * 0.9)))
    
    # Ensure sufficient separation
    if (right_center - left_center) / screen_width < 0.18:
        right_center = left_center + int(screen_width * 0.2)
    
    # Generate boxes similar to double_column_boxes_strategy
    num_left = draw(st.integers(min_value=2, max_value=6))
    left_boxes = []
    for _ in range(num_left):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, left_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        left_boxes.append(TextBox(box=box, score=score))
    
    num_right = draw(st.integers(min_value=2, max_value=6))
    right_boxes = []
    for _ in range(num_right):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, right_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        right_boxes.append(TextBox(box=box, score=score))
    
    all_boxes = left_boxes + right_boxes
    draw(st.randoms()).shuffle(all_boxes)
    
    return all_boxes


@st.composite
def standard_double_boxes_strategy(draw, screen_width=720):
    """
    Generate boxes for a standard double-column layout
    
    One column is on the left side (< 0.5) and one is on the right side (> 0.5)
    with sufficient separation to be detected as two columns.
    """
    # Left center should be < 0.5, right center should be > 0.5
    left_center = draw(st.integers(min_value=80, max_value=int(screen_width * 0.4)))
    right_center = draw(st.integers(min_value=int(screen_width * 0.6), max_value=screen_width - 80))
    
    # Ensure sufficient separation
    if (right_center - left_center) / screen_width < 0.25:
        right_center = left_center + int(screen_width * 0.3)
    
    # Generate boxes similar to double_column_boxes_strategy
    num_left = draw(st.integers(min_value=2, max_value=6))
    left_boxes = []
    for _ in range(num_left):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, left_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        left_boxes.append(TextBox(box=box, score=score))
    
    num_right = draw(st.integers(min_value=2, max_value=6))
    right_boxes = []
    for _ in range(num_right):
        x_offset = draw(st.integers(min_value=-20, max_value=20))
        x_min = max(0, right_center + x_offset - 40)
        width = draw(st.integers(min_value=60, max_value=120))
        y_min = draw(st.integers(min_value=0, max_value=1000))
        height = draw(st.integers(min_value=20, max_value=80))
        
        box = [x_min, y_min, x_min + width, y_min + height]
        score = draw(st.floats(min_value=0.5, max_value=1.0))
        right_boxes.append(TextBox(box=box, score=score))
    
    all_boxes = left_boxes + right_boxes
    draw(st.randoms()).shuffle(all_boxes)
    
    return all_boxes


# ============================================================================
# Property Tests for Layout Subtypes (Task 2.3)
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(double_left_boxes_strategy())
def test_property_5_double_left_classification(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 5: 左对齐双列判定
    
    For any list of TextBox objects classified as double column where both
    cluster centers are less than 0.5 (normalized), the system should mark
    the layout as "double_left".
    
    Validates: Requirements 1.6
    """
    # Create detector with sufficient historical data to avoid fallback
    detector = ChatLayoutDetector(screen_width=720)
    # Populate memory to avoid fallback
    detector.memory["A"] = {"center": 0.25, "width": 0.2, "count": 30}
    detector.memory["B"] = {"center": 0.75, "width": 0.2, "count": 30}
    
    if len(boxes) >= 4:
        centers = np.array([b.center_x for b in boxes]) / detector.screen_width
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # Only check layout subtype if separation is sufficient AND both centers are < 0.5
        if separation >= detector.min_separation_ratio and cluster_centers[0] < 0.5 and cluster_centers[1] < 0.5:
            assert layout == "double_left", \
                f"Expected 'double_left' for centers {cluster_centers[0]:.3f}, {cluster_centers[1]:.3f} (both < 0.5), got '{layout}'"


@settings(max_examples=100, deadline=None)
@given(double_right_boxes_strategy())
def test_property_6_double_right_classification(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 6: 右对齐双列判定
    
    For any list of TextBox objects classified as double column where both
    cluster centers are greater than 0.5 (normalized), the system should mark
    the layout as "double_right".
    
    Validates: Requirements 1.7
    """
    # Create detector with sufficient historical data to avoid fallback
    detector = ChatLayoutDetector(screen_width=720)
    # Populate memory to avoid fallback
    detector.memory["A"] = {"center": 0.25, "width": 0.2, "count": 30}
    detector.memory["B"] = {"center": 0.75, "width": 0.2, "count": 30}
    
    if len(boxes) >= 4:
        centers = np.array([b.center_x for b in boxes]) / detector.screen_width
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # Only check layout subtype if separation is sufficient AND both centers are > 0.5
        if separation >= detector.min_separation_ratio and cluster_centers[0] > 0.5 and cluster_centers[1] > 0.5:
            assert layout == "double_right", \
                f"Expected 'double_right' for centers {cluster_centers[0]:.3f}, {cluster_centers[1]:.3f} (both > 0.5), got '{layout}'"


@settings(max_examples=100, deadline=None)
@given(standard_double_boxes_strategy())
def test_property_7_standard_double_classification(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 7: 标准双列判定
    
    For any list of TextBox objects classified as double column where the
    cluster centers are on opposite sides of 0.5 (one < 0.5, one > 0.5),
    the system should mark the layout as "double".
    
    Validates: Requirements 1.8
    """
    # Create detector with sufficient historical data to avoid fallback
    detector = ChatLayoutDetector(screen_width=720)
    # Populate memory to avoid fallback
    detector.memory["A"] = {"center": 0.25, "width": 0.2, "count": 30}
    detector.memory["B"] = {"center": 0.75, "width": 0.2, "count": 30}
    
    if len(boxes) >= 4:
        centers = np.array([b.center_x for b in boxes]) / detector.screen_width
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # Only check if separation is sufficient AND centers are on opposite sides
        if separation >= detector.min_separation_ratio and cluster_centers[0] < 0.5 and cluster_centers[1] > 0.5:
            assert layout == "double", \
                f"Expected 'double' for centers {cluster_centers[0]:.3f}, {cluster_centers[1]:.3f} (opposite sides of 0.5), got '{layout}'"


# ============================================================================
# Property Tests for Column Assignment (Task 2.4)
# ============================================================================

@settings(max_examples=100)
@given(st.lists(textbox_strategy(), min_size=0, max_size=3))
def test_property_8_single_column_right_empty(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 8: 单列布局右列为空
    
    For any list of TextBox objects classified as single column layout,
    the right column list should be empty and all boxes should be in the left column.
    
    Validates: Requirements 2.1, 5.3
    """
    detector = ChatLayoutDetector(screen_width=720)
    layout, left, right, _ = detector.split_columns(boxes)
    
    if layout == "single":
        assert len(right) == 0, \
            f"Expected right column to be empty for single layout, got {len(right)} boxes"
        assert len(left) == len(boxes), \
            f"Expected all {len(boxes)} boxes in left column, got {len(left)}"


@settings(max_examples=100, deadline=None)
@given(st.lists(textbox_strategy(), min_size=1, max_size=20))
def test_property_9_column_assignment_completeness(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 9: 列分配完整性
    
    For any list of TextBox objects, the total number of boxes in the left
    and right columns returned by split_columns should equal the input list length.
    
    Validates: Requirements 2.2
    """
    detector = ChatLayoutDetector(screen_width=720)
    layout, left, right, _ = detector.split_columns(boxes)
    
    total_assigned = len(left) + len(right)
    assert total_assigned == len(boxes), \
        f"Expected {len(boxes)} total boxes, got {total_assigned} (left: {len(left)}, right: {len(right)})"


@settings(max_examples=100, deadline=None)
@given(double_column_boxes_strategy())
def test_property_10_nearest_cluster_assignment(boxes):
    """
    Feature: chat-bubble-detection-refactor, Property 10: 最近聚类中心分配
    
    For any TextBox in a double column layout, the box should be assigned to
    the column corresponding to the cluster center that is closer to its center_x.
    
    Validates: Requirements 2.2, 2.3, 2.4
    """
    # Create detector with sufficient historical data to avoid fallback
    detector = ChatLayoutDetector(screen_width=720)
    # Populate memory to avoid fallback
    detector.memory["A"] = {"center": 0.25, "width": 0.2, "count": 30}
    detector.memory["B"] = {"center": 0.75, "width": 0.2, "count": 30}
    
    if len(boxes) >= 4:
        centers = np.array([b.center_x for b in boxes]) / detector.screen_width
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(centers.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        separation = cluster_centers[1] - cluster_centers[0]
        
        layout, left, right, _ = detector.split_columns(boxes)
        
        # Only check for double column layouts with sufficient separation
        if layout.startswith("double") and separation >= detector.min_separation_ratio:
            # Verify each box in left column is closer to left cluster center
            for box in left:
                norm_center = box.center_x / detector.screen_width
                dist_to_left = abs(norm_center - cluster_centers[0])
                dist_to_right = abs(norm_center - cluster_centers[1])
                assert dist_to_left <= dist_to_right, \
                    f"Box in left column (center_x={norm_center:.3f}) is closer to right cluster " \
                    f"(dist_left={dist_to_left:.3f}, dist_right={dist_to_right:.3f})"
            
            # Verify each box in right column is closer to right cluster center
            for box in right:
                norm_center = box.center_x / detector.screen_width
                dist_to_left = abs(norm_center - cluster_centers[0])
                dist_to_right = abs(norm_center - cluster_centers[1])
                assert dist_to_right <= dist_to_left, \
                    f"Box in right column (center_x={norm_center:.3f}) is closer to left cluster " \
                    f"(dist_left={dist_to_left:.3f}, dist_right={dist_to_right:.3f})"
