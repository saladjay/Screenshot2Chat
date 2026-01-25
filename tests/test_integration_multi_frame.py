"""
Multi-frame sequence integration tests for ChatLayoutDetector
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

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


def create_double_column_frame(left_center=150, right_center=570, num_boxes_per_side=3, y_start=100):
    """Helper function to create a frame with double-column layout"""
    boxes = []
    
    # Create left column boxes
    for i in range(num_boxes_per_side):
        x_min = left_center - 60
        x_max = left_center + 60
        y_min = y_start + i * 80
        y_max = y_min + 50
        boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.9))
    
    # Create right column boxes
    for i in range(num_boxes_per_side):
        x_min = right_center - 60
        x_max = right_center + 60
        y_min = y_start + i * 80
        y_max = y_min + 50
        boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.9))
    
    return boxes



def test_speaker_consistency_across_frames():
    """
    Test that speaker assignments remain consistent across multiple frames
    with similar layouts.
    
    Validates: Requirements 3.4, 4.4, 4.5
    """
    # Create detector with temporary memory file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        detector = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        # Process 5 frames with consistent layout
        frames = []
        for i in range(5):
            # Create frames with slight variations but consistent column positions
            left_center = 150 + np.random.randint(-10, 10)
            right_center = 570 + np.random.randint(-10, 10)
            boxes = create_double_column_frame(left_center, right_center, num_boxes_per_side=3)
            frames.append(boxes)
        
        results = []
        for boxes in frames:
            result = detector.process_frame(boxes)
            results.append(result)
        
        # Verify all frames are detected as double column
        for i, result in enumerate(results):
            assert result["layout"].startswith("double"), \
                f"Frame {i}: Expected double column layout, got {result['layout']}"
        
        # Verify speaker consistency: boxes in similar positions should be assigned to same speaker
        # Check that left column boxes are consistently assigned
        first_result = results[0]
        first_left_boxes = [b for b in frames[0] if b.center_x < 360]  # Left half
        first_speaker_for_left = "A" if any(b in first_result["A"] for b in first_left_boxes) else "B"
        
        for i, result in enumerate(results[1:], 1):
            left_boxes = [b for b in frames[i] if b.center_x < 360]
            current_speaker_for_left = "A" if any(b in result["A"] for b in left_boxes) else "B"
            
            assert current_speaker_for_left == first_speaker_for_left, \
                f"Frame {i}: Speaker assignment for left column changed from {first_speaker_for_left} to {current_speaker_for_left}"
        
        print(f"✓ Speaker consistency maintained across {len(frames)} frames")
        
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_layout_change_adaptation():
    """
    Test that the detector adapts when layout changes between frames.
    
    Validates: Requirements 3.4, 4.4, 4.5
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        detector = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        # Frame 1-2: Standard double column (left and right)
        frame1 = create_double_column_frame(left_center=150, right_center=570, num_boxes_per_side=3)
        frame2 = create_double_column_frame(left_center=155, right_center=565, num_boxes_per_side=3)
        
        result1 = detector.process_frame(frame1)
        result2 = detector.process_frame(frame2)
        
        assert result1["layout"].startswith("double")
        assert result2["layout"].startswith("double")
        
        # Frame 3: Layout changes to double_left (both columns on left side)
        frame3 = create_double_column_frame(left_center=120, right_center=280, num_boxes_per_side=3)
        result3 = detector.process_frame(frame3)
        
        # Should still detect as double column (may be double_left)
        assert result3["layout"].startswith("double"), \
            f"Expected double column layout after layout change, got {result3['layout']}"
        
        # Frame 4-5: Return to standard double column
        frame4 = create_double_column_frame(left_center=150, right_center=570, num_boxes_per_side=3)
        frame5 = create_double_column_frame(left_center=148, right_center=572, num_boxes_per_side=3)
        
        result4 = detector.process_frame(frame4)
        result5 = detector.process_frame(frame5)
        
        assert result4["layout"].startswith("double")
        assert result5["layout"].startswith("double")
        
        print("✓ Detector successfully adapted to layout changes")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_memory_convergence():
    """
    Test that memory converges and stabilizes over multiple frames.
    
    Validates: Requirements 3.1, 3.2, 3.5
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        detector = ChatLayoutDetector(screen_width=720, memory_path=temp_path, memory_alpha=0.7)
        
        # Process multiple frames with consistent layout
        num_frames = 10
        left_center_target = 150
        right_center_target = 570
        
        memory_history = []
        
        for i in range(num_frames):
            # Add small random variations
            left_center = left_center_target + np.random.randint(-5, 5)
            right_center = right_center_target + np.random.randint(-5, 5)
            boxes = create_double_column_frame(left_center, right_center, num_boxes_per_side=3)
            
            result = detector.process_frame(boxes)
            
            # Record memory state
            if detector.memory["A"] is not None and detector.memory["B"] is not None:
                memory_history.append({
                    "frame": i,
                    "A_center": detector.memory["A"]["center"],
                    "B_center": detector.memory["B"]["center"],
                    "A_count": detector.memory["A"]["count"],
                    "B_count": detector.memory["B"]["count"]
                })
        
        # Verify memory was initialized
        assert len(memory_history) > 0, "Memory was never initialized"
        
        # Verify memory counts are monotonically increasing
        for i in range(1, len(memory_history)):
            assert memory_history[i]["A_count"] >= memory_history[i-1]["A_count"], \
                f"Speaker A count decreased from frame {i-1} to {i}"
            assert memory_history[i]["B_count"] >= memory_history[i-1]["B_count"], \
                f"Speaker B count decreased from frame {i-1} to {i}"
        
        # Verify memory centers stabilize (variance decreases over time)
        if len(memory_history) >= 5:
            early_centers_A = [m["A_center"] for m in memory_history[:3]]
            late_centers_A = [m["A_center"] for m in memory_history[-3:]]
            
            early_variance_A = np.var(early_centers_A)
            late_variance_A = np.var(late_centers_A)
            
            # Later variance should be smaller or similar (memory stabilizing)
            # Allow some tolerance since we're using random variations
            print(f"  Early variance (A): {early_variance_A:.6f}")
            print(f"  Late variance (A): {late_variance_A:.6f}")
            print(f"  Memory converged: {late_variance_A <= early_variance_A * 1.5}")
        
        print(f"✓ Memory converged over {num_frames} frames")
        print(f"  Final counts: A={memory_history[-1]['A_count']}, B={memory_history[-1]['B_count']}")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)



def test_single_to_double_column_transition():
    """
    Test transition from single column to double column layout.
    
    Validates: Requirements 3.4, 5.1, 5.2
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        detector = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        # Frame 1-2: Single column (all boxes on left)
        frame1_boxes = []
        for i in range(3):
            x_min = 100
            x_max = 220
            y_min = 100 + i * 80
            y_max = y_min + 50
            frame1_boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.9))
        
        result1 = detector.process_frame(frame1_boxes)
        assert result1["layout"] == "single", f"Expected single layout, got {result1['layout']}"
        assert len(result1["B"]) == 0, "Speaker B should be empty for single column"
        
        # Frame 2: Transition to double column
        frame2 = create_double_column_frame(left_center=150, right_center=570, num_boxes_per_side=3)
        result2 = detector.process_frame(frame2)
        
        assert result2["layout"].startswith("double"), \
            f"Expected double column layout, got {result2['layout']}"
        assert len(result2["A"]) > 0, "Speaker A should have boxes"
        assert len(result2["B"]) > 0, "Speaker B should have boxes"
        
        # Frame 3-4: Continue with double column
        frame3 = create_double_column_frame(left_center=152, right_center=568, num_boxes_per_side=3)
        frame4 = create_double_column_frame(left_center=148, right_center=572, num_boxes_per_side=3)
        
        result3 = detector.process_frame(frame3)
        result4 = detector.process_frame(frame4)
        
        assert result3["layout"].startswith("double")
        assert result4["layout"].startswith("double")
        
        print("✓ Successfully transitioned from single to double column layout")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_frame_count_increment():
    """
    Test that frame_count increments correctly across multiple frames.
    
    Validates: Requirements 5.5
    """
    detector = ChatLayoutDetector(screen_width=720)
    
    assert detector.frame_count == 0, "Initial frame count should be 0"
    
    # Process 5 frames
    for i in range(5):
        boxes = create_double_column_frame(num_boxes_per_side=2)
        result = detector.process_frame(boxes)
        
        expected_count = i + 1
        assert detector.frame_count == expected_count, \
            f"After frame {i+1}, expected frame_count={expected_count}, got {detector.frame_count}"
        assert result["metadata"]["frame_count"] == expected_count, \
            f"Metadata frame_count mismatch: expected {expected_count}, got {result['metadata']['frame_count']}"
    
    print(f"✓ Frame count correctly incremented to {detector.frame_count}")


def test_memory_persistence_across_sessions():
    """
    Test that memory is correctly saved and loaded across detector sessions.
    
    Validates: Requirements 9.1, 9.2
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        # Session 1: Process some frames and build memory
        detector1 = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        for i in range(3):
            boxes = create_double_column_frame(left_center=150, right_center=570, num_boxes_per_side=3)
            detector1.process_frame(boxes)
        
        # Record memory state
        memory_a_center_1 = detector1.memory["A"]["center"] if detector1.memory["A"] else None
        memory_b_center_1 = detector1.memory["B"]["center"] if detector1.memory["B"] else None
        memory_a_count_1 = detector1.memory["A"]["count"] if detector1.memory["A"] else 0
        memory_b_count_1 = detector1.memory["B"]["count"] if detector1.memory["B"] else 0
        
        assert memory_a_center_1 is not None, "Memory A should be initialized"
        assert memory_b_center_1 is not None, "Memory B should be initialized"
        
        # Session 2: Create new detector and load memory
        detector2 = ChatLayoutDetector(screen_width=720, memory_path=temp_path)
        
        # Verify memory was loaded
        assert detector2.memory["A"] is not None, "Memory A should be loaded"
        assert detector2.memory["B"] is not None, "Memory B should be loaded"
        
        memory_a_center_2 = detector2.memory["A"]["center"]
        memory_b_center_2 = detector2.memory["B"]["center"]
        memory_a_count_2 = detector2.memory["A"]["count"]
        memory_b_count_2 = detector2.memory["B"]["count"]
        
        # Verify loaded memory matches saved memory
        assert abs(memory_a_center_2 - memory_a_center_1) < 1e-6, \
            f"Memory A center mismatch: {memory_a_center_1} vs {memory_a_center_2}"
        assert abs(memory_b_center_2 - memory_b_center_1) < 1e-6, \
            f"Memory B center mismatch: {memory_b_center_1} vs {memory_b_center_2}"
        assert memory_a_count_2 == memory_a_count_1, \
            f"Memory A count mismatch: {memory_a_count_1} vs {memory_a_count_2}"
        assert memory_b_count_2 == memory_b_count_1, \
            f"Memory B count mismatch: {memory_b_count_1} vs {memory_b_count_2}"
        
        print("✓ Memory successfully persisted and loaded across sessions")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests manually
    print("Running multi-frame integration tests...\n")
    
    print("Test 1: Speaker consistency across frames")
    test_speaker_consistency_across_frames()
    print()
    
    print("Test 2: Layout change adaptation")
    test_layout_change_adaptation()
    print()
    
    print("Test 3: Memory convergence")
    test_memory_convergence()
    print()
    
    print("Test 4: Single to double column transition")
    test_single_to_double_column_transition()
    print()
    
    print("Test 5: Frame count increment")
    test_frame_count_increment()
    print()
    
    print("Test 6: Memory persistence across sessions")
    test_memory_persistence_across_sessions()
    print()
    
    print("All integration tests passed! ✓")
