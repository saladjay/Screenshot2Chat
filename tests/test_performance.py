"""
Performance tests for ChatLayoutDetector

Tests single frame processing time and memory usage to ensure
the system meets performance requirements (<100ms per frame).
"""

import time
import tracemalloc
import numpy as np
import pytest
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.processors import TextBox


def generate_test_boxes(num_boxes: int, screen_width: int = 720) -> list:
    """Generate test TextBox objects for performance testing"""
    boxes = []
    for i in range(num_boxes):
        # Alternate between left and right columns
        if i % 2 == 0:
            x_min = np.random.randint(50, 200)
        else:
            x_min = np.random.randint(500, 650)
        
        width = np.random.randint(100, 200)
        y_min = i * 80 + np.random.randint(0, 20)
        height = np.random.randint(40, 80)
        
        box = TextBox(
            box=np.array([x_min, y_min, x_min + width, y_min + height]),
            score=0.95
        )
        boxes.append(box)
    
    return boxes


class TestPerformance:
    """Performance tests for ChatLayoutDetector"""
    
    def test_single_frame_processing_time_small(self):
        """Test processing time for small frame (10 boxes) - should be <100ms"""
        detector = ChatLayoutDetector(screen_width=720)
        boxes = generate_test_boxes(10)
        
        # Warm up
        detector.process_frame(boxes)
        
        # Measure
        start_time = time.perf_counter()
        result = detector.process_frame(boxes)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        print(f"\nSmall frame (10 boxes) processing time: {processing_time_ms:.2f}ms")
        assert processing_time_ms < 100, f"Processing time {processing_time_ms:.2f}ms exceeds 100ms target"
        assert result is not None
    
    def test_single_frame_processing_time_medium(self):
        """Test processing time for medium frame (30 boxes) - should be <100ms"""
        detector = ChatLayoutDetector(screen_width=720)
        boxes = generate_test_boxes(30)
        
        # Warm up
        detector.process_frame(boxes)
        
        # Measure
        start_time = time.perf_counter()
        result = detector.process_frame(boxes)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        print(f"\nMedium frame (30 boxes) processing time: {processing_time_ms:.2f}ms")
        assert processing_time_ms < 100, f"Processing time {processing_time_ms:.2f}ms exceeds 100ms target"
        assert result is not None
    
    def test_single_frame_processing_time_large(self):
        """Test processing time for large frame (50 boxes) - should be <100ms"""
        detector = ChatLayoutDetector(screen_width=720)
        boxes = generate_test_boxes(50)
        
        # Warm up
        detector.process_frame(boxes)
        
        # Measure
        start_time = time.perf_counter()
        result = detector.process_frame(boxes)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        print(f"\nLarge frame (50 boxes) processing time: {processing_time_ms:.2f}ms")
        assert processing_time_ms < 100, f"Processing time {processing_time_ms:.2f}ms exceeds 100ms target"
        assert result is not None
    
    def test_average_processing_time(self):
        """Test average processing time over multiple frames"""
        detector = ChatLayoutDetector(screen_width=720)
        num_iterations = 100
        total_time = 0
        
        for _ in range(num_iterations):
            boxes = generate_test_boxes(np.random.randint(10, 40))
            
            start_time = time.perf_counter()
            detector.process_frame(boxes)
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
        
        avg_time_ms = (total_time / num_iterations) * 1000
        
        print(f"\nAverage processing time over {num_iterations} frames: {avg_time_ms:.2f}ms")
        assert avg_time_ms < 100, f"Average processing time {avg_time_ms:.2f}ms exceeds 100ms target"
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        tracemalloc.start()
        
        detector = ChatLayoutDetector(screen_width=720)
        boxes = generate_test_boxes(30)
        
        # Get baseline memory
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Process multiple frames
        for _ in range(50):
            detector.process_frame(boxes)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_increase_mb = (peak - baseline) / (1024 * 1024)
        
        print(f"\nMemory increase after 50 frames: {memory_increase_mb:.2f}MB")
        print(f"Peak memory usage: {peak / (1024 * 1024):.2f}MB")
        
        # Memory should not grow excessively (< 10MB increase for 50 frames)
        assert memory_increase_mb < 10, f"Memory increase {memory_increase_mb:.2f}MB is too high"
    
    def test_memory_with_persistence(self, tmp_path):
        """Test memory usage with persistence enabled"""
        tracemalloc.start()
        
        memory_path = tmp_path / "test_memory.json"
        detector = ChatLayoutDetector(screen_width=720, memory_path=str(memory_path))
        boxes = generate_test_boxes(30)
        
        # Get baseline memory
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Process multiple frames with auto-save
        for _ in range(50):
            detector.process_frame(boxes)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_increase_mb = (peak - baseline) / (1024 * 1024)
        
        print(f"\nMemory increase with persistence after 50 frames: {memory_increase_mb:.2f}MB")
        
        # Memory should not grow excessively even with persistence
        assert memory_increase_mb < 15, f"Memory increase {memory_increase_mb:.2f}MB is too high with persistence"
    
    def test_processing_time_with_memory_loaded(self, tmp_path):
        """Test processing time when memory is pre-loaded"""
        memory_path = tmp_path / "test_memory.json"
        
        # First, create and save memory
        detector1 = ChatLayoutDetector(screen_width=720, memory_path=str(memory_path))
        for _ in range(20):
            boxes = generate_test_boxes(30)
            detector1.process_frame(boxes)
        
        # Now load memory and test performance
        detector2 = ChatLayoutDetector(screen_width=720, memory_path=str(memory_path))
        boxes = generate_test_boxes(30)
        
        # Warm up
        detector2.process_frame(boxes)
        
        # Measure
        start_time = time.perf_counter()
        result = detector2.process_frame(boxes)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        print(f"\nProcessing time with pre-loaded memory: {processing_time_ms:.2f}ms")
        assert processing_time_ms < 100, f"Processing time {processing_time_ms:.2f}ms exceeds 100ms target"
        assert result is not None


if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s"])
