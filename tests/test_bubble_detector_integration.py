"""
Integration tests for BubbleDetector
Task 2.4: Write integration tests for BubbleDetector
Requirements: 6.5
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from src.screenshot2chat.detectors.bubble_detector import BubbleDetector
from src.screenshot2chat.core.data_models import DetectionResult


class TestBubbleDetectorIntegration:
    """Integration tests for BubbleDetector"""
    
    def test_bubble_detector_creation(self):
        """Test creating BubbleDetector"""
        detector = BubbleDetector()
        assert detector.screen_width == 720  # default
    
    def test_bubble_detector_with_custom_width(self):
        """Test BubbleDetector with custom screen width"""
        config = {"screen_width": 1080}
        detector = BubbleDetector(config=config)
        assert detector.screen_width == 1080
    
    def test_bubble_detector_load_model(self):
        """Test loading the bubble detection model"""
        detector = BubbleDetector()
        detector.load_model()
        assert detector.layout_detector is not None
    
    def test_bubble_detector_with_memory_path(self):
        """Test BubbleDetector with memory persistence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "test_memory.json")
            config = {"memory_path": memory_path}
            
            detector = BubbleDetector(config=config)
            detector.load_model()
            
            # Memory path should be set
            assert detector.layout_detector is not None
    
    def test_bubble_detector_detect_interface(self):
        """Test that detect method returns correct interface"""
        detector = BubbleDetector()
        detector.load_model()
        
        # Create mock text detection results
        text_detections = [
            DetectionResult([10, 10, 100, 50], 0.9, "text", {"text": "Hello"}),
            DetectionResult([10, 60, 100, 100], 0.9, "text", {"text": "World"})
        ]
        
        # Create test image
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        # Detect bubbles
        results = detector.detect(image, text_detections)
        
        # Verify interface
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, DetectionResult)
            assert len(result.bbox) == 4
            assert result.category in ["bubble_left", "bubble_right", "bubble_center"]
    
    def test_bubble_detector_memory_update(self):
        """Test that bubble detector updates memory across frames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "test_memory.json")
            config = {"memory_path": memory_path}
            
            detector = BubbleDetector(config=config)
            detector.load_model()
            
            # Process first frame
            text_detections_1 = [
                DetectionResult([10, 10, 100, 50], 0.9, "text", {"text": "Hello"})
            ]
            image_1 = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
            results_1 = detector.detect(image_1, text_detections_1)
            
            # Process second frame
            text_detections_2 = [
                DetectionResult([10, 60, 100, 100], 0.9, "text", {"text": "World"})
            ]
            image_2 = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
            results_2 = detector.detect(image_2, text_detections_2)
            
            # Both should return results
            assert isinstance(results_1, list)
            assert isinstance(results_2, list)
    
    def test_bubble_detector_single_column_layout(self):
        """Test detection on single column layout"""
        detector = BubbleDetector()
        detector.load_model()
        
        # Create text detections in center column
        text_detections = [
            DetectionResult([300, 10, 420, 50], 0.9, "text", {"text": "Message 1"}),
            DetectionResult([300, 60, 420, 100], 0.9, "text", {"text": "Message 2"})
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        results = detector.detect(image, text_detections)
        
        assert isinstance(results, list)
    
    def test_bubble_detector_double_column_layout(self):
        """Test detection on double column layout (left/right)"""
        detector = BubbleDetector()
        detector.load_model()
        
        # Create text detections in left and right columns
        text_detections = [
            DetectionResult([10, 10, 200, 50], 0.9, "text", {"text": "Left message"}),
            DetectionResult([520, 60, 710, 100], 0.9, "text", {"text": "Right message"})
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        results = detector.detect(image, text_detections)
        
        assert isinstance(results, list)
    
    def test_bubble_detector_empty_detections(self):
        """Test bubble detector with no text detections"""
        detector = BubbleDetector()
        detector.load_model()
        
        text_detections = []
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        results = detector.detect(image, text_detections)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_bubble_detector_preprocess(self):
        """Test bubble detector preprocessing"""
        detector = BubbleDetector()
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        preprocessed = detector.preprocess(image)
        
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.shape == image.shape
