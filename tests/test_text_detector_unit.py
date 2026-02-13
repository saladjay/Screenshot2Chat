"""
Unit tests for TextDetector
Task 2.2: Write unit tests for TextDetector
Requirements: 6.2
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.screenshot2chat.detectors.text_detector import TextDetector
from src.screenshot2chat.core.data_models import DetectionResult


class TestTextDetector:
    """Unit tests for TextDetector"""
    
    def test_text_detector_creation_default(self):
        """Test creating TextDetector with default backend"""
        detector = TextDetector()
        assert detector.backend == "paddleocr"
        assert "paddleocr" in detector.supported_backends
    
    def test_text_detector_creation_with_backend(self):
        """Test creating TextDetector with specific backend"""
        detector = TextDetector(backend="paddleocr")
        assert detector.backend == "paddleocr"
    
    def test_text_detector_unsupported_backend(self):
        """Test that unsupported backend raises error on load"""
        detector = TextDetector(backend="unsupported_backend")
        with pytest.raises(ValueError, match="Unsupported backend"):
            detector.load_model()
    
    def test_text_detector_config_storage(self):
        """Test that TextDetector stores configuration"""
        config = {
            "model_dir": "custom/path",
            "threshold": 0.5
        }
        detector = TextDetector(config=config)
        assert detector.config == config
    
    @patch('src.screenshot2chat.detectors.text_detector.PaddleOCR')
    def test_text_detector_load_model_paddleocr(self, mock_paddle):
        """Test loading PaddleOCR model"""
        detector = TextDetector(backend="paddleocr")
        detector.load_model()
        
        # Verify model was loaded
        assert detector.model is not None
    
    @patch('src.screenshot2chat.detectors.text_detector.PaddleOCR')
    def test_text_detector_detect_returns_list(self, mock_paddle):
        """Test that detect returns a list of DetectionResult"""
        # Mock PaddleOCR response
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [[
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ("Hello", 0.95)]
        ]]
        mock_paddle.return_value = mock_ocr
        
        detector = TextDetector(backend="paddleocr")
        detector.load_model()
        
        # Create test image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Detect
        results = detector.detect(image)
        
        # Verify results
        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], DetectionResult)
    
    @patch('src.screenshot2chat.detectors.text_detector.PaddleOCR')
    def test_text_detector_detect_result_format(self, mock_paddle):
        """Test that detection results have correct format"""
        # Mock PaddleOCR response
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [[
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ("Hello", 0.95)]
        ]]
        mock_paddle.return_value = mock_ocr
        
        detector = TextDetector(backend="paddleocr")
        detector.load_model()
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        results = detector.detect(image)
        
        if len(results) > 0:
            result = results[0]
            assert len(result.bbox) == 4
            assert result.score >= 0 and result.score <= 1
            assert result.category == "text"
            assert isinstance(result.metadata, dict)
    
    def test_text_detector_preprocess(self):
        """Test image preprocessing"""
        detector = TextDetector()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        preprocessed = detector.preprocess(image)
        
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.shape == image.shape
    
    @patch('src.screenshot2chat.detectors.text_detector.PaddleOCR')
    def test_text_detector_empty_image(self, mock_paddle):
        """Test detection on empty/blank image"""
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [[]]  # No detections
        mock_paddle.return_value = mock_ocr
        
        detector = TextDetector(backend="paddleocr")
        detector.load_model()
        
        # Blank image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(image)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    @patch('src.screenshot2chat.detectors.text_detector.PaddleOCR')
    def test_text_detector_multiple_detections(self, mock_paddle):
        """Test detection with multiple text regions"""
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [[
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ("Hello", 0.95)],
            [[[10, 60], [100, 60], [100, 100], [10, 100]], ("World", 0.90)]
        ]]
        mock_paddle.return_value = mock_ocr
        
        detector = TextDetector(backend="paddleocr")
        detector.load_model()
        
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        results = detector.detect(image)
        
        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)
