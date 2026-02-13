"""
Tests for error handling and logging system.

This module tests the exception hierarchy and structured logging functionality.
"""

import pytest
import numpy as np
from src.screenshot2chat.core.exceptions import (
    ScreenshotAnalysisError,
    ConfigurationError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    DetectionError,
    ExtractionError,
    PipelineError,
    ValidationError,
    DataError,
)
from src.screenshot2chat.logging import StructuredLogger


class TestExceptionHierarchy:
    """Test the exception hierarchy."""
    
    def test_base_exception(self):
        """Test that base exception can be raised and caught."""
        with pytest.raises(ScreenshotAnalysisError):
            raise ScreenshotAnalysisError("Test error")
    
    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise ConfigurationError("Config error")
    
    def test_model_error_inheritance(self):
        """Test that ModelError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise ModelError("Model error")
    
    def test_model_load_error_inheritance(self):
        """Test that ModelLoadError inherits from ModelError."""
        with pytest.raises(ModelError):
            raise ModelLoadError("Load error")
        
        with pytest.raises(ScreenshotAnalysisError):
            raise ModelLoadError("Load error")
    
    def test_model_not_found_error_inheritance(self):
        """Test that ModelNotFoundError inherits from ModelError."""
        with pytest.raises(ModelError):
            raise ModelNotFoundError("Not found")
    
    def test_detection_error_inheritance(self):
        """Test that DetectionError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise DetectionError("Detection failed")
    
    def test_extraction_error_inheritance(self):
        """Test that ExtractionError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise ExtractionError("Extraction failed")
    
    def test_pipeline_error_inheritance(self):
        """Test that PipelineError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise PipelineError("Pipeline failed")
    
    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise ValidationError("Validation failed")
    
    def test_data_error_inheritance(self):
        """Test that DataError inherits from ScreenshotAnalysisError."""
        with pytest.raises(ScreenshotAnalysisError):
            raise DataError("Data error")
    
    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        error_message = "This is a test error message"
        
        try:
            raise DetectionError(error_message)
        except DetectionError as e:
            assert str(e) == error_message


class TestStructuredLogger:
    """Test the structured logger."""
    
    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = StructuredLogger("test_module")
        assert logger is not None
        assert logger.logger.name == "test_module"
    
    def test_set_context(self):
        """Test setting context information."""
        logger = StructuredLogger("test_module")
        logger.set_context(user_id="123", session_id="abc")
        
        assert logger.context["user_id"] == "123"
        assert logger.context["session_id"] == "abc"
    
    def test_clear_context(self):
        """Test clearing context information."""
        logger = StructuredLogger("test_module")
        logger.set_context(user_id="123")
        logger.clear_context()
        
        assert len(logger.context) == 0
    
    def test_log_levels(self):
        """Test that all log levels work without errors."""
        logger = StructuredLogger("test_module")
        
        # These should not raise exceptions
        logger.debug("Debug message", key="value")
        logger.info("Info message", key="value")
        logger.warning("Warning message", key="value")
        logger.error("Error message", exc_info=False, key="value")
        logger.critical("Critical message", exc_info=False, key="value")
    
    def test_format_message_with_context(self):
        """Test message formatting with context."""
        logger = StructuredLogger("test_module")
        logger.set_context(request_id="req-123")
        
        formatted = logger._format_message("Test message", user="alice")
        
        # Should contain both context and kwargs
        assert "request_id" in formatted
        assert "user" in formatted
    
    def test_format_message_without_context(self):
        """Test message formatting without context."""
        logger = StructuredLogger("test_module")
        
        formatted = logger._format_message("Test message")
        
        # Should just be the message
        assert formatted == "Test message"
    
    def test_exception_logging(self):
        """Test exception logging."""
        logger = StructuredLogger("test_module")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            # Should not raise
            logger.exception("An error occurred")


class TestErrorHandlingIntegration:
    """Test error handling in actual components."""
    
    def test_detector_with_invalid_image(self):
        """Test that detector raises DataError for invalid image."""
        from src.screenshot2chat.detectors.text_detector import TextDetector
        
        detector = TextDetector(config={"backend": "paddleocr"})
        
        # Mock the model to avoid loading issues
        detector.model = "mock_model"
        
        # Test with None
        with pytest.raises(DataError) as exc_info:
            detector.detect(None)
        
        assert "Input image is None" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_detector_with_wrong_type(self):
        """Test that detector raises DataError for wrong type."""
        from src.screenshot2chat.detectors.text_detector import TextDetector
        
        detector = TextDetector(config={"backend": "paddleocr"})
        
        # Mock the model to avoid loading issues
        detector.model = "mock_model"
        
        # Test with wrong type
        with pytest.raises(DataError) as exc_info:
            detector.detect("not an array")
        
        assert "must be a numpy array" in str(exc_info.value)
    
    def test_detector_with_invalid_dimensions(self):
        """Test that detector raises DataError for invalid dimensions."""
        from src.screenshot2chat.detectors.text_detector import TextDetector
        
        detector = TextDetector(config={"backend": "paddleocr"})
        
        # Test with 1D array
        with pytest.raises(DataError) as exc_info:
            detector.preprocess(np.array([1, 2, 3]))
        
        assert "must be 2D or 3D" in str(exc_info.value)
    
    def test_bubble_detector_without_text_boxes(self):
        """Test that bubble detector raises DataError without text boxes."""
        from src.screenshot2chat.detectors.bubble_detector import BubbleDetector
        
        detector = BubbleDetector()
        detector.load_model()
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(DataError) as exc_info:
            detector.detect(image, text_boxes=None)
        
        assert "text_boxes is required" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_nickname_extractor_without_processor(self):
        """Test that nickname extractor raises DataError without processor."""
        from src.screenshot2chat.extractors.nickname_extractor import NicknameExtractor
        from src.screenshot2chat.core.data_models import DetectionResult
        
        extractor = NicknameExtractor(config={})
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detection_results = [
            DetectionResult(bbox=[10, 10, 50, 30], score=0.9, category="text")
        ]
        
        with pytest.raises(DataError) as exc_info:
            extractor.extract(detection_results, image)
        
        assert "processor is required" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_nickname_extractor_without_image(self):
        """Test that nickname extractor raises DataError without image."""
        from src.screenshot2chat.extractors.nickname_extractor import NicknameExtractor
        from src.screenshot2chat.core.data_models import DetectionResult
        
        # Create a mock processor
        class MockProcessor:
            pass
        
        extractor = NicknameExtractor(config={'processor': MockProcessor()})
        
        detection_results = [
            DetectionResult(bbox=[10, 10, 50, 30], score=0.9, category="text")
        ]
        
        with pytest.raises(DataError) as exc_info:
            extractor.extract(detection_results, None)
        
        assert "image is required" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_speaker_extractor_with_none_input(self):
        """Test that speaker extractor raises DataError with None input."""
        from src.screenshot2chat.extractors.speaker_extractor import SpeakerExtractor
        
        extractor = SpeakerExtractor()
        
        with pytest.raises(DataError) as exc_info:
            extractor.extract(None)
        
        assert "detection_results cannot be None" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_layout_extractor_with_none_input(self):
        """Test that layout extractor raises DataError with None input."""
        from src.screenshot2chat.extractors.layout_extractor import LayoutExtractor
        
        extractor = LayoutExtractor()
        
        with pytest.raises(DataError) as exc_info:
            extractor.extract(None)
        
        assert "detection_results cannot be None" in str(exc_info.value)
        assert "Recovery suggestion" in str(exc_info.value)
    
    def test_pipeline_validation_error(self):
        """Test that pipeline raises PipelineError for invalid configuration."""
        from src.screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
        from src.screenshot2chat.detectors.text_detector import TextDetector
        
        pipeline = Pipeline(name="test")
        
        # Add a step with invalid dependency
        step = PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=TextDetector(),
            depends_on=["nonexistent_step"]
        )
        pipeline.add_step(step)
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Should raise PipelineError (which wraps the validation error)
        with pytest.raises(PipelineError) as exc_info:
            pipeline.execute(image)
        
        # The error message should mention the validation issue
        assert "nonexistent_step" in str(exc_info.value) or "validation" in str(exc_info.value).lower()
    
    def test_pipeline_execution_error_handling(self):
        """Test that pipeline properly handles and wraps execution errors."""
        from src.screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
        from src.screenshot2chat.detectors.text_detector import TextDetector
        
        pipeline = Pipeline(name="test")
        
        # Add a detector step
        detector = TextDetector(config={"backend": "paddleocr"})
        step = PipelineStep(
            name="text_detector",
            step_type=StepType.DETECTOR,
            component=detector
        )
        pipeline.add_step(step)
        
        # Try to execute with invalid image (should raise PipelineError wrapping DataError)
        with pytest.raises(PipelineError) as exc_info:
            pipeline.execute(None)
        
        # The error message should mention the step name
        assert "text_detector" in str(exc_info.value) or "validation" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
