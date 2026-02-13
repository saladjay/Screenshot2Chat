"""
Exception hierarchy for the screenshot analysis system.

This module defines a clear exception hierarchy for different types of errors
that can occur during screenshot analysis operations.
"""


class ScreenshotAnalysisError(Exception):
    """
    Base exception class for all screenshot analysis errors.
    
    All custom exceptions in the system should inherit from this class.
    """
    pass


class ConfigurationError(ScreenshotAnalysisError):
    """
    Exception raised for configuration-related errors.
    
    Examples:
        - Invalid configuration values
        - Missing required configuration keys
        - Configuration file parsing errors
    """
    pass


class ModelError(ScreenshotAnalysisError):
    """
    Base exception class for model-related errors.
    
    This is a parent class for more specific model errors.
    """
    pass


class ModelLoadError(ModelError):
    """
    Exception raised when a model fails to load.
    
    Examples:
        - Model file not found
        - Corrupted model file
        - Incompatible model format
        - Insufficient memory to load model
    """
    pass


class ModelNotFoundError(ModelError):
    """
    Exception raised when a requested model is not found.
    
    Examples:
        - Model not registered in the model manager
        - Model version does not exist
        - Model path does not exist
    """
    pass


class DetectionError(ScreenshotAnalysisError):
    """
    Exception raised during detection operations.
    
    Examples:
        - Detection algorithm failure
        - Invalid input image
        - Backend-specific errors
    """
    pass


class ExtractionError(ScreenshotAnalysisError):
    """
    Exception raised during extraction operations.
    
    Examples:
        - Extraction algorithm failure
        - Invalid detection results
        - Missing required data for extraction
    """
    pass


class PipelineError(ScreenshotAnalysisError):
    """
    Exception raised during pipeline execution.
    
    Examples:
        - Invalid pipeline configuration
        - Step execution failure
        - Dependency resolution errors
        - Circular dependencies
    """
    pass


class ValidationError(ScreenshotAnalysisError):
    """
    Exception raised when validation fails.
    
    Examples:
        - Invalid configuration values
        - Invalid pipeline structure
        - Invalid data format
        - Schema validation failure
    """
    pass


class DataError(ScreenshotAnalysisError):
    """
    Exception raised for data-related errors.
    
    Examples:
        - Invalid image format
        - Corrupted image data
        - Missing required data fields
        - Data serialization/deserialization errors
    """
    pass
