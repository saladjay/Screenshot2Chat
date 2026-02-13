"""
Structured logging system for the screenshot analysis framework.

This module provides a structured logger that supports context information
and multiple log levels for better debugging and monitoring.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLogger:
    """
    Structured logger that records system events with context information.
    
    This logger extends Python's standard logging with structured context
    that can be attached to log messages, making it easier to filter and
    analyze logs in production environments.
    
    Attributes:
        logger: The underlying Python logger instance
        context: Dictionary of context information to include in all logs
    
    Example:
        >>> logger = StructuredLogger("my_module")
        >>> logger.set_context(user_id="123", session_id="abc")
        >>> logger.info("Processing started", image_count=5)
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the structured logger.
        
        Args:
            name: Name of the logger (typically module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.context: Dict[str, Any] = {}
        
        # Add default handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_context(self, **kwargs) -> None:
        """
        Set context information that will be included in all subsequent logs.
        
        Context information persists across multiple log calls until cleared
        or updated. This is useful for tracking request IDs, user IDs, or
        other contextual information.
        
        Args:
            **kwargs: Key-value pairs to add to the context
        
        Example:
            >>> logger.set_context(request_id="req-123", user="alice")
            >>> logger.info("Action performed")  # Will include request_id and user
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """
        Clear all context information.
        
        Example:
            >>> logger.clear_context()
        """
        self.context.clear()
    
    def _format_message(self, message: str, **kwargs) -> str:
        """
        Format a log message with context and additional data.
        
        Args:
            message: The main log message
            **kwargs: Additional key-value pairs for this specific log
        
        Returns:
            Formatted message string with structured data
        """
        # Combine context and kwargs
        data = {**self.context, **kwargs}
        
        if data:
            # Add structured data as JSON
            structured_data = json.dumps(data, default=str)
            return f"{message} | {structured_data}"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a DEBUG level message.
        
        DEBUG level is for detailed diagnostic information, typically only
        interesting when diagnosing problems.
        
        Args:
            message: The log message
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> logger.debug("Variable value", var_name="x", var_value=42)
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.debug(formatted_message)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an INFO level message.
        
        INFO level is for general informational messages that highlight
        the progress of the application.
        
        Args:
            message: The log message
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> logger.info("Processing started", image_count=10)
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.info(formatted_message)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a WARNING level message.
        
        WARNING level indicates something unexpected happened, or a problem
        might occur in the near future (e.g., 'disk space low'). The software
        is still working as expected.
        
        Args:
            message: The log message
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> logger.warning("Low memory", available_mb=100)
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.warning(formatted_message)
    
    def error(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """
        Log an ERROR level message.
        
        ERROR level indicates a more serious problem that prevented the
        software from performing a function.
        
        Args:
            message: The log message
            exc_info: Whether to include exception information (stack trace)
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.error("Operation failed", operation="risky_operation")
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.error(formatted_message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """
        Log a CRITICAL level message.
        
        CRITICAL level indicates a very serious error that may prevent the
        program from continuing to run.
        
        Args:
            message: The log message
            exc_info: Whether to include exception information (stack trace)
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> logger.critical("System failure", component="database")
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.critical(formatted_message, exc_info=exc_info)
    
    def exception(self, message: str, **kwargs) -> None:
        """
        Log an exception with ERROR level.
        
        This is a convenience method that automatically includes exception
        information. Should be called from an exception handler.
        
        Args:
            message: The log message
            **kwargs: Additional structured data for this log entry
        
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception:
            ...     logger.exception("Unexpected error occurred")
        """
        formatted_message = self._format_message(message, **kwargs)
        self.logger.exception(formatted_message)
    
    def set_level(self, level: int) -> None:
        """
        Set the logging level.
        
        Args:
            level: Logging level (logging.DEBUG, logging.INFO, etc.)
        
        Example:
            >>> logger.set_level(logging.DEBUG)
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """
    Factory function to create a structured logger.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level
    
    Returns:
        A configured StructuredLogger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return StructuredLogger(name, level)
