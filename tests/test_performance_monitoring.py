"""
Tests for performance monitoring functionality.

This module tests the PerformanceMonitor class and its integration
with the Pipeline.
"""

import pytest
import numpy as np
import time
from src.screenshot2chat.monitoring.performance_monitor import (
    PerformanceMonitor,
    StepMetrics
)
from src.screenshot2chat.pipeline import Pipeline, PipelineStep, StepType


class MockDetector:
    """Mock detector for testing."""
    
    def __init__(self, config=None, delay=0.0):
        self.config = config or {}
        self.delay = delay
    
    def detect(self, image):
        """Mock detect method with optional delay."""
        if self.delay > 0:
            time.sleep(self.delay)
        return [{'bbox': [0, 0, 100, 100], 'score': 0.9}]


class MockExtractor:
    """Mock extractor for testing."""
    
    def __init__(self, config=None, delay=0.0):
        self.config = config or {}
        self.delay = delay
    
    def extract(self, detection_results, image=None):
        """Mock extract method with optional delay."""
        if self.delay > 0:
            time.sleep(self.delay)
        return {'data': {'extracted': True}, 'confidence': 0.95}


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.is_enabled() is True
        assert len(monitor.metrics) == 0
        assert len(monitor.active_timers) == 0
    
    def test_enable_disable(self):
        """Test enabling and disabling monitoring."""
        monitor = PerformanceMonitor()
        
        monitor.disable()
        assert monitor.is_enabled() is False
        
        monitor.enable()
        assert monitor.is_enabled() is True
    
    def test_start_stop_timer(self):
        """Test starting and stopping timers."""
        monitor = PerformanceMonitor()
        
        # Start timer
        monitor.start_timer("test_step")
        assert "test_step" in monitor.active_timers
        
        # Add small delay
        time.sleep(0.01)
        
        # Stop timer
        duration = monitor.stop_timer("test_step")
        assert duration > 0
        assert "test_step" not in monitor.active_timers
        assert "test_step" in monitor.metrics
        assert len(monitor.metrics["test_step"]) == 1
    
    def test_timer_with_metadata(self):
        """Test timer with metadata."""
        monitor = PerformanceMonitor()
        
        metadata = {'step_type': 'detector', 'enabled': True}
        monitor.start_timer("test_step", metadata=metadata)
        monitor.stop_timer("test_step")
        
        metrics = monitor.get_step_metrics("test_step")
        assert len(metrics) == 1
        assert metrics[0].metadata == metadata
    
    def test_multiple_executions(self):
        """Test recording multiple executions of the same step."""
        monitor = PerformanceMonitor()
        
        for i in range(3):
            monitor.start_timer("test_step")
            time.sleep(0.01)
            monitor.stop_timer("test_step")
        
        metrics = monitor.get_step_metrics("test_step")
        assert len(metrics) == 3
    
    def test_get_stats(self):
        """Test getting statistics for a step."""
        monitor = PerformanceMonitor()
        
        # Record multiple executions
        for i in range(5):
            monitor.start_timer("test_step")
            time.sleep(0.01)
            monitor.stop_timer("test_step")
        
        stats = monitor.get_stats("test_step")
        assert 'count' in stats
        assert stats['count'] == 5
        assert 'duration_mean' in stats
        assert 'duration_std' in stats
        assert 'duration_min' in stats
        assert 'duration_max' in stats
        assert stats['duration_mean'] > 0
    
    def test_get_all_stats(self):
        """Test getting statistics for all steps."""
        monitor = PerformanceMonitor()
        
        # Record executions for multiple steps
        for step_name in ['step1', 'step2', 'step3']:
            monitor.start_timer(step_name)
            time.sleep(0.01)
            monitor.stop_timer(step_name)
        
        all_stats = monitor.get_all_stats()
        assert len(all_stats) == 3
        assert 'step1' in all_stats
        assert 'step2' in all_stats
        assert 'step3' in all_stats
    
    def test_generate_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()
        
        # Record some executions
        monitor.start_timer("test_step")
        time.sleep(0.01)
        monitor.stop_timer("test_step")
        
        report = monitor.generate_report()
        assert "Performance Report" in report
        assert "test_step" in report
        assert "Duration:" in report
    
    def test_export_metrics(self):
        """Test exporting metrics."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test_step")
        time.sleep(0.01)
        monitor.stop_timer("test_step")
        
        exported = monitor.export_metrics()
        assert 'timestamp' in exported
        assert 'steps' in exported
        assert 'test_step' in exported['steps']
        assert 'executions' in exported['steps']['test_step']
        assert 'statistics' in exported['steps']['test_step']
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test_step")
        monitor.stop_timer("test_step")
        
        assert len(monitor.metrics) == 1
        
        monitor.clear()
        assert len(monitor.metrics) == 0
    
    def test_clear_step(self):
        """Test clearing metrics for a specific step."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("step1")
        monitor.stop_timer("step1")
        monitor.start_timer("step2")
        monitor.stop_timer("step2")
        
        assert len(monitor.metrics) == 2
        
        monitor.clear_step("step1")
        assert len(monitor.metrics) == 1
        assert "step1" not in monitor.metrics
        assert "step2" in monitor.metrics
    
    def test_timer_not_started_error(self):
        """Test error when stopping timer that wasn't started."""
        monitor = PerformanceMonitor()
        
        with pytest.raises(ValueError, match="Timer .* was not started"):
            monitor.stop_timer("nonexistent")
    
    def test_timer_already_running_error(self):
        """Test error when starting timer that's already running."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test_step")
        
        with pytest.raises(ValueError, match="Timer .* is already running"):
            monitor.start_timer("test_step")
        
        # Clean up
        monitor.stop_timer("test_step")
    
    def test_disabled_monitoring(self):
        """Test that disabled monitoring doesn't record metrics."""
        monitor = PerformanceMonitor()
        monitor.disable()
        
        monitor.start_timer("test_step")
        duration = monitor.stop_timer("test_step")
        
        assert duration == 0.0
        assert len(monitor.metrics) == 0


class TestPipelineMonitoring:
    """Test cases for Pipeline integration with PerformanceMonitor."""
    
    def test_pipeline_with_monitoring_enabled(self):
        """Test pipeline with monitoring enabled."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        assert pipeline.monitor.is_enabled() is True
    
    def test_pipeline_with_monitoring_disabled(self):
        """Test pipeline with monitoring disabled by default."""
        pipeline = Pipeline(name="test", enable_monitoring=False)
        assert pipeline.monitor.is_enabled() is False
    
    def test_pipeline_execution_records_metrics(self):
        """Test that pipeline execution records metrics."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        # Add a mock detector
        detector = MockDetector(delay=0.01)
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute pipeline
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = pipeline.execute(image)
        
        # Check metrics were recorded
        metrics = pipeline.get_performance_metrics()
        assert 'detector' in metrics
        assert metrics['detector']['duration'] > 0
    
    def test_pipeline_multiple_steps_metrics(self):
        """Test metrics for pipeline with multiple steps."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        # Add multiple steps
        detector = MockDetector(delay=0.01)
        extractor = MockExtractor(delay=0.01)
        
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        pipeline.add_step(PipelineStep(
            name="extractor",
            step_type=StepType.EXTRACTOR,
            component=extractor,
            config={'source': 'detector'}
        ))
        
        # Execute pipeline
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = pipeline.execute(image)
        
        # Check metrics for both steps
        metrics = pipeline.get_performance_metrics()
        assert 'detector' in metrics
        assert 'extractor' in metrics
        assert metrics['detector']['duration'] > 0
        assert metrics['extractor']['duration'] > 0
    
    def test_pipeline_get_performance_stats(self):
        """Test getting performance statistics from pipeline."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        detector = MockDetector(delay=0.01)
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute multiple times
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        for _ in range(3):
            pipeline.execute(image)
        
        # Get stats
        stats = pipeline.get_performance_stats()
        assert 'detector' in stats
        assert stats['detector']['count'] == 3
        assert 'duration_mean' in stats['detector']
    
    def test_pipeline_get_performance_report(self):
        """Test getting performance report from pipeline."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        detector = MockDetector(delay=0.01)
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pipeline.execute(image)
        
        # Get report
        report = pipeline.get_performance_report()
        assert "Performance Report" in report
        assert "detector" in report
    
    def test_pipeline_enable_disable_monitoring(self):
        """Test enabling and disabling monitoring on pipeline."""
        pipeline = Pipeline(name="test", enable_monitoring=False)
        
        detector = MockDetector()
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute with monitoring disabled
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pipeline.execute(image)
        
        stats = pipeline.get_performance_stats()
        assert len(stats) == 0
        
        # Enable monitoring
        pipeline.enable_monitoring()
        pipeline.execute(image)
        
        stats = pipeline.get_performance_stats()
        assert len(stats) > 0
    
    def test_pipeline_clear_metrics(self):
        """Test clearing metrics from pipeline."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        detector = MockDetector()
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute and verify metrics exist
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pipeline.execute(image)
        
        stats = pipeline.get_performance_stats()
        assert len(stats) > 0
        
        # Clear metrics
        pipeline.clear_metrics()
        stats = pipeline.get_performance_stats()
        assert len(stats) == 0
    
    def test_pipeline_export_metrics(self):
        """Test exporting metrics from pipeline."""
        pipeline = Pipeline(name="test", enable_monitoring=True)
        
        detector = MockDetector()
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector
        ))
        
        # Execute
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pipeline.execute(image)
        
        # Export metrics
        exported = pipeline.export_metrics()
        assert 'timestamp' in exported
        assert 'steps' in exported
        assert 'detector' in exported['steps']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
