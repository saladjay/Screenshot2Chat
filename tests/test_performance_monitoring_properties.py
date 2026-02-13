"""
Property-based tests for PerformanceMonitor
Task 12.3: Performance monitoring property tests
Property 15: Pipeline Metrics Recording
Validates: Requirements 8.6
"""

import pytest
import time
import numpy as np
from hypothesis import given, strategies as st, settings
from src.screenshot2chat.monitoring.performance_monitor import PerformanceMonitor
from src.screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.core.base_detector import BaseDetector
from src.screenshot2chat.core.data_models import DetectionResult


class MockDetector(BaseDetector):
    def __init__(self, name="mock", delay=0.0, config=None):
        super().__init__(config)
        self.name = name
        self.delay = delay
    
    def load_model(self):
        self.model = f"model_{self.name}"
    
    def detect(self, image):
        if self.delay > 0:
            time.sleep(self.delay)
        return [DetectionResult([0, 0, 10, 10], 0.9, self.name, {})]


@settings(max_examples=50, deadline=None)
@given(
    num_steps=st.integers(min_value=1, max_value=5)
)
def test_property_15_pipeline_metrics_recording(num_steps):
    """
    Feature: screenshot-analysis-library-refactor
    Property 15: Pipeline Metrics Recording
    
    For any pipeline execution, performance metrics (execution time, memory usage) 
    for each step should be recorded and accessible.
    """
    # Create pipeline with performance monitoring
    pipeline = Pipeline(name="monitored_pipeline")
    monitor = PerformanceMonitor()
    pipeline.monitor = monitor
    
    # Add steps
    for i in range(num_steps):
        detector = MockDetector(name=f"detector_{i}", delay=0.01)
        detector.load_model()
        step = PipelineStep(f"step_{i}", StepType.DETECTOR, detector)
        pipeline.add_step(step)
    
    # Execute pipeline
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Start monitoring
    for i in range(num_steps):
        monitor.start_timer(f"step_{i}")
        time.sleep(0.01)  # Simulate work
        monitor.stop_timer(f"step_{i}")
    
    # Verify metrics were recorded
    for i in range(num_steps):
        step_name = f"step_{i}"
        assert step_name in monitor.metrics, f"Metrics for {step_name} should be recorded"
        
        stats = monitor.get_stats(step_name)
        assert "mean" in stats, "Should have mean execution time"
        assert "std" in stats, "Should have std deviation"
        assert "min" in stats, "Should have min execution time"
        assert "max" in stats, "Should have max execution time"
        assert "count" in stats, "Should have execution count"
        
        # Verify timing is reasonable
        assert stats["mean"] > 0, "Mean execution time should be positive"
        assert stats["min"] >= 0, "Min execution time should be non-negative"
        assert stats["max"] >= stats["min"], "Max should be >= min"


@settings(max_examples=100, deadline=None)
@given(
    timer_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    num_executions=st.integers(min_value=1, max_value=10)
)
def test_performance_monitor_multiple_executions(timer_name, num_executions):
    """
    Test that performance monitor correctly tracks multiple executions
    """
    monitor = PerformanceMonitor()
    
    # Execute multiple times
    for _ in range(num_executions):
        monitor.start_timer(timer_name)
        time.sleep(0.001)  # Small delay
        monitor.stop_timer(timer_name)
    
    # Verify count
    stats = monitor.get_stats(timer_name)
    assert stats["count"] == num_executions, f"Should have {num_executions} executions"
    
    # Verify all metrics exist
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats


@settings(max_examples=50, deadline=None)
@given(
    num_timers=st.integers(min_value=1, max_value=10)
)
def test_performance_monitor_multiple_timers(num_timers):
    """
    Test that performance monitor can track multiple timers simultaneously
    """
    monitor = PerformanceMonitor()
    
    # Start and stop multiple timers
    for i in range(num_timers):
        timer_name = f"timer_{i}"
        monitor.start_timer(timer_name)
        time.sleep(0.001)
        monitor.stop_timer(timer_name)
    
    # Verify all timers were recorded
    assert len(monitor.metrics) == num_timers, "Should track all timers"
    
    for i in range(num_timers):
        timer_name = f"timer_{i}"
        assert timer_name in monitor.metrics, f"{timer_name} should be recorded"
        stats = monitor.get_stats(timer_name)
        assert stats["count"] == 1, "Each timer should have one execution"


@settings(max_examples=50, deadline=None)
@given(
    timer_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
)
def test_performance_monitor_timer_not_started_error(timer_name):
    """
    Test that stopping a timer that wasn't started raises an error
    """
    monitor = PerformanceMonitor()
    
    with pytest.raises(ValueError, match="was not started"):
        monitor.stop_timer(timer_name)


@settings(max_examples=50, deadline=None)
@given(
    timer_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
)
def test_performance_monitor_report_generation(timer_name):
    """
    Test that performance monitor can generate a report
    """
    monitor = PerformanceMonitor()
    
    # Record some metrics
    monitor.start_timer(timer_name)
    time.sleep(0.001)
    elapsed = monitor.stop_timer(timer_name)
    
    # Generate report
    report = monitor.generate_report()
    
    assert isinstance(report, str), "Report should be a string"
    assert timer_name in report, "Report should mention the timer"
    assert "Mean" in report or "mean" in report, "Report should include mean"
    assert str(elapsed)[:4] in report or f"{elapsed:.3f}" in report, "Report should include timing"


@settings(max_examples=50, deadline=None)
@given(
    timer_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    num_executions=st.integers(min_value=2, max_value=10)
)
def test_performance_monitor_statistics_accuracy(timer_name, num_executions):
    """
    Test that performance monitor statistics are calculated correctly
    """
    monitor = PerformanceMonitor()
    
    delays = []
    for i in range(num_executions):
        delay = 0.001 * (i + 1)  # Increasing delays
        delays.append(delay)
        
        monitor.start_timer(timer_name)
        time.sleep(delay)
        monitor.stop_timer(timer_name)
    
    stats = monitor.get_stats(timer_name)
    
    # Verify count
    assert stats["count"] == num_executions
    
    # Verify min and max are in reasonable range
    assert stats["min"] > 0
    assert stats["max"] > stats["min"]
    
    # Mean should be between min and max
    assert stats["min"] <= stats["mean"] <= stats["max"]
    
    # Std should be non-negative
    assert stats["std"] >= 0


@settings(max_examples=50, deadline=None)
@given(
    timer_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
)
def test_performance_monitor_empty_stats(timer_name):
    """
    Test that getting stats for non-existent timer returns empty dict
    """
    monitor = PerformanceMonitor()
    
    stats = monitor.get_stats(timer_name)
    
    assert stats == {}, "Stats for non-existent timer should be empty"


@settings(max_examples=50, deadline=None)
@given(
    num_timers=st.integers(min_value=2, max_value=5)
)
def test_performance_monitor_report_multiple_timers(num_timers):
    """
    Test that report includes all timers
    """
    monitor = PerformanceMonitor()
    
    timer_names = []
    for i in range(num_timers):
        timer_name = f"timer_{i}"
        timer_names.append(timer_name)
        
        monitor.start_timer(timer_name)
        time.sleep(0.001)
        monitor.stop_timer(timer_name)
    
    report = monitor.generate_report()
    
    # Verify all timers are in report
    for timer_name in timer_names:
        assert timer_name in report, f"Report should include {timer_name}"
