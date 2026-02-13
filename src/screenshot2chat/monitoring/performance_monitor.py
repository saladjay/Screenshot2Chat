"""
Performance monitoring for pipeline execution.

This module provides the PerformanceMonitor class for tracking execution time,
memory usage, and other performance metrics during pipeline execution.
"""

import time
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class StepMetrics:
    """
    Metrics for a single pipeline step execution.
    
    Attributes:
        step_name: Name of the pipeline step
        start_time: Timestamp when step started
        end_time: Timestamp when step ended
        duration: Execution duration in seconds
        memory_before: Memory usage before step (MB)
        memory_after: Memory usage after step (MB)
        memory_delta: Change in memory usage (MB)
        metadata: Additional metadata about the step
    """
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, end_time: float, memory_after: float) -> None:
        """
        Finalize metrics after step completion.
        
        Args:
            end_time: Timestamp when step ended
            memory_after: Memory usage after step (MB)
        """
        self.end_time = end_time
        self.duration = end_time - self.start_time
        self.memory_after = memory_after
        if self.memory_before is not None:
            self.memory_delta = memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'step_name': self.step_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_delta': self.memory_delta,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """
    Performance monitoring for pipeline execution.
    
    Tracks execution time, memory usage, and other metrics for each
    pipeline step and provides aggregated statistics.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics: Dict[str, List[StepMetrics]] = {}
        self.active_timers: Dict[str, StepMetrics] = {}
        self.process = psutil.Process(os.getpid())
        self._enabled = True
    
    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._enabled
    
    def start_timer(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start timing a pipeline step.
        
        Args:
            name: Name of the step to time
            metadata: Optional metadata about the step
            
        Raises:
            ValueError: If timer with this name is already running
        """
        if not self._enabled:
            return
        
        if name in self.active_timers:
            raise ValueError(f"Timer '{name}' is already running")
        
        # Get current memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Create metrics object
        metrics = StepMetrics(
            step_name=name,
            start_time=time.time(),
            memory_before=memory_mb,
            metadata=metadata or {}
        )
        
        self.active_timers[name] = metrics
    
    def stop_timer(self, name: str) -> float:
        """
        Stop timing a pipeline step and record metrics.
        
        Args:
            name: Name of the step to stop timing
            
        Returns:
            Elapsed time in seconds
            
        Raises:
            ValueError: If timer was not started
        """
        if not self._enabled:
            return 0.0
        
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")
        
        # Get current memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Finalize metrics
        metrics = self.active_timers[name]
        metrics.finalize(time.time(), memory_mb)
        
        # Store metrics
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metrics)
        
        # Remove from active timers
        del self.active_timers[name]
        
        return metrics.duration
    
    def get_current_memory(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Current memory usage in megabytes
        """
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 * 1024)
    
    def get_step_metrics(self, name: str) -> List[StepMetrics]:
        """
        Get all recorded metrics for a specific step.
        
        Args:
            name: Name of the step
            
        Returns:
            List of metrics for the step
        """
        return self.metrics.get(name, [])
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get aggregated statistics for a specific step.
        
        Args:
            name: Name of the step
            
        Returns:
            Dictionary containing mean, std, min, max, and count
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        durations = [m.duration for m in self.metrics[name] if m.duration is not None]
        memory_deltas = [m.memory_delta for m in self.metrics[name] if m.memory_delta is not None]
        
        stats = {
            'count': len(self.metrics[name])
        }
        
        if durations:
            stats.update({
                'duration_mean': float(np.mean(durations)),
                'duration_std': float(np.std(durations)),
                'duration_min': float(np.min(durations)),
                'duration_max': float(np.max(durations)),
                'duration_total': float(np.sum(durations))
            })
        
        if memory_deltas:
            stats.update({
                'memory_delta_mean': float(np.mean(memory_deltas)),
                'memory_delta_std': float(np.std(memory_deltas)),
                'memory_delta_min': float(np.min(memory_deltas)),
                'memory_delta_max': float(np.max(memory_deltas))
            })
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated statistics for all steps.
        
        Returns:
            Dictionary mapping step names to their statistics
        """
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def generate_report(self, detailed: bool = False) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            detailed: If True, include detailed metrics for each execution
            
        Returns:
            Formatted performance report string
        """
        report = []
        report.append("=" * 70)
        report.append("Performance Report")
        report.append("=" * 70)
        report.append("")
        
        if not self.metrics:
            report.append("No metrics recorded.")
            return "\n".join(report)
        
        # Summary statistics
        all_stats = self.get_all_stats()
        
        for step_name in sorted(self.metrics.keys()):
            stats = all_stats[step_name]
            report.append(f"Step: {step_name}")
            report.append("-" * 70)
            
            if 'count' in stats:
                report.append(f"  Executions: {stats['count']}")
            
            if 'duration_mean' in stats:
                report.append(f"  Duration:")
                report.append(f"    Mean:  {stats['duration_mean']:.3f}s")
                report.append(f"    Std:   {stats['duration_std']:.3f}s")
                report.append(f"    Min:   {stats['duration_min']:.3f}s")
                report.append(f"    Max:   {stats['duration_max']:.3f}s")
                report.append(f"    Total: {stats['duration_total']:.3f}s")
            
            if 'memory_delta_mean' in stats:
                report.append(f"  Memory Delta:")
                report.append(f"    Mean: {stats['memory_delta_mean']:+.2f} MB")
                report.append(f"    Std:  {stats['memory_delta_std']:.2f} MB")
                report.append(f"    Min:  {stats['memory_delta_min']:+.2f} MB")
                report.append(f"    Max:  {stats['memory_delta_max']:+.2f} MB")
            
            report.append("")
        
        # Detailed metrics if requested
        if detailed:
            report.append("=" * 70)
            report.append("Detailed Metrics")
            report.append("=" * 70)
            report.append("")
            
            for step_name in sorted(self.metrics.keys()):
                report.append(f"Step: {step_name}")
                report.append("-" * 70)
                
                for i, metrics in enumerate(self.metrics[step_name], 1):
                    report.append(f"  Execution {i}:")
                    report.append(f"    Duration: {metrics.duration:.3f}s")
                    if metrics.memory_delta is not None:
                        report.append(f"    Memory Delta: {metrics.memory_delta:+.2f} MB")
                    if metrics.metadata:
                        report.append(f"    Metadata: {metrics.metadata}")
                
                report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics in a structured format.
        
        Returns:
            Dictionary containing all metrics data
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        for step_name, metrics_list in self.metrics.items():
            export_data['steps'][step_name] = {
                'executions': [m.to_dict() for m in metrics_list],
                'statistics': self.get_stats(step_name)
            }
        
        return export_data
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        self.active_timers.clear()
    
    def clear_step(self, name: str) -> None:
        """
        Clear metrics for a specific step.
        
        Args:
            name: Name of the step to clear
        """
        if name in self.metrics:
            del self.metrics[name]
        if name in self.active_timers:
            del self.active_timers[name]
    
    def __repr__(self) -> str:
        """String representation of the monitor."""
        step_count = len(self.metrics)
        total_executions = sum(len(metrics) for metrics in self.metrics.values())
        return f"PerformanceMonitor(steps={step_count}, executions={total_executions})"
