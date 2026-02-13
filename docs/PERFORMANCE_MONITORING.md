# Performance Monitoring

The screenshot analysis library includes comprehensive performance monitoring capabilities to help you track and optimize pipeline execution.

## Overview

The `PerformanceMonitor` class tracks execution time, memory usage, and other metrics for each pipeline step. It's integrated into the `Pipeline` class and can be enabled or disabled as needed.

## Features

- **Execution Time Tracking**: Measure how long each step takes to execute
- **Memory Usage Tracking**: Monitor memory consumption before and after each step
- **Aggregated Statistics**: Get mean, std, min, max statistics across multiple executions
- **Detailed Reports**: Generate human-readable performance reports
- **Export Capabilities**: Export metrics in structured format for further analysis
- **Enable/Disable**: Turn monitoring on or off without changing code

## Basic Usage

### Enabling Monitoring

```python
from screenshot2chat import Pipeline

# Create pipeline with monitoring enabled
pipeline = Pipeline(name="my_pipeline", enable_monitoring=True)

# Or enable it later
pipeline.enable_monitoring()
```

### Executing and Getting Metrics

```python
import numpy as np

# Execute pipeline
image = np.random.randint(0, 255, (1080, 720, 3), dtype=np.uint8)
results = pipeline.execute(image)

# Get metrics from last execution
metrics = pipeline.get_performance_metrics()
print(f"Text detection took: {metrics['text_detection']['duration']:.3f}s")
```

### Getting Statistics

After multiple executions, you can get aggregated statistics:

```python
# Execute multiple times
for i in range(10):
    pipeline.execute(image)

# Get statistics
stats = pipeline.get_performance_stats()

for step_name, step_stats in stats.items():
    print(f"{step_name}:")
    print(f"  Average: {step_stats['duration_mean']:.3f}s")
    print(f"  Min: {step_stats['duration_min']:.3f}s")
    print(f"  Max: {step_stats['duration_max']:.3f}s")
```

### Generating Reports

```python
# Generate a human-readable report
report = pipeline.get_performance_report()
print(report)

# Generate detailed report with all executions
detailed_report = pipeline.get_performance_report(detailed=True)
print(detailed_report)
```

Example output:

```
======================================================================
Performance Report
======================================================================

Step: text_detection
----------------------------------------------------------------------
  Executions: 10
  Duration:
    Mean:  0.245s
    Std:   0.012s
    Min:   0.230s
    Max:   0.265s
    Total: 2.450s
  Memory Delta:
    Mean: +15.23 MB
    Std:  2.45 MB
    Min:  +12.50 MB
    Max:  +18.90 MB

Step: nickname_extraction
----------------------------------------------------------------------
  Executions: 10
  Duration:
    Mean:  0.045s
    Std:   0.003s
    Min:   0.042s
    Max:   0.050s
    Total: 0.450s
  Memory Delta:
    Mean: +2.15 MB
    Std:  0.35 MB
    Min:  +1.80 MB
    Max:  +2.50 MB

======================================================================
```

## Advanced Usage

### Exporting Metrics

Export metrics in structured format for further analysis:

```python
# Export all metrics
exported = pipeline.export_metrics()

# Save to JSON file
import json
with open('metrics.json', 'w') as f:
    json.dump(exported, f, indent=2)
```

The exported format includes:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "steps": {
    "text_detection": {
      "executions": [
        {
          "step_name": "text_detection",
          "start_time": 1705315845.123,
          "end_time": 1705315845.368,
          "duration": 0.245,
          "memory_before": 150.5,
          "memory_after": 165.73,
          "memory_delta": 15.23,
          "metadata": {
            "step_type": "detector",
            "enabled": true
          }
        }
      ],
      "statistics": {
        "count": 10,
        "duration_mean": 0.245,
        "duration_std": 0.012,
        "duration_min": 0.230,
        "duration_max": 0.265,
        "duration_total": 2.450,
        "memory_delta_mean": 15.23,
        "memory_delta_std": 2.45,
        "memory_delta_min": 12.50,
        "memory_delta_max": 18.90
      }
    }
  }
}
```

### Clearing Metrics

```python
# Clear all metrics
pipeline.clear_metrics()

# Or clear metrics for a specific step
pipeline.monitor.clear_step("text_detection")
```

### Disabling Monitoring

```python
# Disable monitoring (no overhead)
pipeline.disable_monitoring()

# Execute without recording metrics
results = pipeline.execute(image)

# Re-enable when needed
pipeline.enable_monitoring()
```

## Direct PerformanceMonitor Usage

You can also use the `PerformanceMonitor` class directly:

```python
from screenshot2chat import PerformanceMonitor

monitor = PerformanceMonitor()

# Start timing
monitor.start_timer("my_operation", metadata={'type': 'custom'})

# Do some work
result = expensive_operation()

# Stop timing
duration = monitor.stop_timer("my_operation")
print(f"Operation took: {duration:.3f}s")

# Get statistics
stats = monitor.get_stats("my_operation")
print(f"Average: {stats['duration_mean']:.3f}s")

# Generate report
report = monitor.generate_report()
print(report)
```

## Performance Overhead

When monitoring is **enabled**:
- Minimal overhead (~0.1-0.5ms per step)
- Memory tracking uses `psutil` which is efficient
- Suitable for production use

When monitoring is **disabled**:
- Zero overhead
- No metrics are recorded
- Timers return immediately

## Best Practices

1. **Enable monitoring during development** to identify bottlenecks
2. **Disable monitoring in production** if you don't need metrics
3. **Clear metrics periodically** if running long-term processes
4. **Export metrics** for offline analysis and visualization
5. **Use detailed reports** for debugging specific issues
6. **Monitor memory** to detect memory leaks

## Integration with Pipeline Configuration

You can enable monitoring via configuration:

```yaml
name: "monitored_pipeline"
monitoring:
  enabled: true

steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    # ... rest of config
```

## API Reference

### Pipeline Methods

- `enable_monitoring()`: Enable performance monitoring
- `disable_monitoring()`: Disable performance monitoring
- `get_performance_metrics()`: Get metrics from last execution
- `get_performance_stats()`: Get aggregated statistics
- `get_performance_report(detailed=False)`: Generate human-readable report
- `clear_metrics()`: Clear all recorded metrics
- `export_metrics()`: Export metrics in structured format

### PerformanceMonitor Methods

- `start_timer(name, metadata=None)`: Start timing a step
- `stop_timer(name)`: Stop timing and record metrics
- `get_step_metrics(name)`: Get all metrics for a step
- `get_stats(name)`: Get statistics for a step
- `get_all_stats()`: Get statistics for all steps
- `generate_report(detailed=False)`: Generate report
- `export_metrics()`: Export all metrics
- `clear()`: Clear all metrics
- `clear_step(name)`: Clear metrics for specific step
- `enable()`: Enable monitoring
- `disable()`: Disable monitoring
- `is_enabled()`: Check if monitoring is enabled

## Examples

See `examples/performance_monitoring_demo.py` for a complete working example.

## Requirements

The performance monitoring feature requires:
- `psutil`: For memory usage tracking
- `numpy`: For statistical calculations

These are included in the standard installation.
