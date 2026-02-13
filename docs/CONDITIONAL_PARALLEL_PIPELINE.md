# Conditional Branching and Parallel Execution in Pipeline

This document describes the conditional branching and parallel execution features added to the Pipeline system.

## Overview

The Pipeline system now supports two advanced features:

1. **Conditional Branching**: Execute steps only when certain conditions are met based on intermediate results
2. **Parallel Execution**: Execute independent steps concurrently to improve performance

These features can be used independently or combined for complex processing workflows.

## Conditional Branching

### Basic Concept

Conditional branching allows you to specify conditions that must be met for a step to execute. This enables dynamic pipeline behavior based on intermediate results.

### Condition Syntax

Conditions are specified as string expressions that are evaluated at runtime. The following syntax is supported:

#### Length Comparisons
```python
'len(result.step_name) > 5'
'len(result.step_name) <= 10'
'len(result.step_name) == 0'
```

#### Field Comparisons
```python
'result.step_name.field == value'
'result.step_name.field > 100'
'result.step_name.field != "error"'
```

#### Null Checks
```python
'result.step_name is not None'
'result.step_name is None'
```

### Example: Conditional Branching

```python
from screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors.text_detector import TextDetector
from screenshot2chat.extractors.nickname_extractor import NicknameExtractor

# Create pipeline
pipeline = Pipeline(name="conditional_example")

# Step 1: Text detection
text_step = PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=TextDetector(),
    config={}
)
pipeline.add_step(text_step)

# Step 2: Extract nicknames only if many text boxes found
nickname_step = PipelineStep(
    name="nickname_extraction",
    step_type=StepType.EXTRACTOR,
    component=NicknameExtractor(),
    config={'source': 'text_detection'},
    depends_on=['text_detection'],
    condition='len(result.text_detection) > 5'  # Only run if > 5 text boxes
)
pipeline.add_step(nickname_step)

# Execute
results = pipeline.execute(image)
```

### YAML Configuration

Conditions can also be specified in YAML configuration files:

```yaml
name: "conditional_pipeline"
steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    enabled: true
  
  - name: "nickname_extraction"
    type: "extractor"
    class: "NicknameExtractor"
    depends_on: ["text_detection"]
    condition: "len(result.text_detection) > 5"
    config:
      source: "text_detection"
```

## Parallel Execution

### Basic Concept

Parallel execution allows multiple independent steps to run concurrently, improving overall pipeline performance. Steps are grouped using a `parallel_group` identifier.

### Configuration

Parallel execution is configured at the pipeline level:

```python
pipeline = Pipeline(
    name="parallel_example",
    parallel_executor="thread",  # or "process"
    max_workers=4
)
```

- `parallel_executor`: Choose between "thread" (ThreadPoolExecutor) or "process" (ProcessPoolExecutor)
- `max_workers`: Maximum number of concurrent workers

### Example: Parallel Execution

```python
from screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors.text_detector import TextDetector
from screenshot2chat.detectors.bubble_detector import BubbleDetector

# Create pipeline with parallel execution
pipeline = Pipeline(
    name="parallel_example",
    parallel_executor="thread",
    max_workers=3
)

# Add multiple detectors that can run in parallel
text_step = PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=TextDetector(),
    parallel_group="detection_group"  # Same group = parallel execution
)
pipeline.add_step(text_step)

bubble_step = PipelineStep(
    name="bubble_detection",
    step_type=StepType.DETECTOR,
    component=BubbleDetector(),
    parallel_group="detection_group"  # Same group = parallel execution
)
pipeline.add_step(bubble_step)

# Execute - both detectors run concurrently
results = pipeline.execute(image)
```

### YAML Configuration

```yaml
name: "parallel_pipeline"
parallel_executor: "thread"
max_workers: 4

steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    parallel_group: "detection_group"
  
  - name: "bubble_detection"
    type: "detector"
    class: "BubbleDetector"
    parallel_group: "detection_group"
```

## Combined Usage

Conditional branching and parallel execution can be combined for sophisticated workflows:

```python
pipeline = Pipeline(
    name="combined_example",
    parallel_executor="thread",
    max_workers=2
)

# Step 1: Initial detection
initial_step = PipelineStep(
    name="initial_detection",
    step_type=StepType.DETECTOR,
    component=TextDetector()
)
pipeline.add_step(initial_step)

# Steps 2-3: Parallel processing, only if initial detection found results
parallel_step1 = PipelineStep(
    name="parallel_processor_1",
    step_type=StepType.EXTRACTOR,
    component=Extractor1(),
    depends_on=['initial_detection'],
    condition='len(result.initial_detection) > 0',
    parallel_group="conditional_parallel"
)
pipeline.add_step(parallel_step1)

parallel_step2 = PipelineStep(
    name="parallel_processor_2",
    step_type=StepType.EXTRACTOR,
    component=Extractor2(),
    depends_on=['initial_detection'],
    condition='len(result.initial_detection) > 0',
    parallel_group="conditional_parallel"
)
pipeline.add_step(parallel_step2)

# Execute
results = pipeline.execute(image)
```

### YAML Configuration

```yaml
name: "combined_pipeline"
parallel_executor: "thread"
max_workers: 2

steps:
  - name: "initial_detection"
    type: "detector"
    class: "TextDetector"
  
  - name: "parallel_processor_1"
    type: "extractor"
    class: "Extractor1"
    depends_on: ["initial_detection"]
    condition: "len(result.initial_detection) > 0"
    parallel_group: "conditional_parallel"
  
  - name: "parallel_processor_2"
    type: "extractor"
    class: "Extractor2"
    depends_on: ["initial_detection"]
    condition: "len(result.initial_detection) > 0"
    parallel_group: "conditional_parallel"
```

## Performance Considerations

### Thread vs Process Executor

- **ThreadPoolExecutor** (`parallel_executor="thread"`):
  - Best for I/O-bound tasks (API calls, file operations)
  - Lower overhead
  - Shares memory between threads
  - Limited by Python's GIL for CPU-bound tasks

- **ProcessPoolExecutor** (`parallel_executor="process"`):
  - Best for CPU-bound tasks (heavy computation)
  - Higher overhead (process creation, IPC)
  - True parallelism (not limited by GIL)
  - Each process has separate memory

### Best Practices

1. **Use parallel execution for independent steps**: Steps in the same parallel group should not depend on each other's results

2. **Choose appropriate max_workers**: 
   - For I/O-bound: Can use more workers (e.g., 10-20)
   - For CPU-bound: Use number of CPU cores (e.g., 4-8)

3. **Combine with conditions wisely**: Conditional checks happen before parallel execution, so conditions are evaluated sequentially

4. **Monitor performance**: Use `enable_monitoring=True` to track execution times and identify bottlenecks

## Monitoring

Performance metrics include information about parallel execution:

```python
pipeline = Pipeline(name="example", enable_monitoring=True)
# ... add steps and execute ...

metrics = pipeline.get_performance_metrics()
for step_name, metric in metrics.items():
    print(f"{step_name}:")
    print(f"  Duration: {metric['duration']:.3f}s")
    print(f"  Parallel: {metric.get('parallel', False)}")
```

## Error Handling

### Condition Evaluation Errors

If a condition expression is invalid or references non-existent results, a `RuntimeError` is raised:

```python
try:
    results = pipeline.execute(image)
except RuntimeError as e:
    print(f"Condition evaluation failed: {e}")
```

### Parallel Execution Errors

If any step in a parallel group fails, the entire group fails and a `RuntimeError` is raised with details about which step failed.

## Limitations

1. **Condition Complexity**: Conditions support basic comparisons but not complex boolean logic (AND/OR)
2. **Parallel Dependencies**: Steps in a parallel group cannot depend on each other
3. **Serialization**: When using ProcessPoolExecutor, components must be picklable
4. **Shared State**: Parallel steps cannot safely modify shared state

## Future Enhancements

Potential future improvements:

- Support for complex boolean conditions (AND, OR, NOT)
- Conditional parallel groups (entire group executes based on condition)
- Dynamic worker pool sizing
- Async/await support for I/O-bound operations
- Better error recovery in parallel execution

## See Also

- [Pipeline Documentation](PIPELINE_IMPLEMENTATION.md)
- [Performance Monitoring](PERFORMANCE_MONITORING.md)
- [Configuration Management](CONFIG_MANAGER.md)
