# Pipeline Implementation Documentation

## Overview

The Pipeline system provides a flexible and configurable way to compose detection and extraction steps for screenshot analysis. It supports dependency management, execution order control, validation, and configuration persistence.

## Implementation Status

✅ **Task 5.1**: PipelineStep and Pipeline基础类 - COMPLETED
✅ **Task 5.2**: 流水线配置加载 - COMPLETED  
✅ **Task 5.4**: 流水线执行顺序控制 - COMPLETED
✅ **Task 5.6**: 流水线验证 - COMPLETED

## Core Components

### 1. StepType Enum

Defines the three types of pipeline steps:
- `DETECTOR`: Detection components (e.g., TextDetector, BubbleDetector)
- `EXTRACTOR`: Extraction components (e.g., NicknameExtractor, SpeakerExtractor)
- `PROCESSOR`: Processing components (future use)

### 2. PipelineStep Class

Represents a single step in the pipeline with the following attributes:

```python
@dataclass
class PipelineStep:
    name: str                      # Unique identifier
    step_type: StepType            # Type of step
    component: Any                 # The actual component instance
    config: Dict[str, Any]         # Configuration parameters
    enabled: bool = True           # Whether to execute this step
    depends_on: List[str] = []     # List of dependency step names
```

### 3. Pipeline Class

Main orchestrator that manages step execution:

**Key Methods:**
- `add_step(step)`: Add a step to the pipeline
- `execute(image)`: Execute all steps on an image
- `validate()`: Validate pipeline configuration
- `save(path)`: Save configuration to YAML/JSON
- `load(path)`: Load configuration from file
- `from_config(config)`: Create pipeline from config dict/file

## Features Implemented

### ✅ Dependency Management

The pipeline supports declaring dependencies between steps using the `depends_on` attribute:

```python
step = PipelineStep(
    name="nickname_extraction",
    step_type=StepType.EXTRACTOR,
    component=nickname_extractor,
    depends_on=["text_detection"]  # Depends on text_detection step
)
```

### ✅ Execution Order Control

Steps are executed in dependency order using topological sort:

1. Steps with no dependencies execute first
2. Steps execute only after their dependencies complete
3. Circular dependencies are detected and rejected

**Example:**
```
Steps added: [step_c, step_a, step_d, step_b]
Dependencies: step_c→[a,b], step_b→[a], step_d→[c]
Execution order: [step_a, step_b, step_c, step_d]
```

### ✅ Configuration Loading

Supports loading pipelines from YAML or JSON configuration files:

**YAML Example:**
```yaml
name: "chat_analysis_pipeline"
steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    config:
      backend: "paddleocr"
    enabled: true
  
  - name: "layout_extraction"
    type: "extractor"
    class: "LayoutExtractor"
    config:
      source: "text_detection"
    depends_on: ["text_detection"]
    enabled: true
```

**Usage:**
```python
pipeline = Pipeline.from_config("config.yaml")
```

### ✅ Comprehensive Validation

The `validate()` method checks:

1. **Non-empty pipeline**: At least one step exists
2. **Unique names**: No duplicate step names
3. **Valid dependencies**: All referenced dependencies exist
4. **No circular dependencies**: Detects circular dependency chains
5. **Valid components**: All components have required methods
6. **Method existence**: Detectors have `detect()`, extractors have `extract()`

**Error Messages:**
- Clear, descriptive error messages for each validation failure
- Includes context (step names, dependency chains, etc.)

### ✅ Configuration Persistence

Save and load pipeline configurations:

```python
# Save to YAML
pipeline.save("my_pipeline.yaml")

# Load from YAML
pipeline = Pipeline.load("my_pipeline.yaml")
```

### ✅ Dynamic Component Loading

Components are dynamically imported and instantiated based on class names:

```python
# Config specifies class name
{
    "class": "TextDetector",
    "type": "detector"
}

# Pipeline automatically imports and instantiates
from src.screenshot2chat.detectors.text_detector import TextDetector
component = TextDetector(config=step_config)
```

## Usage Examples

### Example 1: Basic Pipeline

```python
from src.screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.detectors.text_detector import TextDetector

# Create pipeline
pipeline = Pipeline(name="basic_pipeline")

# Add text detection step
text_detector = TextDetector()
step = PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=text_detector
)
pipeline.add_step(step)

# Execute
import numpy as np
image = np.zeros((100, 100, 3), dtype=np.uint8)
results = pipeline.execute(image)
```

### Example 2: Pipeline with Dependencies

```python
pipeline = Pipeline(name="complex_pipeline")

# Step 1: Text detection
text_detector = TextDetector()
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=text_detector
))

# Step 2: Layout extraction (depends on text detection)
layout_extractor = LayoutExtractor()
pipeline.add_step(PipelineStep(
    name="layout_extraction",
    step_type=StepType.EXTRACTOR,
    component=layout_extractor,
    config={'source': 'text_detection'},
    depends_on=["text_detection"]
))

# Validate and execute
pipeline.validate()
results = pipeline.execute(image)
```

### Example 3: Load from Configuration

```python
# Load pipeline from YAML file
pipeline = Pipeline.from_config("pipeline_config.yaml")

# Execute
results = pipeline.execute(image)
```

## Testing

### Basic Tests (test_pipeline_basic.py)

All basic tests pass:
- ✅ Pipeline creation
- ✅ Adding steps
- ✅ Pipeline execution
- ✅ Dependency ordering
- ✅ Validation
- ✅ Circular dependency detection
- ✅ Configuration save/load

### Example Tests (examples/pipeline_usage_example.py)

All examples run successfully:
- ✅ Basic pipeline creation
- ✅ Complex pipeline with dependencies
- ✅ Save and load configuration
- ✅ Validation error handling
- ✅ Execution order verification

## Architecture

### Data Flow

```
Input Image
    ↓
Pipeline.execute()
    ↓
Validate Pipeline
    ↓
Get Execution Order (Topological Sort)
    ↓
For each step in order:
    ↓
    Execute Component (detect/extract)
    ↓
    Store Results in Context
    ↓
Return All Results
```

### Context Management

The pipeline maintains a context dictionary during execution:

```python
context = {
    'image': np.ndarray,           # Original input image
    'results': {
        'step_name_1': [...],      # Results from step 1
        'step_name_2': {...},      # Results from step 2
        ...
    }
}
```

Steps can access results from previous steps via the context.

## Error Handling

### Validation Errors

```python
try:
    pipeline.validate()
except ValueError as e:
    # Handle validation error
    print(f"Validation failed: {e}")
```

**Common validation errors:**
- Missing dependencies
- Circular dependencies
- Duplicate step names
- Missing component methods

### Execution Errors

```python
try:
    results = pipeline.execute(image)
except RuntimeError as e:
    # Handle execution error
    print(f"Execution failed: {e}")
```

**Common execution errors:**
- Component method failures
- Missing required data
- Invalid image format

## Integration with Existing Components

The pipeline integrates seamlessly with existing detectors and extractors:

### Detectors
- ✅ TextDetector
- ✅ BubbleDetector

### Extractors
- ✅ NicknameExtractor
- ✅ SpeakerExtractor
- ✅ LayoutExtractor

All components follow the BaseDetector/BaseExtractor interface and work with the pipeline system.

## Configuration Format

### YAML Configuration

```yaml
name: "pipeline_name"
version: "1.0"  # Optional

steps:
  - name: "step_name"
    type: "detector|extractor|processor"
    class: "ComponentClassName"
    config:
      param1: value1
      param2: value2
    enabled: true  # Optional, default: true
    depends_on:    # Optional
      - "dependency_step_1"
      - "dependency_step_2"
```

### JSON Configuration

```json
{
  "name": "pipeline_name",
  "steps": [
    {
      "name": "step_name",
      "type": "detector",
      "class": "TextDetector",
      "config": {
        "param1": "value1"
      },
      "enabled": true,
      "depends_on": ["dependency_step"]
    }
  ]
}
```

## Performance Considerations

### Execution Order Optimization

The topological sort ensures:
- Minimal execution time (no unnecessary waiting)
- Correct dependency resolution
- Early detection of configuration errors

### Context Management

- Results are stored in memory during execution
- Large intermediate results may impact memory usage
- Consider clearing context between pipeline runs for batch processing

## Future Enhancements

The current implementation provides a solid foundation for future features:

### Planned Features (from tasks.md)
- ⏳ **Task 5.3**: Property tests for configuration round-trip
- ⏳ **Task 5.5**: Property tests for execution order
- ⏳ **Task 5.7**: Property tests for validation

### Potential Enhancements
- Parallel execution of independent steps
- Conditional branching based on intermediate results
- Performance monitoring and metrics collection
- Step result caching
- Streaming execution for large datasets
- Pipeline composition (nested pipelines)

## Requirements Validation

This implementation satisfies the following requirements:

### Requirement 8.1: Pipeline Configuration
✅ Supports YAML/JSON configuration files
✅ Defines execution order of detectors and extractors
✅ Provides pipeline validation functionality

### Requirement 8.2: Execution Order Control
✅ Supports `depends_on` dependency declarations
✅ Sorts steps by dependency relationships
✅ Executes steps in correct order

### Requirement 8.5: Pipeline Validation
✅ Checks step dependencies
✅ Checks configuration completeness
✅ Provides clear error messages

### Requirement 8.7: Configuration Persistence
✅ Supports saving pipeline configuration
✅ Supports loading pipeline configuration
✅ Supports pipeline reuse

## Conclusion

The Pipeline implementation provides a robust, flexible, and well-tested foundation for orchestrating screenshot analysis workflows. It successfully implements all core requirements and integrates seamlessly with existing components.

**Key Achievements:**
- ✅ Complete implementation of all non-optional subtasks
- ✅ Comprehensive validation and error handling
- ✅ Full configuration persistence support
- ✅ Dependency management with topological sorting
- ✅ Integration with existing detectors and extractors
- ✅ Extensive testing and examples

The system is ready for use and provides a solid foundation for future enhancements.
