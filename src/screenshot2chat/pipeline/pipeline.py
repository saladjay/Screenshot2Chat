"""
Pipeline orchestration system for screenshot analysis.

This module provides the core pipeline functionality for composing
and executing detection and extraction steps in a configurable order.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import yaml
import json
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from ..monitoring.performance_monitor import PerformanceMonitor
from ..core.exceptions import PipelineError, ValidationError, DetectionError, ExtractionError
from ..logging import StructuredLogger


class StepType(Enum):
    """Pipeline step types."""
    DETECTOR = "detector"
    EXTRACTOR = "extractor"
    PROCESSOR = "processor"


@dataclass
class PipelineStep:
    """
    Represents a single step in the processing pipeline.
    
    Attributes:
        name: Unique identifier for this step
        step_type: Type of step (detector, extractor, or processor)
        component: The actual detector/extractor/processor instance
        config: Configuration parameters for this step
        enabled: Whether this step should be executed
        depends_on: List of step names this step depends on
        condition: Optional condition expression for conditional execution
        parallel_group: Optional group ID for parallel execution
    """
    name: str
    step_type: StepType
    component: Any
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    parallel_group: Optional[str] = None
    
    def __post_init__(self):
        """Validate step configuration."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not isinstance(self.step_type, StepType):
            raise ValueError(f"Invalid step_type: {self.step_type}")


class Pipeline:
    """
    Processing pipeline for screenshot analysis.
    
    The pipeline manages the execution of multiple detection and extraction
    steps in a configurable order, handling data flow between steps.
    """
    
    def __init__(self, name: str = "default", enable_monitoring: bool = False,
                 parallel_executor: str = "thread", max_workers: int = 4):
        """
        Initialize a new pipeline.
        
        Args:
            name: Name identifier for this pipeline
            enable_monitoring: Whether to enable performance monitoring
            parallel_executor: Type of executor for parallel execution ("thread" or "process")
            max_workers: Maximum number of workers for parallel execution
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.context: Dict[str, Any] = {}
        self._step_map: Dict[str, PipelineStep] = {}
        self.monitor = PerformanceMonitor()
        if enable_monitoring:
            self.monitor.enable()
        else:
            self.monitor.disable()
        
        # Parallel execution configuration
        self.parallel_executor = parallel_executor
        self.max_workers = max_workers
        
        # Initialize logger
        self.logger = StructuredLogger(__name__)
        self.logger.set_context(pipeline_name=name)
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Add a processing step to the pipeline.
        
        Args:
            step: The pipeline step to add
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If a step with the same name already exists
        """
        if step.name in self._step_map:
            raise ValueError(f"Step with name '{step.name}' already exists")
        
        self.steps.append(step)
        self._step_map[step.name] = step
        return self
    
    def _evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a condition expression based on current context.
        
        Supported expressions:
        - result.step_name.field == value
        - result.step_name.field > value
        - result.step_name.field < value
        - result.step_name.field >= value
        - result.step_name.field <= value
        - result.step_name.field != value
        - len(result.step_name) > value
        - result.step_name is not None
        - result.step_name is None
        
        Args:
            condition: Condition expression string
            
        Returns:
            Boolean result of the condition evaluation
            
        Raises:
            ValueError: If condition syntax is invalid
        """
        if not condition:
            return True
        
        # Replace 'result.' with 'self.context["results"].'
        # This allows accessing results from previous steps
        condition_eval = condition.strip()
        
        # Handle 'is None' and 'is not None' checks
        if ' is not None' in condition_eval:
            parts = condition_eval.split(' is not None')[0].strip()
            if parts.startswith('result.'):
                step_name = parts.replace('result.', '').split('.')[0]
                if step_name in self.context['results']:
                    result = self.context['results'][step_name]
                    return result is not None
                return False
            return False
        
        if ' is None' in condition_eval:
            parts = condition_eval.split(' is None')[0].strip()
            if parts.startswith('result.'):
                step_name = parts.replace('result.', '').split('.')[0]
                if step_name in self.context['results']:
                    result = self.context['results'][step_name]
                    return result is None
                return True
            return True
        
        # Handle len() function
        if condition_eval.startswith('len('):
            match = re.match(r'len\(result\.(\w+)\)\s*([><=!]+)\s*(\d+)', condition_eval)
            if match:
                step_name, operator, value = match.groups()
                if step_name in self.context['results']:
                    result = self.context['results'][step_name]
                    result_len = len(result) if hasattr(result, '__len__') else 0
                    value_int = int(value)
                    
                    if operator == '>':
                        return result_len > value_int
                    elif operator == '<':
                        return result_len < value_int
                    elif operator == '>=':
                        return result_len >= value_int
                    elif operator == '<=':
                        return result_len <= value_int
                    elif operator == '==':
                        return result_len == value_int
                    elif operator == '!=':
                        return result_len != value_int
                return False
        
        # Handle field comparisons: result.step_name.field op value
        match = re.match(r'result\.(\w+)\.(\w+)\s*([><=!]+)\s*(.+)', condition_eval)
        if match:
            step_name, field, operator, value = match.groups()
            
            if step_name not in self.context['results']:
                return False
            
            result = self.context['results'][step_name]
            
            # Get field value
            if isinstance(result, dict):
                field_value = result.get(field)
            elif hasattr(result, field):
                field_value = getattr(result, field)
            elif hasattr(result, 'data') and isinstance(result.data, dict):
                field_value = result.data.get(field)
            else:
                return False
            
            if field_value is None:
                return False
            
            # Parse value (handle strings, numbers, booleans)
            value = value.strip()
            if value.startswith('"') or value.startswith("'"):
                value = value.strip('"\'')
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            else:
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
            
            # Perform comparison
            if operator == '==':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            elif operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
        
        raise ValueError(f"Invalid condition syntax: {condition}")
    
    
    def _execute_step(self, step: PipelineStep, image: np.ndarray) -> tuple:
        """
        Execute a single pipeline step.
        
        Args:
            step: The step to execute
            image: Input image
            
        Returns:
            Tuple of (step_name, results, duration)
            
        Raises:
            PipelineError: If step execution fails
        """
        try:
            self.logger.info(
                "Executing step",
                step_name=step.name,
                step_type=step.step_type.value
            )
            
            if step.step_type == StepType.DETECTOR:
                results = step.component.detect(image)
            
            elif step.step_type == StepType.EXTRACTOR:
                detection_results = self._get_detection_results(step)
                results = step.component.extract(detection_results, image)
            
            elif step.step_type == StepType.PROCESSOR:
                input_data = self._get_processor_input(step)
                results = step.component.process(input_data, image)
            
            self.logger.info(
                "Step completed successfully",
                step_name=step.name
            )
            
            return (step.name, results, None)
            
        except DetectionError as e:
            self.logger.error(
                "Detection step failed",
                step_name=step.name,
                error_type="DetectionError"
            )
            raise PipelineError(
                f"Detection step '{step.name}' failed: {str(e)}. "
                f"Recovery suggestion: Check detector configuration and input data."
            ) from e
        except ExtractionError as e:
            self.logger.error(
                "Extraction step failed",
                step_name=step.name,
                error_type="ExtractionError"
            )
            raise PipelineError(
                f"Extraction step '{step.name}' failed: {str(e)}. "
                f"Recovery suggestion: Verify detection results are available."
            ) from e
        except Exception as e:
            self.logger.error(
                "Step execution failed",
                step_name=step.name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise PipelineError(
                f"Error executing step '{step.name}': {str(e)}. "
                f"Recovery suggestion: Check step configuration and dependencies."
            ) from e
    
    def _execute_parallel_group(self, steps: List[PipelineStep], image: np.ndarray) -> Dict[str, Any]:
        """
        Execute a group of steps in parallel.
        
        Args:
            steps: List of steps to execute in parallel
            image: Input image
            
        Returns:
            Dictionary mapping step names to their results
            
        Raises:
            RuntimeError: If any step execution fails
        """
        results = {}
        
        # Choose executor type
        if self.parallel_executor == "process":
            ExecutorClass = ProcessPoolExecutor
        else:
            ExecutorClass = ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all steps
            future_to_step = {}
            for step in steps:
                # Start monitoring
                step_metadata = {
                    'step_type': step.step_type.value,
                    'enabled': step.enabled,
                    'depends_on': step.depends_on,
                    'condition': step.condition,
                    'parallel_group': step.parallel_group
                }
                self.monitor.start_timer(step.name, metadata=step_metadata)
                
                future = executor.submit(self._execute_step, step, image)
                future_to_step[future] = step
            
            # Collect results as they complete
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    step_name, step_results, _ = future.result()
                    
                    # Stop monitoring and record duration
                    duration = self.monitor.stop_timer(step_name)
                    
                    results[step_name] = step_results
                    self.context['results'][step_name] = step_results
                    self.context['metrics'][step_name] = {
                        'duration': duration,
                        'step_type': step.step_type.value,
                        'parallel': True
                    }
                    
                except Exception as e:
                    # Stop timer if it's still running
                    if step.name in self.monitor.active_timers:
                        self.monitor.stop_timer(step.name)
                    raise RuntimeError(
                        f"Error in parallel execution of step '{step.name}': {str(e)}"
                    ) from e
        
        return results
    
    def execute(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute the pipeline on an input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing results from all executed steps
            
        Raises:
            ValidationError: If pipeline validation fails
            PipelineError: If step execution fails
        """
        try:
            self.logger.info(
                "Starting pipeline execution",
                num_steps=len(self.steps),
                enabled_steps=sum(1 for s in self.steps if s.enabled)
            )
            
            # Validate pipeline before execution
            if not self.validate():
                raise ValidationError(
                    "Pipeline validation failed. "
                    "Recovery suggestion: Check step dependencies and configuration."
                )
            
            # Initialize context
            self.context = {
                'image': image,
                'results': {},
                'metrics': {}
            }
            
            # Get execution order
            execution_order = self._get_execution_order()
            
            # Execute steps, handling parallel groups
            processed_groups = set()
            
            for step in execution_order:
                if not step.enabled:
                    self.logger.debug("Skipping disabled step", step_name=step.name)
                    continue
                
                # Evaluate condition if present
                if step.condition:
                    try:
                        condition_met = self._evaluate_condition(step.condition)
                        if not condition_met:
                            self.logger.info(
                                "Skipping step due to condition",
                                step_name=step.name,
                                condition=step.condition
                            )
                            continue
                    except Exception as e:
                        self.logger.error(
                            "Condition evaluation failed",
                            step_name=step.name,
                            condition=step.condition
                        )
                        raise PipelineError(
                            f"Error evaluating condition for step '{step.name}': {str(e)}. "
                            f"Recovery suggestion: Check condition syntax."
                        ) from e
                
                group_id = step.parallel_group
                
                # If this step is part of a parallel group
                if group_id is not None:
                    # Skip if we've already processed this group
                    if group_id in processed_groups:
                        continue
                    
                    # Collect all steps in this parallel group that should execute
                    group_steps = []
                    for s in execution_order:
                        if not s.enabled:
                            continue
                        if s.parallel_group != group_id:
                            continue
                        
                        # Check condition for this step
                        if s.condition:
                            try:
                                if not self._evaluate_condition(s.condition):
                                    continue
                            except Exception:
                                continue
                        
                        group_steps.append(s)
                    
                    # Execute all steps in this parallel group
                    if group_steps:
                        self._execute_parallel_group(group_steps, image)
                    processed_groups.add(group_id)
                
                else:
                    # Execute step sequentially
                    # Start monitoring
                    step_metadata = {
                        'step_type': step.step_type.value,
                        'enabled': step.enabled,
                        'depends_on': step.depends_on,
                        'condition': step.condition
                    }
                    self.monitor.start_timer(step.name, metadata=step_metadata)
                    
                    try:
                        if step.step_type == StepType.DETECTOR:
                            results = step.component.detect(image)
                            self.context['results'][step.name] = results
                        
                        elif step.step_type == StepType.EXTRACTOR:
                            detection_results = self._get_detection_results(step)
                            results = step.component.extract(detection_results, image)
                            self.context['results'][step.name] = results
                        
                        elif step.step_type == StepType.PROCESSOR:
                            # Get input data based on dependencies
                            input_data = self._get_processor_input(step)
                            results = step.component.process(input_data, image)
                            self.context['results'][step.name] = results
                        
                        # Stop monitoring and record duration
                        duration = self.monitor.stop_timer(step.name)
                        self.context['metrics'][step.name] = {
                            'duration': duration,
                            'step_type': step.step_type.value,
                            'parallel': False
                        }
                            
                    except Exception as e:
                        # Stop timer if it's still running
                        if step.name in self.monitor.active_timers:
                            self.monitor.stop_timer(step.name)
                        
                        self.logger.error(
                            "Step execution failed",
                            step_name=step.name,
                            error_type=type(e).__name__
                        )
                        raise PipelineError(
                            f"Error executing step '{step.name}': {str(e)}. "
                            f"Recovery suggestion: Check step configuration and dependencies."
                        ) from e
            
            self.logger.info(
                "Pipeline execution completed",
                num_results=len(self.context['results'])
            )
            
            return self.context['results']
        
        except (ValidationError, PipelineError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "Pipeline execution failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise PipelineError(
                f"Pipeline execution failed: {e}. "
                f"Recovery suggestion: Check pipeline configuration and input data."
            ) from e
    
    def _get_execution_order(self) -> List[PipelineStep]:
        """
        Determine the execution order based on dependencies.
        
        Returns:
            List of steps in execution order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # If no dependencies, return steps in original order
        if not any(step.depends_on for step in self.steps):
            return self.steps
        
        # Topological sort
        visited = set()
        temp_mark = set()
        ordered = []
        
        def visit(step: PipelineStep):
            if step.name in temp_mark:
                raise ValueError(
                    f"Circular dependency detected involving step '{step.name}'"
                )
            if step.name in visited:
                return
            
            temp_mark.add(step.name)
            
            # Visit dependencies first
            for dep_name in step.depends_on:
                if dep_name not in self._step_map:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep_name}'"
                    )
                visit(self._step_map[dep_name])
            
            temp_mark.remove(step.name)
            visited.add(step.name)
            ordered.append(step)
        
        for step in self.steps:
            if step.name not in visited:
                visit(step)
        
        return ordered
    
    def _get_detection_results(self, step: PipelineStep) -> List[Any]:
        """
        Get detection results for an extractor step.
        
        Args:
            step: The extractor step
            
        Returns:
            List of detection results from the source step
        """
        source = step.config.get('source')
        
        if source:
            if source not in self.context['results']:
                raise ValueError(
                    f"Step '{step.name}' requires results from '{source}' "
                    f"which has not been executed"
                )
            return self.context['results'][source]
        
        # If no source specified, try to use the first dependency
        if step.depends_on:
            source = step.depends_on[0]
            if source in self.context['results']:
                return self.context['results'][source]
        
        # Return empty list if no source found
        return []
    
    def _get_processor_input(self, step: PipelineStep) -> Dict[str, Any]:
        """
        Get input data for a processor step.
        
        Args:
            step: The processor step
            
        Returns:
            Dictionary of input data from dependent steps
        """
        input_data = {}
        for dep_name in step.depends_on:
            if dep_name in self.context['results']:
                input_data[dep_name] = self.context['results'][dep_name]
        return input_data
    
    def validate(self) -> bool:
        """
        Validate the pipeline configuration.
        
        Checks:
        - All step dependencies exist
        - No circular dependencies
        - All steps have valid components
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        if not self.steps:
            raise ValueError("Pipeline has no steps")
        
        # Check for duplicate names
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate step names found: {duplicates}")
        
        # Check dependencies exist
        for step in self.steps:
            for dep_name in step.depends_on:
                if dep_name not in self._step_map:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep_name}'"
                    )
        
        # Check for circular dependencies by attempting to get execution order
        try:
            self._get_execution_order()
        except ValueError as e:
            raise ValueError(f"Pipeline validation failed: {str(e)}") from e
        
        # Check all steps have valid components
        for step in self.steps:
            if step.component is None:
                raise ValueError(f"Step '{step.name}' has no component")
            
            # Check component has required methods
            if step.step_type == StepType.DETECTOR:
                if not hasattr(step.component, 'detect'):
                    raise ValueError(
                        f"Detector step '{step.name}' component missing 'detect' method"
                    )
            elif step.step_type == StepType.EXTRACTOR:
                if not hasattr(step.component, 'extract'):
                    raise ValueError(
                        f"Extractor step '{step.name}' component missing 'extract' method"
                    )
            elif step.step_type == StepType.PROCESSOR:
                if not hasattr(step.component, 'process'):
                    raise ValueError(
                        f"Processor step '{step.name}' component missing 'process' method"
                    )
        
        return True
    
    @classmethod
    def from_config(cls, config: Union[str, Dict[str, Any]]) -> 'Pipeline':
        """
        Create a pipeline from a configuration file or dictionary.
        
        Args:
            config: Path to YAML/JSON config file or config dictionary
            
        Returns:
            Configured Pipeline instance
            
        Raises:
            ValueError: If config format is invalid
            FileNotFoundError: If config file doesn't exist
        """
        # Load config from file if string path provided
        if isinstance(config, str):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported config format: {config_path.suffix}"
                    )
        else:
            config_dict = config
        
        # Create pipeline
        pipeline_name = config_dict.get('name', 'default')
        enable_monitoring = config_dict.get('enable_monitoring', False)
        parallel_executor = config_dict.get('parallel_executor', 'thread')
        max_workers = config_dict.get('max_workers', 4)
        
        pipeline = cls(
            name=pipeline_name,
            enable_monitoring=enable_monitoring,
            parallel_executor=parallel_executor,
            max_workers=max_workers
        )
        
        # Parse and add steps
        steps_config = config_dict.get('steps', [])
        if not steps_config:
            raise ValueError("Config must contain 'steps' list")
        
        for step_config in steps_config:
            step = cls._create_step_from_config(step_config)
            pipeline.add_step(step)
        
        return pipeline
    
    @classmethod
    def _create_step_from_config(cls, step_config: Dict[str, Any]) -> PipelineStep:
        """
        Create a PipelineStep from configuration dictionary.
        
        Args:
            step_config: Step configuration dictionary
            
        Returns:
            Configured PipelineStep instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        required_fields = ['name', 'type', 'class']
        for field in required_fields:
            if field not in step_config:
                raise ValueError(f"Step config missing required field: {field}")
        
        # Parse step type
        step_type_str = step_config['type']
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            raise ValueError(f"Invalid step type: {step_type_str}")
        
        # Get component class name
        class_name = step_config['class']
        
        # Import and instantiate component
        component = cls._instantiate_component(
            class_name,
            step_type,
            step_config.get('config', {})
        )
        
        # Create step
        return PipelineStep(
            name=step_config['name'],
            step_type=step_type,
            component=component,
            config=step_config.get('config', {}),
            enabled=step_config.get('enabled', True),
            depends_on=step_config.get('depends_on', []),
            condition=step_config.get('condition'),
            parallel_group=step_config.get('parallel_group')
        )
    
    @classmethod
    def _instantiate_component(
        cls,
        class_name: str,
        step_type: StepType,
        config: Dict[str, Any]
    ) -> Any:
        """
        Instantiate a component from its class name.
        
        Args:
            class_name: Name of the component class
            step_type: Type of step (detector/extractor/processor)
            config: Configuration for the component
            
        Returns:
            Instantiated component
            
        Raises:
            ImportError: If component class cannot be imported
            ValueError: If component class is not found
        """
        # Determine module path based on step type
        if step_type == StepType.DETECTOR:
            module_path = 'src.screenshot2chat.detectors'
        elif step_type == StepType.EXTRACTOR:
            module_path = 'src.screenshot2chat.extractors'
        elif step_type == StepType.PROCESSOR:
            module_path = 'src.screenshot2chat.processors'
        else:
            raise ValueError(f"Unknown step type: {step_type}")
        
        # Try to import the component class
        try:
            # Convert class name to module name (e.g., TextDetector -> text_detector)
            module_name = ''.join(
                ['_' + c.lower() if c.isupper() else c for c in class_name]
            ).lstrip('_')
            
            full_module_path = f"{module_path}.{module_name}"
            
            # Dynamic import
            import importlib
            module = importlib.import_module(full_module_path)
            component_class = getattr(module, class_name)
            
            # Instantiate with config
            return component_class(config=config)
            
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import component '{class_name}' from '{module_path}': {str(e)}"
            ) from e
    
    def save(self, config_path: str) -> None:
        """
        Save pipeline configuration to a file.
        
        Args:
            config_path: Path to save the configuration
            
        Raises:
            ValueError: If file format is not supported
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build config dictionary
        config_dict = {
            'name': self.name,
            'parallel_executor': self.parallel_executor,
            'max_workers': self.max_workers,
            'steps': []
        }
        
        for step in self.steps:
            step_dict = {
                'name': step.name,
                'type': step.step_type.value,
                'class': step.component.__class__.__name__,
                'config': step.config,
                'enabled': step.enabled
            }
            if step.depends_on:
                step_dict['depends_on'] = step.depends_on
            if step.condition:
                step_dict['condition'] = step.condition
            if step.parallel_group:
                step_dict['parallel_group'] = step.parallel_group
            config_dict['steps'].append(step_dict)
        
        # Save to file
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
            elif config_path.suffix == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @classmethod
    def load(cls, config_path: str) -> 'Pipeline':
        """
        Load a pipeline from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded Pipeline instance
        """
        return cls.from_config(config_path)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the last execution.
        
        Returns:
            Dictionary containing metrics for each step
        """
        return self.context.get('metrics', {})
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated performance statistics across all executions.
        
        Returns:
            Dictionary mapping step names to their statistics
        """
        return self.monitor.get_all_stats()
    
    def get_performance_report(self, detailed: bool = False) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            detailed: If True, include detailed metrics for each execution
            
        Returns:
            Formatted performance report string
        """
        return self.monitor.generate_report(detailed=detailed)
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring for this pipeline."""
        self.monitor.enable()
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring for this pipeline."""
        self.monitor.disable()
    
    def clear_metrics(self) -> None:
        """Clear all recorded performance metrics."""
        self.monitor.clear()
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all performance metrics in a structured format.
        
        Returns:
            Dictionary containing all metrics data
        """
        return self.monitor.export_metrics()
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline(name='{self.name}', steps={len(self.steps)})"
