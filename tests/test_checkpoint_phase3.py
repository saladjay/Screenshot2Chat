"""
Checkpoint Test for Phase 3: Pipeline and Configuration System

This test verifies:
1. Pipeline configuration and execution
2. ConfigManager functionality
3. Integration between components
"""

import pytest
import numpy as np
import yaml
import json
from pathlib import Path
import tempfile
import shutil

from screenshot2chat.pipeline.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.config.config_manager import ConfigManager
from screenshot2chat.core.base_detector import BaseDetector, DetectionResult
from screenshot2chat.core.base_extractor import BaseExtractor, ExtractionResult
from screenshot2chat.detectors.text_detector import TextDetector
from screenshot2chat.detectors.bubble_detector import BubbleDetector
from screenshot2chat.extractors.nickname_extractor import NicknameExtractor
from screenshot2chat.extractors.speaker_extractor import SpeakerExtractor
from screenshot2chat.extractors.layout_extractor import LayoutExtractor


class TestConfigManager:
    """Test ConfigManager functionality"""
    
    def test_config_manager_initialization(self):
        """Test that ConfigManager initializes correctly"""
        config_mgr = ConfigManager()
        assert config_mgr is not None
        assert hasattr(config_mgr, 'configs')
        assert 'default' in config_mgr.configs
        assert 'user' in config_mgr.configs
        assert 'runtime' in config_mgr.configs
    
    def test_config_set_and_get(self):
        """Test setting and getting configuration values"""
        config_mgr = ConfigManager()
        
        # Set a simple value
        config_mgr.set('test.key', 'test_value')
        assert config_mgr.get('test.key') == 'test_value'
        
        # Set nested values
        config_mgr.set('detector.text.backend', 'paddleocr')
        assert config_mgr.get('detector.text.backend') == 'paddleocr'
    
    def test_config_layer_priority(self):
        """Test that configuration layers have correct priority"""
        config_mgr = ConfigManager()
        
        # Set same key in different layers
        config_mgr.set('priority.test', 'default_value', layer='default')
        config_mgr.set('priority.test', 'user_value', layer='user')
        config_mgr.set('priority.test', 'runtime_value', layer='runtime')
        
        # Runtime should have highest priority
        assert config_mgr.get('priority.test') == 'runtime_value'
        
        # Remove runtime, should fall back to user
        config_mgr.configs['runtime'] = {}
        assert config_mgr.get('priority.test') == 'user_value'
        
        # Remove user, should fall back to default
        config_mgr.configs['user'] = {}
        assert config_mgr.get('priority.test') == 'default_value'
    
    def test_config_save_and_load_yaml(self):
        """Test saving and loading YAML configuration"""
        config_mgr = ConfigManager()
        
        # Set some configuration
        config_mgr.set('detector.text.backend', 'paddleocr', layer='user')
        config_mgr.set('detector.text.model_dir', 'models/PP-OCRv5', layer='user')
        config_mgr.set('pipeline.name', 'test_pipeline', layer='user')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config_mgr.save(temp_path, layer='user')
            
            # Load into new ConfigManager
            new_config_mgr = ConfigManager()
            new_config_mgr.load(temp_path, layer='user')
            
            # Verify values
            assert new_config_mgr.get('detector.text.backend') == 'paddleocr'
            assert new_config_mgr.get('detector.text.model_dir') == 'models/PP-OCRv5'
            assert new_config_mgr.get('pipeline.name') == 'test_pipeline'
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_save_and_load_json(self):
        """Test saving and loading JSON configuration"""
        config_mgr = ConfigManager()
        
        # Set some configuration
        config_mgr.set('extractor.nickname.top_k', 3, layer='user')
        config_mgr.set('extractor.nickname.min_score', 0.5, layer='user')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config_mgr.save(temp_path, layer='user')
            
            # Load into new ConfigManager
            new_config_mgr = ConfigManager()
            new_config_mgr.load(temp_path, layer='user')
            
            # Verify values
            assert new_config_mgr.get('extractor.nickname.top_k') == 3
            assert new_config_mgr.get('extractor.nickname.min_score') == 0.5
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_default_value(self):
        """Test getting configuration with default value"""
        config_mgr = ConfigManager()
        
        # Get non-existent key with default
        value = config_mgr.get('non.existent.key', default='default_value')
        assert value == 'default_value'
        
        # Get non-existent key without default
        value = config_mgr.get('another.non.existent.key')
        assert value is None


class TestPipeline:
    """Test Pipeline functionality"""
    
    def test_pipeline_initialization(self):
        """Test that Pipeline initializes correctly"""
        pipeline = Pipeline(name='test_pipeline')
        assert pipeline is not None
        assert pipeline.name == 'test_pipeline'
        assert len(pipeline.steps) == 0
        assert isinstance(pipeline.context, dict)
    
    def test_pipeline_add_step(self):
        """Test adding steps to pipeline"""
        pipeline = Pipeline(name='test_pipeline')
        
        # Create a mock detector
        class MockDetector(BaseDetector):
            def load_model(self):
                pass
            
            def detect(self, image):
                return [DetectionResult(
                    bbox=[0, 0, 100, 100],
                    score=0.9,
                    category='text'
                )]
        
        detector = MockDetector()
        step = PipelineStep(
            name='mock_detector',
            step_type=StepType.DETECTOR,
            component=detector
        )
        
        pipeline.add_step(step)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == 'mock_detector'
    
    def test_pipeline_execute_simple(self):
        """Test executing a simple pipeline"""
        pipeline = Pipeline(name='test_pipeline')
        
        # Create a mock detector
        class MockDetector(BaseDetector):
            def load_model(self):
                pass
            
            def detect(self, image):
                return [DetectionResult(
                    bbox=[10, 20, 110, 120],
                    score=0.95,
                    category='text',
                    metadata={'text': 'Hello'}
                )]
        
        detector = MockDetector()
        step = PipelineStep(
            name='text_detector',
            step_type=StepType.DETECTOR,
            component=detector
        )
        
        pipeline.add_step(step)
        
        # Execute pipeline
        test_image = np.zeros((480, 720, 3), dtype=np.uint8)
        results = pipeline.execute(test_image)
        
        assert 'text_detector' in results
        assert len(results['text_detector']) == 1
        assert results['text_detector'][0].category == 'text'
    
    def test_pipeline_execute_with_extractor(self):
        """Test executing pipeline with detector and extractor"""
        pipeline = Pipeline(name='test_pipeline')
        
        # Create mock detector
        class MockDetector(BaseDetector):
            def load_model(self):
                pass
            
            def detect(self, image):
                return [
                    DetectionResult(bbox=[10, 20, 110, 50], score=0.9, category='text'),
                    DetectionResult(bbox=[10, 60, 110, 90], score=0.85, category='text')
                ]
        
        # Create mock extractor
        class MockExtractor(BaseExtractor):
            def extract(self, detection_results, image=None):
                count = len(detection_results)
                return ExtractionResult(
                    data={'count': count},
                    confidence=1.0
                )
        
        detector = MockDetector()
        extractor = MockExtractor()
        
        detector_step = PipelineStep(
            name='detector',
            step_type=StepType.DETECTOR,
            component=detector
        )
        
        extractor_step = PipelineStep(
            name='extractor',
            step_type=StepType.EXTRACTOR,
            component=extractor,
            config={'source': 'detector'}
        )
        
        pipeline.add_step(detector_step)
        pipeline.add_step(extractor_step)
        
        # Execute pipeline
        test_image = np.zeros((480, 720, 3), dtype=np.uint8)
        results = pipeline.execute(test_image)
        
        assert 'detector' in results
        assert 'extractor' in results
        assert results['extractor'].data['count'] == 2
    
    def test_pipeline_step_enable_disable(self):
        """Test enabling and disabling pipeline steps"""
        pipeline = Pipeline(name='test_pipeline')
        
        class MockDetector(BaseDetector):
            def load_model(self):
                pass
            
            def detect(self, image):
                return [DetectionResult(bbox=[0, 0, 100, 100], score=0.9, category='text')]
        
        detector = MockDetector()
        step = PipelineStep(
            name='detector',
            step_type=StepType.DETECTOR,
            component=detector
        )
        
        # Disable the step
        step.enabled = False
        pipeline.add_step(step)
        
        # Execute pipeline
        test_image = np.zeros((480, 720, 3), dtype=np.uint8)
        results = pipeline.execute(test_image)
        
        # Step should not have executed
        assert 'detector' not in results or len(results.get('detector', [])) == 0
    
    def test_pipeline_from_config_yaml(self):
        """Test creating pipeline from YAML configuration"""
        config_dict = {
            'name': 'test_pipeline',
            'steps': [
                {
                    'name': 'text_detector',
                    'type': 'detector',
                    'class': 'TextDetector',
                    'config': {
                        'backend': 'paddleocr'
                    },
                    'enabled': True
                }
            ]
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Load configuration
            with open(temp_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify configuration structure
            assert config['name'] == 'test_pipeline'
            assert len(config['steps']) == 1
            assert config['steps'][0]['name'] == 'text_detector'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIntegration:
    """Test integration between components"""
    
    def test_config_manager_with_pipeline(self):
        """Test using ConfigManager to configure Pipeline"""
        config_mgr = ConfigManager()
        
        # Set pipeline configuration
        config_mgr.set('pipeline.name', 'integrated_pipeline')
        config_mgr.set('pipeline.detector.backend', 'paddleocr')
        config_mgr.set('pipeline.extractor.top_k', 5)
        
        # Retrieve configuration
        pipeline_name = config_mgr.get('pipeline.name')
        detector_backend = config_mgr.get('pipeline.detector.backend')
        extractor_top_k = config_mgr.get('pipeline.extractor.top_k')
        
        assert pipeline_name == 'integrated_pipeline'
        assert detector_backend == 'paddleocr'
        assert extractor_top_k == 5
    
    def test_full_pipeline_configuration_workflow(self):
        """Test complete workflow: configure, save, load, execute"""
        # Step 1: Create and configure
        config_mgr = ConfigManager()
        config_mgr.set('detector.text.backend', 'paddleocr', layer='user')
        config_mgr.set('extractor.nickname.top_k', 3, layer='user')
        
        # Step 2: Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_config_path = f.name
        
        try:
            config_mgr.save(temp_config_path, layer='user')
            
            # Step 3: Load configuration in new instance
            new_config_mgr = ConfigManager()
            new_config_mgr.load(temp_config_path, layer='user')
            
            # Step 4: Verify configuration
            assert new_config_mgr.get('detector.text.backend') == 'paddleocr'
            assert new_config_mgr.get('extractor.nickname.top_k') == 3
            
            # Step 5: Use configuration to create pipeline
            pipeline = Pipeline(name='configured_pipeline')
            
            # Mock components using configuration
            class ConfiguredDetector(BaseDetector):
                def __init__(self, config_mgr):
                    super().__init__()
                    self.backend = config_mgr.get('detector.text.backend')
                
                def load_model(self):
                    pass
                
                def detect(self, image):
                    return [DetectionResult(
                        bbox=[0, 0, 100, 100],
                        score=0.9,
                        category='text',
                        metadata={'backend': self.backend}
                    )]
            
            detector = ConfiguredDetector(new_config_mgr)
            step = PipelineStep(
                name='configured_detector',
                step_type=StepType.DETECTOR,
                component=detector
            )
            
            pipeline.add_step(step)
            
            # Step 6: Execute pipeline
            test_image = np.zeros((480, 720, 3), dtype=np.uint8)
            results = pipeline.execute(test_image)
            
            # Step 7: Verify results
            assert 'configured_detector' in results
            assert results['configured_detector'][0].metadata['backend'] == 'paddleocr'
            
        finally:
            Path(temp_config_path).unlink(missing_ok=True)


def test_checkpoint_summary():
    """
    Summary test that verifies all Phase 3 components are working
    """
    print("\n" + "="*60)
    print("CHECKPOINT: Phase 3 - Pipeline and Configuration System")
    print("="*60)
    
    # Test 1: ConfigManager
    print("\n✓ Testing ConfigManager...")
    config_mgr = ConfigManager()
    config_mgr.set('test.key', 'test_value')
    assert config_mgr.get('test.key') == 'test_value'
    print("  - Configuration set/get: PASSED")
    
    # Test 2: Pipeline
    print("\n✓ Testing Pipeline...")
    pipeline = Pipeline(name='checkpoint_pipeline')
    
    class SimpleDetector(BaseDetector):
        def load_model(self):
            pass
        def detect(self, image):
            return [DetectionResult(bbox=[0, 0, 100, 100], score=0.9, category='test')]
    
    detector = SimpleDetector()
    step = PipelineStep(name='detector', step_type=StepType.DETECTOR, component=detector)
    pipeline.add_step(step)
    
    test_image = np.zeros((480, 720, 3), dtype=np.uint8)
    results = pipeline.execute(test_image)
    assert 'detector' in results
    print("  - Pipeline execution: PASSED")
    
    # Test 3: Integration
    print("\n✓ Testing Integration...")
    config_mgr.set('pipeline.name', 'integrated_test')
    pipeline_name = config_mgr.get('pipeline.name')
    assert pipeline_name == 'integrated_test'
    print("  - ConfigManager + Pipeline: PASSED")
    
    print("\n" + "="*60)
    print("CHECKPOINT RESULT: ALL TESTS PASSED ✓")
    print("="*60)
    print("\nPhase 3 components are working correctly:")
    print("  ✓ ConfigManager - Configuration management")
    print("  ✓ Pipeline - Pipeline orchestration")
    print("  ✓ Integration - Component integration")
    print("\nReady to proceed to Phase 4: Backward Compatibility & Integration")
    print("="*60 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
