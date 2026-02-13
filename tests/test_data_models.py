"""
Unit tests for data models (DetectionResult and ExtractionResult)
Task 1.3: Write unit tests for data models
Requirements: 1.5
"""

import pytest
import json
import numpy as np
from src.screenshot2chat.core.data_models import DetectionResult, ExtractionResult


class TestDetectionResult:
    """Test DetectionResult data model"""
    
    def test_detection_result_creation(self):
        """Test creating a DetectionResult instance"""
        bbox = [10.0, 20.0, 100.0, 200.0]
        score = 0.95
        category = "text"
        metadata = {"language": "en"}
        
        result = DetectionResult(bbox, score, category, metadata)
        
        assert result.bbox == bbox
        assert result.score == score
        assert result.category == category
        assert result.metadata == metadata
    
    def test_detection_result_without_metadata(self):
        """Test creating DetectionResult without metadata"""
        bbox = [10.0, 20.0, 100.0, 200.0]
        score = 0.95
        category = "text"
        
        result = DetectionResult(bbox, score, category)
        
        assert result.metadata == {}
    
    def test_detection_result_serialization(self):
        """Test DetectionResult to_json serialization"""
        bbox = [10.0, 20.0, 100.0, 200.0]
        score = 0.95
        category = "text"
        metadata = {"language": "en", "confidence": 0.9}
        
        result = DetectionResult(bbox, score, category, metadata)
        json_data = result.to_json()
        
        assert json_data["bbox"] == bbox
        assert json_data["score"] == score
        assert json_data["category"] == category
        assert json_data["metadata"] == metadata
    
    def test_detection_result_json_roundtrip(self):
        """Test that DetectionResult can be serialized and deserialized"""
        bbox = [10.0, 20.0, 100.0, 200.0]
        score = 0.95
        category = "text"
        metadata = {"language": "en"}
        
        result = DetectionResult(bbox, score, category, metadata)
        json_str = json.dumps(result.to_json())
        loaded_data = json.loads(json_str)
        
        assert loaded_data["bbox"] == bbox
        assert loaded_data["score"] == score
        assert loaded_data["category"] == category
        assert loaded_data["metadata"] == metadata


class TestExtractionResult:
    """Test ExtractionResult data model"""
    
    def test_extraction_result_creation(self):
        """Test creating an ExtractionResult instance"""
        data = {"nickname": "John", "position": "top"}
        confidence = 0.85
        
        result = ExtractionResult(data, confidence)
        
        assert result.data == data
        assert result.confidence == confidence
    
    def test_extraction_result_default_confidence(self):
        """Test ExtractionResult with default confidence"""
        data = {"nickname": "John"}
        
        result = ExtractionResult(data)
        
        assert result.confidence == 1.0
    
    def test_extraction_result_to_json(self):
        """Test ExtractionResult to_json method"""
        data = {"nickname": "John", "score": 95}
        confidence = 0.85
        
        result = ExtractionResult(data, confidence)
        json_data = result.to_json()
        
        assert json_data["data"] == data
        assert json_data["confidence"] == confidence
    
    def test_extraction_result_json_roundtrip(self):
        """Test ExtractionResult JSON serialization roundtrip"""
        data = {"nickname": "John", "position": "top", "score": 95}
        confidence = 0.85
        
        result = ExtractionResult(data, confidence)
        json_str = json.dumps(result.to_json())
        loaded_data = json.loads(json_str)
        
        assert loaded_data["data"] == data
        assert loaded_data["confidence"] == confidence
    
    def test_extraction_result_complex_data(self):
        """Test ExtractionResult with complex nested data"""
        data = {
            "nicknames": ["John", "Jane", "Bob"],
            "scores": [95, 90, 85],
            "metadata": {
                "method": "scoring",
                "version": "1.0"
            }
        }
        confidence = 0.9
        
        result = ExtractionResult(data, confidence)
        json_data = result.to_json()
        
        assert json_data["data"]["nicknames"] == data["nicknames"]
        assert json_data["data"]["scores"] == data["scores"]
        assert json_data["data"]["metadata"] == data["metadata"]
        assert json_data["confidence"] == confidence
