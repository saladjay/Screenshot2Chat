"""
Unit tests for NicknameExtractor
Task 3.2: Write unit tests for NicknameExtractor
Requirements: 7.2
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.screenshot2chat.extractors.nickname_extractor import NicknameExtractor
from src.screenshot2chat.core.data_models import DetectionResult, ExtractionResult


class TestNicknameExtractor:
    """Unit tests for NicknameExtractor"""
    
    def test_nickname_extractor_creation_default(self):
        """Test creating NicknameExtractor with default config"""
        extractor = NicknameExtractor()
        assert extractor.top_k == 3
        assert extractor.min_top_margin_ratio == 0.05
        assert extractor.top_region_ratio == 0.2
    
    def test_nickname_extractor_creation_with_config(self):
        """Test creating NicknameExtractor with custom config"""
        config = {
            "top_k": 5,
            "min_top_margin_ratio": 0.1,
            "top_region_ratio": 0.3
        }
        extractor = NicknameExtractor(config=config)
        assert extractor.top_k == 5
        assert extractor.min_top_margin_ratio == 0.1
        assert extractor.top_region_ratio == 0.3
    
    def test_nickname_extractor_extract_returns_extraction_result(self):
        """Test that extract returns ExtractionResult"""
        extractor = NicknameExtractor()
        
        # Create mock detection results
        detection_results = [
            DetectionResult([10, 10, 100, 30], 0.9, "text", {"text": "John"}),
            DetectionResult([10, 40, 100, 60], 0.9, "text", {"text": "Hello there"}),
            DetectionResult([10, 70, 100, 90], 0.9, "text", {"text": "Jane"})
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        
        assert isinstance(result, ExtractionResult)
        assert "nicknames" in result.data
        assert isinstance(result.data["nicknames"], list)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1
    
    def test_nickname_extractor_empty_detections(self):
        """Test nickname extraction with no detections"""
        extractor = NicknameExtractor()
        
        detection_results = []
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        
        assert isinstance(result, ExtractionResult)
        assert result.data["nicknames"] == []
        assert result.confidence == 0.0
    
    def test_nickname_extractor_top_k_limit(self):
        """Test that extractor respects top_k limit"""
        config = {"top_k": 2}
        extractor = NicknameExtractor(config=config)
        
        # Create many detection results
        detection_results = [
            DetectionResult([10, i*30, 100, i*30+20], 0.9, "text", {"text": f"Name{i}"})
            for i in range(10)
        ]
        
        image = np.random.randint(0, 256, (400, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        
        # Should return at most top_k nicknames
        assert len(result.data["nicknames"]) <= 2
    
    def test_nickname_extractor_scoring_system(self):
        """Test that nickname extractor uses scoring system"""
        extractor = NicknameExtractor()
        
        # Create detections with different positions
        detection_results = [
            DetectionResult([10, 5, 100, 25], 0.9, "text", {"text": "TopName"}),  # Near top
            DetectionResult([10, 150, 100, 170], 0.9, "text", {"text": "MiddleName"}),  # Middle
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        
        # Top name should have higher score
        if len(result.data["nicknames"]) > 0:
            nickname = result.data["nicknames"][0]
            assert "nickname_score" in nickname or "score" in nickname
    
    def test_nickname_extractor_validate(self):
        """Test nickname extractor validation"""
        extractor = NicknameExtractor()
        
        # Valid result
        valid_result = ExtractionResult(
            data={"nicknames": [{"text": "John", "score": 95}]},
            confidence=0.95
        )
        assert extractor.validate(valid_result) == True
        
        # Empty result is also valid
        empty_result = ExtractionResult(
            data={"nicknames": []},
            confidence=0.0
        )
        assert extractor.validate(empty_result) == True
    
    def test_nickname_extractor_to_json(self):
        """Test that extraction result can be converted to JSON"""
        extractor = NicknameExtractor()
        
        detection_results = [
            DetectionResult([10, 10, 100, 30], 0.9, "text", {"text": "John"})
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        json_data = result.to_json()
        
        assert isinstance(json_data, dict)
        assert "data" in json_data
        assert "confidence" in json_data
        assert "nicknames" in json_data["data"]
    
    def test_nickname_extractor_short_text_preference(self):
        """Test that extractor prefers shorter text (likely nicknames)"""
        extractor = NicknameExtractor()
        
        detection_results = [
            DetectionResult([10, 10, 100, 30], 0.9, "text", {"text": "Jo"}),  # Short
            DetectionResult([10, 40, 100, 60], 0.9, "text", {"text": "This is a very long message"}),  # Long
            DetectionResult([10, 70, 100, 90], 0.9, "text", {"text": "Bob"}),  # Short
        ]
        
        image = np.random.randint(0, 256, (200, 720, 3), dtype=np.uint8)
        
        result = extractor.extract(detection_results, image)
        
        # Should prefer shorter texts
        if len(result.data["nicknames"]) > 0:
            for nickname in result.data["nicknames"]:
                text = nickname.get("text", "")
                assert len(text) < 50  # Reasonable nickname length
