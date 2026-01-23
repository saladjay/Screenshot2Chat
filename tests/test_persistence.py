"""
Tests for memory persistence functionality (_save_memory and _load_memory)

This module tests the JSON serialization/deserialization, file handling,
and error handling for the ChatLayoutDetector persistence methods.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src directory to path
src_path = Path(__file__).parent.parent / "src" / "screenshotanalysis"
sys.path.insert(0, str(src_path))

# Import chat_layout_detector directly
import importlib.util
spec_detector = importlib.util.spec_from_file_location(
    "chat_layout_detector", 
    src_path / "chat_layout_detector.py"
)
chat_layout_detector = importlib.util.module_from_spec(spec_detector)
spec_detector.loader.exec_module(chat_layout_detector)
ChatLayoutDetector = chat_layout_detector.ChatLayoutDetector


class TestMemoryPersistence:
    """Test suite for memory persistence functionality"""
    
    def test_save_memory_creates_file(self):
        """Test that _save_memory creates a JSON file with correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "test_memory.json")
            
            # Create detector with memory path
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            
            # Set some memory data
            detector.memory["A"] = {"center": 0.25, "width": 0.15, "count": 10}
            detector.memory["B"] = {"center": 0.75, "width": 0.18, "count": 8}
            
            # Save memory
            detector._save_memory()
            
            # Verify file exists
            assert os.path.exists(memory_path), "Memory file was not created"
            
            # Verify file content
            with open(memory_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert "A" in saved_data
            assert "B" in saved_data
            assert "version" in saved_data
            assert "last_updated" in saved_data
            
            assert saved_data["A"]["center"] == 0.25
            assert saved_data["A"]["width"] == 0.15
            assert saved_data["A"]["count"] == 10
            
            assert saved_data["B"]["center"] == 0.75
            assert saved_data["B"]["width"] == 0.18
            assert saved_data["B"]["count"] == 8
    
    def test_save_memory_creates_directory(self):
        """Test that _save_memory creates parent directories if they don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a nested path that doesn't exist
            memory_path = os.path.join(tmpdir, "subdir1", "subdir2", "memory.json")
            
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            detector.memory["A"] = {"center": 0.3, "width": 0.2, "count": 5}
            
            # Save should create directories
            detector._save_memory()
            
            # Verify file exists
            assert os.path.exists(memory_path), "Memory file was not created in nested directory"
    
    def test_load_memory_file_not_exists(self):
        """Test that _load_memory handles non-existent file gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "nonexistent.json")
            
            # Should not raise exception
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            
            # Memory should be empty
            assert detector.memory["A"] is None
            assert detector.memory["B"] is None
    
    def test_load_memory_corrupted_file(self):
        """Test that _load_memory handles corrupted JSON file gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "corrupted.json")
            
            # Create a corrupted JSON file
            with open(memory_path, 'w') as f:
                f.write("{ this is not valid JSON }")
            
            # Should not raise exception
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            
            # Memory should be empty
            assert detector.memory["A"] is None
            assert detector.memory["B"] is None
    
    def test_load_memory_invalid_format(self):
        """Test that _load_memory handles invalid format gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "invalid.json")
            
            # Create a valid JSON but with wrong structure
            with open(memory_path, 'w') as f:
                json.dump({"wrong": "structure"}, f)
            
            # Should not raise exception
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            
            # Memory should be empty
            assert detector.memory["A"] is None
            assert detector.memory["B"] is None
    
    def test_roundtrip_persistence(self):
        """Test that save and load preserve memory data correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "roundtrip.json")
            
            # Create first detector and save memory
            detector1 = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            detector1.memory["A"] = {"center": 0.3, "width": 0.2, "count": 15}
            detector1.memory["B"] = {"center": 0.7, "width": 0.25, "count": 12}
            detector1._save_memory()
            
            # Create second detector and load memory
            detector2 = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            
            # Verify memory was loaded correctly
            assert detector2.memory["A"] is not None
            assert detector2.memory["B"] is not None
            
            assert detector2.memory["A"]["center"] == 0.3
            assert detector2.memory["A"]["width"] == 0.2
            assert detector2.memory["A"]["count"] == 15
            
            assert detector2.memory["B"]["center"] == 0.7
            assert detector2.memory["B"]["width"] == 0.25
            assert detector2.memory["B"]["count"] == 12
    
    def test_save_memory_with_none_path(self):
        """Test that _save_memory does nothing when memory_path is None"""
        # Should not raise exception
        detector = ChatLayoutDetector(screen_width=720, memory_path=None)
        detector.memory["A"] = {"center": 0.3, "width": 0.2, "count": 5}
        
        # Should do nothing (no exception)
        detector._save_memory()
    
    def test_load_memory_with_none_path(self):
        """Test that _load_memory does nothing when memory_path is None"""
        # Should not raise exception
        detector = ChatLayoutDetector(screen_width=720, memory_path=None)
        
        # Memory should be empty (not loaded from anywhere)
        assert detector.memory["A"] is None
        assert detector.memory["B"] is None
    
    def test_save_memory_with_null_values(self):
        """Test that _save_memory handles None values in memory correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "null_values.json")
            
            detector = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
            # Leave memory as None (default state)
            detector._save_memory()
            
            # Verify file exists and contains null values
            assert os.path.exists(memory_path)
            
            with open(memory_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["A"] is None
            assert saved_data["B"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
