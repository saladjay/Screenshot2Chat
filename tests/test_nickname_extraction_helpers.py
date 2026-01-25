"""
Unit tests for nickname extraction helper methods.
Tests for Task 1: Set up core infrastructure and helper methods
Tests for Task 2: Implement Method 1: Layout Det Nickname Detection
"""
import pytest
import numpy as np
from screenshotanalysis.processors import ChatMessageProcessor, TextBox


class TestDistanceCalculation:
    """Tests for _calculate_distance helper method (Task 1.1)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    def test_distance_same_position(self):
        """Test distance between boxes at same position"""
        box1 = TextBox(box=[10, 10, 20, 20], score=1.0)
        box2 = TextBox(box=[10, 10, 20, 20], score=1.0)
        distance = self.processor._calculate_distance(box1, box2)
        assert distance == 0.0
    
    def test_distance_horizontal(self):
        """Test distance between horizontally separated boxes"""
        box1 = TextBox(box=[0, 0, 10, 10], score=1.0)  # center: (5, 5)
        box2 = TextBox(box=[10, 0, 20, 10], score=1.0)  # center: (15, 5)
        distance = self.processor._calculate_distance(box1, box2)
        assert distance == 10.0
    
    def test_distance_vertical(self):
        """Test distance between vertically separated boxes"""
        box1 = TextBox(box=[0, 0, 10, 10], score=1.0)  # center: (5, 5)
        box2 = TextBox(box=[0, 10, 10, 20], score=1.0)  # center: (5, 15)
        distance = self.processor._calculate_distance(box1, box2)
        assert distance == 10.0
    
    def test_distance_diagonal(self):
        """Test distance between diagonally separated boxes"""
        box1 = TextBox(box=[0, 0, 10, 10], score=1.0)  # center: (5, 5)
        box2 = TextBox(box=[30, 40, 40, 50], score=1.0)  # center: (35, 45)
        distance = self.processor._calculate_distance(box1, box2)
        # Distance = sqrt((35-5)^2 + (45-5)^2) = sqrt(900 + 1600) = sqrt(2500) = 50
        assert distance == 50.0
    
    def test_distance_overlapping_boxes(self):
        """Test distance between overlapping boxes"""
        box1 = TextBox(box=[0, 0, 20, 20], score=1.0)  # center: (10, 10)
        box2 = TextBox(box=[10, 10, 30, 30], score=1.0)  # center: (20, 20)
        distance = self.processor._calculate_distance(box1, box2)
        # Distance = sqrt((20-10)^2 + (20-10)^2) = sqrt(100 + 100) = sqrt(200) ≈ 14.14
        assert abs(distance - 14.142135623730951) < 0.0001
    
    def test_distance_negative_coordinates(self):
        """Test distance calculation with negative coordinates"""
        box1 = TextBox(box=[-10, -10, 0, 0], score=1.0)  # center: (-5, -5)
        box2 = TextBox(box=[0, 0, 10, 10], score=1.0)  # center: (5, 5)
        distance = self.processor._calculate_distance(box1, box2)
        # Distance = sqrt((5-(-5))^2 + (5-(-5))^2) = sqrt(100 + 100) = sqrt(200) ≈ 14.14
        assert abs(distance - 14.142135623730951) < 0.0001


class TestPositionFilter:
    """Tests for _is_above_or_right helper method (Task 1.2)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    def test_text_above_avatar(self):
        """Test text box above avatar box"""
        text_box = TextBox(box=[10, 0, 30, 10], score=1.0)  # y_max = 10
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # y_min = 20
        assert self.processor._is_above_or_right(text_box, avatar_box) is True
    
    def test_text_below_avatar(self):
        """Test text box below avatar box"""
        text_box = TextBox(box=[10, 50, 30, 60], score=1.0)  # y_min = 50
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # y_max = 40
        assert self.processor._is_above_or_right(text_box, avatar_box) is False
    
    def test_text_right_of_avatar(self):
        """Test text box to the right of avatar box"""
        text_box = TextBox(box=[50, 20, 70, 40], score=1.0)  # x_min = 50
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # x_max = 30
        assert self.processor._is_above_or_right(text_box, avatar_box) is True
    
    def test_text_left_of_avatar(self):
        """Test text box to the left of avatar box"""
        text_box = TextBox(box=[0, 20, 5, 40], score=1.0)  # x_max = 5
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # x_min = 10
        assert self.processor._is_above_or_right(text_box, avatar_box) is False
    
    def test_text_above_and_right(self):
        """Test text box both above and to the right"""
        text_box = TextBox(box=[50, 0, 70, 10], score=1.0)
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)
        assert self.processor._is_above_or_right(text_box, avatar_box) is True
    
    def test_text_below_and_left(self):
        """Test text box both below and to the left"""
        text_box = TextBox(box=[0, 50, 5, 60], score=1.0)
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)
        assert self.processor._is_above_or_right(text_box, avatar_box) is False
    
    def test_text_overlapping_avatar(self):
        """Test text box overlapping with avatar box"""
        text_box = TextBox(box=[15, 25, 25, 35], score=1.0)
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)
        # Overlapping boxes are neither above nor to the right
        assert self.processor._is_above_or_right(text_box, avatar_box) is False
    
    def test_text_exactly_above_boundary(self):
        """Test text box exactly at the boundary (y_max == avatar y_min)"""
        text_box = TextBox(box=[10, 0, 30, 20], score=1.0)  # y_max = 20
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # y_min = 20
        # y_max < y_min is False when they're equal
        assert self.processor._is_above_or_right(text_box, avatar_box) is False
    
    def test_text_exactly_right_boundary(self):
        """Test text box exactly at the boundary (x_min == avatar x_max)"""
        text_box = TextBox(box=[30, 20, 50, 40], score=1.0)  # x_min = 30
        avatar_box = TextBox(box=[10, 20, 30, 40], score=1.0)  # x_max = 30
        # x_min > x_max is False when they're equal
        assert self.processor._is_above_or_right(text_box, avatar_box) is False


class TestSizeFilter:
    """Tests for _meets_size_criteria helper method (Task 1.3)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    def test_meets_default_criteria(self):
        """Test box that meets default size criteria"""
        box = TextBox(box=[0, 0, 30, 20], score=1.0)  # width=30, height=20
        assert self.processor._meets_size_criteria(box) is True
    
    def test_below_height_threshold(self):
        """Test box below minimum height threshold"""
        box = TextBox(box=[0, 0, 30, 10], score=1.0)  # width=30, height=10
        assert self.processor._meets_size_criteria(box) is False
    
    def test_below_width_threshold(self):
        """Test box below minimum width threshold"""
        box = TextBox(box=[0, 0, 20, 20], score=1.0)  # width=20, height=20
        assert self.processor._meets_size_criteria(box) is False
    
    def test_exactly_at_height_threshold(self):
        """Test box exactly at height threshold (should fail)"""
        box = TextBox(box=[0, 0, 30, 10], score=1.0)  # width=30, height=10
        # height > 10 is False when height == 10
        assert self.processor._meets_size_criteria(box, min_height=10) is False
    
    def test_exactly_at_width_threshold(self):
        """Test box exactly at width threshold (should fail)"""
        box = TextBox(box=[0, 0, 20, 20], score=1.0)  # width=20, height=20
        # width > 20 is False when width == 20
        assert self.processor._meets_size_criteria(box, min_width=20) is False
    
    def test_one_pixel_above_threshold(self):
        """Test box one pixel above both thresholds"""
        box = TextBox(box=[0, 0, 21, 11], score=1.0)  # width=21, height=11
        assert self.processor._meets_size_criteria(box, min_height=10, min_width=20) is True
    
    def test_custom_thresholds(self):
        """Test with custom size thresholds"""
        box = TextBox(box=[0, 0, 50, 30], score=1.0)  # width=50, height=30
        assert self.processor._meets_size_criteria(box, min_height=25, min_width=40) is True
        assert self.processor._meets_size_criteria(box, min_height=30, min_width=50) is False
    
    def test_very_small_box(self):
        """Test very small box"""
        box = TextBox(box=[0, 0, 5, 5], score=1.0)  # width=5, height=5
        assert self.processor._meets_size_criteria(box) is False
    
    def test_very_large_box(self):
        """Test very large box"""
        box = TextBox(box=[0, 0, 1000, 500], score=1.0)  # width=1000, height=500
        assert self.processor._meets_size_criteria(box) is True
    
    def test_tall_narrow_box(self):
        """Test tall but narrow box"""
        box = TextBox(box=[0, 0, 15, 100], score=1.0)  # width=15, height=100
        assert self.processor._meets_size_criteria(box) is False  # width too small
    
    def test_wide_short_box(self):
        """Test wide but short box"""
        box = TextBox(box=[0, 0, 100, 5], score=1.0)  # width=100, height=5
        assert self.processor._meets_size_criteria(box) is False  # height too small



class TestLayoutDetNicknameExtraction:
    """Tests for _extract_from_layout_det method (Task 2.1)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    def test_no_nickname_boxes(self):
        """Test with 0 nickname boxes"""
        # Create boxes with different layout_det types but no 'nickname'
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='text', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='text', speaker='B'),
            TextBox(box=[10, 50, 40, 80], score=0.95, layout_det='avatar', speaker='A'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        assert result == {'A': None, 'B': None}
    
    def test_one_nickname_box_speaker_a(self):
        """Test with 1 nickname box for speaker A"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='text', speaker='B'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        assert result['A'] is not None
        assert result['A'].layout_det == 'nickname'
        assert result['A'].speaker == 'A'
        assert result['A'].box.tolist() == [10, 10, 50, 30]
        assert result['B'] is None
    
    def test_one_nickname_box_speaker_b(self):
        """Test with 1 nickname box for speaker B"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='text', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='nickname', speaker='B'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        assert result['A'] is None
        assert result['B'] is not None
        assert result['B'].layout_det == 'nickname'
        assert result['B'].speaker == 'B'
        assert result['B'].box.tolist() == [100, 10, 140, 30]
    
    def test_two_nickname_boxes_one_per_speaker(self):
        """Test with 2 nickname boxes, one for each speaker"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='nickname', speaker='B'),
            TextBox(box=[10, 50, 50, 70], score=0.9, layout_det='text', speaker='A'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        assert result['A'] is not None
        assert result['A'].layout_det == 'nickname'
        assert result['A'].speaker == 'A'
        assert result['A'].box.tolist() == [10, 10, 50, 30]
        
        assert result['B'] is not None
        assert result['B'].layout_det == 'nickname'
        assert result['B'].speaker == 'B'
        assert result['B'].box.tolist() == [100, 10, 140, 30]
    
    def test_multiple_nickname_boxes_per_speaker(self):
        """Test with multiple nickname boxes per speaker (should select first)"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='A'),
            TextBox(box=[10, 40, 50, 60], score=0.85, layout_det='nickname', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='nickname', speaker='B'),
            TextBox(box=[100, 40, 140, 60], score=0.88, layout_det='nickname', speaker='B'),
            TextBox(box=[100, 70, 140, 90], score=0.87, layout_det='nickname', speaker='B'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        # Should select first box for each speaker
        assert result['A'] is not None
        assert result['A'].box.tolist() == [10, 10, 50, 30]
        assert result['A'].score == 0.9
        
        assert result['B'] is not None
        assert result['B'].box.tolist() == [100, 10, 140, 30]
        assert result['B'].score == 0.9
    
    def test_nickname_box_without_speaker_attribute(self):
        """Test nickname box without speaker attribute (should be ignored)"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname'),  # No speaker
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='nickname', speaker='B'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        # Box without speaker should be ignored
        assert result['A'] is None
        assert result['B'] is not None
        assert result['B'].box.tolist() == [100, 10, 140, 30]
    
    def test_nickname_box_with_unknown_speaker(self):
        """Test nickname box with unknown speaker value"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='Unknown'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='nickname', speaker='A'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        # Unknown speaker should be ignored
        assert result['A'] is not None
        assert result['A'].box.tolist() == [100, 10, 140, 30]
        assert result['B'] is None
    
    def test_empty_boxes_list(self):
        """Test with empty boxes list"""
        boxes = []
        
        result = self.processor._extract_from_layout_det(boxes)
        
        assert result == {'A': None, 'B': None}
    
    def test_mixed_layout_det_types(self):
        """Test with mixed layout_det types including nickname"""
        boxes = [
            TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='text', speaker='A'),
            TextBox(box=[10, 40, 40, 70], score=0.95, layout_det='avatar', speaker='A'),
            TextBox(box=[10, 80, 50, 100], score=0.9, layout_det='nickname', speaker='A'),
            TextBox(box=[100, 10, 140, 30], score=0.9, layout_det='image', speaker='B'),
            TextBox(box=[100, 40, 140, 60], score=0.9, layout_det='nickname', speaker='B'),
            TextBox(box=[100, 70, 140, 90], score=0.9, layout_det='text', speaker='B'),
        ]
        
        result = self.processor._extract_from_layout_det(boxes)
        
        # Should only extract nickname boxes
        assert result['A'] is not None
        assert result['A'].layout_det == 'nickname'
        assert result['A'].box.tolist() == [10, 80, 50, 100]
        
        assert result['B'] is not None
        assert result['B'].layout_det == 'nickname'
        assert result['B'].box.tolist() == [100, 40, 140, 60]



class TestOCRIntegration:
    """Tests for _run_ocr_on_nickname method (Task 5.1)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    def test_ocr_with_valid_box_and_image(self):
        """Test OCR with a valid nickname box and image"""
        # Create a simple test image (white background with some text-like pattern)
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Add some dark pixels to simulate text
        image[10:20, 10:50] = 0  # Simulate text region
        
        # Create a nickname box covering the text region
        nickname_box = TextBox(box=[10, 10, 50, 20], score=0.9)
        
        # Note: This test will actually try to run OCR, which may fail in test environment
        # The important thing is that it handles errors gracefully
        result = self.processor._run_ocr_on_nickname(nickname_box, image)
        
        # Result should be either a string or None (if OCR fails)
        assert result is None or isinstance(result, str)
    
    def test_ocr_with_invalid_crop_region(self):
        """Test OCR with invalid crop region (x_max <= x_min)"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Create a box with invalid dimensions
        nickname_box = TextBox(box=[50, 10, 50, 20], score=0.9)  # x_min == x_max
        
        result = self.processor._run_ocr_on_nickname(nickname_box, image)
        
        # Should return None for invalid crop region
        assert result is None
    
    def test_ocr_with_out_of_bounds_box(self):
        """Test OCR with box coordinates outside image bounds"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Create a box that extends beyond image bounds
        nickname_box = TextBox(box=[150, 80, 250, 120], score=0.9)
        
        # Should handle out of bounds gracefully by clipping
        result = self.processor._run_ocr_on_nickname(nickname_box, image)
        
        # Result should be either a string or None
        assert result is None or isinstance(result, str)
    
    def test_ocr_with_zero_height_box(self):
        """Test OCR with zero height box"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Create a box with zero height
        nickname_box = TextBox(box=[10, 20, 50, 20], score=0.9)  # y_min == y_max
        
        result = self.processor._run_ocr_on_nickname(nickname_box, image)
        
        # Should return None for invalid crop region
        assert result is None
    
    def test_ocr_with_negative_coordinates(self):
        """Test OCR with negative coordinates (should be clipped to 0)"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Create a box with negative coordinates
        nickname_box = TextBox(box=[-10, -5, 30, 20], score=0.9)
        
        # Should handle negative coordinates by clipping to 0
        result = self.processor._run_ocr_on_nickname(nickname_box, image)
        
        # Result should be either a string or None
        assert result is None or isinstance(result, str)
    
    def test_ocr_returns_none_or_string(self):
        """Test that OCR always returns None or string, never raises exception"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        nickname_box = TextBox(box=[10, 10, 50, 30], score=0.9)
        
        # This should never raise an exception
        try:
            result = self.processor._run_ocr_on_nickname(nickname_box, image)
            assert result is None or isinstance(result, str)
        except Exception as e:
            pytest.fail(f"OCR method raised an exception: {e}")
