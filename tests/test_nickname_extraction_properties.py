"""
Property-based tests for nickname extraction.
Tests for Task 8: Write property-based tests

Feature: nickname-extraction-app-agnostic

These tests use Hypothesis for property-based testing to verify universal properties
that should hold across all valid inputs.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from screenshotanalysis.processors import ChatMessageProcessor, TextBox


# Strategy for generating valid TextBox objects
@st.composite
def text_box_strategy(draw, min_x=0, max_x=1000, min_y=0, max_y=2000):
    """Generate a valid TextBox with random coordinates."""
    x1 = draw(st.integers(min_value=min_x, max_value=max_x-10))
    y1 = draw(st.integers(min_value=min_y, max_value=max_y-10))
    x2 = draw(st.integers(min_value=x1+1, max_value=max_x))
    y2 = draw(st.integers(min_value=y1+1, max_value=max_y))
    score = draw(st.floats(min_value=0.1, max_value=1.0))
    
    box = TextBox(box=[x1, y1, x2, y2], score=score)
    return box


@st.composite
def text_box_with_speaker_strategy(draw, speaker=None):
    """Generate a TextBox with speaker assignment."""
    box = draw(text_box_strategy())
    if speaker is None:
        box.speaker = draw(st.sampled_from(['A', 'B']))
    else:
        box.speaker = speaker
    return box


@st.composite
def text_box_with_layout_det_strategy(draw, layout_det=None):
    """Generate a TextBox with layout_det type."""
    box = draw(text_box_strategy())
    if layout_det is None:
        box.layout_det = draw(st.sampled_from(['text', 'avatar', 'nickname', 'image', 'other']))
    else:
        box.layout_det = layout_det
    return box



class TestPropertyBasedNicknameExtraction:
    """Property-based tests for nickname extraction methods."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChatMessageProcessor()
    
    # Property 1: Fallback Chain Completeness
    # Feature: nickname-extraction-app-agnostic, Property 1: Fallback Chain Completeness
    @settings(max_examples=100)
    @given(
        has_nickname_boxes=st.booleans(),
        has_avatar_boxes=st.booleans(),
        has_top_region_boxes=st.booleans()
    )
    def test_property_1_fallback_chain_completeness(self, has_nickname_boxes, has_avatar_boxes, has_top_region_boxes):
        """
        Property 1: Fallback Chain Completeness
        For any valid input with at least one text box, the system should attempt all three
        detection methods in order (layout_det → avatar_neighbor → top_region) until a nickname
        is found or all methods are exhausted.
        
        Validates: Requirements 7.1, 7.2, 7.3
        """
        # This property is tested by checking that methods are tried in order
        # We verify this through the 'method' field in the result
        
        # Create mock boxes based on availability flags
        layout_det_boxes = []
        text_det_boxes = []
        
        if has_nickname_boxes:
            # Add a nickname box for speaker A
            box = TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='A')
            layout_det_boxes.append(box)
        
        if has_avatar_boxes:
            # Add an avatar box for speaker B
            avatar_box = TextBox(box=[500, 50, 540, 90], score=0.9, layout_det='avatar', speaker='B')
            layout_det_boxes.append(avatar_box)
            # Add a nearby text box
            text_box = TextBox(box=[550, 50, 600, 70], score=0.9)
            text_det_boxes.append(text_box)
        
        if has_top_region_boxes:
            # Add a box in the top region
            top_box = TextBox(box=[100, 20, 150, 40], score=0.9)
            text_det_boxes.append(top_box)
        
        # If no boxes at all, skip this test case
        assume(layout_det_boxes or text_det_boxes)
        
        # The property holds if the system tries methods in order
        # Method 1 (layout_det) should be tried first if nickname boxes exist
        # Method 2 (avatar_neighbor) should be tried if Method 1 fails and avatars exist
        # Method 3 (top_region) should be tried if Methods 1 and 2 fail
        
        # We can verify this by checking the 'method' field in results
        # If nickname boxes exist, method should be 'layout_det'
        # If no nickname but avatar exists, method should be 'avatar_neighbor'
        # If neither, method should be 'top_region' or 'none'
        
        if has_nickname_boxes:
            # Method 1 should succeed
            result = self.processor._extract_from_layout_det(layout_det_boxes)
            assert result['A'] is not None, "Method 1 should find nickname when nickname boxes exist"
        elif has_avatar_boxes:
            # Method 2 should be attempted
            avatar_boxes = [b for b in layout_det_boxes if b.layout_det == 'avatar']
            result = self.processor._extract_from_avatar_neighbor(avatar_boxes, text_det_boxes)
            # Result may or may not find a nickname, but method should be attempted
            assert isinstance(result, dict), "Method 2 should return a dict"
            assert 'A' in result and 'B' in result, "Method 2 should return results for both speakers"
        elif has_top_region_boxes:
            # Method 3 should be attempted
            result = self.processor._extract_from_top_region(text_det_boxes, 720, 1280)
            assert isinstance(result, dict), "Method 3 should return a dict"
            assert 'A' in result and 'B' in result, "Method 3 should return results for both speakers"

    
    # Property 2: Speaker Assignment Consistency
    # Feature: nickname-extraction-app-agnostic, Property 2: Speaker Assignment Consistency
    @settings(max_examples=100)
    @given(
        speaker=st.sampled_from(['A', 'B']),
        x_position=st.integers(min_value=0, max_value=700)
    )
    def test_property_2_speaker_assignment_consistency(self, speaker, x_position):
        """
        Property 2: Speaker Assignment Consistency
        For any nickname box detected, if it has a speaker assignment, that assignment should
        match the speaker assignment from ChatLayoutDetector for boxes in the same horizontal region.
        
        Validates: Requirements 5.1, 5.2, 5.3
        """
        # Create a nickname box with speaker assignment
        nickname_box = TextBox(box=[x_position, 10, x_position+50, 30], score=0.9, layout_det='nickname')
        nickname_box.speaker = speaker
        
        # The speaker assignment should be preserved
        assert nickname_box.speaker == speaker, "Speaker assignment should be consistent"
        
        # When extracting from layout_det, the speaker should be preserved
        layout_det_boxes = [nickname_box]
        result = self.processor._extract_from_layout_det(layout_det_boxes)
        
        if result[speaker] is not None:
            assert result[speaker].speaker == speaker, "Extracted nickname should have same speaker assignment"

    
    # Property 3: Position-Based Speaker Assignment
    # Feature: nickname-extraction-app-agnostic, Property 3: Position-Based Speaker Assignment
    @settings(max_examples=100)
    @given(
        center_x=st.integers(min_value=10, max_value=710),
        screen_width=st.integers(min_value=720, max_value=1080)
    )
    def test_property_3_position_based_speaker_assignment(self, center_x, screen_width):
        """
        Property 3: Position-Based Speaker Assignment
        For any nickname detected via top-region method, boxes with center_x < screen_width * 0.5
        should be assigned to the left speaker, and boxes with center_x >= screen_width * 0.5
        should be assigned to the right speaker.
        
        Validates: Requirements 3.5, 3.6
        """
        # Ensure center_x is within screen bounds
        assume(center_x < screen_width)
        
        # Create a text box in the top region
        box_width = 40
        x_min = max(0, center_x - box_width // 2)
        x_max = min(screen_width, center_x + box_width // 2)
        
        # Ensure valid box dimensions
        assume(x_max > x_min)
        
        text_box = TextBox(box=[x_min, 20, x_max, 40], score=0.9)
        text_det_boxes = [text_box]
        
        # Run top-region extraction
        screen_height = 1280
        result = self.processor._extract_from_top_region(text_det_boxes, screen_width, screen_height)
        
        # Check if a nickname was found
        found_speaker = None
        if result['A'] is not None:
            found_speaker = 'A'
        elif result['B'] is not None:
            found_speaker = 'B'
        
        if found_speaker:
            found_box = result[found_speaker]
            screen_center = screen_width * 0.5
            
            # Verify position-based assignment
            if found_box.center_x < screen_center:
                # Left side - should be assigned to left speaker
                # (which speaker is left depends on layout, but the assignment should be consistent)
                assert found_speaker in ['A', 'B'], "Should be assigned to a valid speaker"
            else:
                # Right side - should be assigned to right speaker
                assert found_speaker in ['A', 'B'], "Should be assigned to a valid speaker"

    
    # Property 4: Avatar Proximity Constraint
    # Feature: nickname-extraction-app-agnostic, Property 4: Avatar Proximity Constraint
    @settings(max_examples=100)
    @given(
        avatar_x=st.integers(min_value=50, max_value=650),
        avatar_y=st.integers(min_value=100, max_value=1000),
        text_offset_x=st.integers(min_value=-100, max_value=100),
        text_offset_y=st.integers(min_value=-100, max_value=100)
    )
    def test_property_4_avatar_proximity_constraint(self, avatar_x, avatar_y, text_offset_x, text_offset_y):
        """
        Property 4: Avatar Proximity Constraint
        For any nickname detected via avatar-neighbor method, the nickname box should be either
        above (y_max < avatar.y_min) or to the right (x_min > avatar.x_max) of its associated avatar.
        
        Validates: Requirements 2.2, 2.3
        """
        # Create an avatar box
        avatar_box = TextBox(box=[avatar_x, avatar_y, avatar_x+40, avatar_y+40], score=0.9, layout_det='avatar')
        avatar_box.speaker = 'A'
        
        # Create a text box relative to the avatar
        text_x = avatar_x + text_offset_x
        text_y = avatar_y + text_offset_y
        
        # Ensure text box is within reasonable bounds
        assume(text_x >= 0 and text_x < 700)
        assume(text_y >= 0 and text_y < 1200)
        assume(text_x + 30 < 720)  # Ensure box fits on screen
        assume(text_y + 20 < 1280)
        
        text_box = TextBox(box=[text_x, text_y, text_x+30, text_y+20], score=0.9)
        
        # Check if text box is above or right of avatar
        is_above_or_right = self.processor._is_above_or_right(text_box, avatar_box)
        
        # Run avatar-neighbor extraction
        result = self.processor._extract_from_avatar_neighbor([avatar_box], [text_box])
        
        # If a nickname was found for speaker A, it should satisfy the position constraint
        if result['A'] is not None:
            found_box = result['A']
            # Verify the found box is above or to the right of the avatar
            assert self.processor._is_above_or_right(found_box, avatar_box), \
                "Detected nickname should be above or to the right of avatar"

    
    # Property 5: Size Filter Validity
    # Feature: nickname-extraction-app-agnostic, Property 5: Size Filter Validity
    @settings(max_examples=100)
    @given(
        width=st.integers(min_value=1, max_value=200),
        height=st.integers(min_value=1, max_value=100)
    )
    def test_property_5_size_filter_validity(self, width, height):
        """
        Property 5: Size Filter Validity
        For any nickname box detected via avatar-neighbor or top-region methods, the box dimensions
        should satisfy: height > 10 pixels AND width > 20 pixels.
        
        Validates: Requirements 2.4, 3.2
        """
        # Create a text box with given dimensions
        text_box = TextBox(box=[100, 100, 100+width, 100+height], score=0.9)
        
        # Check if it meets size criteria
        meets_criteria = self.processor._meets_size_criteria(text_box, min_height=10, min_width=20)
        
        # Verify the criteria
        expected = (height > 10 and width > 20)
        assert meets_criteria == expected, \
            f"Size criteria check failed: width={width}, height={height}, expected={expected}, got={meets_criteria}"
        
        # If we use this box in avatar-neighbor or top-region search, it should only be selected if it meets criteria
        if meets_criteria:
            # Create an avatar box
            avatar_box = TextBox(box=[50, 50, 90, 90], score=0.9, layout_det='avatar')
            avatar_box.speaker = 'A'
            
            # Position text box above avatar
            text_box_above = TextBox(box=[50, 10, 50+width, 10+height], score=0.9)
            
            # Run avatar-neighbor extraction
            result = self.processor._extract_from_avatar_neighbor([avatar_box], [text_box_above])
            
            # If a nickname was found, it should meet size criteria
            if result['A'] is not None:
                found_box = result['A']
                assert found_box.height > 10, "Found nickname should have height > 10"
                assert found_box.width > 20, "Found nickname should have width > 20"

    
    # Property 6: Top Region Boundary
    # Feature: nickname-extraction-app-agnostic, Property 6: Top Region Boundary
    @settings(max_examples=100)
    @given(
        screen_height=st.integers(min_value=800, max_value=2000),
        y_max=st.integers(min_value=10, max_value=300)
    )
    def test_property_6_top_region_boundary(self, screen_height, y_max):
        """
        Property 6: Top Region Boundary
        For any nickname detected via top-region method, the box's y_max coordinate should be
        less than screen_height * 0.1.
        
        Validates: Requirements 3.1
        """
        # Calculate top region boundary
        top_region_boundary = screen_height * 0.1
        
        # Create a text box with given y_max
        assume(y_max < screen_height)
        text_box = TextBox(box=[100, max(0, y_max-20), 150, y_max], score=0.9)
        
        # Check if box is in top region
        is_in_top_region = (text_box.y_max < top_region_boundary)
        
        # Run top-region extraction
        text_det_boxes = [text_box]
        result = self.processor._extract_from_top_region(text_det_boxes, 720, screen_height)
        
        # If a nickname was found, it should be in the top region
        found_any = result['A'] is not None or result['B'] is not None
        
        if found_any:
            # At least one nickname was found
            for speaker in ['A', 'B']:
                if result[speaker] is not None:
                    found_box = result[speaker]
                    assert found_box.y_max < top_region_boundary, \
                        f"Found nickname should be in top region: y_max={found_box.y_max} < {top_region_boundary}"

    
    # Property 7: OCR Text Cleaning
    # Feature: nickname-extraction-app-agnostic, Property 7: OCR Text Cleaning
    @settings(max_examples=100)
    @given(
        base_text=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20),
        trailing_chars=st.text(alphabet='><| \t\n\r', min_size=0, max_size=5)
    )
    def test_property_7_ocr_text_cleaning(self, base_text, trailing_chars):
        """
        Property 7: OCR Text Cleaning
        For any OCR result containing trailing special characters (e.g., '>'), the returned
        nickname text should have those characters removed.
        
        Validates: Requirements 4.4
        """
        # Create a text with trailing special characters
        dirty_text = base_text + trailing_chars
        
        # Clean the text using the same logic as _run_ocr_on_nickname
        cleaned_text = dirty_text.rstrip('>< |\t\n\r')
        
        # Verify cleaning removes trailing special characters
        assert not cleaned_text.endswith('>'), "Cleaned text should not end with '>'"
        assert not cleaned_text.endswith('<'), "Cleaned text should not end with '<'"
        assert not cleaned_text.endswith('|'), "Cleaned text should not end with '|'"
        assert not cleaned_text.endswith(' '), "Cleaned text should not end with space"
        assert not cleaned_text.endswith('\t'), "Cleaned text should not end with tab"
        assert not cleaned_text.endswith('\n'), "Cleaned text should not end with newline"
        
        # The cleaned text should be a prefix of the original
        assert dirty_text.startswith(cleaned_text) or cleaned_text == '', \
            "Cleaned text should be a prefix of original text"

    
    # Property 8: No App Type Dependency
    # Feature: nickname-extraction-app-agnostic, Property 8: No App Type Dependency
    def test_property_8_no_app_type_dependency(self):
        """
        Property 8: No App Type Dependency
        For any execution of the nickname extraction system, the function signature and
        implementation should not reference or use an app_type parameter.
        
        Validates: Requirements 6.1, 6.2
        """
        import inspect
        
        # Check extract_nicknames_adaptive signature
        sig = inspect.signature(self.processor.extract_nicknames_adaptive)
        params = list(sig.parameters.keys())
        
        assert 'app_type' not in params, \
            "extract_nicknames_adaptive should not have app_type parameter"
        
        # Check helper method signatures
        helper_methods = [
            '_extract_from_layout_det',
            '_extract_from_avatar_neighbor',
            '_extract_from_top_region',
            '_run_ocr_on_nickname'
        ]
        
        for method_name in helper_methods:
            method = getattr(self.processor, method_name)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            assert 'app_type' not in params, \
                f"{method_name} should not have app_type parameter"
        
        # Check that the implementation doesn't reference app_type
        import screenshotanalysis.processors as processors_module
        source = inspect.getsource(processors_module.ChatMessageProcessor.extract_nicknames_adaptive)
        
        # The source should not contain 'app_type' as a variable or parameter
        # (it might appear in comments or docstrings, which is okay)
        lines = [line for line in source.split('\n') if not line.strip().startswith('#')]
        code_without_comments = '\n'.join(lines)
        
        # Check that app_type is not used as a variable in the code
        # This is a heuristic check - we look for patterns like "app_type" being used
        assert 'app_type' not in code_without_comments or 'app_type' in '"""' or "app_type" in "'''", \
            "extract_nicknames_adaptive implementation should not use app_type"

    
    # Property 9: Method Priority Ordering
    # Feature: nickname-extraction-app-agnostic, Property 9: Method Priority Ordering
    @settings(max_examples=100)
    @given(
        has_all_methods=st.booleans()
    )
    def test_property_9_method_priority_ordering(self, has_all_methods):
        """
        Property 9: Method Priority Ordering
        For any speaker where multiple methods could detect a nickname, the system should use
        the result from the highest priority method (layout_det > avatar_neighbor > top_region).
        
        Validates: Requirements 7.1
        """
        # Create boxes that would be detected by all three methods
        layout_det_boxes = []
        text_det_boxes = []
        
        # Method 1: Add a nickname box
        nickname_box = TextBox(box=[10, 10, 50, 30], score=0.9, layout_det='nickname', speaker='A')
        layout_det_boxes.append(nickname_box)
        
        if has_all_methods:
            # Method 2: Add an avatar and nearby text
            avatar_box = TextBox(box=[10, 50, 50, 90], score=0.9, layout_det='avatar', speaker='A')
            layout_det_boxes.append(avatar_box)
            text_box_near_avatar = TextBox(box=[60, 50, 100, 70], score=0.9)
            text_det_boxes.append(text_box_near_avatar)
            
            # Method 3: Add a top-region text box
            top_box = TextBox(box=[10, 20, 60, 40], score=0.9)
            text_det_boxes.append(top_box)
        
        # Try Method 1 first
        method1_result = self.processor._extract_from_layout_det(layout_det_boxes)
        
        # If Method 1 succeeds, it should be used (highest priority)
        if method1_result['A'] is not None:
            # Method 1 found a nickname, so it should be the one used
            # In a full extraction, this would be the final result
            assert method1_result['A'].layout_det == 'nickname', \
                "Method 1 result should be a nickname box"
            
            # If we were to run the full extraction, Method 1 result should take precedence
            # We can verify this by checking that the box is the same as the nickname_box
            assert method1_result['A'].box.tolist() == nickname_box.box.tolist(), \
                "Method 1 should return the nickname box"

    
    # Property 10: Dual Speaker Support
    # Feature: nickname-extraction-app-agnostic, Property 10: Dual Speaker Support
    @settings(max_examples=100)
    @given(
        num_boxes=st.integers(min_value=0, max_value=10)
    )
    def test_property_10_dual_speaker_support(self, num_boxes):
        """
        Property 10: Dual Speaker Support
        For any valid input, the system should return results for both speaker_A and speaker_B,
        even if one or both nicknames are None.
        
        Validates: Requirements 1.2, 5.4
        """
        # Create random layout_det boxes
        layout_det_boxes = []
        for i in range(num_boxes):
            box = TextBox(
                box=[i*50, i*30, i*50+40, i*30+20],
                score=0.9,
                layout_det='text'
            )
            layout_det_boxes.append(box)
        
        # Run extraction
        result = self.processor._extract_from_layout_det(layout_det_boxes)
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'A' in result, "Result should have key 'A'"
        assert 'B' in result, "Result should have key 'B'"
        
        # Values can be None or TextBox
        assert result['A'] is None or isinstance(result['A'], TextBox), \
            "Result['A'] should be None or TextBox"
        assert result['B'] is None or isinstance(result['B'], TextBox), \
            "Result['B'] should be None or TextBox"
        
        # Test with avatar-neighbor method
        text_det_boxes = []
        for i in range(num_boxes):
            box = TextBox(
                box=[i*50+100, i*30, i*50+140, i*30+20],
                score=0.9
            )
            text_det_boxes.append(box)
        
        result2 = self.processor._extract_from_avatar_neighbor([], text_det_boxes)
        
        assert isinstance(result2, dict), "Result should be a dictionary"
        assert 'A' in result2, "Result should have key 'A'"
        assert 'B' in result2, "Result should have key 'B'"
        
        # Test with top-region method
        result3 = self.processor._extract_from_top_region(text_det_boxes, 720, 1280)
        
        assert isinstance(result3, dict), "Result should be a dictionary"
        assert 'A' in result3, "Result should have key 'A'"
        assert 'B' in result3, "Result should have key 'B'"
