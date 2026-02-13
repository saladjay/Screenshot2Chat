"""
Test script for extract_nicknames_adaptive method
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from screenshotanalysis.processors import ChatMessageProcessor, TextBox

def test_extract_nicknames_adaptive_basic():
    """Test basic functionality of extract_nicknames_adaptive"""
    print("Testing extract_nicknames_adaptive method...")
    
    # Create processor
    processor = ChatMessageProcessor()
    
    # Create mock layout_det_results with nickname boxes
    layout_det_results = [{
        'boxes': [
            {
                'label': 'nickname',
                'score': 0.95,
                'coordinate': [10, 10, 100, 30]
            },
            {
                'label': 'nickname',
                'score': 0.93,
                'coordinate': [620, 10, 710, 30]
            },
            {
                'label': 'text',
                'score': 0.90,
                'coordinate': [10, 50, 200, 70]
            },
            {
                'label': 'text',
                'score': 0.88,
                'coordinate': [520, 50, 710, 70]
            }
        ]
    }]
    
    # Create mock text_det_results
    text_det_results = [{
        'dt_polys': [
            [[10, 10], [100, 10], [100, 30], [10, 30]],
            [[620, 10], [710, 10], [710, 30], [620, 30]],
            [[10, 50], [200, 50], [200, 70], [10, 70]],
            [[520, 50], [710, 50], [710, 70], [520, 70]]
        ],
        'dt_scores': [0.95, 0.93, 0.90, 0.88]
    }]
    
    # Create mock image (720x1280 RGB)
    image = np.zeros((1280, 720, 3), dtype=np.uint8)
    screen_width = 720
    
    # Test without log file
    print("\n1. Testing without log file...")
    result = processor.extract_nicknames_adaptive(
        layout_det_results=layout_det_results,
        text_det_results=text_det_results,
        image=image,
        screen_width=screen_width
    )
    
    # Verify result structure
    assert 'speaker_A' in result, "Result should have 'speaker_A' key"
    assert 'speaker_B' in result, "Result should have 'speaker_B' key"
    assert 'metadata' in result, "Result should have 'metadata' key"
    
    # Verify speaker structure
    for speaker in ['speaker_A', 'speaker_B']:
        assert 'nickname' in result[speaker], f"{speaker} should have 'nickname' key"
        assert 'box' in result[speaker], f"{speaker} should have 'box' key"
        assert 'method' in result[speaker], f"{speaker} should have 'method' key"
    
    # Verify metadata structure
    assert 'layout' in result['metadata'], "Metadata should have 'layout' key"
    assert 'confidence' in result['metadata'], "Metadata should have 'confidence' key"
    assert 'frame_count' in result['metadata'], "Metadata should have 'frame_count' key"
    
    print("✓ Result structure is correct")
    print(f"  Speaker A method: {result['speaker_A']['method']}")
    print(f"  Speaker B method: {result['speaker_B']['method']}")
    print(f"  Layout: {result['metadata']['layout']}")
    
    # Test with log file
    print("\n2. Testing with log file...")
    with open('test_extract_nicknames_adaptive.log', 'w', encoding='utf-8') as log_file:
        result = processor.extract_nicknames_adaptive(
            layout_det_results=layout_det_results,
            text_det_results=text_det_results,
            image=image,
            screen_width=screen_width,
            log_file=log_file
        )
    
    print("✓ Log file created successfully")
    print("  Check test_extract_nicknames_adaptive.log for details")
    
    # Test with no nickname boxes (fallback to method 2 and 3)
    print("\n3. Testing fallback methods...")
    layout_det_results_no_nickname = [{
        'boxes': [
            {
                'label': 'text',
                'score': 0.90,
                'coordinate': [10, 50, 200, 70]
            },
            {
                'label': 'text',
                'score': 0.88,
                'coordinate': [520, 50, 710, 70]
            },
            {
                'label': 'avatar',
                'score': 0.92,
                'coordinate': [10, 100, 60, 150]
            }
        ]
    }]
    
    result = processor.extract_nicknames_adaptive(
        layout_det_results=layout_det_results_no_nickname,
        text_det_results=text_det_results,
        image=image,
        screen_width=screen_width
    )
    
    print("✓ Fallback methods executed")
    print(f"  Speaker A method: {result['speaker_A']['method']}")
    print(f"  Speaker B method: {result['speaker_B']['method']}")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_extract_nicknames_adaptive_basic()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
