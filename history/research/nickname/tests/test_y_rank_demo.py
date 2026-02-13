"""
Demo script to show Y-rank scoring in action with real nickname detection
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_with_real_image():
    """Test Y-rank scoring with a real chat screenshot"""
    
    # Import after path setup
    from screenshotanalysis import ChatLayoutAnalyzer
    from screenshotanalysis.processors import ChatMessageProcessor
    from screenshotanalysis.utils import ImageLoader, letterbox
    
    # Use a test image
    test_image_path = Path("test_images/test_whatsapp.png")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        print("Available test images:")
        for img in Path("test_images").glob("*.png"):
            print(f"  - {img}")
        return
    
    print(f"Testing Y-rank scoring with: {test_image_path}")
    print("=" * 60)
    
    # Load and preprocess image
    original_image = ImageLoader.load_image(str(test_image_path))
    if original_image.mode == 'RGBA':
        original_image = original_image.convert("RGB")
    
    image_array = np.array(original_image)
    processed_image, padding = letterbox(image_array)
    
    print(f"Image shape: {processed_image.shape}")
    
    # Initialize detector
    print("\nInitializing text detection model...")
    text_analyzer = ChatLayoutAnalyzer(model_name='PP-OCRv5_server_det')
    text_analyzer.load_model()
    
    # Run detection
    print("Running text detection...")
    text_det_results = text_analyzer.model.predict(processed_image)
    
    # Initialize processor
    processor = ChatMessageProcessor()
    
    # Create log file
    log_path = Path("test_output/y_rank_demo.log")
    log_path.parent.mkdir(exist_ok=True)
    
    print("Extracting nicknames with Y-rank scoring...")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        # Extract nicknames with smart detection (which uses Y-rank scoring)
        result = processor.extract_nicknames_smart(
            text_det_results, 
            processed_image, 
            log_file=log_file
        )
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result['top_candidate']:
        top = result['top_candidate']
        print(f"\nTop Nickname: '{top['text']}'")
        print(f"Total Score: {top['score']:.1f}")
        print(f"Y Rank: {top.get('y_rank', 'N/A')}")
        print(f"\nScore Breakdown:")
        for key, value in top['score_breakdown'].items():
            print(f"  {key:12s}: {value:5.1f}")
    
    print(f"\n\nTop 5 Candidates:")
    print("-" * 60)
    for i, candidate in enumerate(result['candidates'][:5], 1):
        print(f"\n{i}. '{candidate['text']}'")
        print(f"   Total Score: {candidate['score']:.1f}")
        print(f"   Y Rank: {candidate.get('y_rank', 'N/A')}")
        print(f"   Y Position: {candidate['y_min']:.0f}px")
        breakdown = candidate['score_breakdown']
        print(f"   Breakdown: pos={breakdown['position']:.1f}, "
              f"text={breakdown['text']:.1f}, "
              f"y_pos={breakdown['y_position']:.1f}, "
              f"height={breakdown['height']:.1f}, "
              f"y_rank={breakdown['y_rank']:.1f}")
    
    print("\n" + "=" * 60)
    print(f"✓ Test complete! Log saved to: {log_path}")
    print("=" * 60)

if __name__ == "__main__":
    test_with_real_image()
