"""
Test script to verify Y-direction ranking score implementation
"""
import cv2
import numpy as np
from pathlib import Path
from screenshotanalysis.processors import ChatMessageProcessor

def test_y_rank_scoring():
    """Test that Y-direction ranking scores are correctly applied"""
    
    # Use a test image
    test_image_path = Path("test_images/test_whatsapp.png")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    # Load image
    image = cv2.imread(str(test_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Testing with image: {test_image_path}")
    print(f"Image shape: {image.shape}")
    
    # Initialize processor
    processor = ChatMessageProcessor()
    
    # Load detection results (you'll need to run detection first)
    # For now, let's just test the scoring function directly
    from screenshotanalysis.processors import TextBox
    
    screen_height, screen_width = image.shape[:2]
    
    # Create mock text boxes at different Y positions
    boxes = [
        TextBox(box=[100, 50, 200, 80], score=0.9),   # Top box (rank 1)
        TextBox(box=[100, 100, 200, 130], score=0.9), # Middle box (rank 2)
        TextBox(box=[100, 150, 200, 180], score=0.9), # Lower box (rank 3)
        TextBox(box=[100, 200, 200, 230], score=0.9), # Even lower (rank 4)
    ]
    
    print("\nTesting Y-rank scoring:")
    print("-" * 60)
    
    for i, box in enumerate(boxes, 1):
        # Test with different ranks
        score, breakdown = processor._calculate_nickname_score(
            box, 
            f"TestName{i}", 
            screen_width, 
            screen_height,
            y_rank=i
        )
        
        print(f"\nBox {i} (y_min={box.y_min}):")
        print(f"  Y Rank: {i}")
        print(f"  Y Rank Score: {breakdown['y_rank']:.1f}")
        print(f"  Total Score: {breakdown['total']:.1f}")
        print(f"  Breakdown: {breakdown}")
        
        # Verify expected Y rank scores
        if i == 1:
            assert breakdown['y_rank'] == 20, f"Expected rank 1 to get 20 points, got {breakdown['y_rank']}"
        elif i == 2:
            assert breakdown['y_rank'] == 15, f"Expected rank 2 to get 15 points, got {breakdown['y_rank']}"
        elif i == 3:
            assert breakdown['y_rank'] == 10, f"Expected rank 3 to get 10 points, got {breakdown['y_rank']}"
        else:
            assert breakdown['y_rank'] == 0, f"Expected rank 4+ to get 0 points, got {breakdown['y_rank']}"
    
    print("\n" + "=" * 60)
    print("✓ All Y-rank scoring tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_y_rank_scoring()
