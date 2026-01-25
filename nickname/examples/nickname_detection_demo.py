#!/usr/bin/env python3
"""
Simple Nickname Detection Demo

Demonstrates the smart nickname detection functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor


def main():
    """Main function"""
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"ERROR: Directory not found: {test_images_dir}")
        return 1
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"ERROR: No images found in {test_images_dir}")
        return 1
    
    print("Initializing models...")
    
    # Initialize detector
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    
    print(f"\nFound {len(image_files)} images\n")
    
    # Process each image
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        
        print(f"{'='*60}")
        print(f"Image: {filename}")
        print(f"{'='*60}")
        
        try:
            # Load image
            original_image = ImageLoader.load_image(image_path)
            if original_image.mode == 'RGBA':
                original_image = original_image.convert("RGB")
            
            image_array = np.array(original_image)
            processed_image, padding = letterbox(image_array)
            
            # Detect text
            text_det_results = text_analyzer.model.predict(processed_image)
            
            # Extract nicknames using smart detection
            result = processor.extract_nicknames_smart(text_det_results, processed_image)
            
            # Display results
            if result['top_candidate']:
                top = result['top_candidate']
                print(f"Detected Nickname: '{top['text']}'")
                print(f"  Score: {top['score']:.1f}/100")
                
                # Display score breakdown
                if 'score_breakdown' in top:
                    bd = top['score_breakdown']
                    print(f"  Breakdown: Position={bd['position']:.1f}, "
                          f"Text={bd['text']:.1f}, Y={bd['y_position']:.1f}, Height={bd['height']:.1f}")
                
                print(f"  Position: ({top['center_x']:.0f}, {top['y_min']:.0f})")
                
                if len(result['candidates']) > 1:
                    print(f"\nOther candidates:")
                    for i, c in enumerate(result['candidates'][1:4], 2):
                        print(f"  {i}. '{c['text']}' (score: {c['score']:.1f})")
                        if 'score_breakdown' in c:
                            bd = c['score_breakdown']
                            print(f"     Breakdown: Pos={bd['position']:.1f}, "
                                  f"Text={bd['text']:.1f}, Y={bd['y_position']:.1f}, Height={bd['height']:.1f}")
            else:
                print("No nickname detected")
            
            print()
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
