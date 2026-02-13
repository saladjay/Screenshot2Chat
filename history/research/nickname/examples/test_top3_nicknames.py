#!/usr/bin/env python3
"""
测试方法3：找到y轴方向最大的三个框进行nickname检测

使用顶部区域搜索方法，找到最靠近顶部的3个文本框，
对它们进行OCR识别，输出到命令行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.core import ChatTextRecognition


def extract_top3_nicknames(image_path, text_analyzer, processor):
    """从单张图片提取顶部3个文本框的内容"""
    # 加载图片
    original_image = ImageLoader.load_image(image_path)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert("RGB")
    
    image_array = np.array(original_image)
    processed_image, padding = letterbox(image_array)
    
    # 进行文本检测
    text_det_results = text_analyzer.model.predict(processed_image)
    
    # 获取所有文本框
    text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results)
    
    screen_width = processed_image.shape[1]
    screen_height = processed_image.shape[0]
    
    print(f"\n{'='*80}")
    print(f"图片: {os.path.basename(image_path)}")
    print(f"屏幕尺寸: {screen_width}x{screen_height}")
    print(f"检测到 {len(text_det_boxes)} 个文本框")
    print(f"{'='*80}")
    
    # 按y_min排序（从上到下）
    sorted_boxes = sorted(text_det_boxes, key=lambda b: b.y_min)
    
    # 取前3个
    top3_boxes = sorted_boxes[:3]
    
    print(f"\n找到顶部3个文本框:")
    for i, box in enumerate(top3_boxes, 1):
        print(f"  {i}. 位置: {box.box.tolist()}")
        print(f"     y_min: {box.y_min:.1f}, 高度: {box.height:.1f}, 宽度: {box.width:.1f}")
    
    # 对每个框进行OCR
    print(f"\n开始OCR识别:")
    print(f"{'-'*80}")
    
    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()
    
    results = []
    for i, box in enumerate(top3_boxes, 1):
        # 裁剪图像
        x_min, y_min, x_max, y_max = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
        
        # 确保坐标在范围内
        h, w = processed_image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            print(f"\n框 {i}: ❌ 无效的裁剪区域")
            results.append(None)
            continue
        
        # 裁剪
        cropped_image = processed_image[y_min:y_max, x_min:x_max]
        
        # OCR
        try:
            ocr_result = text_rec.predict_text(cropped_image)
            
            if ocr_result and len(ocr_result) > 0:
                first_result = ocr_result[0]
                
                if isinstance(first_result, dict):
                    text = first_result.get('rec_text', '')
                    score = first_result.get('rec_score', 0.0)
                elif isinstance(first_result, tuple):
                    text = first_result[0]
                    score = first_result[1] if len(first_result) > 1 else 0.0
                else:
                    text = str(first_result)
                    score = 0.0
                
                # 清理文本
                cleaned_text = text.rstrip('>< |\t\n\r')
                
                print(f"\n框 {i}: ✓ '{cleaned_text}'")
                print(f"     置信度: {score:.3f}")
                print(f"     位置: [{x_min}, {y_min}, {x_max}, {y_max}]")
                
                results.append({
                    'text': cleaned_text,
                    'score': score,
                    'box': box.box.tolist(),
                    'position': i
                })
            else:
                print(f"\n框 {i}: ⚠ OCR返回空结果")
                results.append(None)
                
        except Exception as e:
            print(f"\n框 {i}: ❌ OCR失败: {e}")
            results.append(None)
    
    return results


def main():
    """主函数"""
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 目录不存在: {test_images_dir}")
        return 1
    
    # 获取所有图片文件
    image_files = sorted([f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"❌ {test_images_dir} 目录下没有图片")
        return 1
    
    print("正在初始化模型...")
    
    # 初始化检测器
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    
    print(f"\n找到 {len(image_files)} 张图片")
    
    # 处理每张图片
    all_results = {}
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        
        try:
            results = extract_top3_nicknames(image_path, text_analyzer, processor)
            all_results[filename] = results
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[filename] = None
    
    # 输出汇总
    print(f"\n\n{'='*80}")
    print("汇总结果")
    print(f"{'='*80}\n")
    
    for filename, results in all_results.items():
        print(f"📷 {filename}")
        if results:
            valid_results = [r for r in results if r is not None]
            if valid_results:
                for r in valid_results:
                    print(f"   {r['position']}. '{r['text']}' (置信度: {r['score']:.3f})")
            else:
                print(f"   ⚠ 未识别出文本")
        else:
            print(f"   ❌ 处理失败")
        print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
