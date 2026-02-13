#!/usr/bin/env python3
"""
测试方法3（增强版）：过滤边缘框后找到顶部3个框进行nickname检测

增加筛选规则：
1. 过滤左上角的框（系统时间）
2. 过滤右上角的框（信号、电池等）
3. 这些框都非常靠近屏幕边缘
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import re
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.core import ChatTextRecognition


def is_edge_box(box, screen_width, screen_height):
    """
    判断文本框是否在屏幕边缘（系统UI区域）
    
    边缘定义：
    - 左上角：x < 20% 屏幕宽度 且 y < 10% 屏幕高度
    - 右上角：x > 80% 屏幕宽度 且 y < 10% 屏幕高度
    """
    # 左边缘阈值（屏幕宽度的20%）
    left_edge_threshold = screen_width * 0.20
    # 右边缘阈值（屏幕宽度的80%）
    right_edge_threshold = screen_width * 0.80
    # 顶部阈值（屏幕高度的10%）
    top_edge_threshold = screen_height * 0.10
    
    # 检查是否在左上角
    is_left_top = (box.x_min < left_edge_threshold and 
                   box.y_max < top_edge_threshold)
    
    # 检查是否在右上角
    is_right_top = (box.x_max > right_edge_threshold and 
                    box.y_max < top_edge_threshold)
    
    return is_left_top or is_right_top


def is_likely_system_text(text):
    """
    判断文本是否像系统UI文本（时间、信号等）
    
    系统文本特征：
    - 时间格式：HH:MM 或 HH:MM:SS
    - 纯数字
    - 单个字符或符号
    - 常见系统文本：5G, 4G, WiFi等
    """
    if not text or len(text.strip()) == 0:
        return True
    
    text = text.strip()
    
    # 时间格式
    if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', text):
        return True
    
    # 纯数字（如电池百分比）
    if text.replace('%', '').replace('.', '').isdigit():
        return True
    
    # 单个字符或符号
    if len(text) <= 1:
        return True
    
    # 常见系统文本
    system_keywords = ['5G', '4G', '3G', 'LTE', 'WiFi', 'WIFI']
    if text.upper() in system_keywords:
        return True
    
    return False


def extract_top3_nicknames_filtered(image_path, text_analyzer, processor):
    """从单张图片提取顶部3个文本框的内容（过滤边缘框）"""
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
    
    # 过滤边缘框
    filtered_boxes = []
    edge_boxes = []
    
    for box in text_det_boxes:
        if is_edge_box(box, screen_width, screen_height):
            edge_boxes.append(box)
        else:
            filtered_boxes.append(box)
    
    print(f"过滤掉 {len(edge_boxes)} 个边缘框（系统UI）")
    print(f"保留 {len(filtered_boxes)} 个候选框")
    print(f"{'='*80}")
    
    # 显示被过滤的边缘框
    if edge_boxes:
        print(f"\n被过滤的边缘框:")
        for i, box in enumerate(edge_boxes[:5], 1):  # 只显示前5个
            print(f"  {i}. 位置: {box.box.tolist()}, y_min: {box.y_min:.1f}")
        if len(edge_boxes) > 5:
            print(f"  ... 还有 {len(edge_boxes) - 5} 个")
    
    # 按y_min排序（从上到下）
    sorted_boxes = sorted(filtered_boxes, key=lambda b: b.y_min)
    
    # 取前3个
    top3_boxes = sorted_boxes[:min(3, len(sorted_boxes))]
    
    print(f"\n找到顶部{len(top3_boxes)}个文本框（已过滤边缘）:")
    for i, box in enumerate(top3_boxes, 1):
        print(f"  {i}. 位置: {box.box.tolist()}")
        print(f"     y_min: {box.y_min:.1f}, 高度: {box.height:.1f}, 宽度: {box.width:.1f}")
        print(f"     center_x: {box.center_x:.1f} (屏幕中心: {screen_width/2:.1f})")
    
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
            print(f"\n框 {i}: ERROR 无效的裁剪区域")
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
                
                # 检查是否是系统文本
                is_system = is_likely_system_text(cleaned_text)
                system_flag = " [系统文本]" if is_system else ""
                
                print(f"\n框 {i}: OK '{cleaned_text}'{system_flag}")
                print(f"     置信度: {score:.3f}")
                print(f"     位置: [{x_min}, {y_min}, {x_max}, {y_max}]")
                
                results.append({
                    'text': cleaned_text,
                    'score': score,
                    'box': box.box.tolist(),
                    'position': i,
                    'is_system': is_system
                })
            else:
                print(f"\n框 {i}: WARN OCR返回空结果")
                results.append(None)
                
        except Exception as e:
            print(f"\n框 {i}: ERROR OCR失败: {e}")
            results.append(None)
    
    return results


def main():
    """主函数"""
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"ERROR 目录不存在: {test_images_dir}")
        return 1
    
    # 获取所有图片文件
    image_files = sorted([f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"ERROR {test_images_dir} 目录下没有图片")
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
            results = extract_top3_nicknames_filtered(image_path, text_analyzer, processor)
            all_results[filename] = results
            
        except Exception as e:
            print(f"\nERROR 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[filename] = None
    
    # 输出汇总
    print(f"\n\n{'='*80}")
    print("汇总结果（已过滤系统UI）")
    print(f"{'='*80}\n")
    
    for filename, results in all_results.items():
        print(f"[图片] {filename}")
        if results:
            # 只显示非系统文本
            nickname_results = [r for r in results if r is not None and not r.get('is_system', False)]
            if nickname_results:
                print(f"   [OK] 检测到 {len(nickname_results)} 个可能的昵称:")
                for r in nickname_results:
                    print(f"      - '{r['text']}' (置信度: {r['score']:.3f})")
            else:
                print(f"   [WARN] 未检测到昵称（所有文本都是系统UI）")
        else:
            print(f"   [ERROR] 处理失败")
        print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
