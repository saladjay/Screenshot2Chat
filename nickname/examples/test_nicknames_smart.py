#!/usr/bin/env python3
"""
智能Nickname检测（优化版 - 使用新的Y-rank评分系统）

改进：
1. 更精细的边缘过滤：只过滤极端边缘的小框
2. 基于位置优先级：优先选择靠近屏幕中心的文本框
3. 综合评分系统：位置 + 尺寸 + 文本特征 + Y排名（新增）
4. Y-rank评分：第1名20分，第2名15分，第3名10分

注意：现在使用 processor._calculate_nickname_score 方法，包含完整的评分系统
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import re
import cv2
from pathlib import Path
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.core import ChatTextRecognition


# 注意：以下函数已废弃，现在使用 processor 的方法
# 保留仅供参考

def is_extreme_edge_box_OLD(box, screen_width, screen_height):
    """[已废弃] 请使用 processor._is_extreme_edge_box"""
    pass


def is_likely_system_text_OLD(text):
    """[已废弃] 请使用 processor._is_likely_system_text"""
    pass


# 注意：此函数已废弃，现在使用 processor._calculate_nickname_score 方法
# 该方法包含新的 Y-rank 评分系统（0-20分）
# 保留此函数仅供参考
def calculate_nickname_score_OLD(box, text, screen_width, screen_height):
    """
    [已废弃] 计算文本框作为昵称的综合得分
    
    请使用 processor._calculate_nickname_score(box, text, screen_width, screen_height, y_rank=y_rank)
    新方法包含 Y-rank 评分（0-20分）
    """
    pass


def draw_top3_results(image_path, top_candidates, output_dir="test_output/smart_nicknames"):
    """
    绘制得分前三的候选框到图片上
    
    Args:
        image_path: 原始图片路径
        top_candidates: 前三名候选者列表
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载原始图片
    original_image = ImageLoader.load_image(image_path)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert("RGB")
    
    image_array = np.array(original_image)
    processed_image, padding = letterbox(image_array)
    
    # 转换为BGR用于OpenCV
    draw_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    
    # 定义颜色（BGR格式）
    colors = [
        (0, 255, 0),    # 绿色 - 第1名
        (0, 165, 255),  # 橙色 - 第2名
        (0, 0, 255),    # 红色 - 第3名
    ]
    
    # 绘制每个候选框
    for i, candidate in enumerate(top_candidates[:3]):
        box = candidate['box']
        x_min, y_min, x_max, y_max = map(int, box)
        
        color = colors[i]
        rank = i + 1
        
        # 绘制矩形框（加粗）
        cv2.rectangle(draw_image, (x_min, y_min), (x_max, y_max), color, 3)
        
        # 准备标签文本
        text = candidate['text']
        score = candidate['nickname_score']
        y_rank = candidate.get('y_rank', 'N/A')
        
        # 标签：排名 + 文本 + 得分
        label = f"#{rank}: {text} ({score:.1f})"
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制标签背景
        label_y = y_min - 10
        if label_y < text_height + 10:
            label_y = y_max + text_height + 10
        
        cv2.rectangle(draw_image, 
                     (x_min, label_y - text_height - 5), 
                     (x_min + text_width + 10, label_y + 5), 
                     color, -1)
        
        # 绘制标签文字
        cv2.putText(draw_image, label, 
                   (x_min + 5, label_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 在框内绘制Y排名
        rank_label = f"Y-Rank: {y_rank}"
        cv2.putText(draw_image, rank_label,
                   (x_min + 5, y_min + 20),
                   font, 0.5, color, 2)
    
    # 保存图片
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"top3_{filename}")
    cv2.imwrite(output_path, draw_image)
    
    return output_path


def extract_nicknames_smart(image_path, text_analyzer, processor, draw_results=False, output_dir="test_output/smart_nicknames"):
    """智能提取昵称（基于综合评分）- 使用新的Y-rank评分系统"""
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
    
    # 过滤极端边缘框
    filtered_boxes = []
    edge_boxes = []
    
    for box in text_det_boxes:
        if processor._is_extreme_edge_box(box, screen_width, screen_height):
            edge_boxes.append(box)
        else:
            filtered_boxes.append(box)
    
    print(f"过滤掉 {len(edge_boxes)} 个极端边缘框")
    print(f"保留 {len(filtered_boxes)} 个候选框")
    
    # 只处理顶部区域的框（前20%）
    top_region_boundary = screen_height * 0.20
    top_boxes = [box for box in filtered_boxes if box.y_min < top_region_boundary]
    
    print(f"顶部区域候选框: {len(top_boxes)} 个")
    
    if not top_boxes:
        print("没有找到候选框")
        print(f"{'='*80}")
        return []
    
    # 按Y位置排序以计算排名
    sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)
    box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}
    
    # 初始化OCR
    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()
    
    # 对每个候选框进行OCR并计算得分
    candidates = []
    
    for box in top_boxes:
        # 裁剪图像
        x_min, y_min, x_max, y_max = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
        
        # 确保坐标在范围内
        h, w = processed_image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
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
                    ocr_score = first_result.get('rec_score', 0.0)
                elif isinstance(first_result, tuple):
                    text = first_result[0]
                    ocr_score = first_result[1] if len(first_result) > 1 else 0.0
                else:
                    text = str(first_result)
                    ocr_score = 0.0
                
                # 清理文本
                cleaned_text = text.rstrip('>< |\t\n\r')
                
                if not cleaned_text:
                    continue
                
                # 获取Y排名
                y_rank = box_to_rank.get(id(box), None)
                
                # 使用processor的新评分方法（包含Y-rank得分）
                nickname_score, score_breakdown = processor._calculate_nickname_score(
                    box, cleaned_text, screen_width, screen_height, y_rank=y_rank
                )
                
                candidates.append({
                    'text': cleaned_text,
                    'ocr_score': ocr_score,
                    'nickname_score': nickname_score,
                    'score_breakdown': score_breakdown,
                    'box': box.box.tolist(),
                    'center_x': box.center_x,
                    'y_min': box.y_min,
                    'y_rank': y_rank
                })
                
        except Exception as e:
            continue
    
    # 按得分排序，选择前3个
    candidates.sort(key=lambda x: x['nickname_score'], reverse=True)
    top_candidates = candidates[:3]
    
    print(f"\n最终选择（按得分排序）:")
    for i, c in enumerate(top_candidates, 1):
        breakdown_str = ', '.join([f"{k}={v:.1f}" for k, v in c['score_breakdown'].items()])
        print(f"  {i}. '{c['text']}' (得分: {c['nickname_score']:.1f}/100, Y排名: {c.get('y_rank', 'N/A')})")
        print(f"     细项: {breakdown_str}")
    
    print(f"{'='*80}")
    
    # 绘制结果
    if draw_results and top_candidates:
        output_path = draw_top3_results(image_path, top_candidates, output_dir)
        print(f"结果已保存到: {output_path}")
    
    return top_candidates


def main():
    """主函数"""
    test_images_dir = "test_images"
    output_dir = "test_output/smart_nicknames"
    
    if not os.path.exists(test_images_dir):
        print(f"ERROR 目录不存在: {test_images_dir}")
        return 1
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
    print(f"结果将保存到: {output_dir}")
    
    # 处理每张图片
    all_results = {}
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        
        try:
            results = extract_nicknames_smart(image_path, text_analyzer, processor, 
                                            draw_results=True, output_dir=output_dir)
            all_results[filename] = results
            
        except Exception as e:
            print(f"\nERROR 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[filename] = None
    
    # 输出汇总
    print(f"\n\n{'='*80}")
    print("汇总结果（智能评分 - 包含Y-rank评分）")
    print(f"{'='*80}\n")
    
    for filename, results in all_results.items():
        print(f"[图片] {filename}")
        if results:
            print(f"   [OK] 检测到 {len(results)} 个可能的昵称:")
            for r in results:
                breakdown_str = ', '.join([f"{k}={v:.1f}" for k, v in r['score_breakdown'].items()])
                print(f"      - '{r['text']}' (得分: {r['nickname_score']:.1f}/100, Y排名: {r.get('y_rank', 'N/A')})")
                print(f"        细项: {breakdown_str}")
        else:
            print(f"   [WARN] 未检测到昵称")
        print()
    
    print(f"{'='*80}")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
