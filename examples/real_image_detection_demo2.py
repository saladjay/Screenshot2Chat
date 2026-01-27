#!/usr/bin/env python3
"""
使用layout_det筛选后绘制检测框

使用text_det检测文本框，经过layout_det筛选，然后分配说话者并绘制
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor, TextBox


def draw_boxes(original_image, filtered_boxes, output_path, image_info):
    """在原始图片上绘制筛选后的检测框"""
    image = original_image.copy()
    draw = ImageDraw.Draw(image)
    
    # 定义颜色
    color_a = (255, 50, 50)      # 红色 - 说话者A
    color_b = (50, 150, 255)     # 蓝色 - 说话者B
    color_unknown = (150, 150, 150)  # 灰色 - 未分类
    
    # 加载字体
    try:
        font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except:
        font_small = ImageFont.load_default()
    
    # 获取坐标转换参数
    left, top, right, bottom = image_info['padding']
    scale_x, scale_y = image_info['scale_ratio']
    
    # 统计说话者
    count_a = sum(1 for box in filtered_boxes if getattr(box, 'speaker', None) == 'A')
    count_b = sum(1 for box in filtered_boxes if getattr(box, 'speaker', None) == 'B')
    
    # 绘制每个文本框
    for box in filtered_boxes:
        speaker = getattr(box, 'speaker', 'Unknown')
        color = color_a if speaker == 'A' else (color_b if speaker == 'B' else color_unknown)
        
        # 坐标转换：letterbox -> 原始图片
        x_min_orig = (box.x_min - left) * scale_x
        y_min_orig = (box.y_min - top) * scale_y
        x_max_orig = (box.x_max - left) * scale_x
        y_max_orig = (box.y_max - top) * scale_y
        
        # 确保坐标在范围内
        x_min_orig = max(0, min(x_min_orig, image.width))
        y_min_orig = max(0, min(y_min_orig, image.height))
        x_max_orig = max(0, min(x_max_orig, image.width))
        y_max_orig = max(0, min(y_max_orig, image.height))
        
        # 绘制半透明矩形
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x_min_orig, y_min_orig, x_max_orig, y_max_orig], 
                              fill=(*color, 30), outline=color, width=4)
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # 绘制标签
        label = f"{speaker}"
        label_y = max(0, y_min_orig - 25)
        try:
            text_bbox = draw.textbbox((x_min_orig, label_y), label, font=font_small)
        except:
            text_width, text_height = draw.textsize(label, font=font_small)
            text_bbox = (x_min_orig, label_y, x_min_orig + text_width, label_y + text_height)
        
        padding = 4
        draw.rectangle([text_bbox[0] - padding, text_bbox[1] - padding, 
                       text_bbox[2] + padding, text_bbox[3] + padding], fill=color)
        draw.text((x_min_orig, label_y), label, fill=(255, 255, 255), font=font_small)
    
    # 绘制信息
    info_lines = [
        f"Filtered by layout_det",
        f"Speaker A: {count_a} boxes",
        f"Speaker B: {count_b} boxes"
    ]
    
    y_offset = 10
    for line in info_lines:
        try:
            text_bbox = draw.textbbox((10, y_offset), line, font=font_small)
        except:
            text_width, text_height = draw.textsize(line, font=font_small)
            text_bbox = (10, y_offset, 10 + text_width, y_offset + text_height)
        
        padding = 5
        draw.rectangle([text_bbox[0] - padding, text_bbox[1] - padding, 
                       text_bbox[2] + padding, text_bbox[3] + padding], fill=(0, 0, 0, 200))
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font_small)
        y_offset += 30
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"已保存: {output_path}")
    print(f"  Speaker A: {count_a} 个框")
    print(f"  Speaker B: {count_b} 个框")


def main():
    """主函数"""
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 目录不存在: {test_images_dir}")
        return 1
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ {test_images_dir} 目录下没有图片")
        return 1
    
    print(f"找到 {len(image_files)} 张图片")
    print("=" * 70)
    
    # 初始化检测器（只需要初始化一次）
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    layout_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    
    # 处理每张图片
    success_count = 0
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(test_images_dir, filename)
        print(f"\n[{idx}/{len(image_files)}] 正在处理: {filename}")
        
        try:
            # 1. 加载图片
            original_image = ImageLoader.load_image(image_path)
            if original_image.mode == 'RGBA':
                original_image = original_image.convert("RGB")
            
            image_array = np.array(original_image)
            processed_image, padding = letterbox(image_array)
            
            # 2. 进行检测
            text_det_results = text_analyzer.model.predict(processed_image)
            layout_det_results = layout_analyzer.model.predict(processed_image)
            
            # 3. 获取所有文本框
            text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results)
            layout_det_boxes = processor._get_all_boxes_from_layout_det(layout_det_results, special_types=['text'])
            
            print(f"  text_det检测到: {len(text_det_boxes)} 个框")
            print(f"  layout_det检测到: {len(layout_det_boxes)} 个框")
            
            # 4. 使用layout_det筛选和分配说话者
            screen_width = processed_image.shape[1]
            
            # 先为layout_det框分配说话者
            layout_det_with_speakers = processor.assign_speakers_to_layout_det_boxes(
                layout_det_boxes,
                screen_width=screen_width
            )
            
            # 使用layout_det结果筛选text_det框
            filtered_boxes = processor.filter_text_boxes_by_layout_det(
                text_det_boxes,
                layout_det_with_speakers,
                coverage_threshold=0.2,
                screen_width=screen_width
            )
            
            print(f"  筛选后保留: {len(filtered_boxes)} 个框")
            print(f"  过滤率: {(len(text_det_boxes) - len(filtered_boxes)) / len(text_det_boxes) * 100:.1f}%")
            
            # 5. 计算坐标转换参数
            left, top, right, bottom = padding
            processed_h, processed_w = processed_image.shape[:2]
            original_h, original_w = original_image.height, original_image.width
            
            content_w = processed_w - left - right
            content_h = processed_h - top - bottom
            
            scale_x = original_w / content_w if content_w > 0 else 1.0
            scale_y = original_h / content_h if content_h > 0 else 1.0
            
            image_info = {
                'padding': padding,
                'scale_ratio': (scale_x, scale_y)
            }
            
            # 6. 绘制结果
            output_filename = f"{os.path.splitext(filename)[0]}_filtered.png"
            output_path = f"test_new_way_det/{output_filename}"
            draw_boxes(original_image, filtered_boxes, output_path, image_info)
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n" + "=" * 70)
    print(f"✓ 完成！成功处理 {success_count}/{len(image_files)} 张图片")
    print(f"结果保存在: test_new_way_det/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
