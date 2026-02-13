#!/usr/bin/env python3
"""
提取所有测试图片的沟通对象nickname

遍历test_images目录下的所有图片，提取每张图片中的沟通对象昵称
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor


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
    print("=" * 80)
    
    # 初始化检测器（只需要初始化一次）
    print("正在初始化模型...")
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    layout_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    print("模型初始化完成\n")
    
    # 存储所有结果
    all_results = []
    
    # 处理每张图片
    success_count = 0
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(test_images_dir, filename)
        print(f"[{idx}/{len(image_files)}] 正在处理: {filename}")
        
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
            
            # 5. 提取nickname
            nickname_result = processor.extract_nicknames_adaptive(
                layout_det_results=layout_det_results,
                text_det_results=text_det_results,
                image=processed_image,
                screen_width=screen_width
            )
            
            # 6. 输出结果
            nicknames = []
            if nickname_result.get('speaker_A', {}).get('nickname'):
                nicknames.append(nickname_result['speaker_A']['nickname'])
            if nickname_result.get('speaker_B', {}).get('nickname'):
                nicknames.append(nickname_result['speaker_B']['nickname'])
            
            if nicknames:
                print(f"  ✓ 检测到沟通对象: {', '.join(nicknames)}")
                all_results.append({
                    'filename': filename,
                    'nicknames': nicknames,
                    'status': 'success'
                })
            else:
                print(f"  ⚠ 未检测到沟通对象昵称")
                all_results.append({
                    'filename': filename,
                    'nicknames': [],
                    'status': 'no_nicknames'
                })
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            all_results.append({
                'filename': filename,
                'nicknames': [],
                'status': 'error',
                'error': str(e)
            })
        
        print()
    
    # 输出汇总结果
    print("=" * 80)
    print("汇总结果:")
    print("=" * 80)
    
    for result in all_results:
        filename = result['filename']
        if result['status'] == 'success' and result['nicknames']:
            nicknames_str = ', '.join(result['nicknames'])
            print(f"✓ {filename:40s} -> {nicknames_str}")
        elif result['status'] == 'no_nicknames':
            print(f"⚠ {filename:40s} -> (未检测到昵称)")
        else:
            print(f"❌ {filename:40s} -> (处理失败)")
    
    print("\n" + "=" * 80)
    print(f"处理完成: {success_count}/{len(image_files)} 张图片成功")
    
    # 统计有昵称的图片数量
    with_nicknames = sum(1 for r in all_results if r['status'] == 'success' and r['nicknames'])
    print(f"成功提取昵称: {with_nicknames}/{len(image_files)} 张图片")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
