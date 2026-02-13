#!/usr/bin/env python3
"""
显示所有测试图片的沟通对象nickname

简洁版本：遍历test_images目录，输出每张图片检测到的昵称
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor


def extract_nicknames_from_image(image_path, text_analyzer, layout_analyzer, processor):
    """从单张图片提取昵称"""
    # 加载图片
    original_image = ImageLoader.load_image(image_path)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert("RGB")
    
    image_array = np.array(original_image)
    processed_image, padding = letterbox(image_array)
    
    # 进行检测
    text_det_results = text_analyzer.model.predict(processed_image)
    layout_det_results = layout_analyzer.model.predict(processed_image)
    
    # 提取nickname
    screen_width = processed_image.shape[1]
    nickname_result = processor.extract_nicknames_adaptive(
        layout_det_results=layout_det_results,
        text_det_results=text_det_results,
        image=processed_image,
        screen_width=screen_width
    )
    
    # 提取昵称列表
    nicknames = []
    for speaker in ['A', 'B']:
        speaker_key = f'speaker_{speaker}'
        if nickname_result.get(speaker_key, {}).get('nickname'):
            nicknames.append(nickname_result[speaker_key]['nickname'])
    
    return nicknames


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
    
    layout_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    
    print(f"\n找到 {len(image_files)} 张图片\n")
    print("=" * 80)
    print(f"{'图片文件':<45} {'检测到的昵称'}")
    print("=" * 80)
    
    # 处理每张图片
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        
        try:
            nicknames = extract_nicknames_from_image(
                image_path, text_analyzer, layout_analyzer, processor
            )
            
            if nicknames:
                nicknames_str = ', '.join(nicknames)
                print(f"{filename:<45} {nicknames_str}")
            else:
                print(f"{filename:<45} (未检测到)")
                
        except Exception as e:
            print(f"{filename:<45} (错误: {str(e)[:20]}...)")
    
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
