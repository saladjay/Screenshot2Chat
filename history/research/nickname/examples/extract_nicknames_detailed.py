#!/usr/bin/env python3
"""
提取所有测试图片的沟通对象nickname（详细版本）

遍历test_images目录下的所有图片，提取每张图片中的沟通对象昵称
并输出详细的检测信息
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
    output_dir = "test_nickname_extraction"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 目录不存在: {test_images_dir}")
        return 1
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 创建日志文件
        log_path = os.path.join(output_dir, f"{filename}.log")
        
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
            
            # 3. 提取nickname（带日志）
            screen_width = processed_image.shape[1]
            
            with open(log_path, 'w', encoding='utf-8') as log_file:
                nickname_result = processor.extract_nicknames_adaptive(
                    layout_det_results=layout_det_results,
                    text_det_results=text_det_results,
                    image=processed_image,
                    screen_width=screen_width,
                    log_file=log_file
                )
            
            # 4. 提取昵称
            nicknames = []
            speaker_info = {}
            
            for speaker in ['A', 'B']:
                speaker_key = f'speaker_{speaker}'
                if nickname_result.get(speaker_key, {}).get('nickname'):
                    nickname = nickname_result[speaker_key]['nickname']
                    method = nickname_result[speaker_key]['method']
                    nicknames.append(nickname)
                    speaker_info[speaker] = {
                        'nickname': nickname,
                        'method': method
                    }
            
            # 5. 输出结果
            if nicknames:
                print(f"  ✓ 检测到沟通对象:")
                for speaker, info in speaker_info.items():
                    print(f"    Speaker {speaker}: '{info['nickname']}' (方法: {info['method']})")
                
                all_results.append({
                    'filename': filename,
                    'nicknames': nicknames,
                    'speaker_info': speaker_info,
                    'metadata': nickname_result.get('metadata', {}),
                    'status': 'success',
                    'log_file': log_path
                })
            else:
                print(f"  ⚠ 未检测到沟通对象昵称")
                all_results.append({
                    'filename': filename,
                    'nicknames': [],
                    'speaker_info': {},
                    'metadata': nickname_result.get('metadata', {}),
                    'status': 'no_nicknames',
                    'log_file': log_path
                })
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'filename': filename,
                'nicknames': [],
                'speaker_info': {},
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
            layout = result['metadata'].get('layout', 'unknown')
            print(f"✓ {filename:40s} -> {nicknames_str:30s} (布局: {layout})")
        elif result['status'] == 'no_nicknames':
            layout = result['metadata'].get('layout', 'unknown')
            print(f"⚠ {filename:40s} -> (未检测到昵称) (布局: {layout})")
        else:
            print(f"❌ {filename:40s} -> (处理失败)")
    
    print("\n" + "=" * 80)
    print(f"处理完成: {success_count}/{len(image_files)} 张图片成功")
    
    # 统计有昵称的图片数量
    with_nicknames = sum(1 for r in all_results if r['status'] == 'success' and r['nicknames'])
    print(f"成功提取昵称: {with_nicknames}/{len(image_files)} 张图片")
    print(f"\n详细日志保存在: {output_dir}/")
    
    # 按检测方法统计
    print("\n检测方法统计:")
    method_stats = {}
    for result in all_results:
        if result['status'] == 'success':
            for speaker, info in result.get('speaker_info', {}).items():
                method = info['method']
                method_stats[method] = method_stats.get(method, 0) + 1
    
    for method, count in sorted(method_stats.items()):
        print(f"  {method}: {count} 次")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
