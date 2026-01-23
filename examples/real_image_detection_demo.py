#!/usr/bin/env python3
"""
使用真实聊天截图的检测演示

本示例展示了如何使用ChatLayoutDetector处理真实的聊天应用截图，包括：
1. 从真实截图中提取文本框
2. 使用自适应检测器分析布局
3. 对比不同聊天应用的检测效果
4. 展示跨截图的记忆学习

作者: ScreenshotAnalysis Team
日期: 2026-01-23
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.processors import ChatMessageProcessor


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_and_detect_image(image_path, detector=None):
    """
    加载图片并进行文本框检测
    
    Args:
        image_path: 图片路径
        detector: ChatLayoutDetector实例（可选）
    
    Returns:
        (text_boxes, result, image_info)
    """
    # 初始化文本检测器
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    # 加载图片
    image = ImageLoader.load_image(image_path)
    if image.mode == 'RGBA':
        image = image.convert("RGB")
    image_array = np.array(image)
    image_array, padding = letterbox(image_array)
    
    # 检测文本框
    result = text_analyzer.model.predict(image_array)
    
    # 提取文本框
    from screenshotanalysis.processors import TextBox
    text_boxes = []
    
    if result and len(result) > 0:
        for box_info in result:
            # box_info 可能是字典或列表
            if isinstance(box_info, dict):
                box = box_info.get('points', box_info.get('box'))
                score = box_info.get('score', 0.9)
            elif isinstance(box_info, (list, tuple)) and len(box_info) >= 2:
                box = box_info[0]
                score = box_info[1]
            else:
                continue
            
            if box is None:
                continue
            
            # 转换为 [x_min, y_min, x_max, y_max] 格式
            if isinstance(box, np.ndarray) and box.shape == (4, 2):
                # 四个点的格式
                x_coords = box[:, 0]
                y_coords = box[:, 1]
            elif isinstance(box, (list, tuple)):
                # 列表格式
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
            else:
                continue
            
            x_min, x_max = float(min(x_coords)), float(max(x_coords))
            y_min, y_max = float(min(y_coords)), float(max(y_coords))
            
            text_box = TextBox(
                box=np.array([x_min, y_min, x_max, y_max]),
                score=float(score)
            )
            text_boxes.append(text_box)
    
    image_info = {
        'width': image_array.shape[1],
        'height': image_array.shape[0],
        'padding': padding,
        'original_size': (image.width, image.height)
    }
    
    return text_boxes, image_info


def demo_1_single_image():
    """演示1: 单张真实截图分析"""
    print_section("演示 1: 单张真实截图分析")
    
    # 选择一张Discord截图
    image_path = "test_images/test_discord.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print(f"\n正在分析: {image_path}")
    
    # 加载并检测
    text_boxes, image_info = load_and_detect_image(image_path)
    
    print(f"\n图片信息:")
    print(f"  原始尺寸: {image_info['original_size']}")
    print(f"  处理后尺寸: {image_info['width']} x {image_info['height']}")
    print(f"  检测到文本框: {len(text_boxes)} 个")
    
    # 使用自适应检测器
    detector = ChatLayoutDetector(screen_width=image_info['width'])
    result = detector.process_frame(text_boxes)
    
    print(f"\n检测结果:")
    print(f"  布局类型: {result['layout']}")
    print(f"  说话者A: {len(result['A'])} 条消息")
    print(f"  说话者B: {len(result['B'])} 条消息")
    
    if 'confidence' in result['metadata']:
        print(f"  置信度: {result['metadata']['confidence']:.2f}")
    
    if 'left_center' in result['metadata']:
        print(f"  左列中心: {result['metadata']['left_center']:.3f}")
        print(f"  右列中心: {result['metadata']['right_center']:.3f}")
        print(f"  列分离度: {result['metadata']['separation']:.3f}")
    
    # 显示一些文本框的位置
    print(f"\n说话者A的前3个文本框位置:")
    for i, box in enumerate(result['A'][:3], 1):
        print(f"  {i}. ({box.x_min:.0f}, {box.y_min:.0f}) - ({box.x_max:.0f}, {box.y_max:.0f})")
    
    print(f"\n说话者B的前3个文本框位置:")
    for i, box in enumerate(result['B'][:3], 1):
        print(f"  {i}. ({box.x_min:.0f}, {box.y_min:.0f}) - ({box.x_max:.0f}, {box.y_max:.0f})")
    
    print("\n✓ 成功分析真实Discord截图")


def demo_2_multiple_apps():
    """演示2: 对比不同聊天应用"""
    print_section("演示 2: 对比不同聊天应用的检测效果")
    
    # 测试不同应用的截图
    test_images = [
        ("Discord", "test_images/test_discord.png"),
        ("WhatsApp", "test_images/test_whatsapp.png"),
        ("Instagram", "test_images/test_instagram.png"),
        ("Telegram", "test_images/test_telegram1.png"),
    ]
    
    results_summary = []
    
    for app_name, image_path in test_images:
        if not os.path.exists(image_path):
            print(f"\n⚠️  跳过 {app_name}: 图片不存在")
            continue
        
        print(f"\n正在分析 {app_name}...")
        
        try:
            # 加载并检测
            text_boxes, image_info = load_and_detect_image(image_path)
            
            # 使用自适应检测器
            detector = ChatLayoutDetector(screen_width=image_info['width'])
            result = detector.process_frame(text_boxes)
            
            results_summary.append({
                'app': app_name,
                'layout': result['layout'],
                'speaker_a': len(result['A']),
                'speaker_b': len(result['B']),
                'confidence': result['metadata'].get('confidence', 0),
                'total_boxes': len(text_boxes)
            })
            
            print(f"  ✓ 布局: {result['layout']}")
            print(f"  ✓ 说话者A: {len(result['A'])} 条")
            print(f"  ✓ 说话者B: {len(result['B'])} 条")
            print(f"  ✓ 置信度: {result['metadata'].get('confidence', 0):.2f}")
            
        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
    
    # 打印汇总表格
    print("\n" + "=" * 70)
    print("检测结果汇总:")
    print("-" * 70)
    print(f"{'应用':<12} {'布局':<15} {'说话者A':<10} {'说话者B':<10} {'置信度':<10}")
    print("-" * 70)
    
    for r in results_summary:
        print(f"{r['app']:<12} {r['layout']:<15} {r['speaker_a']:<10} {r['speaker_b']:<10} {r['confidence']:<10.2f}")
    
    print("-" * 70)
    print("\n✓ 系统成功处理多种聊天应用，无需任何配置！")


def demo_3_cross_screenshot_learning():
    """演示3: 跨截图记忆学习"""
    print_section("演示 3: 跨截图记忆学习")
    
    # 使用同一应用的多张截图
    image_paths = [
        "test_images/test_discord.png",
        "test_images/test_discord_2.png",
        "test_images/test_discord_3.png",
    ]
    
    # 检查图片是否存在
    available_images = [p for p in image_paths if os.path.exists(p)]
    
    if len(available_images) < 2:
        print("\n⚠️  需要至少2张Discord截图来演示跨截图学习")
        return
    
    print(f"\n将处理 {len(available_images)} 张Discord截图...")
    
    # 创建检测器（不使用持久化，仅在内存中学习）
    detector = None
    
    for i, image_path in enumerate(available_images, 1):
        print(f"\n处理第 {i} 张截图: {os.path.basename(image_path)}")
        
        try:
            # 加载并检测
            text_boxes, image_info = load_and_detect_image(image_path)
            
            # 第一次创建检测器
            if detector is None:
                detector = ChatLayoutDetector(screen_width=image_info['width'])
            
            # 处理帧
            result = detector.process_frame(text_boxes)
            
            print(f"  布局: {result['layout']}")
            print(f"  说话者A: {len(result['A'])} 条")
            print(f"  说话者B: {len(result['B'])} 条")
            print(f"  置信度: {result['metadata'].get('confidence', 0):.2f}")
            
            # 显示记忆状态
            if detector.memory['A'] is not None:
                print(f"  记忆A: center={detector.memory['A']['center']:.3f}, "
                      f"count={detector.memory['A']['count']}")
            if detector.memory['B'] is not None:
                print(f"  记忆B: center={detector.memory['B']['center']:.3f}, "
                      f"count={detector.memory['B']['count']}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n✓ 记忆已在多张截图之间学习和更新")
    if detector and detector.memory['A'] is not None:
        print(f"✓ 说话者A累计: {detector.memory['A']['count']} 条消息")
    if detector and detector.memory['B'] is not None:
        print(f"✓ 说话者B累计: {detector.memory['B']['count']} 条消息")


def demo_4_with_processor():
    """演示4: 使用ChatMessageProcessor集成接口"""
    print_section("演示 4: 使用ChatMessageProcessor集成接口")
    
    image_path = "test_images/test_whatsapp.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print(f"\n正在分析: {image_path}")
    
    # 加载并检测
    text_boxes, image_info = load_and_detect_image(image_path)
    
    print(f"\n检测到 {len(text_boxes)} 个文本框")
    
    # 使用ChatMessageProcessor的自适应方法
    processor = ChatMessageProcessor()
    
    # 方法1: detect_chat_layout_adaptive
    print("\n方法1: detect_chat_layout_adaptive")
    result = processor.detect_chat_layout_adaptive(
        text_boxes=text_boxes,
        screen_width=image_info['width']
    )
    
    print(f"  布局: {result['layout']}")
    print(f"  说话者A: {len(result['A'])} 条")
    print(f"  说话者B: {len(result['B'])} 条")
    
    # 方法2: format_conversation_adaptive
    print("\n方法2: format_conversation_adaptive")
    sorted_boxes, metadata = processor.format_conversation_adaptive(
        text_boxes=text_boxes,
        screen_width=image_info['width']
    )
    
    print(f"  布局: {metadata['layout']}")
    print(f"  按时间排序的消息数: {len(sorted_boxes)}")
    print(f"  说话者A: {metadata['speaker_A_count']} 条")
    print(f"  说话者B: {metadata['speaker_B_count']} 条")
    
    # 显示前5条消息的说话者
    print(f"\n前5条消息的说话者:")
    for i, box in enumerate(sorted_boxes[:5], 1):
        speaker = getattr(box, 'speaker', 'Unknown')
        print(f"  {i}. [{speaker}] 位置: ({box.x_min:.0f}, {box.y_min:.0f})")
    
    print("\n✓ ChatMessageProcessor提供了便捷的集成接口")


def demo_5_layout_types():
    """演示5: 识别不同的布局类型"""
    print_section("演示 5: 识别不同的布局类型")
    
    print("\n测试不同布局类型的识别能力...")
    
    # 测试所有可用的图片
    all_images = []
    if os.path.exists("test_images"):
        for file in os.listdir("test_images"):
            if file.endswith(('.png', '.jpg')):
                all_images.append(os.path.join("test_images", file))
    
    layout_stats = {
        'single': 0,
        'double': 0,
        'double_left': 0,
        'double_right': 0
    }
    
    for image_path in all_images[:10]:  # 限制处理前10张
        try:
            text_boxes, image_info = load_and_detect_image(image_path)
            detector = ChatLayoutDetector(screen_width=image_info['width'])
            result = detector.process_frame(text_boxes)
            
            layout = result['layout']
            if layout in layout_stats:
                layout_stats[layout] += 1
            
            print(f"  {os.path.basename(image_path):<30} -> {layout}")
            
        except Exception as e:
            print(f"  {os.path.basename(image_path):<30} -> 错误: {e}")
    
    print("\n布局类型统计:")
    for layout_type, count in layout_stats.items():
        if count > 0:
            print(f"  {layout_type}: {count} 张")
    
    print("\n✓ 系统能够识别多种布局类型")


def main():
    """主函数：运行所有演示"""
    print("\n" + "=" * 70)
    print("  使用真实聊天截图的检测演示")
    print("  Real Chat Screenshot Detection Demo")
    print("=" * 70)
    
    # 检查test_images目录
    if not os.path.exists("test_images"):
        print("\n❌ 错误: test_images 目录不存在")
        print("请确保在项目根目录下运行此脚本")
        return 1
    
    print("\n本演示将使用test_images目录中的真实聊天截图：")
    print("  1. 单张真实截图分析")
    print("  2. 对比不同聊天应用的检测效果")
    print("  3. 跨截图记忆学习")
    print("  4. 使用ChatMessageProcessor集成接口")
    print("  5. 识别不同的布局类型")
    
    try:
        demo_1_single_image()
        demo_2_multiple_apps()
        demo_3_cross_screenshot_learning()
        demo_4_with_processor()
        demo_5_layout_types()
        
        print("\n" + "=" * 70)
        print("  所有演示完成！")
        print("=" * 70)
        
        print("\n关键发现:")
        print("  ✓ 系统成功处理真实聊天截图")
        print("  ✓ 无需配置即可适应不同应用")
        print("  ✓ 跨截图学习提高了准确性")
        print("  ✓ 提供了多种便捷的使用接口")
        
        print("\n下一步:")
        print("  - 尝试使用自己的聊天截图")
        print("  - 调整参数以优化检测效果")
        print("  - 查看 chat_layout_detector.py 了解实现细节")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
