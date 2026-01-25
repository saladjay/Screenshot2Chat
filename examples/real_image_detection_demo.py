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
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.processors import ChatMessageProcessor


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def map_box_to_original(box, image_info, image):
    left, top, right, bottom = image_info['padding']
    scale_x, scale_y = image_info['scale_ratio']

    x_min_orig = (box.x_min - left) * scale_x
    y_min_orig = (box.y_min - top) * scale_y
    x_max_orig = (box.x_max - left) * scale_x
    y_max_orig = (box.y_max - top) * scale_y

    x_min_orig = max(0, min(x_min_orig, image.width))
    y_min_orig = max(0, min(y_min_orig, image.height))
    x_max_orig = max(0, min(x_max_orig, image.width))
    y_max_orig = max(0, min(y_max_orig, image.height))

    return [float(x_min_orig), float(y_min_orig), float(x_max_orig), float(y_max_orig)]


def build_mapped_boxes(text_boxes, result, image_info, image):
    box_to_speaker = {}
    for box in result['A']:
        box_to_speaker[id(box)] = 'A'
    for box in result['B']:
        box_to_speaker[id(box)] = 'B'

    mapped_boxes = []
    for box in text_boxes:
        speaker = box_to_speaker.get(id(box), 'Unknown')
        mapped_boxes.append({
            'speaker': speaker,
            'coordinates': map_box_to_original(box, image_info, image)
        })
    return mapped_boxes


def save_detection_coords(coords_dir, image_path, mapped_boxes, result, groups_payload=None):
    os.makedirs(coords_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(coords_dir, f"{base_name}.json")
    payload = {
        'image': os.path.basename(image_path),
        'layout': result['layout'],
        'metadata': result.get('metadata', {}),
        'boxes': mapped_boxes,
        'groups': groups_payload
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_against_gt(image_path, mapped_boxes, gt_dir="test_images_answer", iou_threshold=0.5):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gt_path = os.path.join(gt_dir, f"{base_name}.json")
    if not os.path.exists(gt_path):
        return None

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_items = json.load(f)

    gt_boxes = [item['coordinates'] for item in gt_items if 'coordinates' in item]
    det_boxes = [item['coordinates'] for item in mapped_boxes]

    if not gt_boxes:
        return None

    matched_det = set()
    tp = 0
    for gt_box in gt_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, det_box in enumerate(det_boxes):
            if idx in matched_det:
                continue
            iou = compute_iou(gt_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_det.add(best_idx)
            tp += 1

    precision = tp / len(det_boxes) if det_boxes else 0.0
    recall = tp / len(gt_boxes) if gt_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'gt_count': len(gt_boxes),
        'det_count': len(det_boxes),
        'true_positive': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_path': gt_path
    }


def draw_detection_result(original_image, text_boxes, result, output_path, image_info, image_path, coords_dir):
    """
    在原始图片上绘制检测结果并保存
    
    Args:
        original_image: 原始PIL图片对象
        text_boxes: 所有文本框列表（letterbox处理后的坐标）
        result: ChatLayoutDetector的检测结果
        output_path: 输出图片路径
        image_info: 图片信息，包含padding和scale_ratio
    """
    # 复制图片以避免修改原图
    image = original_image.copy()
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 定义颜色 - 使用更鲜艳的颜色
    color_a = (255, 50, 50)      # 红色 - 说话者A
    color_b = (50, 150, 255)     # 蓝色 - 说话者B
    color_unknown = (150, 150, 150)  # 灰色 - 未分类
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            # Windows系统字体
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    mapped_boxes = build_mapped_boxes(text_boxes, result, image_info, image)
    grouped_boxes = None
    groups_payload = None
    if 'A' in result and 'B' in result:
        processor = ChatMessageProcessor()
        grouped_boxes = processor.group_text_boxes_by_y(result['A'] + result['B'])
    if grouped_boxes:
        groups_payload = [
            [map_box_to_original(box, image_info, image) for box in group]
            for group in grouped_boxes
        ]
    
    # 绘制每个文本框
    drawn_count = {'A': 0, 'B': 0, 'Unknown': 0}
    for item in mapped_boxes:
        speaker = item['speaker']
        drawn_count[speaker] += 1
        
        # 选择颜色
        if speaker == 'A':
            color = color_a
        elif speaker == 'B':
            color = color_b
        else:
            color = color_unknown
        
        x_min_orig, y_min_orig, x_max_orig, y_max_orig = item['coordinates']
        
        # 绘制矩形框 - 使用更粗的线条
        # 绘制填充的半透明矩形
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x_min_orig, y_min_orig, x_max_orig, y_max_orig], 
                              fill=(*color, 30),  # 半透明填充
                              outline=color, 
                              width=4)
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # 绘制说话者标签
        label = f"{speaker}"
        # 计算标签位置和大小
        label_y = max(0, y_min_orig - 25)
        try:
            text_bbox = draw.textbbox((x_min_orig, label_y), label, font=font_small)
        except:
            # 如果textbbox不可用，使用textsize
            text_width, text_height = draw.textsize(label, font=font_small)
            text_bbox = (x_min_orig, label_y, x_min_orig + text_width, label_y + text_height)
        
        # 绘制标签背景
        padding = 4
        draw.rectangle([text_bbox[0] - padding, text_bbox[1] - padding, 
                       text_bbox[2] + padding, text_bbox[3] + padding], 
                      fill=color)
        # 绘制标签文字
        draw.text((x_min_orig, label_y), label, fill=(255, 255, 255), font=font_small)
    
    # 在图片顶部绘制检测信息
    info_lines = [
        f"Layout: {result['layout']}",
        f"Speaker A (红色): {len(result['A'])} boxes",
        f"Speaker B (蓝色): {len(result['B'])} boxes"
    ]
    
    if 'confidence' in result['metadata']:
        info_lines.append(f"Confidence: {result['metadata']['confidence']:.2f}")
    
    # 绘制信息背景和文字
    y_offset = 10
    for line in info_lines:
        try:
            text_bbox = draw.textbbox((10, y_offset), line, font=font_small)
        except:
            text_width, text_height = draw.textsize(line, font=font_small)
            text_bbox = (10, y_offset, 10 + text_width, y_offset + text_height)
        
        # 绘制半透明黑色背景
        padding = 5
        draw.rectangle([text_bbox[0] - padding, text_bbox[1] - padding, 
                       text_bbox[2] + padding, text_bbox[3] + padding], 
                      fill=(0, 0, 0, 200))
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font_small)
        y_offset += 30
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"  已保存检测结果到: {output_path}")
    print(f"    绘制了 {drawn_count['A']} 个A气泡, {drawn_count['B']} 个B气泡, {drawn_count['Unknown']} 个未分类气泡")
    coords_path = save_detection_coords(coords_dir, image_path, mapped_boxes, result, groups_payload)
    print(f"    坐标已保存到: {coords_path}")
    return mapped_boxes



def load_and_detect_image(image_path, detector=None):
    """
    加载图片并进行文本框检测
    
    Args:
        image_path: 图片路径
        detector: ChatLayoutDetector实例（可选）
    
    Returns:
        (text_boxes, image_info, original_image)
    """
    # 初始化文本检测器
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    # 加载原始图片
    original_image = ImageLoader.load_image(image_path)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert("RGB")
    
    # 转换为numpy数组并进行letterbox处理
    image_array = np.array(original_image)
    processed_image, padding = letterbox(image_array)
    
    # 检测文本框
    result = text_analyzer.model.predict(processed_image)
    
    # 提取文本框 - 使用正确的格式
    from screenshotanalysis.processors import TextBox
    text_boxes = []
    
    if result and len(result) > 0:
        for element in result:
            # 文本检测模型返回的格式包含 dt_polys 和 dt_scores
            if 'dt_polys' in element and 'dt_scores' in element:
                for i, box in enumerate(element['dt_polys']):
                    points = [box[0], box[1], box[2], box[3]]
                    min_x = min([p[0] for p in points])
                    max_x = max([p[0] for p in points])
                    min_y = min([p[1] for p in points])
                    max_y = max([p[1] for p in points])
                    
                    text_box = TextBox(
                        box=np.array([min_x, min_y, max_x, max_y]),
                        score=float(element['dt_scores'][i])
                    )
                    text_boxes.append(text_box)
    
    image_info = {
        'width': processed_image.shape[1],
        'height': processed_image.shape[0],
        'padding': padding,
        'original_size': (original_image.width, original_image.height),
        'scale_ratio': None  # 将在下面计算
    }
    
    # 计算缩放比例
    left, top, right, bottom = padding
    processed_h, processed_w = processed_image.shape[:2]
    original_h, original_w = original_image.height, original_image.width
    
    # 实际内容区域（去除padding）
    content_w = processed_w - left - right
    content_h = processed_h - top - bottom
    
    # 缩放比例
    scale_x = original_w / content_w if content_w > 0 else 1.0
    scale_y = original_h / content_h if content_h > 0 else 1.0
    
    image_info['scale_ratio'] = (scale_x, scale_y)
    
    return text_boxes, image_info, original_image


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
    text_boxes, image_info, original_image = load_and_detect_image(image_path)
    
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
    
    # 绘制并保存结果
    output_path = f"test_new_way_det/{os.path.basename(image_path)}"
    mapped_boxes = draw_detection_result(
        original_image,
        text_boxes,
        result,
        output_path,
        image_info,
        image_path,
        coords_dir="test_new_way_coords"
    )
    metrics = evaluate_against_gt(image_path, mapped_boxes)
    if metrics:
        print(
            f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
            f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
        )
    
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
            text_boxes, image_info, original_image = load_and_detect_image(image_path)
            
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
            
            # 绘制并保存结果
            output_path = f"test_new_way_det/{os.path.basename(image_path)}"
            mapped_boxes = draw_detection_result(
                original_image,
                text_boxes,
                result,
                output_path,
                image_info,
                image_path,
                coords_dir="test_new_way_coords"
            )
            metrics = evaluate_against_gt(image_path, mapped_boxes)
            if metrics:
                print(
                    f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
                    f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
                )
            
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
            text_boxes, image_info, original_image = load_and_detect_image(image_path)
            
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
            
            # 绘制并保存结果
            output_path = f"test_new_way_det/{os.path.basename(image_path)}"
            mapped_boxes = draw_detection_result(
                original_image,
                text_boxes,
                result,
                output_path,
                image_info,
                image_path,
                coords_dir="test_new_way_coords"
            )
            metrics = evaluate_against_gt(image_path, mapped_boxes)
            if metrics:
                print(
                    f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
                    f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
                )
            
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
    text_boxes, image_info, original_image = load_and_detect_image(image_path)
    
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

    output_path = f"test_new_way_det/{os.path.basename(image_path)}"
    mapped_boxes = draw_detection_result(
        original_image,
        text_boxes,
        result,
        output_path,
        image_info,
        image_path,
        coords_dir="test_new_way_coords"
    )
    metrics = evaluate_against_gt(image_path, mapped_boxes)
    if metrics:
        print(
            f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
            f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
        )
    
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
            text_boxes, image_info, original_image = load_and_detect_image(image_path)
            detector = ChatLayoutDetector(screen_width=image_info['width'])
            result = detector.process_frame(text_boxes)
            
            layout = result['layout']
            if layout in layout_stats:
                layout_stats[layout] += 1
            
            print(f"  {os.path.basename(image_path):<30} -> {layout}")
            
            # 绘制并保存结果
            output_path = f"test_new_way_det/{os.path.basename(image_path)}"
            mapped_boxes = draw_detection_result(
                original_image,
                text_boxes,
                result,
                output_path,
                image_info,
                image_path,
                coords_dir="test_new_way_coords"
            )
            metrics = evaluate_against_gt(image_path, mapped_boxes)
            if metrics:
                print(
                    f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
                    f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
                )
            
        except Exception as e:
            print(f"  {os.path.basename(image_path):<30} -> 错误: {e}")
    
    print("\n布局类型统计:")
    for layout_type, count in layout_stats.items():
        if count > 0:
            print(f"  {layout_type}: {count} 张")
    
    print("\n✓ 系统能够识别多种布局类型")
    print(f"✓ 所有检测结果已保存到 test_new_way_det/ 文件夹")


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
    
    # 创建输出目录
    os.makedirs("test_new_way_det", exist_ok=True)
    os.makedirs("test_new_way_coords", exist_ok=True)
    print("\n检测结果将保存到: test_new_way_det/")
    print("坐标结果将保存到: test_new_way_coords/")
    
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
        print("  ✓ 所有检测结果已保存到 test_new_way_det/ 文件夹")
        
        print("\n下一步:")
        print("  - 查看 test_new_way_det/ 文件夹中的可视化结果")
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
