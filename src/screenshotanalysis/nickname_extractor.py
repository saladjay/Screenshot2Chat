"""
智能昵称提取模块

本模块提供基于综合评分系统的昵称检测功能，使用以下策略：
1. 过滤极端边缘的小框（系统UI区域）
2. 基于位置优先级：优先选择靠近屏幕中心的文本框
3. 综合评分系统：位置 + 尺寸 + 文本特征 + Y排名
4. Y-rank评分：第1名20分，第2名15分，第3名10分

主要功能：
- extract_nicknames_smart: 智能提取昵称的主函数
- draw_top3_results: 可视化前三名候选结果
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from screenshotanalysis.utils import ImageLoader, letterbox


logger = logging.getLogger(__name__)


def draw_top3_results(
    image_path: str,
    top_candidates: List[Dict[str, Any]],
    output_dir: str = "test_output/smart_nicknames"
) -> str:
    """
    绘制得分前三的候选框到图片上
    
    Args:
        image_path: 原始图片路径
        top_candidates: 前三名候选者列表，每个元素包含text、nickname_score、box、y_rank等字段
        output_dir: 输出目录
    
    Returns:
        输出图片的路径
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
        
        cv2.rectangle(
            draw_image,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 10, label_y + 5),
            color, -1
        )
        
        # 绘制标签文字
        cv2.putText(
            draw_image, label,
            (x_min + 5, label_y),
            font, font_scale, (255, 255, 255), thickness
        )
        
        # 在框内绘制Y排名
        rank_label = f"Y-Rank: {y_rank}"
        cv2.putText(
            draw_image, rank_label,
            (x_min + 5, y_min + 20),
            font, 0.5, color, 2
        )
    
    # 保存图片
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"top3_{filename}")
    cv2.imwrite(output_path, draw_image)
    
    logger.info(f"结果已保存到: {output_path}")
    return output_path


def extract_nicknames_smart(
    image_path: str,
    text_analyzer,
    processor,
    text_rec=None,
    draw_results: bool = False,
    output_dir: str = "test_output/smart_nicknames",
    min_top_margin_ratio: float = 0.05
) -> List[Dict[str, Any]]:
    """
    智能提取昵称（基于综合评分）- 使用新的Y-rank评分系统
    
    优化说明：
    1. 减少模型调用：复用传入的text_rec模型，避免重复加载
    2. 统一日志：使用全局logger而不是print
    
    Args:
        image_path: 图片路径
        text_analyzer: 文本检测分析器（ChatLayoutAnalyzer实例）
        processor: 消息处理器（ChatMessageProcessor实例）
        text_rec: 可选的文本识别模型，如果为None则创建新实例
        draw_results: 是否绘制可视化结果
        output_dir: 输出目录
    
    Returns:
        前三名候选者列表，每个元素包含：
        - text: 识别的文本
        - ocr_score: OCR置信度
        - nickname_score: 昵称综合得分
        - score_breakdown: 得分细项
        - box: 边界框坐标
        - center_x: 中心X坐标
        - y_min: 顶部Y坐标
        - y_rank: Y方向排名
    """
    # 加载图片
    if not isinstance(image_path, np.ndarray):
        original_image = ImageLoader.load_image(image_path)
        if original_image.mode == 'RGBA':
            original_image = original_image.convert("RGB")
    
        image_array = np.array(original_image)
    else:
        image_array = image_path
    processed_image, padding = letterbox(image_array)
    
    # 进行文本检测（只调用一次）
    text_det_results = text_analyzer.model.predict(processed_image)
    
    # 获取所有文本框
    text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results)
    
    screen_width = processed_image.shape[1]
    screen_height = processed_image.shape[0]
    
    logger.info(f"{'='*80}")
    logger.info(f"图片: {os.path.basename(image_path)}")
    logger.info(f"屏幕尺寸: {screen_width}x{screen_height}")
    logger.info(f"检测到 {len(text_det_boxes)} 个文本框")
    
    # 过滤极端边缘框
    filtered_boxes = []
    edge_boxes = []
    
    for box in text_det_boxes:
        if processor._is_extreme_edge_box(box, screen_width, screen_height):
            edge_boxes.append(box)
        else:
            filtered_boxes.append(box)
    
    logger.info(f"过滤掉 {len(edge_boxes)} 个极端边缘框")
    logger.info(f"保留 {len(filtered_boxes)} 个候选框")
    
    # 只处理顶部区域的框（前20%），并排除最顶部边缘区域
    top_region_boundary = screen_height * 0.20
    min_top_margin = screen_height * min_top_margin_ratio
    top_boxes = [
        box for box in filtered_boxes
        if min_top_margin <= box.y_min < top_region_boundary
    ]
    
    logger.info(f"顶部区域候选框: {len(top_boxes)} 个")
    
    if not top_boxes:
        logger.warning("没有找到候选框")
        logger.info(f"{'='*80}")
        return []
    
    # 按Y位置排序以计算排名
    sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)
    box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}
    
    # 初始化OCR（如果未提供则创建新实例）
    if text_rec is None:
        from screenshotanalysis.core import ChatTextRecognition
        text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
        text_rec.load_model()
        logger.info("创建新的文本识别模型实例")
    else:
        logger.info("复用已有的文本识别模型实例")
    
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
            logger.warning(f"OCR处理失败: {e}")
            continue
    
    # 按得分排序，选择前3个
    candidates.sort(key=lambda x: x['nickname_score'], reverse=True)
    top_candidates = candidates[:3]
    
    logger.info(f"\n最终选择（按得分排序）:")
    for i, c in enumerate(top_candidates, 1):
        breakdown_str = ', '.join([f"{k}={v:.1f}" for k, v in c['score_breakdown'].items()])
        logger.info(f"  {i}. '{c['text']}' (得分: {c['nickname_score']:.1f}/100, Y排名: {c.get('y_rank', 'N/A')})")
        logger.info(f"     细项: {breakdown_str}")
    
    logger.info(f"{'='*80}")
    
    # 绘制结果
    if draw_results and top_candidates:
        output_path = draw_top3_results(image_path, top_candidates, output_dir)
        logger.info(f"可视化结果已保存到: {output_path}")
    
    return top_candidates


def extract_nicknames_from_text_boxes(
    text_boxes: List,
    image: np.ndarray,
    processor,
    text_rec=None,
    ocr_reader=None,
    draw_results: bool = False,
    output_dir: str = "test_output/smart_nicknames",
    top_k: int = 3,
    image_path: str | None = None,
    min_top_margin_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Extract nickname candidates from precomputed text boxes.

    Args:
        text_boxes: List of TextBox objects from text_det.
        image: Letterboxed image array.
        processor: ChatMessageProcessor instance.
        text_rec: Optional OCR model (required if ocr_reader is None).
        ocr_reader: Optional callable returning (text, score) for a TextBox.
        draw_results: Whether to draw visualization results.
        output_dir: Output directory for visualization.
        top_k: Number of top candidates to return.
    """
    if not text_boxes:
        return []

    screen_height, screen_width = image.shape[:2]

    if ocr_reader is None:
        if text_rec is None:
            from screenshotanalysis.core import ChatTextRecognition

            text_rec = ChatTextRecognition(model_name="PP-OCRv5_server_rec", lang="en")
            text_rec.load_model()

        def ocr_reader(box):
            x_min, y_min, x_max, y_max = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            if x_max <= x_min or y_max <= y_min:
                return "", 0.0
            cropped_image = image[y_min:y_max, x_min:x_max]
            ocr_result = text_rec.predict_text(cropped_image)
            if not ocr_result:
                return "", 0.0
            first_result = ocr_result[0]
            if isinstance(first_result, dict):
                return first_result.get("rec_text", ""), first_result.get("rec_score", 0.0)
            if isinstance(first_result, tuple):
                return first_result[0], first_result[1] if len(first_result) > 1 else 0.0
            return str(first_result), 0.0

    filtered_boxes = []
    edge_boxes = []
    for box in text_boxes:
        if processor._is_extreme_edge_box(box, screen_width, screen_height):
            edge_boxes.append(box)
        else:
            filtered_boxes.append(box)

    top_region_boundary = screen_height * 0.20
    min_top_margin = screen_height * min_top_margin_ratio
    top_boxes = [
        box for box in filtered_boxes
        if min_top_margin <= box.y_min < top_region_boundary
    ]

    if not top_boxes:
        return []

    sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)
    box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}

    candidates = []
    for box in top_boxes:
        text, ocr_score = ocr_reader(box)
        cleaned_text = text.rstrip(">< |\t\n\r")
        if not cleaned_text:
            continue
        y_rank = box_to_rank.get(id(box), None)
        nickname_score, score_breakdown = processor._calculate_nickname_score(
            box, cleaned_text, screen_width, screen_height, y_rank=y_rank
        )
        candidates.append(
            {
                "text": cleaned_text,
                "ocr_score": ocr_score,
                "nickname_score": nickname_score,
                "score_breakdown": score_breakdown,
                "box": box.box.tolist(),
                "center_x": box.center_x,
                "y_min": box.y_min,
                "y_rank": y_rank,
            }
        )

    candidates.sort(key=lambda x: x["nickname_score"], reverse=True)
    top_candidates = candidates[:top_k]

    if draw_results and top_candidates and image_path:
        output_path = draw_top3_results(image_path, top_candidates, output_dir)
        logger.info(f"可视化结果已保存到: {output_path}")

    return top_candidates
