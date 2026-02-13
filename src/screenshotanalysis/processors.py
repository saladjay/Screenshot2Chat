import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List
import cv2
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from copy import deepcopy
from screenshotanalysis.experience_formula import SpeakerPositionKMeans, concat_data
from screenshotanalysis.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.basemodel import TextBox, OTHER, UNKNOWN, USER

NAME_LINE = 'name_line' # 用户名字
MULTI_LINE = 'multi_line' # 多行聊天框
SINGLE_LINE = 'single_line' # 单行聊天框

LAYOUT_DET = 'layout_det'
TEXT_DET = 'text_det'
MAGIC_MARGIN = 5
MAGIC_MARGIN_PERCENTAGE = 0.01

MAIN_HEIGHT_TOLERANCE = 0.25
MAIN_AREA_TOLERANCE = 0.2


class ChatMessageProcessor:
    """聊天消息后处理器"""
    
    def __init__(self, model_name=None):
        self.message_templates = {
            'text': '文本消息',
            'image': '图片消息', 
            'table': '表格分享',
            'formula': '公式消息',
            'chart': '图表分享'
        }
        self.model_name = model_name

    def draw_all_text_boxes(self, image, results, save_path, enable_log=False, model_name=None):
        if model_name:
            self.model_name = model_name
        if self.model_name in ['PP-DocLayoutV2', 'PP-DocLayout-L']:
            self._layout_det_draw_all_text_boxes(image, results, save_path, enable_log)
        else:
            self._text_det_draw_all_text_boxes(image, results, save_path, enable_log)

    def _text_det_draw_all_text_boxes(self, image, results, save_path, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')
        
        for element in results:
            for i, box in enumerate(element['dt_polys']):
                points = [box[0], box[1], box[2], box[3]]
                min_x = min([p[0] for p in points])
                max_x = max([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_y = max([p[1] for p in points])
                boxes = [int(min_x), int(min_y), int(max_x), int(max_y)]
                # print(f'{file} - {box["label"]}: {box["score"]} at {box["coordinate"]}', file=f)
                
                image = cv2.rectangle(image, 
                                (boxes[0], boxes[1]), 
                                (boxes[2], boxes[3]), 
                                (0, 255, 255), 2)
                image = cv2.putText(image, 
                                f"{element['dt_scores'][i]:.2f}", 
                                (boxes[0], boxes[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 0, 0), 2)
                if log_file:
                    
                    boxes = list(map(str, boxes))
                    print(f"{' '.join(boxes)} text {int(boxes[3]) - int(boxes[1])}", file=log_file)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()
    
    def _get_all_text_boxes_from_text_det(self, text_det_results):
        text_boxes = []
        for element in text_det_results:
            for i, box in enumerate(element['dt_polys']):
                points = [box[0], box[1], box[2], box[3]]
                min_x = min([p[0] for p in points])
                max_x = max([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_y = max([p[1] for p in points])
                text_box = TextBox(box=[min_x, min_y, max_x, max_y], score=element['dt_scores'][i], source=TEXT_DET, layout_det='text')
                text_boxes.append(text_box)
        return text_boxes

    def _layout_det_draw_all_text_boxes(self, image, results, save_path, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')
        
        for element in results:
            for box in element['boxes']:
                assert 'label' in box
                assert 'score' in box
                assert 'coordinate' in box
                # print(f'{file} - {box["label"]}: {box["score"]} at {box["coordinate"]}', file=f)
                if box['label'] == 'text':
                    image = cv2.rectangle(image, 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1])), 
                                    (int(box['coordinate'][2]), int(box['coordinate'][3])), 
                                    (0, 255, 255), 2)
                    image = cv2.putText(image, 
                                    f"{box['label']}:{box['score']:.2f}", 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (255, 0, 0), 2)
                    if log_file:
                        boxes = [int(box['coordinate'][0]), int(box['coordinate'][1]), int(box['coordinate'][2]), int(box['coordinate'][3])]
                        boxes = list(map(str, boxes))
                        print(f"{' '.join(boxes)} {box['label']}", file=log_file)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()
    
    def _get_all_boxes_from_layout_det(self, layout_det_results, special_types:list=None, excluded_types:list=None):
        text_boxes = []
        for element in layout_det_results:
            for box in element['boxes']:
                if special_types is not None:
                    if box['label'] not in special_types:
                        continue
                if excluded_types is not None:
                    if box['label'] in excluded_types:
                        continue
                p1 = [int(box['coordinate'][0]), int(box['coordinate'][1])]
                p2 = [int(box['coordinate'][2]), int(box['coordinate'][3])]
                min_x, min_y, max_x, max_y = min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]), 
                text_box = TextBox(box=[min_x, min_y, max_x, max_y], score=box['score'], source=LAYOUT_DET, layout_det=box['label'])
                text_boxes.append(text_box)
        return text_boxes

    def draw_all_image_boxes(self, image, results, save_path, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')
        
        for element in results:
            for box in element['boxes']:
                assert 'label' in box
                assert 'score' in box
                assert 'coordinate' in box
                if box['label'] == 'image':
                    image = cv2.rectangle(image, 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1])), 
                                    (int(box['coordinate'][2]), int(box['coordinate'][3])), 
                                    (0, 255, 255), 2)
                    image = cv2.putText(image, 
                                    f"{box['label']}:{box['score']:.2f}", 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (255, 0, 0), 2)
                    if log_file:
                        boxes = [int(box['coordinate'][0]), int(box['coordinate'][1]), int(box['coordinate'][2]), int(box['coordinate'][3])]
                        boxes = list(map(str, boxes))
                        print(f"{' '.join(boxes)} {box['label']}", file=log_file)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()

    def _get_all_avatar_boxes_from_layout_det(self, results, log_file=None):
        # find all image boxes
        image_boxes = []
        for element in results:
            for box in element['boxes']:
                assert 'label' in box
                assert 'score' in box
                assert 'coordinate' in box
                # print(f'{file} - {box["label"]}: {box["score"]} at {box["coordinate"]}', file=f)
                if box['label'] == 'image':
                    image_box = [int(box['coordinate'][0]), int(box['coordinate'][1]),int(box['coordinate'][2]), int(box['coordinate'][3])]
                    image_boxes.append(TextBox(box=image_box, score=box['score']))
                    if log_file:
                        print(f'{image_box}  {image_boxes[-1].width * image_boxes[-1].height}', file=log_file)
        main_area_value = self.estimate_main_box_area(image_boxes, bin_size=100, log_file=log_file)
        # print(f'main_area:{main_area_value}', file=log_file)
        filtered_image_boxes = []
        for box in image_boxes:
            if self.filter_by_image_avatar_area(main_area_value, box):
                box.layout_det = 'avatar'
                filtered_image_boxes.append(box)
        return filtered_image_boxes

    def draw_all_avatar_boxes(self, image, results, save_path, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')

        for box in self._get_all_avatar_boxes_from_layout_det(results, log_file):
            image = cv2.rectangle(image, 
                                (box.x_min, box.y_min), 
                                (box.x_max, box.y_max), 
                                (0, 255, 255), 2)
            image = cv2.putText(image, 
                                f"{box.score:.2f}", 
                                (box.x_min, box.y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 0, 0), 2)

        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()

    def draw_all_text_info_boxes(self, image, results, save_path, padding, image_sizes, ratios, app_type, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')
        text_det_text_boxes = self._get_all_text_boxes_from_text_det(results)
        sorted_text_det_text_boxes = self.sort_boxes_by_y(text_det_text_boxes)
        if log_file:
            for b in sorted_text_det_text_boxes:
                print(b.box.tolist(), file=log_file)
        filtered_text_boxes = self.filter_by_min_x_and_max_x_and_main_height(sorted_text_det_text_boxes, padding, image_sizes, ratios, app_type, log_file)
        for box in filtered_text_boxes:
            image = cv2.rectangle(image, 
                                (box.x_min, box.y_min), 
                                (box.x_max, box.y_max), 
                                (0, 255, 255), 2)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()

    def draw_all_other_boxes(self, image, results, save_path, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')
        
        for element in results:
            for box in element['boxes']:
                assert 'label' in box
                assert 'score' in box
                assert 'coordinate' in box
                # print(f'{file} - {box["label"]}: {box["score"]} at {box["coordinate"]}', file=f)
                if box['label'] not in ['image', 'text']:
                    image = cv2.rectangle(image, 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1])), 
                                    (int(box['coordinate'][2]), int(box['coordinate'][3])), 
                                    (0, 255, 255), 2)
                    image = cv2.putText(image, 
                                    f"{box['label']}:{box['score']:.2f}", 
                                    (int(box['coordinate'][0]), int(box['coordinate'][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (255, 0, 0), 2)
                    if log_file:
                        boxes = [int(box['coordinate'][0]), int(box['coordinate'][1]), int(box['coordinate'][2]), int(box['coordinate'][3])]
                        boxes = list(map(str, boxes))
                        print(f"{' '.join(boxes)} {box['label']}", file=log_file)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()

    def get_nickname_box_from_text_det_boxes(self, results, padding, image_sizes, ratios, app_type, log_file=None):
        text_det_text_boxes = self._get_all_text_boxes_from_text_det(results)
        sorted_text_det_text_boxes = self.sort_boxes_by_y(text_det_text_boxes)
        filtered_text_boxes = self.filter_by_rules_to_find_nickname(sorted_text_det_text_boxes, padding, image_sizes, ratios, app_type, log_file)

        main_height = self.estimate_main_text_height(text_det_text_boxes, bin_size=2, log_file=log_file)
        height_lower = main_height * (1 - MAIN_HEIGHT_TOLERANCE)
        height_upper = main_height * (1 + MAIN_HEIGHT_TOLERANCE)
        if log_file:
            print(f'main height:{main_height}', file=log_file)
            print(f'height range:{height_lower} - {height_upper}', file=log_file)
        main_min_x = self.estimate_main_value(text_det_text_boxes, 'x_min', bin_size=4, log_file=log_file)
        minimum_offset = main_min_x + 1.1 * MAGIC_MARGIN
        if log_file:
            print(f'main min x:{main_min_x}', file=log_file)
            print(f'the minimun left offset of nickname:{minimum_offset}', file=log_file)
        nickname_box = None
        for text_box in filtered_text_boxes:
            if height_lower <= text_box.height:
                if text_box.x_min > minimum_offset:
                    nickname_box = deepcopy(text_box)
                    break
        return nickname_box

    def draw_all_nickname_from_det_boxes(self, image, results, save_path,  padding, image_sizes, ratios, app_type, enable_log=False):
        log_file = None
        if enable_log:
            log_file = open(save_path+'.txt', 'w', encoding='utf-8')

        nickname_box = self.get_nickname_box_from_text_det_boxes(results, padding, image_sizes, ratios, app_type, log_file)
        if nickname_box:
            image = cv2.rectangle(image, 
                                    (nickname_box.x_min, nickname_box.y_min), 
                                    (nickname_box.x_max, nickname_box.y_max), 
                                    (0, 255, 255), 2)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if log_file:
            log_file.close()

    def _boxes_coverage(self, boxes1, boxes2):
        """
        计算boxes2与boxes1的覆盖率 = 交集面积 / boxes1面积
        用于评估boxes2覆盖boxes1的程度
        
        Args:
            boxes1: [n, 4] 基准矩形框集合 (通常是需要被覆盖的框)
            boxes2: [m, 4] 覆盖矩形框集合
        
        Returns:
            coverage_matrix: [n, m] 覆盖率矩阵
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        # 计算每个边界框的面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (n,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (m,)
        
        # 使用广播机制计算交集区域的坐标
        # 增加维度以便广播计算: (n,1,2) 和 (m,2) -> (n,m,2)
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # 左上角交点 (n,m,2)
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角交点 (n,m,2)
        
        # 计算交集区域的宽高并确保非负
        wh = np.maximum(rb - lt, 0)  # (n,m,2)
        
        # 计算交集面积
        inter = wh[:, :, 0] * wh[:, :, 1]  # (n,m)
        
        # 计算覆盖率 = 交集面积 / boxes1面积
        coverage_matrix = inter / np.maximum(area1[:, None], 1e-8)  # (n,m)
        
        return coverage_matrix
    
    def filter_text_boxes_by_layout_det(self, text_det_boxes: List[TextBox], 
                                       layout_det_boxes: List[TextBox],
                                       coverage_threshold: float = 0.2,
                                       screen_width: int = None,
                                       log_file=None) -> List[TextBox]:
        """
        使用layout_det的结果筛选text_det的文本框，并根据位置信息赋值所属对象
        
        Args:
            text_det_boxes: text_det检测到的文本框列表
            layout_det_boxes: layout_det检测到的文本框列表（应该只包含'text'类型）
            coverage_threshold: 覆盖率阈值，text_det框与layout_det框的覆盖率超过此值才保留
            screen_width: 屏幕宽度，用于判断左右位置
            log_file: 日志文件对象
            
        Returns:
            筛选后的text_det文本框列表，每个框都被赋予了speaker属性（'A'或'B'）
        """
        if not text_det_boxes or not layout_det_boxes:
            return []
        
        # 转换为numpy数组用于计算覆盖率
        text_det_boxes_np = np.array([box.box for box in text_det_boxes])
        layout_det_boxes_np = np.array([box.box for box in layout_det_boxes])
        
        # 计算覆盖率矩阵
        coverage_matrix = self._boxes_coverage(text_det_boxes_np, layout_det_boxes_np)
        
        if log_file:
            print(f'text_det shape {text_det_boxes_np.shape} layout_det shape {layout_det_boxes_np.shape} coverage matrix {coverage_matrix.shape}', file=log_file)
        
        # 筛选有效的text_det框
        filtered_boxes = []
        for i, text_box in enumerate(text_det_boxes):
            # 找到与当前text_det框覆盖率最高的layout_det框
            max_coverage_idx = coverage_matrix[i].argmax()
            max_coverage = coverage_matrix[i, max_coverage_idx]
            
            # 如果覆盖率超过阈值，保留这个框
            if max_coverage > coverage_threshold:
                # 获取对应的layout_det框
                matched_layout_box = layout_det_boxes[max_coverage_idx]
                
                # 复制text_box以避免修改原始对象
                filtered_box = text_box
                
                # 从layout_det框继承位置信息
                if hasattr(matched_layout_box, 'speaker'):
                    filtered_box.speaker = matched_layout_box.speaker
                
                # 如果layout_det框有layout_det属性，也继承过来
                if hasattr(matched_layout_box, 'layout_det'):
                    filtered_box.layout_det = matched_layout_box.layout_det
                
                filtered_boxes.append(filtered_box)
                
                if log_file:
                    print(f'text_det box {text_box.box.tolist()} matched layout_det box {matched_layout_box.box.tolist()} '
                          f'with coverage {max_coverage:.3f}, speaker: {getattr(filtered_box, "speaker", "Unknown")}', 
                          file=log_file)
            else:
                if log_file:
                    print(f'text_det box {text_box.box.tolist()} filtered out (max coverage {max_coverage:.3f} < {coverage_threshold})', 
                          file=log_file)
        
        return filtered_boxes
    
    def assign_speakers_to_layout_det_boxes(self, layout_det_boxes: List[TextBox],
                                           screen_width: int,
                                           memory_path: str = None,
                                           log_file=None) -> List[TextBox]:
        """
        使用ChatLayoutDetector为layout_det的文本框分配说话者
        
        Args:
            layout_det_boxes: layout_det检测到的文本框列表
            screen_width: 屏幕宽度
            memory_path: 可选的记忆持久化路径
            log_file: 日志文件对象
            
        Returns:
            带有speaker属性的文本框列表
        """
        if not layout_det_boxes:
            return []
        
        # 使用ChatLayoutDetector进行说话者分配
        detector = ChatLayoutDetector(screen_width=screen_width, memory_path=memory_path)
        result = detector.process_frame(layout_det_boxes)
        
        # 为每个框添加speaker属性
        a_ids = {id(box) for box in result['A']}
        b_ids = {id(box) for box in result['B']}
        
        for box in layout_det_boxes:
            if id(box) in a_ids:
                box.speaker = OTHER
            elif id(box) in b_ids:
                box.speaker = USER
            else:
                box.speaker = UNKNOWN
        
        if log_file:
            print(f'Assigned speakers: A={len(result["A"])}, B={len(result["B"])}', file=log_file)
            print(f'Layout type: {result["layout"]}', file=log_file)
        
        return layout_det_boxes

    def _boxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        # 计算每个边界框的面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (n,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (m,)
        
        # 使用广播机制计算交集区域的坐标
        # 增加维度以便广播计算: (n,1,2) 和 (m,2) -> (n,m,2)
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # 左上角交点 (n,m,2)
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角交点 (n,m,2)
        
        # 计算交集区域的宽高并确保非负
        wh = np.maximum(rb - lt, 0)  # (n,m,2)
        
        # 计算交集面积
        inter = wh[:, :, 0] * wh[:, :, 1]  # (n,m)
        
        # 计算并集面积
        union = area1[:, None] + area2 - inter   # (n,m)
        
        # 计算IoU，避免除以0
        iou_matrix = inter / np.maximum(union, 1e-8)  # (n,m)
        
        return iou_matrix

    def group_chat_message(self):
        pass

    def format_conversation(self, layout_det_results, text_det_results, padding, image_sizes, ratios=None, app_type=None, log_file=None, use_adaptive=False, screen_width=None):
        """
        格式化对话内容
        
        Args:
            layout_det_results: layout检测结果
            text_det_results: text检测结果
            padding: 图片padding信息
            image_sizes: 图片尺寸
            ratios: 比例信息（如果为None则返回原始boxes用于计算ratios）
            app_type: 应用类型（DISCORD等）
            log_file: 日志文件
            use_adaptive: 是否使用自适应检测器（新方法）
            screen_width: 屏幕宽度（use_adaptive=True时需要）
            
        Returns:
            如果ratios为None: 返回(layout_det_boxes_np, text_det_boxes_np)
            否则: 返回(sorted_boxes, original_text_boxes)
        """
        layout_det_text_boxes = self._get_all_boxes_from_layout_det(layout_det_results, special_types=['text'])
        text_det_text_boxes = self._get_all_text_boxes_from_text_det(text_det_results)
        layout_det_text_boxes_np = np.array([text_box.box for text_box in layout_det_text_boxes])
        text_det_text_boxes_np = np.array([text_box.box for text_box in text_det_text_boxes])
        
        if ratios is None: # 如果没输入ratios，就代表需要计算ratios，直接返回boxes
            return layout_det_text_boxes_np, text_det_text_boxes_np
        else:
            original_text_boxes = [layout_det_text_boxes_np, text_det_text_boxes_np]
        
        sorted_text_det_text_boxes = self.sort_boxes_by_y(text_det_text_boxes)
        if log_file:
            for b in sorted_text_det_text_boxes:
                print(b.box.tolist(), file=log_file)
        if app_type == DISCORD and self._has_dominant_xmin_bin(sorted_text_det_text_boxes):

            filtered_text_boxes = self.filter_by_min_x_and_max_x_and_main_height(sorted_text_det_text_boxes, padding, image_sizes, ratios, app_type, log_file)
            avatar_boxes = self._get_all_avatar_boxes_from_layout_det(layout_det_results)
            sorted_avatar_boxes = self.sort_boxes_by_y(avatar_boxes)
            sorted_box = self.discord_group_text(sorted_avatar_boxes, filtered_text_boxes, log_file)
            return sorted_box, original_text_boxes
        else:
            iou_matrix = self._boxes_coverage(text_det_text_boxes_np, layout_det_text_boxes_np)
            if log_file:
                print(f'text_det shape {text_det_text_boxes_np.shape} layout_det shape {layout_det_text_boxes_np.shape} iou matrix {iou_matrix.shape}', file=log_file)

            def get_matches_for_each_box1(iou_matrix, threshold=0.2):
                mask = iou_matrix > threshold
                result = []
                for i in range(iou_matrix.shape[0]):
                    result.append(np.where(mask[i])[0].tolist())
                return result
            
            filtered_text_det_boxes = []
            for i, layout_det_match_indices in enumerate(get_matches_for_each_box1(iou_matrix)):
                if len(layout_det_match_indices)>0:
                    for index in layout_det_match_indices:
                        if log_file:
                            print(f'{text_det_text_boxes[i].box.tolist()} : {layout_det_text_boxes[index].layout_det} iou:{iou_matrix[i, index]}', file=log_file) 
                    filtered_text_det_boxes.append(text_det_text_boxes[i])
            return self.sort_boxes_by_y(filtered_text_det_boxes), original_text_boxes

    def format_conversation_app_agnostic(
        self,
        layout_det_results,
        text_det_results,
        screen_width: int,
        memory_path: str = None,
        coverage_threshold: float = 0.15,
        coverage_keep_ratio: float = 0.35,
        enable_height_filter: bool = True,
        height_bin_size: int = 3,
        height_tolerance_px: int = 4,
        min_height_keep_ratio: float = 0.4,
        padding: list[float] | None = None,
        image_sizes: list[float] | None = None,
        ocr_reader=None,
        talker_nickname: str | None = None,
        log_file=None
    ) -> tuple:
        """
        App无关的对话格式化：使用layout_det过滤text_det，再调用自适应检测器。

        Args:
            layout_det_results: layout_det模型输出
            text_det_results: text_det模型输出
            screen_width: 屏幕宽度（像素）
            memory_path: 可选的记忆持久化路径
            coverage_threshold: coverage过滤阈值
            coverage_keep_ratio: coverage过滤后的最小保留比例
            enable_height_filter: 是否启用高度过滤
            height_bin_size: 文本高度量化的bin大小
            height_tolerance_px: 文本高度容忍度（像素）
            min_height_keep_ratio: 最小高度保持比率
            padding: 图片padding信息
            image_sizes: 图片尺寸
            ocr_reader: 可选OCR读取函数，返回(text, score)
            talker_nickname: 可选的对方昵称，用于单列时区分说话者
            log_file: 可选日志文件

        Returns:
            (sorted_boxes, metadata) 元组
        """
        layout_det_text_boxes = self._get_all_boxes_from_layout_det(layout_det_results, special_types=['text'])
        text_det_text_boxes = self._get_all_text_boxes_from_text_det(text_det_results)

        if not text_det_text_boxes:
            return [], {
                'layout': 'single',
                'speaker_A_count': 0,
                'speaker_B_count': 0,
                'confidence': 0.0,
                'frame_count': 0
            }

        if not layout_det_text_boxes:
            return self.format_conversation_adaptive(
                text_boxes=text_det_text_boxes,
                screen_width=screen_width,
                memory_path=memory_path,
                layout_det_results=layout_det_results,
                text_det_results=text_det_results,
                padding=padding,
                image_sizes=image_sizes,
                ocr_reader=ocr_reader,
                talker_nickname=talker_nickname,
                log_file=log_file
            )

        main_height = self.estimate_main_text_height(text_det_text_boxes, bin_size=height_bin_size)
        if enable_height_filter and main_height is not None:
            height_filtered = [
                box for box in text_det_text_boxes
                if abs(box.height - main_height) <= height_tolerance_px
            ]
            if height_filtered:
                text_det_text_boxes = height_filtered
                keep_ratio = len(height_filtered) / max(len(text_det_text_boxes), 1)
                if keep_ratio >= min_height_keep_ratio:
                    text_det_text_boxes = height_filtered

        layout_det_text_boxes_np = np.array([text_box.box for text_box in layout_det_text_boxes])
        text_det_text_boxes_np = np.array([text_box.box for text_box in text_det_text_boxes])
        coverage_matrix = self._boxes_coverage(text_det_text_boxes_np, layout_det_text_boxes_np)

        if log_file:
            print(
                f'text_det shape {text_det_text_boxes_np.shape} '
                f'layout_det shape {layout_det_text_boxes_np.shape} '
                f'coverage matrix {coverage_matrix.shape}',
                file=log_file
            )

        mask = coverage_matrix > coverage_threshold
        filtered_text_det_boxes = []
        for i in range(coverage_matrix.shape[0]):
            if np.any(mask[i]):
                for j in range(coverage_matrix.shape[1]):
                    if mask[i, j]:
                        text_det_text_boxes[i].related_layout_boxes.append(layout_det_text_boxes[j])
                filtered_text_det_boxes.append(text_det_text_boxes[i])

        if not filtered_text_det_boxes:
            filtered_text_det_boxes = text_det_text_boxes
        else:
            keep_ratio = len(filtered_text_det_boxes) / max(len(text_det_text_boxes), 1)
            if keep_ratio < coverage_keep_ratio:
                filtered_text_det_boxes = text_det_text_boxes

        return self.format_conversation_adaptive(
            text_boxes=filtered_text_det_boxes,
            screen_width=screen_width,
            memory_path=memory_path,
            layout_det_results=layout_det_results,
            text_det_results=text_det_results,
            padding=padding,
            image_sizes=image_sizes,
            ocr_reader=ocr_reader,
            talker_nickname=talker_nickname,
            log_file=log_file
        )

    def sort_boxes_by_y(self, boxes:list[TextBox]) -> list[TextBox]:
        return sorted(boxes, key=lambda b: b.y_min)

    def discord_group_text(self, avatar_boxes, text_boxes, log_file):
        sorted_boxes = []
        a_boxes = deepcopy(avatar_boxes)
        t_boxes = deepcopy(text_boxes)

        sorted_boxes = a_boxes + t_boxes
        sorted_boxes = sorted(sorted_boxes, key=lambda b: b.y_min)
        if log_file:
            print('sorted list start', file=log_file)
        for box in sorted_boxes:
            if log_file:
                print(f'{box.box.tolist()} {box.layout_det}', file=log_file)
        if log_file:
            print('sorted list end', file=log_file)

        for (i, j) in zip(range(len(sorted_boxes)-1), range(1, len(sorted_boxes))):
            if (sorted_boxes[j].layout_det == 'avatar' and sorted_boxes[i].layout_det == 'text'):
                # 排序靠后的头像 和 靠前昵称 需要交换位置
                if abs(sorted_boxes[i].y_min - sorted_boxes[j].y_min) < MAGIC_MARGIN:
                    sorted_boxes[i], sorted_boxes[j] = sorted_boxes[j], sorted_boxes[i]
        
        for (i, j) in zip(range(len(sorted_boxes)-1), range(1, len(sorted_boxes))):
            if (sorted_boxes[i].layout_det == 'avatar' and sorted_boxes[j].layout_det == 'text'):
                # 靠近头像的文本框是nickname
                if abs(sorted_boxes[i].y_min - sorted_boxes[j].y_min) < MAGIC_MARGIN:
                    sorted_boxes[j].layout_det = 'nickname'
        if log_file:
            print("*"*20, file=log_file)
            print('sorted list start', file=log_file)
        for box in sorted_boxes:
            if log_file:
                print(f'{box.box.tolist()} {box.layout_det}', file=log_file)
        if log_file:
            print('sorted list end', file=log_file)

        while sorted_boxes[0].layout_det != 'avatar':
            sorted_boxes.pop(0) 
        return sorted_boxes


    def is_same_group(prev: TextBox, curr: TextBox, y_threshold: float, x_threshold: float):
        return curr.y_min <= (prev.y_max + y_threshold)

    def group_text_boxes_by_y(
        self,
        boxes: list[TextBox],
        y_threshold: float = 12.0,
        height_ratio: float = 0.6
    ) -> list[list[TextBox]]:
        if not boxes:
            return []
        main_height = self.estimate_main_text_height(boxes)
        if main_height is not None:
            y_threshold = max(y_threshold, main_height * height_ratio)
        sorted_boxes = sorted(boxes, key=lambda b: b.y_min)
        groups: list[list[TextBox]] = []
        current_group: list[TextBox] = []
        current_group_max_y = None

        for box in sorted_boxes:
            if not current_group:
                current_group = [box]
                current_group_max_y = box.y_max
                continue

            if box.y_min <= (current_group_max_y + y_threshold):
                current_group.append(box)
                current_group_max_y = max(current_group_max_y, box.y_max)
            else:
                groups.append(current_group)
                current_group = [box]
                current_group_max_y = box.y_max

        if current_group:
            groups.append(current_group)

        return groups
    def estimate_main_value(self, boxes:list[TextBox], selected_property, bin_size=2, log_file=None):
        assert selected_property in ['x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'area', 'center_x', 'center_y'], 'estimate_main_value函数只能作用于代码里设定好的属性'
        def collect_areas(boxes):
            return [b.height * b.width for b in boxes]

        def collect_property(boxes, selected_property):
            return [getattr(b, selected_property) for b in boxes]

        if selected_property == 'area':
            unquantized_values = collect_areas(boxes)
        else:
            unquantized_values = collect_property(boxes, selected_property)

        if not unquantized_values:
            return None
        if log_file:
            print(f'{selected_property}:{unquantized_values}', file=log_file)
        unquantized_values = np.array(unquantized_values)
        quantized = (unquantized_values // bin_size) * bin_size
        values, counts = np.unique(quantized, return_counts=True)
        if log_file:
            print(f"values:{values} counts:{counts}", file=log_file)
        main_values = values[counts.argmax()]
        return main_values

    def estimate_main_box_area(self, boxes:list[TextBox], bin_size=2, log_file=None):
        """
        返回：主头像面积（像素）
        """
        def collect_areas(boxes):
            return [b.height * b.width for b in boxes]
        areas = collect_areas(boxes)

        if not areas:
            return None
        if log_file:
            print(f"areas:{areas}", file=log_file)
        areas = np.array(areas)
        # 量化面积，减少 OCR 抖动
        quantized = (areas // bin_size) * bin_size
        values, counts = np.unique(quantized, return_counts=True)
        if log_file:
            print(f"values:{values} counts:{counts}", file=log_file)
        main_area = values[counts.argmax()]
        return main_area

    def filter_by_image_avatar_area(self, main_area, box:TextBox):
        lower = main_area * (1 - MAIN_AREA_TOLERANCE)
        upper = main_area * (1 + MAIN_AREA_TOLERANCE)
        return lower <= (box.height * box.width) <= upper

    def estimate_main_text_height(self, boxes:list[TextBox], bin_size=2):
        """
        返回：主文本高度（像素）
        """
        def collect_heights(boxes):
            return [b.height for b in boxes if b.height > 0]
 
        heights = collect_heights(boxes)

        if not heights:
            return None

        heights = np.array(heights)
        # 量化高度，减少 OCR 抖动
        quantized = (heights // bin_size) * bin_size
        values, counts = np.unique(quantized, return_counts=True)
        main_height = values[counts.argmax()]
        return main_height

    def filter_by_text_height(self, main_height, box:TextBox):
        lower = main_height * (1 - MAIN_HEIGHT_TOLERANCE)
        upper = main_height * (1 + MAIN_HEIGHT_TOLERANCE)
        return lower <= box.height <= upper

    def _has_dominant_xmin_bin(self, boxes: list[TextBox], bin_size: int = 4, min_ratio: float = 0.35) -> bool:
        if not boxes:
            return False
        xmins = np.array([box.x_min for box in boxes])
        bins = (xmins // bin_size) * bin_size
        _, counts = np.unique(bins, return_counts=True)
        if counts.size == 0:
            return False
        return counts.max() / len(boxes) >= min_ratio

    def _has_dominant_xmax_bin(self, boxes: list[TextBox], bin_size: int = 4, min_ratio: float = 0.35) -> bool:
        if not boxes:
            return False
        xmaxs = np.array([box.x_max for box in boxes])
        bins = (xmaxs // bin_size) * bin_size
        _, counts = np.unique(bins, return_counts=True)
        if counts.size == 0:
            return False
        return counts.max() / len(boxes) >= min_ratio

    def filter_by_rules_to_find_nickname(self, boxes_from_text_det:list[TextBox], padding, image_sizes, ratios, app_type, log_file=None):
        unfiltered_text_boxes = boxes_from_text_det
        filtered_text_boxes = []
        w, h = image_sizes

        main_height = self.estimate_main_text_height(boxes_from_text_det)
        if log_file:
            print(f"main_height:{main_height}", file = log_file)
        if app_type == DISCORD and self._has_dominant_xmin_bin(unfiltered_text_boxes):

            # discord 都是聊天气泡靠左
            box_left = (w - padding[0] - padding[2]) * ratios[0] + padding[0] # 统计的气泡左边起点在新图片上的像素数值
            if log_file:
                print(f"box_left:{box_left}", file = log_file)
            left_margin = MAGIC_MARGIN * 8
            for text_box in unfiltered_text_boxes:
                if log_file:
                    print(f'current box: {text_box.box.tolist()}', file=log_file)
                    print(f'condition (box_left - left_margin) < text_box.x_min: {(box_left - left_margin) < text_box.x_min}', file=log_file)
                    print(f'condition text_box.y_max < 0.25 * h:{text_box.y_max < (0.25 * h)}', file=log_file)
                if (box_left - left_margin) < text_box.x_min and text_box.y_max < (0.25 * h):
                    filtered_text_boxes.append(text_box)
            return filtered_text_boxes
        else:
            box_left = (w - padding[0] - padding[2]) * ratios[2] + padding[0] # 统计的气泡左边起点在新图片上的像素数值, talker left start
            box_right = (w - padding[0] - padding[2]) * ratios[1] + padding[0] # 统计的气泡右边终点在新图片上的像素数值, user right end
            for text_box in unfiltered_text_boxes:
                if text_box.x_min > (box_left - MAGIC_MARGIN) and text_box.x_min < (box_left + MAGIC_MARGIN):
                    filtered_text_boxes.append(text_box)
                elif text_box.x_max > (box_right - MAGIC_MARGIN) and text_box.x_min < (box_left + MAGIC_MARGIN):
                    filtered_text_boxes.append(text_box)
            return filtered_text_boxes

    def filter_by_min_x_and_max_x_and_main_height(self, boxes_from_text_det:list[TextBox], padding, image_sizes, ratios, app_type, log_file=None):
        unfiltered_text_boxes = boxes_from_text_det
        filtered_text_boxes = []
        w, h = image_sizes

        main_height = self.estimate_main_text_height(boxes_from_text_det)
        if log_file:
            print(f"main_height:{main_height}", file = log_file)
        if app_type == DISCORD and self._has_dominant_xmin_bin(unfiltered_text_boxes):
            # discord 都是聊天气泡靠左
            box_left = (w - padding[0] - padding[2]) * ratios[0] + padding[0]
            if log_file:
                print(f"box_left:{box_left}", file = log_file)
            left_margin = MAGIC_MARGIN * 8
            height_lower = main_height * 0.5 if main_height is not None else None
            height_upper = main_height * 2.2 if main_height is not None else None

            for text_box in unfiltered_text_boxes:
                if log_file:
                    print(f'current box: {text_box.box.tolist()}', file=log_file)
                    print(f'condition (box_left - left_margin) < text_box.x_min: {(box_left - left_margin) < text_box.x_min}', file=log_file)
                    print(f'condition (box_left + left_margin) < text_box.x_min: {text_box.x_min < (box_left + left_margin)}', file=log_file)
                    print(f'condition self.filter_by_text_height(main_height, text_box): {self.filter_by_text_height(main_height, text_box)}', file=log_file)

                if (box_left - left_margin) < text_box.x_min < (box_left + left_margin) and (height_lower is None or height_lower <= text_box.height) and (height_upper is None or text_box.height <= height_upper):
                    filtered_text_boxes.append(text_box)
            return filtered_text_boxes
        else:
            box_left = (w - padding[0] - padding[2]) * ratios[2] + padding[0] # 统计的气泡左边起点在新图片上的像素数值, talker left start
            box_right = (w - padding[0] - padding[2]) * ratios[1] + padding[0] # 统计的气泡右边终点在新图片上的像素数值, user right end
            if log_file:
                print(f"box_left:{box_left}  box_right:{box_right}", file = log_file)
            for text_box in unfiltered_text_boxes:
                if log_file:
                    print(f'current box: {text_box.box.tolist()}', file=log_file)
                    print(f'lower {box_left - MAGIC_MARGIN} x_min {text_box.x_min} upper {box_left + MAGIC_MARGIN}', file=log_file)
                    print(f'lower {box_right - MAGIC_MARGIN} x_max {text_box.x_max} upper {box_right + MAGIC_MARGIN}', file=log_file)
                    print(f'condition self.filter_by_text_height(main_height, text_box): {self.filter_by_text_height(main_height, text_box)}', file=log_file)
                    print('*'*20, file=log_file)
                if (box_left - MAGIC_MARGIN) < text_box.x_min < (box_left + MAGIC_MARGIN):# and self.filter_by_text_height(main_height, text_box):
                    filtered_text_boxes.append(text_box)
                elif (box_right - MAGIC_MARGIN) < text_box.x_max < (box_right + MAGIC_MARGIN):# and self.filter_by_text_height(main_height, text_box):
                    filtered_text_boxes.append(text_box)
            return filtered_text_boxes
        
    def format_message(self, message_group: List[Dict]) -> Dict[str, Any]:
        """格式化单条聊天消息"""
        if not message_group:
            return {}
            
        primary_element = message_group[0]
        message_type = self._detect_message_type(message_group)
        
        return {
            'type': message_type,
            'position': primary_element['bbox'],
            'elements': message_group,
            'timestamp': self._estimate_timestamp(message_group),
            'sender': self._estimate_sender(message_group)
        }
    
    def _detect_message_type(self, message_group: List[Dict]) -> str:
        """检测消息类型"""
        categories = [elem['category'] for elem in message_group]
        
        if 'image' in categories:
            return 'image'
        elif 'table' in categories:
            return 'table'
        elif 'formula' in categories:
            return 'formula'
        elif 'chart' in categories:
            return 'chart'
        else:
            return 'text'
    
    def _estimate_timestamp(self, message_group: List[Dict]) -> str:
        """估计消息时间戳（基于位置）"""
        # 简化实现：实际应用中可能需要OCR识别时间戳
        return "estimated_time"
    
    def _estimate_sender(self, message_group: List[Dict]) -> str:
        """估计发送者（基于位置）"""
        # 简化实现：左侧通常为对方，右侧为自己
        bbox = message_group[0]['bbox']
        if bbox[0] < 200:  # 假设屏幕宽度>400
            return 'contact'
        else:
            return 'user'

    def detect_chat_layout_adaptive(self, text_boxes: List[TextBox], layout_det_text_boxes: List[TextBox], text_det_text_boxes: List[TextBox], screen_width: int, 
                                    memory_path: str = None, log_file=None) -> Dict:
        """
        使用新的自适应ChatLayoutDetector进行聊天布局检测
        
        这是一个应用无关的检测方法，不需要app_type参数。
        它使用几何学习和历史记忆来自动识别说话者。
        
        Args:
            text_boxes: TextBox对象列表
            screen_width: 屏幕宽度（像素）
            memory_path: 可选的记忆持久化路径
            log_file: 可选的日志文件对象
            
        Returns:
            包含以下字段的字典:
            - layout: 布局类型 ("single", "double", "double_left", "double_right")
            - A: Speaker A的文本框列表
            - B: Speaker B的文本框列表
            - metadata: 包含置信度、帧计数等元数据
            
        Example:
            >>> processor = ChatMessageProcessor()
            >>> result = processor.detect_chat_layout_adaptive(text_boxes, 720)
            >>> print(f"Layout: {result['layout']}")
            >>> print(f"Speaker A has {len(result['A'])} messages")
            >>> print(f"Speaker B has {len(result['B'])} messages")
        """
        # 创建检测器实例
        detector = ChatLayoutDetector(
            screen_width=screen_width,
            memory_path=memory_path
        )
        
        # 处理当前帧
        result = detector.process_frame(text_boxes, layout_det_text_boxes, text_det_text_boxes)
        
        # 可选：记录日志
        if log_file:
            print(f"Layout detected: {result['layout']}", file=log_file)
            print(f"Speaker A boxes: {len(result['A'])}", file=log_file)
            print(f"Speaker B boxes: {len(result['B'])}", file=log_file)
            print(f"Metadata: {result['metadata']}", file=log_file)
        
        return result
    
    def format_conversation_adaptive(self, text_boxes: List[TextBox], screen_width: int,
                                    memory_path: str = None,
                                    layout_det_results=None,
                                    text_det_results=None,
                                    padding: list[float] | None = None,
                                    image_sizes: list[float] | None = None,
                                    ocr_reader=None,
                                    talker_nickname: str | None = None,
                                    log_file=None) -> tuple:
        """
        使用自适应检测器格式化对话
        
        这个方法是format_conversation的应用无关版本。
        它返回与现有代码兼容的格式，但使用新的检测逻辑。
        
        Args:
            text_boxes: TextBox对象列表
            screen_width: 屏幕宽度（像素）
            memory_path: 可选的记忆持久化路径
            layout_det_results: layout_det模型输出
            text_det_results: text_det模型输出
            padding: 图片padding信息
            image_sizes: 图片尺寸
            ocr_reader: 可选OCR读取函数，返回(text, score)
            talker_nickname: 可选的对方昵称，用于单列时区分说话者
            log_file: 可选的日志文件对象
            
        Returns:
            (sorted_boxes, metadata) 元组:
            - sorted_boxes: 按y坐标排序的文本框列表
            - metadata: 包含布局信息和说话者分配的字典
        """
        # 使用新检测器
        result = self.detect_chat_layout_adaptive(text_boxes, layout_det_results, text_det_results, screen_width, memory_path, log_file)

        # 单列布局时，如果提供了Discord上下文信息，复用Discord分组逻辑
        if (
            result.get('layout') == 'single'
            and layout_det_results is not None
            and text_det_results is not None
            and padding is not None
            and image_sizes is not None
        ):
            avatar_boxes = self._get_all_avatar_boxes_from_layout_det(layout_det_results)
            sorted_avatar_boxes = self.sort_boxes_by_y(avatar_boxes)
            sorted_box = self.discord_group_text(sorted_avatar_boxes, text_boxes, log_file)

            layout_text_boxes = []
            new_speaker_group_flag = False
            current_speaker = None
            last_avatar_center_x = None
            for box in sorted_box:
                if box.layout_det == 'avatar':
                    new_speaker_group_flag = True
                    last_avatar_center_x = box.center_x
                    if talker_nickname is None:
                        current_speaker = OTHER if box.center_x <= screen_width / 2 else USER

                    continue
                if box.layout_det == 'nickname':
                    if not new_speaker_group_flag:
                        continue
                    speaker_name = ""
                    if ocr_reader is not None:
                        speaker_name, _ = ocr_reader(box)
                    if talker_nickname and self._is_nickname_match(speaker_name, talker_nickname):
                        current_speaker = OTHER
                    elif speaker_name:
                        current_speaker = USER
                    elif last_avatar_center_x is not None:
                        current_speaker = OTHER if last_avatar_center_x <= screen_width / 2 else USER

                    continue
                if box.layout_det == 'text':
                    if current_speaker is None and last_avatar_center_x is not None:
                        current_speaker = OTHER if last_avatar_center_x <= screen_width / 2 else USER

                    if current_speaker is None:
                        continue
                    box.speaker = current_speaker
                    layout_text_boxes.append(box)

            grouped_boxes = self.group_text_boxes_by_y(layout_text_boxes)
            metadata = {
                'layout': result.get('layout', 'single'),
                'speaker_A_count': len([b for b in layout_text_boxes if b.speaker == OTHER]),
                'speaker_B_count': len([b for b in layout_text_boxes if b.speaker == USER]),

                'confidence': result.get('metadata', {}).get('confidence', 1.0),
                'frame_count': result.get('metadata', {}).get('frame_count', 0),
                'group_count': len(grouped_boxes),
                'groups': grouped_boxes
            }
            return layout_text_boxes, metadata
        # 合并所有文本框并按y坐标排序
        all_boxes = result['A'] + result['B']
        sorted_boxes = self.sort_boxes_by_y(all_boxes)
        
        # 为每个文本框添加说话者标记
        a_ids = {id(box) for box in result['A']}
        for box in sorted_boxes:
            if id(box) in a_ids:
                box.speaker = OTHER
            else:
                box.speaker = USER
        
        grouped_boxes = self.group_text_boxes_by_y(sorted_boxes)

        # 构建元数据
        metadata = {
            'layout': result['layout'],
            'speaker_A_count': len(result['A']),
            'speaker_B_count': len(result['B']),
            'confidence': result['metadata'].get('confidence', 1.0),
            'frame_count': result['metadata'].get('frame_count', 0),
            'group_count': len(grouped_boxes),
            'groups': grouped_boxes
        }
        
        return sorted_boxes, metadata

    @staticmethod
    def _normalize_nickname(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text).lower()

    def _is_nickname_match(self, speaker_name: str, talker_nickname: str, min_ratio: float = 0.8) -> bool:
        normalized_speaker = self._normalize_nickname(speaker_name)
        normalized_talker = self._normalize_nickname(talker_nickname)
        if not normalized_speaker or not normalized_talker:
            return False
        if normalized_speaker in normalized_talker or normalized_talker in normalized_speaker:
            return True
        ratio = SequenceMatcher(None, normalized_speaker, normalized_talker).ratio()
        return ratio >= min_ratio

    # Helper methods for nickname extraction
    def _calculate_distance(self, box1: TextBox, box2: TextBox) -> float:
        """
        Calculate Euclidean distance between centers of two boxes.
        
        Args:
            box1: First TextBox
            box2: Second TextBox
            
        Returns:
            Euclidean distance between box centers
        """
        dx = box1.center_x - box2.center_x
        dy = box1.center_y - box2.center_y
        return (dx**2 + dy**2) ** 0.5

    def _is_above_or_right(self, text_box: TextBox, avatar_box: TextBox) -> bool:
        """
        Check if text box is above or to the right of avatar box.
        
        Args:
            text_box: TextBox to check position of
            avatar_box: Reference avatar TextBox
            
        Returns:
            True if text_box is above or to the right of avatar_box
        """
        # Above: text box bottom is above avatar box top
        is_above = text_box.y_max < avatar_box.y_min
        # Right: text box left is to the right of avatar box right
        is_right = text_box.x_min > avatar_box.x_max
        return is_above or is_right

    def _meets_size_criteria(self, box: TextBox, min_height: int = 10, min_width: int = 20) -> bool:
        """
        Check if box meets minimum size criteria.
        
        Args:
            box: TextBox to check
            min_height: Minimum height in pixels (default: 10)
            min_width: Minimum width in pixels (default: 20)
            
        Returns:
            True if box meets both height and width criteria
        """
        return box.height > min_height and box.width > min_width

    def _extract_from_layout_det(self, layout_det_boxes: List[TextBox], log_file=None) -> Dict[str, TextBox]:
        """
        Extract nickname boxes directly from layout_det results.
        
        This is Method 1 of the nickname extraction fallback chain.
        It looks for boxes that are explicitly labeled as 'nickname' by the layout_det model.
        
        Args:
            layout_det_boxes: All boxes from layout_det with speaker assignments
            log_file: Optional logging file
            
        Returns:
            Dictionary with keys 'A' and 'B', values are TextBox or None
            {
                'A': TextBox or None,
                'B': TextBox or None
            }
            
        Requirements:
            - 8.1: Log method entry and parameters
            - 8.2: Log candidate boxes found
            - 8.3: Log filtering steps and results
            - 8.4: Log final selection for each speaker
        """
        if log_file:
            print("=== Method 1: Layout Det Nickname Detection ===", file=log_file)
            print(f"Input: {len(layout_det_boxes)} layout_det boxes", file=log_file)
        
        # Initialize result dictionary
        result = {'A': None, 'B': None}
        
        # Filter boxes where layout_det == 'nickname'
        nickname_boxes = [box for box in layout_det_boxes if box.layout_det == 'nickname']
        
        if log_file:
            print(f"\nFiltering step: Found {len(nickname_boxes)} boxes with layout_det='nickname'", file=log_file)
            for i, box in enumerate(nickname_boxes):
                speaker_info = f"speaker={box.speaker}" if hasattr(box, 'speaker') else "no speaker"
                print(f"  Candidate {i+1}: {box.box.tolist()}, {speaker_info}, score={box.score:.3f}", file=log_file)
        
        # Handle case where no nickname boxes exist
        if not nickname_boxes:
            if log_file:
                print("\nResult: No nickname boxes found in layout_det results", file=log_file)
            return result
        
        # Group nickname boxes by speaker attribute
        speaker_a_boxes = []
        speaker_b_boxes = []
        unassigned_boxes = []
        
        if log_file:
            print("\nGrouping by speaker:", file=log_file)
        
        for box in nickname_boxes:
            if hasattr(box, 'speaker'):
                if box.speaker == OTHER:
                    speaker_a_boxes.append(box)
                    if log_file:
                        print(f"  Speaker A: {box.box.tolist()}", file=log_file)
                elif box.speaker == USER:
                    speaker_b_boxes.append(box)
                    if log_file:
                        print(f"  Speaker B: {box.box.tolist()}", file=log_file)
                else:
                    unassigned_boxes.append(box)
                    if log_file:
                        print(f"  Warning: Unknown speaker '{box.speaker}': {box.box.tolist()}", file=log_file)
            else:
                unassigned_boxes.append(box)
                if log_file:
                    print(f"  Warning: No speaker attribute: {box.box.tolist()}", file=log_file)
        
        if log_file:
            print(f"\nGrouping results: A={len(speaker_a_boxes)}, B={len(speaker_b_boxes)}, Unassigned={len(unassigned_boxes)}", file=log_file)
        
        # Select first box for each speaker if multiple exist
        if log_file:
            print("\nFinal selection:", file=log_file)
        
        if speaker_a_boxes:
            result['A'] = speaker_a_boxes[0]
            if log_file:
                print(f"  Speaker A: {result['A'].box.tolist()}, score={result['A'].score:.3f}", file=log_file)
                if len(speaker_a_boxes) > 1:
                    print(f"    (selected first of {len(speaker_a_boxes)} boxes)", file=log_file)
                    for i, box in enumerate(speaker_a_boxes[1:], start=2):
                        print(f"    Alternative {i}: {box.box.tolist()}, score={box.score:.3f}", file=log_file)
        else:
            if log_file:
                print("  Speaker A: None", file=log_file)
        
        if speaker_b_boxes:
            result['B'] = speaker_b_boxes[0]
            if log_file:
                print(f"  Speaker B: {result['B'].box.tolist()}, score={result['B'].score:.3f}", file=log_file)
                if len(speaker_b_boxes) > 1:
                    print(f"    (selected first of {len(speaker_b_boxes)} boxes)", file=log_file)
                    for i, box in enumerate(speaker_b_boxes[1:], start=2):
                        print(f"    Alternative {i}: {box.box.tolist()}, score={box.score:.3f}", file=log_file)
        else:
            if log_file:
                print("  Speaker B: None", file=log_file)
        
        return result

    def _extract_from_avatar_neighbor(self, avatar_boxes: List[TextBox], 
                                      text_det_boxes: List[TextBox], 
                                      log_file=None) -> Dict[str, TextBox]:
        """
        Find nicknames near avatar boxes.
        
        This is Method 2 of the nickname extraction fallback chain.
        For each avatar with a speaker assignment, it finds the nearest text_det box
        that is above or to the right of the avatar and meets size criteria.
        
        Args:
            avatar_boxes: Boxes with layout_det == 'avatar' and speaker assigned
            text_det_boxes: All text_det boxes
            log_file: Optional logging file
            
        Returns:
            Dictionary with keys 'A' and 'B', values are TextBox or None
            {
                'A': TextBox or None,
                'B': TextBox or None
            }
            
        Requirements:
            - 8.1: Log method entry and parameters
            - 8.2: Log candidate boxes found
            - 8.3: Log filtering steps and results
            - 8.4: Log final selection for each speaker
        """
        if log_file:
            print("=== Method 2: Avatar-Neighbor Search ===", file=log_file)
            print(f"Input: {len(avatar_boxes)} avatar boxes, {len(text_det_boxes)} text_det boxes", file=log_file)
        
        # Initialize result dictionary
        result = {'A': None, 'B': None}
        
        # Handle edge cases
        if not avatar_boxes:
            if log_file:
                print("\nResult: No avatar boxes provided", file=log_file)
            return result
        
        if not text_det_boxes:
            if log_file:
                print("\nResult: No text_det boxes provided", file=log_file)
            return result
        
        if log_file:
            print(f"\nAvatar boxes to process:", file=log_file)
            for i, box in enumerate(avatar_boxes):
                speaker_info = f"speaker={box.speaker}" if hasattr(box, 'speaker') else "no speaker"
                print(f"  Avatar {i+1}: {box.box.tolist()}, {speaker_info}", file=log_file)
        
        # Process each avatar box
        avatar_count = 0
        for avatar_box in avatar_boxes:
            # Skip avatars without speaker assignment
            if not hasattr(avatar_box, 'speaker') or avatar_box.speaker not in [OTHER, USER]:
                if log_file:
                    print(f"\nSkipping avatar without valid speaker: {avatar_box.box.tolist()}", file=log_file)
                continue
            
            avatar_count += 1
            speaker = avatar_box.speaker
            speaker_key = "A" if speaker == OTHER else "B"
            
            if log_file:
                print(f"\n--- Processing Avatar {avatar_count} for Speaker {speaker} ---", file=log_file)
                print(f"Avatar position: {avatar_box.box.tolist()}", file=log_file)
                print(f"Avatar center: ({avatar_box.center_x:.1f}, {avatar_box.center_y:.1f})", file=log_file)
            
            # Find candidate text boxes
            candidates = []
            position_filtered = 0
            size_filtered = 0
            
            for text_box in text_det_boxes:
                # Filter by position (above or right of avatar)
                if not self._is_above_or_right(text_box, avatar_box):
                    position_filtered += 1
                    continue
                
                # Filter by size
                if not self._meets_size_criteria(text_box):
                    size_filtered += 1
                    if log_file:
                        print(f"  Filtered (size): {text_box.box.tolist()}, h={text_box.height:.1f}, w={text_box.width:.1f}", file=log_file)
                    continue
                
                # Calculate distance
                distance = self._calculate_distance(text_box, avatar_box)
                candidates.append((text_box, distance))
                
                if log_file:
                    print(f"  Candidate: {text_box.box.tolist()}, distance={distance:.2f}, h={text_box.height:.1f}, w={text_box.width:.1f}", file=log_file)
            
            if log_file:
                print(f"\nFiltering results:", file=log_file)
                print(f"  Position filtered: {position_filtered} boxes (not above or right)", file=log_file)
                print(f"  Size filtered: {size_filtered} boxes (too small)", file=log_file)
                print(f"  Valid candidates: {len(candidates)} boxes", file=log_file)
            
            # Select nearest box
            if candidates:
                # Sort by distance and select the nearest
                candidates.sort(key=lambda x: x[1])
                nearest_box, nearest_distance = candidates[0]
                
                if log_file:
                    print(f"\nCandidate ranking (by distance):", file=log_file)
                    for i, (box, dist) in enumerate(candidates[:5], start=1):  # Show top 5
                        print(f"  {i}. {box.box.tolist()}, distance={dist:.2f}", file=log_file)
                    if len(candidates) > 5:
                        print(f"  ... and {len(candidates) - 5} more", file=log_file)
                
                # Create a copy and assign speaker
                nickname_box = nearest_box
                nickname_box.speaker = speaker
                
                # Store result (only if we haven't found one for this speaker yet)
                if result[speaker_key] is None:
                    result[speaker_key] = nickname_box
                    if log_file:
                        print(f"\nSelection: Nearest box for Speaker {speaker}", file=log_file)
                        print(f"  Box: {nickname_box.box.tolist()}", file=log_file)
                        print(f"  Distance: {nearest_distance:.2f}", file=log_file)
                else:
                    if log_file:
                        print(f"\nSkipping: Speaker {speaker} already has a nickname from previous avatar", file=log_file)
            else:
                if log_file:
                    print(f"\nNo valid candidates found for Speaker {speaker}", file=log_file)
        
        # Summary
        if log_file:
            print("\n" + "=" * 60, file=log_file)
            print("=== Avatar-Neighbor Search Results ===", file=log_file)
            print("=" * 60, file=log_file)
            for speaker in ['A', 'B']:
                if result[speaker]:
                    print(f"Speaker {speaker}: {result[speaker].box.tolist()}", file=log_file)
                else:
                    print(f"Speaker {speaker}: None", file=log_file)
        
        return result

    def _extract_from_top_region(self, text_det_boxes: List[TextBox], 
                                 screen_width: int, 
                                 screen_height: int,
                                 layout_det_boxes: List[TextBox] = None,
                                 log_file=None) -> Dict[str, TextBox]:
        """
        Find nicknames in the top 10% of screen.
        
        This is Method 3 of the nickname extraction fallback chain.
        It searches for text boxes in the top region of the screen and assigns
        them to speakers based on horizontal position (left vs right).
        
        Args:
            text_det_boxes: All text_det boxes
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            layout_det_boxes: Optional layout_det boxes with speaker assignments for mapping
            log_file: Optional logging file
            
        Returns:
            Dictionary with keys 'A' and 'B', values are TextBox or None
            {
                'A': TextBox or None,
                'B': TextBox or None
            }
            
        Requirements:
            - 8.1: Log method entry and parameters
            - 8.2: Log candidate boxes found
            - 8.3: Log filtering steps and results
            - 8.4: Log final selection for each speaker
        """
        if log_file:
            print("=== Method 3: Top-Region Search ===", file=log_file)
            print(f"Input parameters:", file=log_file)
            print(f"  Screen dimensions: {screen_width}x{screen_height}", file=log_file)
            print(f"  Text_det boxes: {len(text_det_boxes)}", file=log_file)
            print(f"  Layout_det boxes: {len(layout_det_boxes) if layout_det_boxes else 0}", file=log_file)
        
        # Initialize result dictionary
        result = {'A': None, 'B': None}
        
        # Handle edge cases
        if not text_det_boxes:
            if log_file:
                print("\nResult: No text_det boxes provided", file=log_file)
            return result
        
        # Calculate top region boundary (top 10% of screen)
        top_region_boundary = screen_height * 0.1
        
        if log_file:
            print(f"\nStep 1: Filter by top region", file=log_file)
            print(f"  Top region boundary: y_max < {top_region_boundary:.2f} (10% of {screen_height})", file=log_file)
        
        # Filter boxes in top region
        top_region_boxes = []
        below_boundary_count = 0
        
        for box in text_det_boxes:
            if box.y_max < top_region_boundary:
                top_region_boxes.append(box)
                if log_file:
                    print(f"  In region: {box.box.tolist()}, y_max={box.y_max:.1f}", file=log_file)
            else:
                below_boundary_count += 1
        
        if log_file:
            print(f"\nFiltering result: {len(top_region_boxes)} boxes in top region, {below_boundary_count} below boundary", file=log_file)
        
        if not top_region_boxes:
            if log_file:
                print("\nResult: No boxes found in top region", file=log_file)
            return result
        
        # Apply size filters
        max_width = screen_width * 0.4
        
        if log_file:
            print(f"\nStep 2: Apply size filters", file=log_file)
            print(f"  Minimum height: 10 pixels", file=log_file)
            print(f"  Minimum width: 20 pixels", file=log_file)
            print(f"  Maximum width: {max_width:.1f} pixels (40% of {screen_width})", file=log_file)
        
        filtered_boxes = []
        too_small_count = 0
        too_wide_count = 0
        
        for box in top_region_boxes:
            # Check minimum size criteria
            if not self._meets_size_criteria(box, min_height=10, min_width=20):
                too_small_count += 1
                if log_file:
                    print(f"  Filtered (too small): {box.box.tolist()}, h={box.height:.1f}, w={box.width:.1f}", file=log_file)
                continue
            
            # Check maximum width (exclude headers)
            if box.width > max_width:
                too_wide_count += 1
                if log_file:
                    print(f"  Filtered (too wide): {box.box.tolist()}, w={box.width:.1f} > {max_width:.1f}", file=log_file)
                continue
            
            filtered_boxes.append(box)
            if log_file:
                print(f"  Valid candidate: {box.box.tolist()}, h={box.height:.1f}, w={box.width:.1f}, center_x={box.center_x:.1f}", file=log_file)
        
        if log_file:
            print(f"\nFiltering result: {len(filtered_boxes)} valid boxes, {too_small_count} too small, {too_wide_count} too wide", file=log_file)
        
        if not filtered_boxes:
            if log_file:
                print("\nResult: No boxes passed size filters", file=log_file)
            return result
        
        # Assign speaker by horizontal position
        screen_center = screen_width * 0.5
        left_boxes = []
        right_boxes = []
        
        if log_file:
            print(f"\nStep 3: Assign to left/right by horizontal position", file=log_file)
            print(f"  Screen center: {screen_center:.1f}", file=log_file)
        
        for box in filtered_boxes:
            if box.center_x < screen_center:
                left_boxes.append(box)
                if log_file:
                    print(f"  Left: {box.box.tolist()}, center_x={box.center_x:.1f}", file=log_file)
            else:
                right_boxes.append(box)
                if log_file:
                    print(f"  Right: {box.box.tolist()}, center_x={box.center_x:.1f}", file=log_file)
        
        if log_file:
            print(f"\nPosition assignment: {len(left_boxes)} left, {len(right_boxes)} right", file=log_file)
        
        # Select topmost box for each side (minimum y_min)
        left_nickname = None
        right_nickname = None
        
        if log_file:
            print(f"\nStep 4: Select topmost box for each side", file=log_file)
        
        if left_boxes:
            left_boxes.sort(key=lambda b: b.y_min)
            left_nickname = left_boxes[0]
            if log_file:
                print(f"  Left side candidates (sorted by y_min):", file=log_file)
                for i, box in enumerate(left_boxes[:3], start=1):  # Show top 3
                    print(f"    {i}. {box.box.tolist()}, y_min={box.y_min:.1f}", file=log_file)
                if len(left_boxes) > 3:
                    print(f"    ... and {len(left_boxes) - 3} more", file=log_file)
                print(f"  Selected: {left_nickname.box.tolist()}, y_min={left_nickname.y_min:.1f}", file=log_file)
        else:
            if log_file:
                print(f"  Left side: No candidates", file=log_file)
        
        if right_boxes:
            right_boxes.sort(key=lambda b: b.y_min)
            right_nickname = right_boxes[0]
            if log_file:
                print(f"  Right side candidates (sorted by y_min):", file=log_file)
                for i, box in enumerate(right_boxes[:3], start=1):  # Show top 3
                    print(f"    {i}. {box.box.tolist()}, y_min={box.y_min:.1f}", file=log_file)
                if len(right_boxes) > 3:
                    print(f"    ... and {len(right_boxes) - 3} more", file=log_file)
                print(f"  Selected: {right_nickname.box.tolist()}, y_min={right_nickname.y_min:.1f}", file=log_file)
        else:
            if log_file:
                print(f"  Right side: No candidates", file=log_file)
        
        # Map left/right to A/B using existing speaker assignments
        # We need to determine which speaker (A or B) is on the left vs right
        # Use layout_det_boxes with speaker assignments if available
        left_is_a = True  # Default assumption
        
        if log_file:
            print(f"\nStep 5: Map left/right to Speaker A/B", file=log_file)
        
        if layout_det_boxes:
            # Count which speaker appears more on the left vs right
            a_left_count = 0
            a_right_count = 0
            b_left_count = 0
            b_right_count = 0
            
            if log_file:
                print(f"  Analyzing {len(layout_det_boxes)} layout_det boxes for speaker positions:", file=log_file)
            
            for box in layout_det_boxes:
                if hasattr(box, 'speaker') and box.speaker in [OTHER, USER]:
                    if box.center_x < screen_center:
                        if box.speaker == OTHER:
                            a_left_count += 1
                        else:
                            b_left_count += 1
                    else:
                        if box.speaker == OTHER:
                            a_right_count += 1
                        else:
                            b_right_count += 1
            
            # Determine which speaker is predominantly on the left
            if a_left_count + b_right_count > b_left_count + a_right_count:
                left_is_a = True
            elif b_left_count + a_right_count > a_left_count + b_right_count:
                left_is_a = False
            # else: keep default (left_is_a = True)
            
            if log_file:
                print(f"  Speaker position analysis:", file=log_file)
                print(f"    A: {a_left_count} left, {a_right_count} right", file=log_file)
                print(f"    B: {b_left_count} left, {b_right_count} right", file=log_file)
                print(f"  Determination: left side is Speaker {'A' if left_is_a else 'B'}", file=log_file)
        else:
            if log_file:
                print(f"  No layout_det boxes provided, using default: left=A, right=B", file=log_file)
        
        # Assign to result dictionary
        if left_is_a:
            if left_nickname:
                left_nickname.speaker = OTHER
                result['A'] = left_nickname
            if right_nickname:
                right_nickname.speaker = USER
                result['B'] = right_nickname
        else:
            if left_nickname:
                left_nickname.speaker = USER
                result['B'] = left_nickname
            if right_nickname:
                right_nickname.speaker = OTHER
                result['A'] = right_nickname
        
        # Summary
        if log_file:
            print("\n" + "=" * 60, file=log_file)
            print("=== Top-Region Search Results ===", file=log_file)
            print("=" * 60, file=log_file)
            for speaker in ['A', 'B']:
                if result[speaker]:
                    print(f"Speaker {speaker}: {result[speaker].box.tolist()}", file=log_file)
                else:
                    print(f"Speaker {speaker}: None", file=log_file)
        
        return result

    def _run_ocr_on_nickname(self, nickname_box: TextBox, image: np.ndarray, log_file=None) -> str:
        """
        Extract text from nickname box using OCR.
        
        This method uses the ChatTextRecognition model to perform OCR on a cropped
        region of the image corresponding to the nickname box. It handles OCR errors
        gracefully and cleans the result by removing trailing special characters.
        
        Args:
            nickname_box: TextBox containing the nickname coordinates
            image: Original image array (RGB format)
            log_file: Optional logging file
            
        Returns:
            Extracted and cleaned text string, or None if OCR fails
            
        Requirements:
            - 4.1: Use ChatTextRecognition to perform OCR
            - 4.2: Pass original image and box coordinates to OCR model
            - 4.3: Handle OCR errors gracefully
            - 4.4: Strip trailing special characters
        """
        if log_file:
            print(f"=== Running OCR on nickname box: {nickname_box.box.tolist()} ===", file=log_file)
        
        try:
            # Import ChatTextRecognition
            from screenshotanalysis.core import ChatTextRecognition
            
            # Crop image to nickname box coordinates
            x_min, y_min, x_max, y_max = int(nickname_box.x_min), int(nickname_box.y_min), int(nickname_box.x_max), int(nickname_box.y_max)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if log_file:
                print(f"Cropping region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]", file=log_file)
                print(f"Image dimensions: {w}x{h}", file=log_file)
            
            # Validate crop region
            if x_max <= x_min or y_max <= y_min:
                if log_file:
                    print(f"Invalid crop region: width={x_max-x_min}, height={y_max-y_min}", file=log_file)
                return None
            
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            if log_file:
                print(f"Cropped image shape: {cropped_image.shape}", file=log_file)
            
            # Initialize OCR model
            text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
            text_rec.load_model()
            
            if log_file:
                print("OCR model loaded successfully", file=log_file)
            
            # Run OCR on cropped region
            ocr_result = text_rec.predict_text(cropped_image)
            
            if log_file:
                print(f"Raw OCR result: {ocr_result}", file=log_file)
            
            # Extract text from OCR result
            # The predict_text method returns a list of dictionaries with 'rec_text' key
            if ocr_result and len(ocr_result) > 0:
                # Get the first result
                first_result = ocr_result[0]
                
                # Handle different OCR result formats
                if isinstance(first_result, dict):
                    # Dictionary format: {'rec_text': 'text', 'rec_score': 0.95, ...}
                    text = first_result.get('rec_text', '')
                elif isinstance(first_result, tuple):
                    # Tuple format: (text, confidence)
                    text = first_result[0]
                else:
                    # String format
                    text = str(first_result)
                
                # Clean result: strip trailing special characters
                # Remove trailing '>', '<', '|', and whitespace
                cleaned_text = text.rstrip('>< |\t\n\r')
                
                if log_file:
                    print(f"Cleaned text: '{cleaned_text}'", file=log_file)
                
                return cleaned_text if cleaned_text else None
            else:
                if log_file:
                    print("OCR returned empty result", file=log_file)
                return None
        except Exception as e:
            if log_file:
                print(f"OCR error: {e}", file=log_file)
            return None

    # Smart nickname detection methods (improved scoring-based approach)
    def _is_extreme_edge_box(self, box: TextBox, screen_width: int, screen_height: int) -> bool:
        """
        Check if text box is in extreme edge (system UI region).
        
        More refined edge detection:
        - Left edge: x < 15% AND y < 8% AND width < 25%
        - Right edge: x > 85% AND y < 8% AND width < 15%
        
        Args:
            box: TextBox to check
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            True if box is in extreme edge region
        """
        # Edge thresholds
        left_edge_x = screen_width * 0.15
        right_edge_x = screen_width * 0.85
        top_edge_y = screen_height * 0.08
        
        # Width thresholds
        max_width_left = screen_width * 0.25
        max_width_right = screen_width * 0.15
        
        # Check left top corner
        is_left_top = (box.x_min < left_edge_x and 
                       box.y_max < top_edge_y and
                       box.width < max_width_left)
        
        # Check right top corner (more strict)
        is_right_top = (box.x_max > right_edge_x and 
                        box.y_max < top_edge_y and
                        box.width < max_width_right)
        
        return is_left_top or is_right_top

    def _is_likely_system_text(self, text: str) -> bool:
        """
        Check if text is likely system UI text.
        
        Args:
            text: Text string to check
            
        Returns:
            True if text appears to be system text
        """
        import re
        
        if not text or len(text.strip()) == 0:
            return True
        
        text = text.strip()
        
        # Time format (HH:MM or HH:MM:SS)
        if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', text):
            return True
        
        # Pure numbers (battery percentage, etc.)
        if text.replace('%', '').replace('.', '').isdigit():
            return True
        
        # Single character
        if len(text) <= 1:
            return True
        
        # Common system keywords
        system_keywords = ['5G', '4G', '3G', 'LTE', 'WIFI']
        if text.upper() in system_keywords:
            return True
        
        # Status text (online, typing, last seen, etc.)
        status_keywords = [
            'online', 'offline', 'typing', 'active', 'away',
            '在线', '离线', '正在输入', '活跃', '离开',
            'last seen', 'active now', 'recently',
            '最近在线', '刚刚在线', '活跃中'
        ]
        text_lower = text.lower()
        for keyword in status_keywords:
            if keyword in text_lower:
                return True
        
        return False

    def _calculate_nickname_score(self, box: TextBox, text: str, 
                                  screen_width: int, screen_height: int,
                                  y_rank: int = None):
        """
        Calculate comprehensive score for nickname candidate.
        
        Scoring factors:
        1. Position score (0-15): Closer to screen center = higher score
        2. Text score (0-30): Not system text
        3. Y position score (0-15): In top region but not extreme top
        4. Height score (0-30): Larger font height = higher score (nicknames use larger fonts) ⬆️ 增加权重
        5. Y rank score (0-10): Ranking based on Y position (1st=10, 2nd=7, 3rd=5) ⬇️ 降低权重
        
        总分：100分
        
        Args:
            box: TextBox candidate
            text: OCR text from box
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            y_rank: Y-direction ranking (1=topmost, 2=second, 3=third, etc.)
            
        Returns:
            Tuple of (total_score, score_breakdown_dict)
        """
        # 1. Position score (0-15): closer to center = higher
        screen_center_x = screen_width / 2
        distance_from_center = abs(box.center_x - screen_center_x)
        normalized_distance = distance_from_center / (screen_width / 2)
        position_score = (1 - normalized_distance) * 15
        
        # 2. Text score (0-30): not system text
        if not self._is_likely_system_text(text):
            text_score = 30
        else:
            text_score = 0
        
        # 3. Y position score (0-15): in top region but not extreme top
        top_region = screen_height * 0.15
        if box.y_min < top_region:
            if box.y_min > screen_height * 0.05:
                y_score = 15
            else:
                y_score = 7  # Too close to top edge
        else:
            y_score = 0
        
        # 4. Height score (0-30): larger font height = higher score ⬆️ 从20提高到30
        # Nicknames typically use larger fonts than other text
        # Normalize height relative to screen height
        # Typical nickname height: 2-5% of screen height
        height_ratio = box.height / screen_height
        
        # Ideal height range for nicknames
        ideal_height_min = 0.02  # 2% of screen height
        ideal_height_max = 0.08  # 8% of screen height
        
        if ideal_height_min <= height_ratio <= ideal_height_max:
            # Within ideal range, score based on how large it is
            # Larger within range = higher score
            normalized_height = (height_ratio - ideal_height_min) / (ideal_height_max - ideal_height_min)
            height_score = normalized_height * 30  # ⬆️ 从20改为30
        elif height_ratio < ideal_height_min:
            # Too small, penalize more
            height_score = (height_ratio / ideal_height_min) * 15  # ⬆️ 从10改为15
        else:
            # Too large, penalize
            height_score = (ideal_height_max / height_ratio) * 15  # ⬆️ 从10改为15
        
        # 5. Y rank score (0-10): ranking based on Y position ⬇️ 从20降低到10
        if y_rank is not None:
            if y_rank == 1:
                y_rank_score = 10  # ⬇️ 从20改为10
            elif y_rank == 2:
                y_rank_score = 7   # ⬇️ 从15改为7
            elif y_rank == 3:
                y_rank_score = 5   # ⬇️ 从10改为5
            else:
                y_rank_score = 0
        else:
            y_rank_score = 0
        
        # Calculate total score
        total_score = position_score + text_score + y_score + height_score + y_rank_score
        
        # Build breakdown dictionary
        breakdown = {
            'position': position_score,
            'text': text_score,
            'y_position': y_score,
            'height': height_score,
            'y_rank': y_rank_score,
            'total': total_score
        }
        
        return total_score, breakdown

    def extract_nicknames_smart(self, text_det_results: List, image: np.ndarray, 
                               log_file=None) -> Dict[str, Any]:
        """
        Extract nicknames using smart scoring-based detection.
        
        This method uses a comprehensive scoring system to identify nicknames:
        - Filters extreme edge boxes (system UI)
        - Focuses on top 20% of screen
        - Scores candidates based on position, size, and text characteristics
        - Returns top candidates with scores
        
        Args:
            text_det_results: Raw results from PP-OCRv5_server_det
            image: Original image array (for OCR and dimensions)
            log_file: Optional file object for logging
            
        Returns:
            Dictionary with detected nicknames and scores:
            {
                'candidates': [
                    {
                        'text': str,
                        'score': float,
                        'box': list,
                        'center_x': float,
                        'y_min': float
                    },
                    ...
                ],
                'top_candidate': dict or None,
                'metadata': {
                    'screen_width': int,
                    'screen_height': int,
                    'total_boxes': int,
                    'filtered_boxes': int,
                    'top_region_boxes': int
                }
            }
        """
        if log_file:
            print("=" * 60, file=log_file)
            print("=== Smart Nickname Detection ===", file=log_file)
            print("=" * 60, file=log_file)
        
        # Get screen dimensions
        screen_height, screen_width = image.shape[:2]
        
        if log_file:
            print(f"Screen dimensions: {screen_width}x{screen_height}", file=log_file)
        
        # Extract text boxes
        text_det_boxes = self._get_all_text_boxes_from_text_det(text_det_results)
        
        if log_file:
            print(f"Detected {len(text_det_boxes)} text boxes", file=log_file)
        
        # Filter extreme edge boxes
        filtered_boxes = []
        edge_boxes = []
        
        for box in text_det_boxes:
            if self._is_extreme_edge_box(box, screen_width, screen_height):
                edge_boxes.append(box)
            else:
                filtered_boxes.append(box)
        
        if log_file:
            print(f"Filtered {len(edge_boxes)} extreme edge boxes", file=log_file)
            print(f"Remaining: {len(filtered_boxes)} candidate boxes", file=log_file)
        
        # Focus on top 20% region
        top_region_boundary = screen_height * 0.20
        top_boxes = [box for box in filtered_boxes if box.y_min < top_region_boundary]
        
        if log_file:
            print(f"Top region (20%): {len(top_boxes)} boxes", file=log_file)
        
        if not top_boxes:
            if log_file:
                print("No candidates in top region", file=log_file)
            return {
                'candidates': [],
                'top_candidate': None,
                'metadata': {
                    'screen_width': screen_width,
                    'screen_height': screen_height,
                    'total_boxes': len(text_det_boxes),
                    'filtered_boxes': len(filtered_boxes),
                    'top_region_boxes': 0
                }
            }
        
        # Initialize OCR
        from screenshotanalysis.core import ChatTextRecognition
        text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
        text_rec.load_model()
        
        # Sort boxes by Y position to calculate rankings
        sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)
        
        # Create a mapping from box to its Y rank
        box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}
        
        # Score each candidate
        candidates = []
        
        if log_file:
            print("\nScoring candidates:", file=log_file)
        
        for box in top_boxes:
            # Crop and OCR
            x_min, y_min, x_max, y_max = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
            
            # Ensure valid coordinates
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            
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
                    
                    cleaned_text = text.rstrip('>< |\t\n\r')
                    
                    if not cleaned_text:
                        continue
                    
                    # Get Y rank for this box
                    y_rank = box_to_rank.get(id(box), None)
                    
                    # Calculate score with Y rank
                    nickname_score, score_breakdown = self._calculate_nickname_score(
                        box, cleaned_text, screen_width, screen_height, y_rank=y_rank
                    )
                    
                    candidates.append({
                        'text': cleaned_text,
                        'score': nickname_score,
                        'score_breakdown': score_breakdown,
                        'ocr_score': ocr_score,
                        'box': box.box.tolist(),
                        'center_x': box.center_x,
                        'y_min': box.y_min,
                        'y_rank': y_rank
                    })
                    
                    if log_file:
                        print(f"  '{cleaned_text}' -> score: {nickname_score:.1f} "
                              f"(OCR: {ocr_score:.3f}, pos: {box.center_x:.0f}, y: {box.y_min:.0f}, rank: {y_rank})", 
                              file=log_file)
                        print(f"    Breakdown: pos={score_breakdown['position']:.1f}, "
                              f"text={score_breakdown['text']:.1f}, "
                              f"y={score_breakdown['y_position']:.1f}, "
                              f"height={score_breakdown['height']:.1f}, "
                              f"y_rank={score_breakdown['y_rank']:.1f}", 
                              file=log_file)
                        
            except Exception as e:
                if log_file:
                    print(f"  Error processing box {box.box.tolist()}: {e}", file=log_file)
                continue
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        top_candidate = candidates[0] if candidates else None
        
        if log_file:
            print(f"\nTop candidates:", file=log_file)
            for i, c in enumerate(candidates[:3], 1):
                print(f"  {i}. '{c['text']}' (score: {c['score']:.1f})", file=log_file)
        
        return {
            'candidates': candidates,
            'top_candidate': top_candidate,
            'metadata': {
                'screen_width': screen_width,
                'screen_height': screen_height,
                'total_boxes': len(text_det_boxes),
                'filtered_boxes': len(filtered_boxes),
                'top_region_boxes': len(top_boxes)
            }
        }

    def extract_nicknames_adaptive(self, layout_det_results: List, text_det_results: List,
                                   image: np.ndarray, screen_width: int,
                                   memory_path: str = None, log_file=None) -> Dict[str, Any]:
        """
        Extract nicknames for both speakers using app-agnostic methods.
        
        This is the main entry point for nickname extraction. It uses a three-tier
        fallback strategy: layout_det detection → avatar-neighbor search → top-region search.
        The method is completely app-agnostic and relies only on geometric properties.
        
        Args:
            layout_det_results: Raw results from PP-DocLayoutV2
            text_det_results: Raw results from PP-OCRv5_server_det
            image: Original image array (for OCR)
            screen_width: Screen width in pixels
            memory_path: Optional path for speaker memory persistence
            log_file: Optional file object for logging
            
        Returns:
            Dictionary with the following structure:
            {
                'speaker_A': {
                    'nickname': str or None,
                    'box': TextBox or None,
                    'method': str  # 'layout_det', 'avatar_neighbor', 'top_region', or 'none'
                },
                'speaker_B': {
                    'nickname': str or None,
                    'box': TextBox or None,
                    'method': str
                },
                'metadata': {
                    'layout': str,  # from ChatLayoutDetector
                    'confidence': float,
                    'frame_count': int
                }
            }
            
        Requirements:
            - 5.1, 5.2, 5.3, 5.4: Speaker assignment integration
            - 6.1, 6.2, 6.3: App-agnostic implementation
            - 7.1, 7.2, 7.3, 7.4: Fallback chain execution
            - 8.1, 8.2, 8.3, 8.4, 8.5: Logging and debugging support
        """
        if log_file:
            print("=" * 60, file=log_file)
            print("=== Nickname Extraction (Adaptive) ===", file=log_file)
            print("=" * 60, file=log_file)
            print(f"Screen width: {screen_width}", file=log_file)
        
        # Initialize result structure for both speakers
        result = {
            'speaker_A': {
                'nickname': None,
                'box': None,
                'method': 'none'
            },
            'speaker_B': {
                'nickname': None,
                'box': None,
                'method': 'none'
            },
            'metadata': {
                'layout': 'unknown',
                'confidence': 0.0,
                'frame_count': 0
            }
        }
        
        # Extract layout_det boxes (all types)
        layout_det_boxes = self._get_all_boxes_from_layout_det(layout_det_results)
        
        if log_file:
            print(f"\nExtracted {len(layout_det_boxes)} layout_det boxes", file=log_file)
            layout_types = {}
            for box in layout_det_boxes:
                layout_types[box.layout_det] = layout_types.get(box.layout_det, 0) + 1
            print(f"Layout types: {layout_types}", file=log_file)
        
        # Extract text_det boxes
        text_det_boxes = self._get_all_text_boxes_from_text_det(text_det_results)
        
        if log_file:
            print(f"Extracted {len(text_det_boxes)} text_det boxes", file=log_file)
        
        # Get screen height from image
        screen_height = image.shape[0]
        
        if log_file:
            print(f"Screen dimensions: {screen_width}x{screen_height}", file=log_file)
        
        # Assign speakers using assign_speakers_to_layout_det_boxes
        layout_det_with_speakers = self.assign_speakers_to_layout_det_boxes(
            layout_det_boxes,
            screen_width=screen_width,
            memory_path=memory_path,
            log_file=log_file
        )
        
        # Get layout metadata from ChatLayoutDetector
        detector = ChatLayoutDetector(screen_width=screen_width, memory_path=memory_path)
        layout_result = detector.process_frame(layout_det_boxes)
        result['metadata']['layout'] = layout_result['layout']
        result['metadata']['confidence'] = layout_result['metadata'].get('confidence', 0.0)
        result['metadata']['frame_count'] = layout_result['metadata'].get('frame_count', 0)
        
        if log_file:
            print(f"\nLayout detected: {result['metadata']['layout']}", file=log_file)
            print(f"Confidence: {result['metadata']['confidence']:.2f}", file=log_file)
        
        # Try Method 1: _extract_from_layout_det
        if log_file:
            print("\n" + "=" * 60, file=log_file)
            print("Attempting Method 1: Layout Det Nickname Detection", file=log_file)
            print("=" * 60, file=log_file)
        
        method1_results = self._extract_from_layout_det(layout_det_with_speakers, log_file)
        
        # Store Method 1 results
        for speaker in ['A', 'B']:
            if method1_results[speaker] is not None:
                result[f'speaker_{speaker}']['box'] = method1_results[speaker]
                result[f'speaker_{speaker}']['method'] = 'layout_det'
                if log_file:
                    print(f"Method 1 found nickname for Speaker {speaker}", file=log_file)
        
        # For each speaker without nickname, try Method 2: _extract_from_avatar_neighbor
        speakers_needing_method2 = [s for s in ['A', 'B'] if result[f'speaker_{s}']['box'] is None]
        
        if speakers_needing_method2:
            if log_file:
                print("\n" + "=" * 60, file=log_file)
                print(f"Attempting Method 2: Avatar-Neighbor Search for speakers: {speakers_needing_method2}", file=log_file)
                print("=" * 60, file=log_file)
            
            # Get avatar boxes with speaker assignments
            avatar_boxes = [box for box in layout_det_with_speakers if box.layout_det == 'avatar']
            
            if log_file:
                print(f"Found {len(avatar_boxes)} avatar boxes", file=log_file)
            
            method2_results = self._extract_from_avatar_neighbor(avatar_boxes, text_det_boxes, log_file)
            
            # Store Method 2 results (only for speakers that don't have nicknames yet)
            for speaker in speakers_needing_method2:
                if method2_results[speaker] is not None:
                    result[f'speaker_{speaker}']['box'] = method2_results[speaker]
                    result[f'speaker_{speaker}']['method'] = 'avatar_neighbor'
                    if log_file:
                        print(f"Method 2 found nickname for Speaker {speaker}", file=log_file)
        else:
            if log_file:
                print("\nSkipping Method 2: All speakers already have nicknames", file=log_file)
        
        # For each speaker still without nickname, try Method 3: _extract_from_top_region
        speakers_needing_method3 = [s for s in ['A', 'B'] if result[f'speaker_{s}']['box'] is None]
        
        if speakers_needing_method3:
            if log_file:
                print("\n" + "=" * 60, file=log_file)
                print(f"Attempting Method 3: Top-Region Search for speakers: {speakers_needing_method3}", file=log_file)
                print("=" * 60, file=log_file)
            
            method3_results = self._extract_from_top_region(
                text_det_boxes,
                screen_width,
                screen_height,
                layout_det_boxes=layout_det_with_speakers,
                log_file=log_file
            )
            
            # Store Method 3 results (only for speakers that don't have nicknames yet)
            for speaker in speakers_needing_method3:
                if method3_results[speaker] is not None:
                    result[f'speaker_{speaker}']['box'] = method3_results[speaker]
                    result[f'speaker_{speaker}']['method'] = 'top_region'
                    if log_file:
                        print(f"Method 3 found nickname for Speaker {speaker}", file=log_file)
        else:
            if log_file:
                print("\nSkipping Method 3: All speakers already have nicknames", file=log_file)
        
        # Run OCR on all detected nickname boxes
        if log_file:
            print("\n" + "=" * 60, file=log_file)
            print("Running OCR on detected nickname boxes", file=log_file)
            print("=" * 60, file=log_file)
        
        for speaker in ['A', 'B']:
            nickname_box = result[f'speaker_{speaker}']['box']
            if nickname_box is not None:
                if log_file:
                    print(f"\nProcessing Speaker {speaker}:", file=log_file)
                
                nickname_text = self._run_ocr_on_nickname(nickname_box, image, log_file)
                result[f'speaker_{speaker}']['nickname'] = nickname_text
                
                if log_file:
                    if nickname_text:
                        print(f"Successfully extracted nickname: '{nickname_text}'", file=log_file)
                    else:
                        print("OCR failed to extract text", file=log_file)
            else:
                if log_file:
                    print(f"\nSpeaker {speaker}: No nickname box found", file=log_file)
        
        # Final summary
        if log_file:
            print("\n" + "=" * 60, file=log_file)
            print("=== Final Results ===", file=log_file)
            print("=" * 60, file=log_file)
            for speaker in ['A', 'B']:
                speaker_key = f'speaker_{speaker}'
                nickname = result[speaker_key]['nickname']
                method = result[speaker_key]['method']
                box = result[speaker_key]['box']
                
                if nickname:
                    print(f"Speaker {speaker}: '{nickname}' (method: {method})", file=log_file)
                    if box:
                        print(f"  Box: {box.box.tolist()}", file=log_file)
                else:
                    print(f"Speaker {speaker}: None (method: {method})", file=log_file)
            
            print(f"\nLayout: {result['metadata']['layout']}", file=log_file)
            print(f"Confidence: {result['metadata']['confidence']:.2f}", file=log_file)
            print("=" * 60, file=log_file)
        
        return result

class LayoutVisualizer:
    """版面分析结果可视化器"""
    
    def __init__(self):
        self.colors = {
            'text': (255, 0, 0),      # 红色
            'image': (0, 255, 0),     # 绿色  
            'table': (0, 0, 255),     # 蓝色
            'formula': (255, 255, 0), # 黄色
            'chart': (255, 0, 255)    # 紫色
        }
    
    def draw_layout(self, image_path: str, analysis_result: Dict[str, Any], 
                   output_path: str = None) -> Image.Image:
        """绘制版面分析结果"""
        # 打开原始图像
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
            
        draw = ImageDraw.Draw(image)
        
        # 为每个元素绘制边界框
        for element in analysis_result.get('chat_elements', []):
            self._draw_element(draw, element)
            
        # 绘制消息分组
        self._draw_message_groups(draw, analysis_result.get('message_groups', []))
        
        if output_path:
            image.save(output_path)
            print(f"可视化结果已保存至: {output_path}")
            
        return image
    
    def _draw_element(self, draw: ImageDraw.Draw, element: Dict[str, Any]):
        """绘制单个元素"""
        bbox = element['bbox']
        category = element['category']
        color = self.colors.get(category, (128, 128, 128))  # 默认灰色
        
        # 绘制矩形框
        draw.rectangle(bbox, outline=color, width=3)
        
        # 添加类别标签
        label = f"{category} ({element['confidence']:.2f})"
        draw.text((bbox[0], bbox[1] - 20), label, fill=color)
    
    def _draw_message_groups(self, draw: ImageDraw.Draw, message_groups: List[List[Dict]]):
        """绘制消息分组"""
        for i, group in enumerate(message_groups):
            if not group:
                continue
                
            # 计算消息组的整体边界框
            x1 = min(elem['bbox'][0] for elem in group)
            y1 = min(elem['bbox'][1] for elem in group)
            x2 = max(elem['bbox'][2] for elem in group)
            y2 = max(elem['bbox'][3] for elem in group)
            
            # 绘制消息组边界
            draw.rectangle([x1-5, y1-5, x2+5, y2+5], outline=(0, 255, 255), width=2)
            draw.text((x1, y1-25), f"Message {i+1}", fill=(0, 255, 255))