import json
from typing import Dict, Any, List
import cv2
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from copy import deepcopy
from screenshotanalysis.experience_formula import *
from screenshotanalysis.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
NAME_LINE = 'name_line' # 用户名字
MULTI_LINE = 'multi_line' # 多行聊天框
SINGLE_LINE = 'single_line' # 单行聊天框

LAYOUT_DET = 'layout_det'
TEXT_DET = 'text_det'
MAGIC_MARGIN = 5
MAGIC_MARGIN_PERCENTAGE = 0.01

MAIN_HEIGHT_TOLERANCE = 0.25
MAIN_AREA_TOLERANCE = 0.2
class TextBox:
    def __init__(self, box, score, **kwargs):
        self.box = box
        self.score = score
        if isinstance(self.box, list):
            self.box = np.array(self.box)
        self.text_type = None
        self.source = None
        self.layout_det = None

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.x_min, self.y_min, self.x_max, self.y_max = self.box.tolist()

    @property
    def min_x(self): 
        return self.x_min 

    @property
    def min_y(self): 
        return self.y_min

    @property
    def max_x(self): 
        return self.x_max

    @property
    def max_y(self): 
        return self.y_max

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self):
        return (self.y_min + self.y_max) / 2

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min

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
        print(f'main_area:{main_area_value}', file=log_file)
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

        main_height = self.estimate_main_value(text_det_text_boxes, 'height', bin_size=2, log_file=log_file)
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
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # 左上角交点 (n,m,2)
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角交点 (n,m,2)
        
        # 计算交集区域的宽高并确保非负
        wh = np.maximum(rb - lt, 0)  # (n,m,2)
        
        # 计算交集面积
        inter = wh[:, :, 0] * wh[:, :, 1]  # (n,m)
        
        # 计算覆盖率 = 交集面积 / boxes1面积
        coverage_matrix = inter / np.maximum(area1[:, None], 1e-8)  # (n,m)
        
        return coverage_matrix

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

    def format_conversation(self, layout_det_results, text_det_results, padding, image_sizes, ratios=None, app_type=None, log_file=None):
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
                # keep_ratio = len(height_filtered) / max(len(text_det_text_boxes), 1)
                # if keep_ratio >= min_height_keep_ratio:
                #     text_det_text_boxes = height_filtered

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

    def filter_by_rules_to_find_nickname(self, boxes_from_text_det:list[TextBox], padding, image_sizes, ratios, app_type, log_file=None):
        unfiltered_text_boxes = boxes_from_text_det
        filtered_text_boxes = []
        w, h = image_sizes

        main_height = self.estimate_main_text_height(boxes_from_text_det)
        if log_file:
            print(f"main_height:{main_height}", file = log_file)
        if app_type == DISCORD:
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
        if app_type == DISCORD:
            # discord 都是聊天气泡靠左
            box_left = (w - padding[0] - padding[2]) * ratios[0] + padding[0] # 统计的气泡左边起点在新图片上的像素数值
            if log_file:
                print(f"box_left:{box_left}", file = log_file)
            left_margin = MAGIC_MARGIN * 6
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

    def detect_chat_layout_adaptive(self, text_boxes: List[TextBox], screen_width: int, 
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
        result = detector.process_frame(text_boxes)
        
        # 可选：记录日志
        if log_file:
            print(f"Layout detected: {result['layout']}", file=log_file)
            print(f"Speaker A boxes: {len(result['A'])}", file=log_file)
            print(f"Speaker B boxes: {len(result['B'])}", file=log_file)
            print(f"Metadata: {result['metadata']}", file=log_file)
        
        return result
    
    def format_conversation_adaptive(self, text_boxes: List[TextBox], screen_width: int,
                                    memory_path: str = None, log_file=None) -> tuple:
        """
        使用自适应检测器格式化对话
        
        这个方法是format_conversation的应用无关版本。
        它返回与现有代码兼容的格式，但使用新的检测逻辑。
        
        Args:
            text_boxes: TextBox对象列表
            screen_width: 屏幕宽度（像素）
            memory_path: 可选的记忆持久化路径
            log_file: 可选的日志文件对象
            
        Returns:
            (sorted_boxes, metadata) 元组:
            - sorted_boxes: 按y坐标排序的文本框列表
            - metadata: 包含布局信息和说话者分配的字典
        """
        # 使用新检测器
        result = self.detect_chat_layout_adaptive(text_boxes, screen_width, memory_path, log_file)
        
        # 合并所有文本框并按y坐标排序
        all_boxes = result['A'] + result['B']
        sorted_boxes = self.sort_boxes_by_y(all_boxes)
        
        # 为每个文本框添加说话者标记
        a_ids = {id(box) for box in result['A']}
        for box in sorted_boxes:
            if id(box) in a_ids:
                box.speaker = 'A'
            else:
                box.speaker = 'B'
        
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