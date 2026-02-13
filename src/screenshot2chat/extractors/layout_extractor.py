"""布局类型提取器

检测聊天界面的布局类型（单列/双列/左对齐/右对齐）。
"""

from typing import List, Dict, Any, Optional

import numpy as np

from ..core.base_extractor import BaseExtractor
from ..core.data_models import DetectionResult, ExtractionResult
from ..core.exceptions import ExtractionError, DataError
from ..logging import StructuredLogger
from screenshotanalysis.basemodel import TextBox
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector


class LayoutExtractor(BaseExtractor):
    """布局类型提取器
    
    从文本框检测结果中分析并识别聊天界面的布局类型。
    支持以下布局类型：
    - single: 单列布局（所有消息在同一列）
    - double: 标准双列布局（左右两列，分别代表不同说话者）
    - double_left: 左对齐双列布局（两列都在屏幕左侧）
    - double_right: 右对齐双列布局（两列都在屏幕右侧）
    
    使用ChatLayoutDetector的列分割算法进行布局检测。
    
    Attributes:
        config: 配置参数，包括：
            - screen_width: 屏幕宽度（像素），默认720
            - min_separation_ratio: 最小列分离比例，默认0.18
            - layout_detector: 可选的ChatLayoutDetector实例
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化布局类型提取器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 从配置中获取参数
        self.screen_width = self.get_config('screen_width', 720)
        self.min_separation_ratio = self.get_config('min_separation_ratio', 0.18)
        
        # Initialize logger
        self.logger = StructuredLogger(__name__)
        self.logger.set_context(
            extractor_type="LayoutExtractor",
            screen_width=self.screen_width
        )
        
        # 获取或创建ChatLayoutDetector实例
        self.layout_detector = self.get_config('layout_detector')
        if self.layout_detector is None:
            try:
                # 创建一个不持久化记忆的临时检测器（仅用于布局检测）
                self.layout_detector = ChatLayoutDetector(
                    screen_width=self.screen_width,
                    min_separation_ratio=self.min_separation_ratio,
                    memory_path=None  # 不持久化记忆
                )
                self.logger.info("Created new ChatLayoutDetector instance for layout detection")
            except Exception as e:
                self.logger.error(
                    "Failed to create ChatLayoutDetector",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise ExtractionError(
                    f"Failed to initialize ChatLayoutDetector: {e}. "
                    "Recovery suggestion: Check ChatLayoutDetector dependencies."
                ) from e
        else:
            self.logger.info("Using provided ChatLayoutDetector instance")
    
    def extract(
        self, 
        detection_results: List[DetectionResult], 
        image: Optional[np.ndarray] = None
    ) -> ExtractionResult:
        """从检测结果中提取布局类型
        
        Args:
            detection_results: 文本检测结果列表
            image: 原始图像（可选，此提取器不需要）
        
        Returns:
            提取结果，包含布局类型和列统计信息
            
        Raises:
            ExtractionError: 如果提取失败
            DataError: 如果输入数据无效
        """
        try:
            # Validate input
            if detection_results is None:
                raise DataError(
                    "detection_results cannot be None. "
                    "Recovery suggestion: Provide valid detection results."
                )
            
            self.logger.info(
                "Starting layout extraction",
                num_detections=len(detection_results)
            )
            
            # 将DetectionResult转换为TextBox
            text_boxes = self._convert_to_textboxes(detection_results)
            
            if not text_boxes:
                self.logger.warning("No text boxes to extract layout from")
                return ExtractionResult(
                    data={
                        'layout_type': 'unknown',
                        'is_single_column': False,
                        'is_double_column': False,
                        'num_columns': 0,
                        'left_boxes': [],
                        'right_boxes': []
                    },
                    confidence=0.0,
                    metadata={'reason': 'no_text_boxes'}
                )
            
            # 调用ChatLayoutDetector进行列分割
            layout_type, left_boxes, right_boxes, fallback_metadata = \
                self.layout_detector.split_columns(text_boxes)
            
            # 判断是单列还是双列
            is_single_column = layout_type == 'single'
            is_double_column = layout_type.startswith('double')
            num_columns = 1 if is_single_column else 2
            
            # 计算列统计信息
            left_stats = self._calculate_column_stats(left_boxes)
            right_stats = self._calculate_column_stats(right_boxes)
            
            # 计算置信度
            confidence = self._calculate_confidence(
                layout_type, 
                left_boxes, 
                right_boxes,
                fallback_metadata
            )
            
            # 构建返回数据
            extraction_data = {
                'layout_type': layout_type,
                'is_single_column': is_single_column,
                'is_double_column': is_double_column,
                'num_columns': num_columns,
                'left_boxes': [i for i, box in enumerate(text_boxes) if box in left_boxes],
                'right_boxes': [i for i, box in enumerate(text_boxes) if box in right_boxes],
                'left_stats': left_stats,
                'right_stats': right_stats
            }
            
            # 构建元数据
            metadata = {
                'num_text_boxes': len(text_boxes),
                'num_left_boxes': len(left_boxes),
                'num_right_boxes': len(right_boxes)
            }
            
            # 如果使用了fallback方法，添加相关信息
            if fallback_metadata is not None:
                metadata.update(fallback_metadata)
            
            self.logger.info(
                "Layout extraction completed",
                layout_type=layout_type,
                num_columns=num_columns,
                confidence=confidence
            )
            
            return ExtractionResult(
                data=extraction_data,
                confidence=confidence,
                metadata=metadata
            )
            
        except (DataError, ExtractionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "Layout extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise ExtractionError(
                f"Failed to extract layout: {e}. "
                f"Recovery suggestion: Check detection results format and ChatLayoutDetector configuration."
            ) from e
    
    def _convert_to_textboxes(
        self, 
        detection_results: List[DetectionResult]
    ) -> List[TextBox]:
        """将DetectionResult转换为TextBox对象
        
        Args:
            detection_results: 检测结果列表
        
        Returns:
            TextBox对象列表
        """
        text_boxes = []
        
        for result in detection_results:
            # 只处理文本类别的检测结果
            if result.category != 'text':
                continue
            
            # 创建TextBox对象
            text_box = TextBox(
                box=result.bbox,
                score=result.score
            )
            
            # 如果metadata中有额外信息，设置到TextBox
            if 'text' in result.metadata:
                text_box.text = result.metadata['text']
            if 'text_type' in result.metadata:
                text_box.text_type = result.metadata['text_type']
            if 'source' in result.metadata:
                text_box.source = result.metadata['source']
            
            text_boxes.append(text_box)
        
        return text_boxes
    
    def _calculate_column_stats(
        self, 
        boxes: List[TextBox]
    ) -> Dict[str, float]:
        """计算列的统计信息
        
        Args:
            boxes: TextBox对象列表
        
        Returns:
            统计信息字典，包含center、width、count等
        """
        if not boxes:
            return {
                'center': 0.0,
                'center_normalized': 0.0,
                'width': 0.0,
                'width_normalized': 0.0,
                'count': 0
            }
        
        centers = [box.center_x for box in boxes]
        widths = [box.width for box in boxes]
        
        center = float(np.mean(centers))
        width = float(np.mean(widths))
        
        return {
            'center': center,
            'center_normalized': center / self.screen_width,
            'width': width,
            'width_normalized': width / self.screen_width,
            'count': len(boxes)
        }
    
    def _calculate_confidence(
        self,
        layout_type: str,
        left_boxes: List[TextBox],
        right_boxes: List[TextBox],
        fallback_metadata: Optional[Dict[str, Any]]
    ) -> float:
        """计算布局检测的置信度
        
        Args:
            layout_type: 布局类型
            left_boxes: 左列文本框
            right_boxes: 右列文本框
            fallback_metadata: fallback方法的元数据
        
        Returns:
            置信度值，范围[0.0, 1.0]
        """
        # 如果是单列布局，置信度较高（因为判断相对简单）
        if layout_type == 'single':
            return 0.9
        
        # 如果是双列布局
        if layout_type.startswith('double'):
            # 基础置信度
            base_confidence = 0.7
            
            # 如果使用了fallback方法，降低置信度
            if fallback_metadata is not None:
                base_confidence = 0.6
            
            # 如果两列的文本框数量差异很大，降低置信度
            if left_boxes and right_boxes:
                count_ratio = min(len(left_boxes), len(right_boxes)) / max(len(left_boxes), len(right_boxes))
                if count_ratio < 0.3:
                    base_confidence *= 0.8
            
            # 如果有分离度信息，根据分离度调整置信度
            if fallback_metadata and 'separation' in fallback_metadata:
                separation = fallback_metadata['separation']
                # 分离度越大，置信度越高
                if separation > 0.3:
                    base_confidence = min(1.0, base_confidence * 1.1)
                elif separation < 0.2:
                    base_confidence *= 0.9
            
            return min(1.0, max(0.0, base_confidence))
        
        # 未知布局类型
        return 0.0
    
    def validate(self, result: ExtractionResult) -> bool:
        """验证提取结果的有效性
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果结果有效，否则 False
        """
        # 检查必需字段
        required_fields = [
            'layout_type', 
            'is_single_column', 
            'is_double_column',
            'num_columns',
            'left_boxes',
            'right_boxes'
        ]
        if not all(field in result.data for field in required_fields):
            return False
        
        # 检查layout_type类型
        valid_layouts = ['single', 'double', 'double_left', 'double_right', 'unknown']
        if result.data['layout_type'] not in valid_layouts:
            return False
        
        # 检查布尔字段
        if not isinstance(result.data['is_single_column'], bool):
            return False
        if not isinstance(result.data['is_double_column'], bool):
            return False
        
        # 检查num_columns
        if not isinstance(result.data['num_columns'], int):
            return False
        if result.data['num_columns'] not in [0, 1, 2]:
            return False
        
        # 检查列索引列表
        if not isinstance(result.data['left_boxes'], list):
            return False
        if not isinstance(result.data['right_boxes'], list):
            return False
        
        return True
    
    def get_layout_type(self, result: ExtractionResult) -> str:
        """获取布局类型
        
        Args:
            result: 提取结果
        
        Returns:
            布局类型字符串
        """
        return result.data.get('layout_type', 'unknown')
    
    def is_single_column(self, result: ExtractionResult) -> bool:
        """判断是否为单列布局
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果是单列布局，否则 False
        """
        return result.data.get('is_single_column', False)
    
    def is_double_column(self, result: ExtractionResult) -> bool:
        """判断是否为双列布局
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果是双列布局，否则 False
        """
        return result.data.get('is_double_column', False)
    
    def get_column_boxes(
        self, 
        result: ExtractionResult, 
        column: str
    ) -> List[int]:
        """获取指定列的文本框索引
        
        Args:
            result: 提取结果
            column: 列标识（'left' 或 'right'）
        
        Returns:
            文本框索引列表
        """
        if column == 'left':
            return result.data.get('left_boxes', [])
        elif column == 'right':
            return result.data.get('right_boxes', [])
        else:
            return []
    
    def get_column_stats(
        self, 
        result: ExtractionResult, 
        column: str
    ) -> Dict[str, float]:
        """获取指定列的统计信息
        
        Args:
            result: 提取结果
            column: 列标识（'left' 或 'right'）
        
        Returns:
            统计信息字典
        """
        if column == 'left':
            return result.data.get('left_stats', {})
        elif column == 'right':
            return result.data.get('right_stats', {})
        else:
            return {}
