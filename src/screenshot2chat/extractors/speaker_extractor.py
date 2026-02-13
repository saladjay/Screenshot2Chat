"""说话者识别提取器

基于ChatLayoutDetector的说话者推断逻辑，识别聊天气泡的说话者。
"""

from typing import List, Dict, Any, Optional

import numpy as np

from ..core.base_extractor import BaseExtractor
from ..core.data_models import DetectionResult, ExtractionResult
from ..core.exceptions import ExtractionError, DataError
from ..logging import StructuredLogger
from screenshotanalysis.basemodel import TextBox
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector


class SpeakerExtractor(BaseExtractor):
    """说话者识别提取器
    
    从气泡检测结果中识别说话者身份（Speaker A和Speaker B）。
    使用ChatLayoutDetector的几何学习方法和跨截图记忆来保持说话者身份的一致性。
    
    工作流程：
    1. 接收文本框检测结果
    2. 使用ChatLayoutDetector进行列分割和说话者推断
    3. 返回每个文本框的说话者标识
    
    Attributes:
        config: 配置参数，包括：
            - screen_width: 屏幕宽度（像素），默认720
            - min_separation_ratio: 最小列分离比例，默认0.18
            - memory_alpha: 记忆更新的滑动平均系数，默认0.7
            - memory_path: 记忆数据持久化路径，可选
            - save_interval: 自动保存间隔（帧数），默认10
            - layout_detector: 可选的ChatLayoutDetector实例
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化说话者识别提取器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 从配置中获取参数
        self.screen_width = self.get_config('screen_width', 720)
        self.min_separation_ratio = self.get_config('min_separation_ratio', 0.18)
        self.memory_alpha = self.get_config('memory_alpha', 0.7)
        self.memory_path = self.get_config('memory_path')
        self.save_interval = self.get_config('save_interval', 10)
        
        # Initialize logger
        self.logger = StructuredLogger(__name__)
        self.logger.set_context(
            extractor_type="SpeakerExtractor",
            screen_width=self.screen_width
        )
        
        # 获取或创建ChatLayoutDetector实例
        self.layout_detector = self.get_config('layout_detector')
        if self.layout_detector is None:
            try:
                self.layout_detector = ChatLayoutDetector(
                    screen_width=self.screen_width,
                    min_separation_ratio=self.min_separation_ratio,
                    memory_alpha=self.memory_alpha,
                    memory_path=self.memory_path,
                    save_interval=self.save_interval
                )
                self.logger.info("Created new ChatLayoutDetector instance")
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
        """从检测结果中识别说话者
        
        Args:
            detection_results: 文本检测结果列表
            image: 原始图像（可选，此提取器不需要）
        
        Returns:
            提取结果，包含说话者分配信息和布局类型
            
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
                "Starting speaker extraction",
                num_detections=len(detection_results)
            )
            
            # 将DetectionResult转换为TextBox
            text_boxes = self._convert_to_textboxes(detection_results)
            
            if not text_boxes:
                self.logger.warning("No text boxes to extract speakers from")
                return ExtractionResult(
                    data={
                        'layout': 'single',
                        'speakers': {},
                        'speaker_A': [],
                        'speaker_B': []
                    },
                    confidence=0.0,
                    metadata={'reason': 'no_text_boxes'}
                )
            
            # 调用ChatLayoutDetector进行说话者推断
            result = self.layout_detector.process_frame(text_boxes)
            
            # 提取结果
            layout_type = result['layout']
            speaker_A_boxes = result['A']
            speaker_B_boxes = result['B']
            metadata = result.get('metadata', {})
            
            # 构建说话者映射（box index -> speaker）
            speakers = {}
            
            # 为每个文本框分配说话者标识
            for i, box in enumerate(text_boxes):
                box_id = id(box)
                
                # 检查box是否在speaker_A中
                if any(id(b) == box_id for b in speaker_A_boxes):
                    speakers[i] = 'A'
                # 检查box是否在speaker_B中
                elif any(id(b) == box_id for b in speaker_B_boxes):
                    speakers[i] = 'B'
                else:
                    speakers[i] = 'unknown'
            
            # 计算置信度
            confidence = metadata.get('confidence', 0.5)
            
            # 如果是单列布局，置信度设为1.0（因为没有歧义）
            if layout_type == 'single':
                confidence = 1.0
            
            # 构建返回数据
            extraction_data = {
                'layout': layout_type,
                'speakers': speakers,
                'speaker_A': [i for i, s in speakers.items() if s == 'A'],
                'speaker_B': [i for i, s in speakers.items() if s == 'B'],
                'num_A': len(speaker_A_boxes),
                'num_B': len(speaker_B_boxes)
            }
            
            self.logger.info(
                "Speaker extraction completed",
                layout=layout_type,
                num_A=len(speaker_A_boxes),
                num_B=len(speaker_B_boxes)
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
                "Speaker extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise ExtractionError(
                f"Failed to extract speakers: {e}. "
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
    
    def validate(self, result: ExtractionResult) -> bool:
        """验证提取结果的有效性
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果结果有效，否则 False
        """
        # 检查必需字段
        required_fields = ['layout', 'speakers', 'speaker_A', 'speaker_B']
        if not all(field in result.data for field in required_fields):
            return False
        
        # 检查layout类型
        valid_layouts = ['single', 'double', 'double_left', 'double_right', 'unknown']
        if result.data['layout'] not in valid_layouts:
            return False
        
        # 检查speakers是否为字典
        if not isinstance(result.data['speakers'], dict):
            return False
        
        # 检查speaker_A和speaker_B是否为列表
        if not isinstance(result.data['speaker_A'], list):
            return False
        if not isinstance(result.data['speaker_B'], list):
            return False
        
        return True
    
    def get_speaker_for_box(
        self, 
        result: ExtractionResult, 
        box_index: int
    ) -> Optional[str]:
        """获取指定文本框的说话者标识
        
        Args:
            result: 提取结果
            box_index: 文本框索引
        
        Returns:
            说话者标识（'A', 'B', 'unknown'），如果索引无效则返回None
        """
        speakers = result.data.get('speakers', {})
        return speakers.get(box_index)
    
    def get_layout_type(self, result: ExtractionResult) -> str:
        """获取布局类型
        
        Args:
            result: 提取结果
        
        Returns:
            布局类型字符串
        """
        return result.data.get('layout', 'unknown')
    
    def is_double_column(self, result: ExtractionResult) -> bool:
        """判断是否为双列布局
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果是双列布局，否则 False
        """
        layout = self.get_layout_type(result)
        return layout.startswith('double')
    
    def get_speaker_boxes(
        self, 
        result: ExtractionResult, 
        speaker: str
    ) -> List[int]:
        """获取指定说话者的所有文本框索引
        
        Args:
            result: 提取结果
            speaker: 说话者标识（'A' 或 'B'）
        
        Returns:
            文本框索引列表
        """
        if speaker == 'A':
            return result.data.get('speaker_A', [])
        elif speaker == 'B':
            return result.data.get('speaker_B', [])
        else:
            return []
    
    def get_memory_state(self) -> Dict[str, Any]:
        """获取当前的记忆状态
        
        Returns:
            记忆状态字典，包含Speaker A和B的几何特征
        """
        return {
            'A': self.layout_detector.memory['A'],
            'B': self.layout_detector.memory['B'],
            'frame_count': self.layout_detector.frame_count
        }
    
    def reset_memory(self) -> None:
        """重置跨截图记忆
        
        清除所有历史记忆，从头开始学习。
        """
        self.layout_detector.memory['A'] = None
        self.layout_detector.memory['B'] = None
        self.layout_detector.frame_count = 0
        logger.info("Memory reset")
