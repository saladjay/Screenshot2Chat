"""聊天气泡检测器

包装现有的 ChatLayoutDetector 实现，提供符合 BaseDetector 接口的气泡检测功能。
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from ..core.base_detector import BaseDetector
from ..core.data_models import DetectionResult
from ..core.exceptions import DetectionError, ModelLoadError, DataError
from ..logging import StructuredLogger


class BubbleDetector(BaseDetector):
    """聊天气泡检测器
    
    包装 ChatLayoutDetector 的功能，基于文本框检测结果识别聊天气泡和说话者。
    支持跨截图记忆学习，保持说话者身份的一致性。
    
    Attributes:
        screen_width: 屏幕宽度（像素）
        min_separation_ratio: 最小列分离比例
        memory_alpha: 记忆更新的滑动平均系数
        memory_path: 记忆数据持久化路径
        layout_detector: 内部的 ChatLayoutDetector 实例
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化气泡检测器
        
        Args:
            config: 配置参数，支持以下键：
                - screen_width: 屏幕宽度，默认 720
                - min_separation_ratio: 最小列分离比例，默认 0.18
                - memory_alpha: 记忆更新系数，默认 0.7
                - memory_path: 记忆文件路径，默认 None
                - save_interval: 自动保存间隔，默认 10
                - auto_load: 是否自动加载模型，默认 False
        """
        super().__init__(config)
        
        self.screen_width = self.get_config("screen_width", 720)
        self.min_separation_ratio = self.get_config("min_separation_ratio", 0.18)
        self.memory_alpha = self.get_config("memory_alpha", 0.7)
        self.memory_path = self.get_config("memory_path", None)
        self.save_interval = self.get_config("save_interval", 10)
        
        self.logger = StructuredLogger(__name__)
        self.logger.set_context(
            detector_type="BubbleDetector",
            screen_width=self.screen_width
        )
        self.layout_detector = None
        
        # 如果配置了自动加载，立即加载模型
        if self.get_config("auto_load", False):
            self.load_model()
    
    def load_model(self) -> None:
        """加载 ChatLayoutDetector
        
        初始化内部的 ChatLayoutDetector 实例。
        
        Raises:
            ModelLoadError: 如果模型加载失败
        """
        if self.is_model_loaded():
            self.logger.info("ChatLayoutDetector already loaded, skipping")
            return
        
        try:
            from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
            
            self.logger.info(
                "Initializing ChatLayoutDetector",
                screen_width=self.screen_width,
                memory_path=self.memory_path
            )
            
            self.layout_detector = ChatLayoutDetector(
                screen_width=self.screen_width,
                min_separation_ratio=self.min_separation_ratio,
                memory_alpha=self.memory_alpha,
                memory_path=self.memory_path,
                save_interval=self.save_interval
            )
            
            # 将 layout_detector 设置为 model 以满足基类接口
            self.model = self.layout_detector
            
            self.logger.info("ChatLayoutDetector loaded successfully")
            
        except ImportError as e:
            error_msg = "ChatLayoutDetector is not available"
            self.logger.error(error_msg, error=str(e))
            raise ModelLoadError(
                f"{error_msg}. "
                "Please ensure screenshotanalysis module is installed. "
                "Recovery suggestion: Install the required module."
            ) from e
        except Exception as e:
            self.logger.error(
                "Failed to load ChatLayoutDetector",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise ModelLoadError(f"Failed to load ChatLayoutDetector: {e}") from e
    
    def detect(self, image: np.ndarray, text_boxes: Optional[List[Any]] = None) -> List[DetectionResult]:
        """执行气泡检测
        
        基于文本框检测结果，识别聊天气泡和说话者。
        
        Args:
            image: 输入图像，numpy array 格式 (H, W, C)
            text_boxes: 文本框列表（TextBox 对象或 DetectionResult 对象）
                       如果为 None，则只返回空列表
        
        Returns:
            检测结果列表，每个结果代表一个聊天气泡
            
        Raises:
            DetectionError: 如果检测失败
            DataError: 如果输入数据无效
        """
        # 确保模型已加载
        try:
            self.ensure_model_loaded()
        except Exception as e:
            raise DetectionError(f"Cannot perform detection: {e}") from e
        
        if text_boxes is None:
            raise DataError(
                "text_boxes is required for bubble detection. "
                "Please provide text detection results first. "
                "Recovery suggestion: Run text detection before bubble detection."
            )
        
        if not text_boxes:
            self.logger.debug("No text boxes provided, returning empty results")
            return []
        
        try:
            self.logger.info(
                "Starting bubble detection",
                num_text_boxes=len(text_boxes)
            )
            
            # 转换 DetectionResult 为 TextBox（如果需要）
            converted_boxes = self._convert_to_textboxes(text_boxes)
            
            # 调用 ChatLayoutDetector 的 process_frame 方法
            result = self.layout_detector.process_frame(converted_boxes)
            
            # 后处理结果
            detection_results = self.postprocess(result, image)
            
            self.logger.info(
                "Bubble detection completed",
                num_bubbles=len(detection_results),
                layout_type=result.get('layout', 'unknown')
            )
            
            return detection_results
            
        except (DataError, DetectionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "Bubble detection failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise DetectionError(
                f"Bubble detection failed: {e}. "
                f"Recovery suggestion: Check text box format and detector configuration."
            ) from e
    
    def _convert_to_textboxes(self, boxes: List[Any]) -> List[Any]:
        """将 DetectionResult 转换为 TextBox 对象
        
        Args:
            boxes: DetectionResult 或 TextBox 对象列表
        
        Returns:
            TextBox 对象列表
        """
        from screenshotanalysis.basemodel import TextBox
        
        converted = []
        
        for box in boxes:
            if isinstance(box, DetectionResult):
                # 转换 DetectionResult 为 TextBox
                text_box = TextBox(
                    box=box.bbox,
                    score=box.score,
                    text_type=box.metadata.get("text_type"),
                    source=box.metadata.get("source"),
                )
                converted.append(text_box)
            else:
                # 假设已经是 TextBox 对象
                converted.append(box)
        
        return converted
    
    def postprocess(self, raw_results: Dict[str, Any], image: np.ndarray) -> List[DetectionResult]:
        """后处理气泡检测结果
        
        将 ChatLayoutDetector 的输出转换为标准的 DetectionResult 格式。
        
        Args:
            raw_results: ChatLayoutDetector 的输出字典
            image: 原始图像（用于获取尺寸信息）
        
        Returns:
            标准化的检测结果列表
        """
        detection_results = []
        
        try:
            layout_type = raw_results.get("layout", "unknown")
            metadata_base = raw_results.get("metadata", {})
            
            # 处理 Speaker A 的气泡
            for box in raw_results.get("A", []):
                bbox = self._extract_bbox(box)
                if bbox:
                    detection_results.append(
                        DetectionResult(
                            bbox=bbox,
                            score=getattr(box, 'score', 1.0),
                            category="bubble",
                            metadata={
                                "speaker": "A",
                                "layout": layout_type,
                                "text": getattr(box, 'text', None),
                                **metadata_base
                            }
                        )
                    )
            
            # 处理 Speaker B 的气泡
            for box in raw_results.get("B", []):
                bbox = self._extract_bbox(box)
                if bbox:
                    detection_results.append(
                        DetectionResult(
                            bbox=bbox,
                            score=getattr(box, 'score', 1.0),
                            category="bubble",
                            metadata={
                                "speaker": "B",
                                "layout": layout_type,
                                "text": getattr(box, 'text', None),
                                **metadata_base
                            }
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to postprocess bubble results: {e}")
            raise
        
        return detection_results
    
    def _extract_bbox(self, box: Any) -> Optional[List[float]]:
        """从 TextBox 对象提取边界框
        
        Args:
            box: TextBox 对象
        
        Returns:
            边界框 [x_min, y_min, x_max, y_max]，如果提取失败则返回 None
        """
        try:
            if hasattr(box, 'box'):
                # TextBox 对象
                if isinstance(box.box, np.ndarray):
                    bbox_array = box.box
                else:
                    bbox_array = np.array(box.box)
                
                # 确保是 [x_min, y_min, x_max, y_max] 格式
                if bbox_array.shape == (4,):
                    return bbox_array.tolist()
                elif bbox_array.shape == (4, 2):
                    # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 格式
                    x_coords = bbox_array[:, 0]
                    y_coords = bbox_array[:, 1]
                    return [
                        float(x_coords.min()),
                        float(y_coords.min()),
                        float(x_coords.max()),
                        float(y_coords.max())
                    ]
            
            # 尝试直接访问坐标属性
            if hasattr(box, 'x_min') and hasattr(box, 'y_min') and \
               hasattr(box, 'x_max') and hasattr(box, 'y_max'):
                return [
                    float(box.x_min),
                    float(box.y_min),
                    float(box.x_max),
                    float(box.y_max)
                ]
            
        except Exception as e:
            self.logger.warning(f"Failed to extract bbox from box: {e}")
        
        return None
    
    def get_memory_state(self) -> Dict[str, Any]:
        """获取当前记忆状态
        
        Returns:
            包含说话者记忆信息的字典
        """
        self.ensure_model_loaded()
        
        return {
            "A": self.layout_detector.memory.get("A"),
            "B": self.layout_detector.memory.get("B"),
            "frame_count": self.layout_detector.frame_count
        }
    
    def reset_memory(self) -> None:
        """重置跨截图记忆
        
        清除所有历史记忆，从头开始学习。
        """
        self.ensure_model_loaded()
        
        self.layout_detector.memory["A"] = None
        self.layout_detector.memory["B"] = None
        self.layout_detector.frame_count = 0
        
        self.logger.info("Memory reset successfully")
    
    def save_memory(self) -> None:
        """手动保存记忆到磁盘"""
        self.ensure_model_loaded()
        
        if self.layout_detector.memory_path:
            self.layout_detector._save_memory()
            self.logger.info(f"Memory saved to {self.layout_detector.memory_path}")
        else:
            self.logger.warning("No memory_path configured, cannot save memory")

