"""ChatLayoutDetector 向后兼容包装器

提供与旧版 ChatLayoutDetector API 完全兼容的接口，内部使用新的 BubbleDetector 实现。
在使用时会发出弃用警告，提示用户迁移到新 API。
"""

import warnings
import logging
from typing import List, Dict, Any, Optional

from ..detectors.bubble_detector import BubbleDetector


class ChatLayoutDetector:
    """ChatLayoutDetector 向后兼容包装器
    
    这是一个兼容层，包装新的 BubbleDetector 以保持与旧版 API 的兼容性。
    
    警告:
        此类已弃用，建议使用 Pipeline 配合 BubbleDetector。
        详见迁移指南: docs/MIGRATION_GUIDE.md
    
    示例:
        >>> # 旧版用法（仍然支持，但会发出警告）
        >>> detector = ChatLayoutDetector(screen_width=720)
        >>> result = detector.process_frame(text_boxes)
        
        >>> # 推荐的新用法
        >>> from screenshot2chat import BubbleDetector
        >>> detector = BubbleDetector(config={"screen_width": 720})
        >>> detector.load_model()
        >>> results = detector.detect(image, text_boxes=text_boxes)
    
    Attributes:
        screen_width: 屏幕宽度（像素）
        min_separation_ratio: 最小列分离比例
        memory_alpha: 记忆更新的滑动平均系数
        memory_path: 记忆数据持久化路径
        save_interval: 自动保存间隔
        _detector: 内部的 BubbleDetector 实例
        _warned: 是否已发出弃用警告（避免重复警告）
    """
    
    def __init__(
        self,
        screen_width: int,
        min_separation_ratio: float = 0.18,
        memory_alpha: float = 0.7,
        memory_path: Optional[str] = None,
        save_interval: int = 10
    ):
        """初始化 ChatLayoutDetector 兼容包装器
        
        Args:
            screen_width: 屏幕宽度（像素）
            min_separation_ratio: 最小列分离比例，默认 0.18
            memory_alpha: 记忆更新的滑动平均系数，默认 0.7
            memory_path: 记忆数据持久化路径，默认 None
            save_interval: 自动保存间隔（帧数），默认 10
        """
        # 发出弃用警告
        warnings.warn(
            "ChatLayoutDetector is deprecated and will be removed in version 1.0.0. "
            "Please use Pipeline with BubbleDetector instead. "
            "See docs/MIGRATION_GUIDE.md for migration instructions.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.logger = logging.getLogger(__name__)
        self._warned = True  # 标记已发出警告
        
        # 保存参数以保持兼容性
        self.screen_width = screen_width
        self.min_separation_ratio = min_separation_ratio
        self.memory_alpha = memory_alpha
        self.memory_path = memory_path
        self.save_interval = save_interval
        
        # 创建内部的 BubbleDetector 实例
        config = {
            "screen_width": screen_width,
            "min_separation_ratio": min_separation_ratio,
            "memory_alpha": memory_alpha,
            "memory_path": memory_path,
            "save_interval": save_interval,
            "auto_load": True  # 自动加载模型以保持兼容性
        }
        
        self._detector = BubbleDetector(config=config)
        
        # 为了完全兼容，暴露内部 layout_detector 的属性
        self._layout_detector = self._detector.layout_detector
    
    @property
    def memory(self) -> Dict[str, Optional[Dict[str, float]]]:
        """获取跨截图记忆
        
        Returns:
            包含说话者 A 和 B 记忆的字典
        """
        return self._layout_detector.memory
    
    @property
    def frame_count(self) -> int:
        """获取已处理的帧数
        
        Returns:
            已处理的帧数
        """
        return self._layout_detector.frame_count
    
    def process_frame(
        self, 
        boxes: List[Any], 
        layout_det_boxes: Optional[List[Any]] = None,
        text_det_boxes: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """处理单帧截图（兼容旧版接口）
        
        这是主要的公共接口，整合了列分割、说话者推断和记忆更新。
        
        Args:
            boxes: TextBox 对象列表
            layout_det_boxes: 布局检测框（可选，保持兼容性）
            text_det_boxes: 文本检测框（可选，保持兼容性）
        
        Returns:
            包含以下字段的字典：
            - layout: 布局类型 ("single" | "double" | "double_left" | "double_right")
            - A: Speaker A 的文本框列表
            - B: Speaker B 的文本框列表
            - metadata: 元数据字典，包含 frame_count、置信度等
        """
        # 直接调用内部 layout_detector 的 process_frame 方法
        # 这样可以保持完全的行为一致性
        return self._layout_detector.process_frame(
            boxes=boxes,
            layout_det_boxes=layout_det_boxes,
            text_det_boxes=text_det_boxes
        )
    
    def split_columns(
        self, 
        boxes: List[Any]
    ) -> tuple:
        """分割列并判断布局类型（兼容旧版接口）
        
        Args:
            boxes: TextBox 对象列表
        
        Returns:
            四元组 (layout_type, left_boxes, right_boxes, fallback_metadata)
        """
        return self._layout_detector.split_columns(boxes)
    
    def infer_speaker_in_frame(
        self, 
        left: List[Any], 
        right: List[Any]
    ) -> Dict[str, List[Any]]:
        """单帧内推断说话者（兼容旧版接口）
        
        Args:
            left: 左列的 TextBox 列表
            right: 右列的 TextBox 列表
        
        Returns:
            字典 {"A": List[TextBox], "B": List[TextBox]}
        """
        return self._layout_detector.infer_speaker_in_frame(left, right)
    
    def update_memory(self, assigned: Dict[str, List[Any]]) -> None:
        """更新跨截图记忆（兼容旧版接口）
        
        Args:
            assigned: 说话者分配结果，格式为 {"A": List[TextBox], "B": List[TextBox]}
        """
        self._layout_detector.update_memory(assigned)
    
    def calculate_temporal_confidence(
        self, 
        boxes: List[Any], 
        assigned: Dict[str, List[Any]]
    ) -> float:
        """计算基于时序规律的置信度（兼容旧版接口）
        
        Args:
            boxes: 所有 TextBox 对象列表
            assigned: 说话者分配结果
        
        Returns:
            置信度值，范围 [0.0, 1.0]
        """
        return self._layout_detector.calculate_temporal_confidence(boxes, assigned)
    
    def should_use_fallback(self, threshold: int = 50) -> bool:
        """判断是否应该使用 fallback 方法（兼容旧版接口）
        
        Args:
            threshold: 历史数据阈值，默认 50
        
        Returns:
            True 表示应该使用 fallback 方法
        """
        return self._layout_detector.should_use_fallback(threshold)
    
    def split_columns_median_fallback(
        self, 
        boxes: List[Any]
    ) -> tuple:
        """使用 median 方法的 fallback 分列（兼容旧版接口）
        
        Args:
            boxes: TextBox 对象列表
        
        Returns:
            四元组 (layout_type, left_boxes, right_boxes, fallback_metadata)
        """
        return self._layout_detector.split_columns_median_fallback(boxes)
    
    def _save_memory(self) -> None:
        """保存记忆到磁盘（兼容旧版接口）"""
        self._layout_detector._save_memory()
    
    def _load_memory(self) -> None:
        """从磁盘加载记忆（兼容旧版接口）"""
        self._layout_detector._load_memory()
