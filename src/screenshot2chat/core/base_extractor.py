"""提取器抽象基类

定义所有提取器必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .data_models import DetectionResult, ExtractionResult
from .exceptions import ExtractionError, ValidationError, DataError
from ..logging import StructuredLogger


class BaseExtractor(ABC):
    """提取器抽象基类
    
    所有提取器（昵称提取、说话者识别、布局分析等）都应继承此类。
    提取器从检测结果中提取结构化信息。
    
    Attributes:
        config: 提取器配置参数
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化提取器
        
        Args:
            config: 配置参数字典，可选
        """
        self.config = config or {}
        self.logger = StructuredLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract(
        self, 
        detection_results: List[DetectionResult], 
        image: Optional[np.ndarray] = None
    ) -> ExtractionResult:
        """从检测结果中提取信息
        
        这是提取器的核心方法，子类必须实现。
        
        Args:
            detection_results: 检测结果列表
            image: 可选的原始图像，某些提取器可能需要访问原图
        
        Returns:
            提取结果，包含结构化数据和置信度
            
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        pass
    
    def validate(self, result: ExtractionResult) -> bool:
        """验证提取结果的有效性
        
        这是一个模板方法，子类可以重写以实现自定义的验证逻辑。
        默认实现总是返回 True。
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果结果有效，否则 False
        """
        return True
    
    def to_json(self, result: ExtractionResult) -> Dict[str, Any]:
        """将提取结果转换为 JSON 格式
        
        这是一个便捷方法，调用 ExtractionResult 的 to_json() 方法。
        子类可以重写以实现自定义的序列化逻辑。
        
        Args:
            result: 提取结果
        
        Returns:
            JSON 格式的字典
        """
        return result.to_json()
    
    def __call__(
        self, 
        detection_results: List[DetectionResult], 
        image: Optional[np.ndarray] = None
    ) -> ExtractionResult:
        """使提取器可调用
        
        这是一个便捷方法，等同于调用 extract()。
        
        Args:
            detection_results: 检测结果列表
            image: 可选的原始图像
        
        Returns:
            提取结果
        """
        return self.extract(detection_results, image)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
        
        Returns:
            配置值，如果不存在则返回默认值
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键名
            value: 配置值
        """
        self.config[key] = value
    
    def filter_by_category(
        self, 
        detection_results: List[DetectionResult], 
        category: str
    ) -> List[DetectionResult]:
        """按类别过滤检测结果
        
        这是一个辅助方法，用于从检测结果中筛选特定类别的结果。
        
        Args:
            detection_results: 检测结果列表
            category: 要筛选的类别
        
        Returns:
            筛选后的检测结果列表
        """
        return [r for r in detection_results if r.category == category]
    
    def filter_by_score(
        self, 
        detection_results: List[DetectionResult], 
        min_score: float
    ) -> List[DetectionResult]:
        """按置信度分数过滤检测结果
        
        这是一个辅助方法，用于从检测结果中筛选高置信度的结果。
        
        Args:
            detection_results: 检测结果列表
            min_score: 最小置信度阈值
        
        Returns:
            筛选后的检测结果列表
        """
        return [r for r in detection_results if r.score >= min_score]
    
    def sort_by_position(
        self, 
        detection_results: List[DetectionResult], 
        by: str = "y"
    ) -> List[DetectionResult]:
        """按位置排序检测结果
        
        这是一个辅助方法，用于对检测结果进行空间排序。
        
        Args:
            detection_results: 检测结果列表
            by: 排序依据，可选 "x", "y", "x_center", "y_center"
        
        Returns:
            排序后的检测结果列表
        """
        if by == "x":
            return sorted(detection_results, key=lambda r: r.x_min)
        elif by == "y":
            return sorted(detection_results, key=lambda r: r.y_min)
        elif by == "x_center":
            return sorted(detection_results, key=lambda r: r.center_x)
        elif by == "y_center":
            return sorted(detection_results, key=lambda r: r.center_y)
        else:
            raise ValueError(f"Invalid sort key: {by}")
