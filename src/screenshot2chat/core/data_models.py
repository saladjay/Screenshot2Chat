"""核心数据模型

定义检测和提取过程中使用的标准数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DetectionResult:
    """检测结果数据类
    
    用于表示检测器的输出结果，包含边界框、置信度分数、类别等信息。
    
    Attributes:
        bbox: 边界框坐标 [x_min, y_min, x_max, y_max]
        score: 置信度分数，范围 [0.0, 1.0]
        category: 检测类别（如 "text", "avatar", "emoji", "bubble"）
        metadata: 额外的元数据信息
    """
    bbox: List[float]
    score: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据有效性"""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {len(self.bbox)}")
        
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")
        
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("category must be a non-empty string")
    
    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式
        
        Returns:
            包含所有字段的字典
        """
        return {
            "bbox": self.bbox,
            "score": float(self.score),
            "category": self.category,
            "metadata": self.metadata
        }
    
    @property
    def x_min(self) -> float:
        """边界框左上角 x 坐标"""
        return self.bbox[0]
    
    @property
    def y_min(self) -> float:
        """边界框左上角 y 坐标"""
        return self.bbox[1]
    
    @property
    def x_max(self) -> float:
        """边界框右下角 x 坐标"""
        return self.bbox[2]
    
    @property
    def y_max(self) -> float:
        """边界框右下角 y 坐标"""
        return self.bbox[3]
    
    @property
    def center_x(self) -> float:
        """边界框中心 x 坐标"""
        return (self.x_min + self.x_max) / 2
    
    @property
    def center_y(self) -> float:
        """边界框中心 y 坐标"""
        return (self.y_min + self.y_max) / 2
    
    @property
    def width(self) -> float:
        """边界框宽度"""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """边界框高度"""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        """边界框面积"""
        return self.width * self.height


@dataclass
class ExtractionResult:
    """提取结果数据类
    
    用于表示提取器的输出结果，包含提取的结构化数据和置信度。
    
    Attributes:
        data: 提取的结构化数据（字典格式）
        confidence: 提取结果的置信度，范围 [0.0, 1.0]
        metadata: 额外的元数据信息
    """
    data: Dict[str, Any]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据有效性"""
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
    
    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式
        
        Returns:
            包含所有字段的字典
        """
        return {
            "data": self.data,
            "confidence": float(self.confidence),
            "metadata": self.metadata
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """从 data 中获取值
        
        Args:
            key: 键名
            default: 默认值
            
        Returns:
            对应的值，如果不存在则返回默认值
        """
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问
        
        Args:
            key: 键名
            
        Returns:
            对应的值
            
        Raises:
            KeyError: 如果键不存在
        """
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在
        
        Args:
            key: 键名
            
        Returns:
            True 如果键存在，否则 False
        """
        return key in self.data
