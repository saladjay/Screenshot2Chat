"""检测器抽象基类

定义所有检测器必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .data_models import DetectionResult
from .exceptions import DetectionError, ModelLoadError, DataError
from ..logging import StructuredLogger


class BaseDetector(ABC):
    """检测器抽象基类
    
    所有检测器（文本检测、头像检测、气泡检测等）都应继承此类。
    提供统一的接口和模板方法模式。
    
    Attributes:
        config: 检测器配置参数
        model: 加载的模型实例
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化检测器
        
        Args:
            config: 配置参数字典，可选
        """
        self.config = config or {}
        self.model = None
        self.logger = StructuredLogger(self.__class__.__name__)
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型
        
        子类必须实现此方法来加载具体的模型。
        模型应该存储在 self.model 中。
        
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """执行检测
        
        这是检测器的核心方法，子类必须实现。
        
        Args:
            image: 输入图像，numpy array 格式 (H, W, C)
        
        Returns:
            检测结果列表，每个结果是一个 DetectionResult 对象
            
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像
        
        这是一个模板方法，子类可以重写以实现自定义的预处理逻辑。
        默认实现直接返回原图像。
        
        Args:
            image: 输入图像
        
        Returns:
            预处理后的图像
        """
        return image
    
    def postprocess(self, raw_results: Any) -> List[DetectionResult]:
        """后处理原始结果
        
        这是一个模板方法，子类可以重写以实现自定义的后处理逻辑。
        默认实现假设 raw_results 已经是 List[DetectionResult] 格式。
        
        Args:
            raw_results: 模型的原始输出
        
        Returns:
            标准化的检测结果列表
        """
        if isinstance(raw_results, list) and all(
            isinstance(r, DetectionResult) for r in raw_results
        ):
            return raw_results
        
        # 如果不是标准格式，子类应该重写此方法
        raise NotImplementedError(
            "postprocess must be implemented when raw_results is not List[DetectionResult]"
        )
    
    def __call__(self, image: np.ndarray) -> List[DetectionResult]:
        """使检测器可调用
        
        这是一个便捷方法，等同于调用 detect()。
        
        Args:
            image: 输入图像
        
        Returns:
            检测结果列表
        """
        return self.detect(image)
    
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
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载
        
        Returns:
            True 如果模型已加载，否则 False
        """
        return self.model is not None
    
    def ensure_model_loaded(self) -> None:
        """确保模型已加载
        
        如果模型未加载，则自动加载。
        
        Raises:
            ModelLoadError: 如果模型加载失败
        """
        if not self.is_model_loaded():
            try:
                self.logger.info("Model not loaded, loading now")
                self.load_model()
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(
                    "Failed to load model",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise ModelLoadError(f"Failed to load model: {e}") from e
