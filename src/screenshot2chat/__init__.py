"""Screenshot2Chat - 聊天截图分析库

这是一个模块化的、可扩展的聊天截图分析框架。
"""

# 导出核心抽象类和数据模型
from .core import (
    BaseDetector,
    BaseExtractor,
    DetectionResult,
    ExtractionResult,
)

# 导出检测器
from .detectors import (
    TextDetector,
    BubbleDetector,
)

# 导出提取器
from .extractors import (
    NicknameExtractor,
    SpeakerExtractor,
    LayoutExtractor,
)

# 导出流水线
from .pipeline import (
    Pipeline,
    PipelineStep,
    StepType,
)

# 导出配置管理器
from .config import (
    ConfigManager,
)

# 导出模型管理器
from .models import (
    ModelManager,
    ModelMetadata,
)

# 导出性能监控
from .monitoring import (
    PerformanceMonitor,
)

# 导出向后兼容层（带弃用警告）
from .compat import (
    ChatLayoutDetector as CompatChatLayoutDetector,
)

__version__ = "0.2.0"

__all__ = [
    # 核心抽象类
    "BaseDetector",
    "BaseExtractor",
    
    # 数据模型
    "DetectionResult",
    "ExtractionResult",
    
    # 检测器
    "TextDetector",
    "BubbleDetector",
    
    # 提取器
    "NicknameExtractor",
    "SpeakerExtractor",
    "LayoutExtractor",
    
    # 流水线
    "Pipeline",
    "PipelineStep",
    "StepType",
    
    # 配置管理
    "ConfigManager",
    
    # 模型管理
    "ModelManager",
    "ModelMetadata",
    
    # 性能监控
    "PerformanceMonitor",
    
    # 向后兼容（已弃用）
    "CompatChatLayoutDetector",
]

