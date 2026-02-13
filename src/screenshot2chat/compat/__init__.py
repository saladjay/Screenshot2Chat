"""向后兼容层

提供与旧版 API 兼容的包装器，确保现有代码可以无缝迁移到新架构。
"""

from .chat_layout_detector import ChatLayoutDetector

__all__ = [
    "ChatLayoutDetector",
]
