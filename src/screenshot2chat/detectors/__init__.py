"""检测器模块

包含各种检测器的实现。
"""

from .text_detector import TextDetector
from .bubble_detector import BubbleDetector

__all__ = [
    "TextDetector",
    "BubbleDetector",
]

