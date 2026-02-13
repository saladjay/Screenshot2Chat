"""提取器模块

提供各种提取器实现，用于从检测结果中提取结构化信息。
"""

from .nickname_extractor import NicknameExtractor
from .speaker_extractor import SpeakerExtractor
from .layout_extractor import LayoutExtractor

__all__ = [
    'NicknameExtractor',
    'SpeakerExtractor',
    'LayoutExtractor',
]
