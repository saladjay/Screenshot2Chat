
"""聊天内容定位分析工具包"""
__version__ = "0.1.0"

from .core import ChatLayoutAnalyzer, ChatTextRecognition
from .processors import ChatMessageProcessor, LayoutVisualizer
from .nickname_extractor import extract_nicknames_smart, draw_top3_results

__all__ = [
    "ChatLayoutAnalyzer",
    "ChatMessageProcessor", 
    "LayoutVisualizer",
    "ChatTextRecognition",
    "extract_nicknames_smart",
    "draw_top3_results",
    "layout_det",
    "text_det",
    "en_rec"
]


layout_det = ChatLayoutAnalyzer("PP-DocLayoutV2")
text_det = ChatLayoutAnalyzer("PP-OCRv5_server_det")
en_rec = ChatTextRecognition("PP-OCRv5_server_rec")
