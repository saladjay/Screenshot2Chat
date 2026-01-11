
"""聊天内容定位分析工具包"""
__version__ = "0.1.0"

from .core import ChatLayoutAnalyzer, ChatTextRecognition
from .processors import ChatMessageProcessor, LayoutVisualizer

__all__ = [
    "ChatLayoutAnalyzer",
    "ChatMessageProcessor", 
    "LayoutVisualizer",
    "ChatTextRecognition",
    "layout_det",
    "text_det",
    "en_rec"
]


layout_det = ChatLayoutAnalyzer("PP-DocLayoutV2")
text_det = ChatLayoutAnalyzer("PP-OCRv5_server_det")
en_rec = ChatTextRecognition("PP-OCRv5_server_rec")
