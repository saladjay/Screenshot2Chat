
"""聊天内容定位分析工具包"""
__version__ = "0.1.0"

from .core import ChatLayoutAnalyzer, ChatTextRecognition
from .processors import ChatMessageProcessor, LayoutVisualizer
from .nickname_extractor import extract_nicknames_smart, draw_top3_results
from .app_agnostic_text_boxes import (
    assign_speaker_by_avatar_order,
    assign_speaker_by_center_x,
    assign_speaker_by_edges,
    assign_speaker_by_nearest_avatar,
    assign_speaker_by_nickname_in_layout,
    compute_iou,
    detect_left_only_layout,
    draw_boxes_by_speaker,
    evaluate_against_gt,
    filter_by_frequent_edges,
    filter_center_near_boxes,
    filter_small_layout_boxes,
    save_detection_coords,
    select_layout_text_boxes,
    suppress_nested_boxes,
)

__all__ = [
    "ChatLayoutAnalyzer",
    "ChatMessageProcessor", 
    "LayoutVisualizer",
    "ChatTextRecognition",
    "extract_nicknames_smart",
    "draw_top3_results",
    "assign_speaker_by_avatar_order",
    "assign_speaker_by_center_x",
    "assign_speaker_by_edges",
    "assign_speaker_by_nearest_avatar",
    "assign_speaker_by_nickname_in_layout",
    "compute_iou",
    "detect_left_only_layout",
    "draw_boxes_by_speaker",
    "evaluate_against_gt",
    "filter_by_frequent_edges",
    "filter_center_near_boxes",
    "filter_small_layout_boxes",
    "save_detection_coords",
    "select_layout_text_boxes",
    "suppress_nested_boxes",
    "layout_det",
    "text_det",
    "en_rec"
]


layout_det = ChatLayoutAnalyzer("PP-DocLayoutV2")
text_det = ChatLayoutAnalyzer("PP-OCRv5_server_det")
en_rec = ChatTextRecognition("PP-OCRv5_server_rec")
