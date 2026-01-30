"""聊天内容定位分析工具包"""
__version__ = "0.1.0"

from .core import ChatLayoutAnalyzer, ChatTextRecognition
from .processors import ChatMessageProcessor, LayoutVisualizer
from .nickname_extractor import extract_nicknames_smart, draw_top3_results
from .exceptions import (
    AnalysisError,
    ConfigError,
    NicknameAnalysisError,
    NicknameNotFoundError,
    NicknameScoreTooLowError,
    DialogAnalysisError,
    DialogCountTooLowError,
    UnknownSpeakerTooHighError,
)
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
    "AnalysisError",
    "ConfigError",
    "NicknameAnalysisError",
    "NicknameNotFoundError",
    "NicknameScoreTooLowError",
    "DialogAnalysisError",
    "DialogCountTooLowError",
    "UnknownSpeakerTooHighError",
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
    "get_models",
    "layout_det",
    "text_det",
    "en_rec",
    "ar_rec",
    "pt_rec"
]


layout_det = ChatLayoutAnalyzer("PP-DocLayoutV2")
text_det = ChatLayoutAnalyzer("PP-OCRv5_server_det")
en_rec = ChatTextRecognition("PP-OCRv5_server_rec", lang='en')
ar_rec = ChatTextRecognition("PP-OCRv5_server_rec", lang="ar")
pt_rec = ChatTextRecognition("PP-OCRv5_server_rec", lang="pt")

def get_models():
    global layout_det, text_det, en_rec, ar_rec, pt_rec
    try:
        layout_det.load_model()
        text_det.load_model()
        en_rec.load_model()
        ar_rec.load_model()
        pt_rec.load_model()
    except Exception as e:
        print(f"Error loading models: {e}")
    return {"layout_det": layout_det, "text_det": text_det, "en_rec": en_rec, "ar_rec": ar_rec, "pt_rec": pt_rec, "es_rec": pt_rec}