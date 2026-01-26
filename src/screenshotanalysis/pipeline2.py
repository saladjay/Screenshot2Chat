"""
Single-image chat dialog pipeline (pipeline2).

Mirrors dialog_pipeline.analyze_chat_image but uses the demo "final" output
selection: double layouts use layout_det boxes, single layouts use the
app-agnostic final boxes.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2

from screenshotanalysis import ChatLayoutAnalyzer, ChatTextRecognition
from screenshotanalysis.app_agnostic_text_boxes import (
    assign_speaker_by_edges,
    filter_by_frequent_edges,
    filter_center_near_boxes,
    filter_small_layout_boxes,
    select_layout_text_boxes,
    suppress_nested_boxes,
)
from screenshotanalysis.nickname_extractor import extract_nicknames_from_text_boxes
from screenshotanalysis.processors import ChatMessageProcessor, TextBox
from screenshotanalysis.utils import ImageLoader, letterbox


def analyze_chat_image(
    image_path: str,
    output_path: Optional[str] = None,
    draw_output_path: Optional[str] = None,
    text_det_analyzer: Optional[ChatLayoutAnalyzer] = None,
    layout_det_analyzer: Optional[ChatLayoutAnalyzer] = None,
    text_rec: Optional[ChatTextRecognition] = None,
    processor: Optional[ChatMessageProcessor] = None,
    speaker_map: Optional[Dict[str, str]] = None,
    track_model_calls: bool = True,
) -> Tuple[Dict, Dict[int, Dict[str, int]]]:
    """
    Analyze a single chat image and return dialog JSON + model usage stats.

    Returns:
        output_payload: dict in output_example.json format (with optional model_calls)
        model_calls: mapping from dialog index to model call counts
    """
    if processor is None:
        processor = ChatMessageProcessor()
    if text_det_analyzer is None:
        text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        text_det_analyzer.load_model()
    if layout_det_analyzer is None:
        layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        layout_det_analyzer.load_model()
    if text_rec is None:
        text_rec = ChatTextRecognition(model_name="PP-OCRv5_server_rec", lang="en")
        text_rec.load_model()

    speaker_map = speaker_map or {"A": "talker", "B": "user", None: "user"}

    image = ImageLoader.load_image(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = np.array(image)
    image, padding = letterbox(image)

    preprocess_dict = {"letterbox": True, "padding": padding}
    text_det_results = text_det_analyzer.analyze_chat_screenshot(image, **preprocess_dict)
    layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image, **preprocess_dict)

    screen_width = int(text_det_results["image_size"][0])

    text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results["results"])
    sorted_text_det_boxes = processor.sort_boxes_by_y(text_det_boxes)

    def box_key(box: TextBox) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in box.box.tolist())

    model_call_by_box: Dict[Tuple[int, int, int, int], Dict[str, int]] = {}
    if track_model_calls:
        for box in sorted_text_det_boxes:
            model_call_by_box[box_key(box)] = {
                "text_det": 1,
                "layout_det": 1,
                "text_rec": 0,
            }

    ocr_cache: Dict[int, Tuple[str, float]] = {}

    def ocr_reader(box: TextBox) -> Tuple[str, float]:
        key = box_key(box)
        if key in ocr_cache:
            return ocr_cache[key]
        x_min, y_min, x_max, y_max = box.box.tolist()
        text_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]
        if text_image.size == 0:
            ocr_cache[key] = ("", 0.0)
            return ocr_cache[key]
        text_output = text_rec.predict_text(text_image)
        if track_model_calls:
            model_call_by_box.setdefault(key, {"text_det": 0, "layout_det": 0, "text_rec": 0})
            model_call_by_box[key]["text_rec"] += 1
        if not text_output:
            ocr_cache[key] = ("", 0.0)
            return ocr_cache[key]
        first = text_output[0]
        text_value = first.get("rec_text", "") if isinstance(first, dict) else str(first)
        score = first.get("rec_score", 0.0) if isinstance(first, dict) else 0.0
        ocr_cache[key] = (text_value, score)
        return ocr_cache[key]

    nickname_candidates = extract_nicknames_from_text_boxes(
        text_boxes=sorted_text_det_boxes,
        image=image,
        processor=processor,
        text_rec=text_rec,
        ocr_reader=ocr_reader,
        draw_results=False,
        image_path=image_path,
    )
    talker_nickname = nickname_candidates[0]["text"] if nickname_candidates else ""

    sorted_boxes, metadata = processor.format_conversation_app_agnostic(
        layout_det_results=layout_det_results["results"],
        text_det_results=text_det_results["results"],
        screen_width=screen_width,
        coverage_threshold=0.1,
        coverage_keep_ratio=0.25,
        enable_height_filter=False,
        padding=text_det_results.get("padding"),
        image_sizes=text_det_results.get("image_size"),
        ocr_reader=ocr_reader,
        talker_nickname=talker_nickname or None,
    )

    layout_text_boxes = select_layout_text_boxes(
        processor,
        layout_det_results["results"],
    )
    layout_text_boxes = filter_small_layout_boxes(layout_text_boxes)
    layout_text_boxes = filter_center_near_boxes(
        layout_text_boxes,
        image.shape[1],
        image.shape[0],
    )
    layout_text_boxes = suppress_nested_boxes(layout_text_boxes)
    layout_text_boxes = filter_by_frequent_edges(
        layout_text_boxes,
        image.shape[1],
    )
    assign_speaker_by_edges(layout_text_boxes, image.shape[1])

    layout_name = metadata.get("layout", "")
    final_boxes = layout_text_boxes if layout_name.startswith("double") else sorted_boxes

    dialogs: List[Dict] = []
    model_calls_by_dialog: Dict[int, Dict[str, int]] = {}

    for idx, box in enumerate(final_boxes):
        text_value, _ = ocr_reader(box)
        speaker = speaker_map.get(box.speaker, speaker_map.get(None, "user"))
        dialog = {
            "speaker": speaker,
            "text": text_value,
            "box": [int(v) for v in box.box.tolist()],
        }
        if track_model_calls:
            dialog["model_calls"] = model_call_by_box.get(
                box_key(box), {"text_det": 0, "layout_det": 0, "text_rec": 0}
            )
            model_calls_by_dialog[idx] = dialog["model_calls"]
        dialogs.append(dialog)

    output_payload = {
        "talker_nickname": talker_nickname,
        "dialogs": dialogs,
    }

    if output_path:
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

    if draw_output_path:
        draw_dialog_overlays(
            image=image,
            boxes=final_boxes,
            nickname_candidate=nickname_candidates[0] if nickname_candidates else None,
            output_path=draw_output_path,
            speaker_map=speaker_map,
        )

    return output_payload, model_calls_by_dialog


def write_output_json(image_path: str, output_path: str) -> Dict:
    """Convenience wrapper to write output_example.json format for one image."""
    output_payload, _ = analyze_chat_image(image_path=image_path, output_path=output_path)
    return output_payload


def draw_dialog_overlays(
    image: np.ndarray,
    boxes: List,
    nickname_candidate: Optional[Dict],
    output_path: str,
    speaker_map: Optional[Dict[str, str]] = None,
) -> None:
    speaker_map = speaker_map or {"A": "talker", "B": "user", None: "user"}
    color_map = {
        "talker": (255, 0, 0),  # blue
        "user": (0, 0, 255),
        "Unknown": (128, 128, 128),
    }
    draw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box in boxes:
        if isinstance(box, dict):
            x_min, y_min, x_max, y_max = box["box"]
            speaker = box.get("speaker", "user")
        else:
            x_min, y_min, x_max, y_max = [int(v) for v in box.box.tolist()]
            speaker = speaker_map.get(getattr(box, "speaker", None), "user")
        color = color_map.get(speaker, color_map["Unknown"])
        cv2.rectangle(draw_image, (x_min, y_min), (x_max, y_max), color, 2)

    if nickname_candidate:
        nx_min, ny_min, nx_max, ny_max = [int(v) for v in nickname_candidate["box"]]
        nickname = nickname_candidate.get("text", "")
        cv2.rectangle(draw_image, (nx_min, ny_min), (nx_max, ny_max), (0, 255, 0), 2)
        if nickname:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(nickname, font, font_scale, thickness)
            label_y = ny_min - 5
            if label_y - text_height < 0:
                label_y = ny_max + text_height + 5
            cv2.rectangle(
                draw_image,
                (nx_min, label_y - text_height - 4),
                (nx_min + text_width + 6, label_y + 2),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                draw_image,
                nickname,
                (nx_min + 3, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, draw_image)


def iter_image_paths(input_path: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(input_path, fname)
        return
    if os.path.isfile(input_path):
        yield input_path
