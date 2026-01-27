"""
Single-image chat dialog pipeline.

Runs app-agnostic dialog assignment and nickname extraction with minimal
model inference, returning output in output_example.json format and
tracking per-textbox model usage.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from screenshotanalysis.basemodel import OTHER, UNKNOWN, USER
from screenshotanalysis.processors import ChatMessageProcessor, TextBox, LAYOUT_DET, TEXT_DET
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.config_manager import load_config, update_config


def _load_analysis_config(
    config_path: Optional[str],
    runtime_config: Optional[Dict[str, Any]],
    persist_runtime_config: bool,
) -> Dict[str, Any]:
    if runtime_config:
        return update_config(
            runtime_config,
            config_path=config_path,
            keep_history=persist_runtime_config,
        )
    return load_config(config_path)


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
    split_layout_lines: bool = True,
    multiline_height_ratio: float = 1.8,
    config_path: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    persist_runtime_config: bool = True,
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

    speaker_map = speaker_map or {OTHER: OTHER, USER: USER, UNKNOWN: UNKNOWN, None: UNKNOWN}

    config = _load_analysis_config(config_path, runtime_config, persist_runtime_config)
    nickname_min_score = float(config["nickname"]["min_score"])
    nickname_min_top_margin_ratio = float(config["nickname"].get("min_top_margin_ratio", 0.05))
    nickname_top_region_ratio = float(config["nickname"].get("top_region_ratio", 0.2))
    min_bubble_count = int(config["dialog"]["min_bubble_count"])
    analysis_failures: List[str] = []
    analysis_failure_details: List[Dict[str, Any]] = []

    image = ImageLoader.load_image(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = np.array(image)
    image, padding = letterbox(image)

    preprocess_dict = {"letterbox":True, "padding":padding}
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
        min_top_margin_ratio=nickname_min_top_margin_ratio,
        top_region_ratio=nickname_top_region_ratio,
    )
    top_candidate = nickname_candidates[0] if nickname_candidates else None
    if not top_candidate:
        talker_nickname = ""
        analysis_failures.append("nickname_not_found")
        analysis_failure_details.append(
            {
                "code": "nickname_not_found",
                "actual": None,
                "threshold": nickname_min_score,
            }
        )
    else:
        top_score = float(top_candidate.get("nickname_score", 0.0))
        if top_score < nickname_min_score:
            talker_nickname = ""
            analysis_failures.append("nickname_score_below_threshold")
            analysis_failure_details.append(
                {
                    "code": "nickname_score_below_threshold",
                    "actual": top_score,
                    "threshold": nickname_min_score,
                }
            )
        else:
            talker_nickname = top_candidate.get("text", "")

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

    dialogs: List[Dict] = []
    model_calls_by_dialog: Dict[int, Dict[str, int]] = {}

    for idx, box in enumerate(sorted_boxes):
        text_value, _ = ocr_reader(box)
        speaker = speaker_map.get(box.speaker, speaker_map.get(None, UNKNOWN))
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

    if len(dialogs) < min_bubble_count:
        analysis_failures.append("dialog_count_below_threshold")
        analysis_failure_details.append(
            {
                "code": "dialog_count_below_threshold",
                "actual": len(dialogs),
                "threshold": min_bubble_count,
            }
        )

    output_payload = {
        "talker_nickname": talker_nickname,
        "dialogs": dialogs,
        "analysis_failures": analysis_failures,
        "analysis_failure_details": analysis_failure_details,
        "config": {
            "nickname_min_score": nickname_min_score,
            "nickname_min_top_margin_ratio": nickname_min_top_margin_ratio,
            "nickname_top_region_ratio": nickname_top_region_ratio,
            "min_bubble_count": min_bubble_count,
        },
    }

    if output_path:
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

    if draw_output_path:
        draw_dialog_overlays(
            image=image,
            boxes=layout_text_boxes,
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
    speaker_map = speaker_map or {OTHER: OTHER, USER: USER, UNKNOWN: UNKNOWN, None: UNKNOWN}
    color_map = {
        OTHER: (255, 0, 0),  # blue
        USER: (0, 0, 255),
        UNKNOWN: (128, 128, 128),
    }
    draw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box in boxes:
        if isinstance(box, dict):
            x_min, y_min, x_max, y_max = box["box"]
            speaker = box.get("speaker", UNKNOWN)
        else:
            x_min, y_min, x_max, y_max = [int(v) for v in box.box.tolist()]
            speaker = speaker_map.get(getattr(box, "speaker", None), UNKNOWN)
        color = color_map.get(speaker, color_map[UNKNOWN])
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
        for name in sorted(os.listdir(input_path)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
                continue
            yield os.path.join(input_path, name)
    else:
        yield input_path


def analyze_chat_images(
    input_path: str,
    output_dir: str,
    draw_dir: Optional[str] = None,
    track_model_calls: bool = True,
    split_layout_lines: bool = True,
    multiline_height_ratio: float = 1.8,
    config_path: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    persist_runtime_config: bool = True,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    if draw_dir:
        os.makedirs(draw_dir, exist_ok=True)

    processor = ChatMessageProcessor()
    text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_det_analyzer.load_model()
    layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_det_analyzer.load_model()
    text_rec = ChatTextRecognition(model_name="PP-OCRv5_server_rec", lang="en")
    text_rec.load_model()

    outputs = []
    for image_path in iter_image_paths(input_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")
        draw_output_path = None
        if draw_dir:
            draw_output_path = os.path.join(draw_dir, f"{base_name}.png")
        output_payload, _ = analyze_chat_image(
            image_path=image_path,
            output_path=output_path,
            draw_output_path=draw_output_path,
            text_det_analyzer=text_det_analyzer,
            layout_det_analyzer=layout_det_analyzer,
            text_rec=text_rec,
            processor=processor,
            track_model_calls=track_model_calls,
            split_layout_lines=split_layout_lines,
            multiline_height_ratio=multiline_height_ratio,
            config_path=config_path,
            runtime_config=runtime_config,
            persist_runtime_config=persist_runtime_config,
        )
        outputs.append(output_payload)
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dialog analysis for one image.")
    parser.add_argument("input_path", help="Path to input image or directory")
    parser.add_argument("--output", default="output_example.json", help="Output JSON path (single image)")
    parser.add_argument("--output-dir", default="dialog_outputs", help="Output directory for batch JSON")
    parser.add_argument("--draw-dir", default=None, help="Optional directory for overlay images")
    parser.add_argument(
        "--no-split-layout-lines",
        action="store_true",
        help="Disable splitting tall layout_det text boxes into multiple lines",
    )
    parser.add_argument(
        "--multiline-height-ratio",
        type=float,
        default=1.8,
        help="Height ratio threshold to split layout_det text boxes into multiple lines",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--nickname-min-score", type=float, default=None, help="Runtime nickname score threshold")
    parser.add_argument("--min-bubble-count", type=int, default=None, help="Minimum dialog bubble count threshold")
    args = parser.parse_args()

    runtime_config = {}
    if args.nickname_min_score is not None:
        runtime_config.setdefault("nickname", {})["min_score"] = args.nickname_min_score
    if args.min_bubble_count is not None:
        runtime_config.setdefault("dialog", {})["min_bubble_count"] = args.min_bubble_count
    runtime_config = runtime_config or None

    if os.path.isdir(args.input_path):
        analyze_chat_images(
            input_path=args.input_path,
            output_dir=args.output_dir,
            draw_dir=args.draw_dir,
            split_layout_lines=not args.no_split_layout_lines,
            multiline_height_ratio=args.multiline_height_ratio,
            config_path=args.config,
            runtime_config=runtime_config,
            persist_runtime_config=True,
        )
    else:
        draw_output_path = None
        if args.draw_dir:
            os.makedirs(args.draw_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input_path))[0]
            draw_output_path = os.path.join(args.draw_dir, f"{base_name}.png")
        analyze_chat_image(
            image_path=args.input_path,
            output_path=args.output,
            draw_output_path=draw_output_path,
            split_layout_lines=not args.no_split_layout_lines,
            multiline_height_ratio=args.multiline_height_ratio,
            config_path=args.config,
            runtime_config=runtime_config,
            persist_runtime_config=True,
        )
