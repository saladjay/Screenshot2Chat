#!/usr/bin/env python3
"""
App-agnostic text box detection demo.

Mirrors tests/test_05_chat_analysis.py but uses format_conversation_app_agnostic
and draws detected text boxes to output images.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from screenshotanalysis import ChatLayoutAnalyzer, ChatTextRecognition
from screenshotanalysis.app_agnostic_text_boxes import (
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
from screenshotanalysis.nickname_extractor import extract_nicknames_from_text_boxes
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.utils import ImageLoader, letterbox


def main():
    input_dir = "test_images"
    output_dir = "example_app_agnostic_text_boxes"
    coords_dir = "example_app_agnostic_coords"

    if not os.path.exists(input_dir):
        print("test_images folder not found; skipping demo.")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_det_analyzer.load_model()

    layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_det_analyzer.load_model()

    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()

    processor = ChatMessageProcessor()

    def get_text_from_rec_model(text_box, image):
        x_min, y_min, x_max, y_max = text_box.box.tolist()
        text_image = image[y_min:y_max, x_min:x_max, :]
        if text_image is None or text_image.size == 0:
            return ""
        text_output = text_rec.predict_text(text_image)
        if not text_output:
            return ""
        return text_output[0].get('rec_text', '')

    def draw_badge(image_bgr, label, origin=(8, 8)):
        x, y = origin
        badge_width = max(140, 12 * len(label))
        cv2.rectangle(image_bgr, (x, y), (x + badge_width, y + 28), (0, 0, 0), -1)
        cv2.putText(
            image_bgr,
            label,
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(input_dir, file_name)
        image = ImageLoader.load_image(image_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.array(image)
        image, _ = letterbox(image)

        text_det_results = text_det_analyzer.analyze_chat_screenshot(image_path)
        layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image_path)

        screen_width = int(text_det_results["image_size"][0])
        padding = list(map(float, text_det_results.get('padding', [0, 0, 0, 0])))
        image_sizes = list(map(float, text_det_results.get('image_size', [image.shape[1], image.shape[0]])))
        text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results["results"])
        sorted_text_det_boxes = processor.sort_boxes_by_y(text_det_boxes)

        ocr_cache = {}

        def ocr_reader(box):
            key = tuple(map(int, box.box.tolist()))
            if key in ocr_cache:
                return ocr_cache[key]
            text_value = get_text_from_rec_model(box, image)
            ocr_cache[key] = (text_value, 1.0 if text_value else 0.0)
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
            padding=padding,
            image_sizes=image_sizes,
            ocr_reader=ocr_reader,
            talker_nickname=talker_nickname or None,
        )

        layout_text_boxes = select_layout_text_boxes(
            processor,
            layout_det_results["results"]
        )
        layout_text_boxes = filter_small_layout_boxes(layout_text_boxes)
        layout_text_boxes = filter_center_near_boxes(
            layout_text_boxes,
            image.shape[1],
            image.shape[0]
        )
        layout_text_boxes = suppress_nested_boxes(layout_text_boxes)
        layout_text_boxes = filter_by_frequent_edges(
            layout_text_boxes,
            image.shape[1]
        )
        assign_speaker_by_edges(layout_text_boxes, image.shape[1])

        layout_label = f"layout: {metadata.get('layout', 'unknown')}"

        final_boxes = layout_text_boxes if metadata.get('layout', '').startswith('double') else sorted_boxes
        drawn = draw_boxes_by_speaker(image.copy(), final_boxes)
        final_output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_final.png"
        )
        final_bgr = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)
        draw_badge(final_bgr, f"final | {layout_label}")
        cv2.imwrite(final_output_path, final_bgr)

        layout_drawn = draw_boxes_by_speaker(image.copy(), layout_text_boxes)
        layout_output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_layout_det.png"
        )
        layout_bgr = cv2.cvtColor(layout_drawn, cv2.COLOR_RGB2BGR)
        draw_badge(layout_bgr, f"layout_det | {layout_label}")
        cv2.imwrite(layout_output_path, layout_bgr)

        text_det_drawn = draw_boxes_by_speaker(image.copy(), sorted_boxes)
        text_det_output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_text_det.png"
        )
        text_det_bgr = cv2.cvtColor(text_det_drawn, cv2.COLOR_RGB2BGR)
        draw_badge(text_det_bgr, f"text_det | {layout_label}")
        cv2.imwrite(text_det_output_path, text_det_bgr)
        coords_path = save_detection_coords(coords_dir, image_path, sorted_boxes, metadata)
        metrics = evaluate_against_gt(image_path, sorted_boxes)

        print(f"{file_name}: {metadata['layout']} A={metadata['speaker_A_count']} B={metadata['speaker_B_count']}")
        print(f"  坐标输出: {coords_path}")
        if metrics:
            print(
                f"  精确率: {metrics['precision']:.3f} 召回率: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
                f"(GT={metrics['gt_count']} DET={metrics['det_count']} TP={metrics['true_positive']})"
            )


if __name__ == "__main__":
    main()
