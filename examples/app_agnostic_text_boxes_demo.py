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
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.utils import ImageLoader, letterbox, DISCORD
from screenshotanalysis.experience_formula import load_data


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

    try:
        _, ratios = load_data(DISCORD)
        ratios = ratios.tolist()
    except Exception:
        ratios = None

    def get_text_from_rec_model(text_box, image):
        x_min, y_min, x_max, y_max = text_box.box.tolist()
        text_image = image[y_min:y_max, x_min:x_max, :]
        if text_image is None or text_image.size == 0:
            return ""
        text_output = text_rec.predict_text(text_image)
        if not text_output:
            return ""
        return text_output[0].get('rec_text', '')

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
        should_use_discord = (
            'discord' in file_name.lower()
            and ratios
            and processor._has_dominant_xmin_bin(sorted_text_det_boxes)
        )
        if should_use_discord:
            sorted_boxes, _ = processor.format_conversation(
                layout_det_results=layout_det_results["results"],
                text_det_results=text_det_results["results"],
                padding=padding,
                image_sizes=image_sizes,
                ratios=ratios,
                app_type=DISCORD
            )
            nickname = None
            nickname_box = processor.get_nickname_box_from_text_det_boxes(
                text_det_results["results"],
                padding,
                image_sizes,
                ratios,
                DISCORD
            )
            if nickname_box:
                nickname = get_text_from_rec_model(nickname_box, image)
                if nickname.endswith('>'):
                    nickname = nickname[:-1]
            layout_text_boxes = []
            new_speaker_group_flag = False
            current_speaker = None
            last_avatar_center_x = None
            for box in sorted_boxes:
                if box.layout_det == 'avatar':
                    new_speaker_group_flag = True
                    last_avatar_center_x = box.center_x
                    if nickname is None:
                        current_speaker = 'A' if box.center_x <= image.shape[1] / 2 else 'B'
                    continue
                if box.layout_det == 'nickname':
                    if not new_speaker_group_flag:
                        continue
                    speaker_name = get_text_from_rec_model(box, image)
                    if nickname and speaker_name.startswith(nickname):
                        current_speaker = 'A'
                    elif speaker_name:
                        current_speaker = 'B'
                    elif last_avatar_center_x is not None:
                        current_speaker = 'A' if last_avatar_center_x <= image.shape[1] / 2 else 'B'
                    continue
                if box.layout_det == 'text':
                    if current_speaker is None and last_avatar_center_x is not None:
                        current_speaker = 'A' if last_avatar_center_x <= image.shape[1] / 2 else 'B'
                    if current_speaker is None:
                        continue
                    box.speaker = current_speaker
                    layout_text_boxes.append(box)
            metadata = {
                'layout': 'single',
                'speaker_A_count': len([b for b in layout_text_boxes if b.speaker == 'A']),
                'speaker_B_count': len([b for b in layout_text_boxes if b.speaker == 'B'])
            }
            sorted_boxes = layout_text_boxes
        else:
            sorted_boxes, metadata = processor.format_conversation_app_agnostic(
                layout_det_results=layout_det_results["results"],
                text_det_results=text_det_results["results"],
                screen_width=screen_width,
                coverage_threshold=0.1,
                coverage_keep_ratio=0.25,
                enable_height_filter=False
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

        drawn = draw_boxes_by_speaker(image.copy(), sorted_boxes)
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))

        layout_drawn = draw_boxes_by_speaker(image.copy(), layout_text_boxes)
        layout_output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_layout_det.png"
        )
        cv2.imwrite(layout_output_path, cv2.cvtColor(layout_drawn, cv2.COLOR_RGB2BGR))
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
