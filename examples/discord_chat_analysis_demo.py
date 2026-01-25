#!/usr/bin/env python3
"""
Discord chat analysis demo based on tests/test_05_chat_analysis.py.
"""

import os
import numpy as np
import cv2

from screenshotanalysis import ChatLayoutAnalyzer, ChatTextRecognition
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.utils import ImageLoader, letterbox, DISCORD
from screenshotanalysis.experience_formula import load_data


def draw_boxes_by_speaker(image, boxes):
    speaker_colors = {
        'A': (0, 0, 255),
        'B': (255, 0, 0),
        'Unknown': (128, 128, 128)
    }
    for box in boxes:
        x_min, y_min, x_max, y_max = box.box.tolist()
        speaker = getattr(box, 'speaker', 'Unknown')
        color = speaker_colors.get(speaker, speaker_colors['Unknown'])
        image = cv2.rectangle(
            image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            2
        )
    return image


def main():
    input_dir = "test_images"
    output_dir = "example_discord_text_boxes"
    log_dir = os.path.join(output_dir, "logs")

    if not os.path.exists(input_dir):
        print("test_images folder not found; skipping demo.")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_det_analyzer.load_model()

    layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
    layout_det_analyzer.load_model()

    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()

    processor = ChatMessageProcessor()

    _, ratios = load_data(DISCORD)
    ratios = ratios.tolist()

    def get_text_from_rec_model(text_box, image):
        min_x, min_y, max_x, max_y = text_box.box.tolist()
        text_image = image[min_y:max_y, min_x:max_x, :]
        if text_image is None or text_image.size == 0:
            return ""
        text_output = text_rec.predict_text(text_image)
        return text_output[0]['rec_text'] if text_output else ""

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if "discord" not in file_name.lower():
            continue

        image_path = os.path.join(input_dir, file_name)
        image = ImageLoader.load_image(image_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.array(image)
        image, _ = letterbox(image)

        text_det_results = text_det_analyzer.analyze_chat_screenshot(image_path)
        layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image_path)
        padding = list(map(float, text_det_results['padding']))
        image_sizes = list(map(float, text_det_results['image_size']))

        log_path = os.path.join(log_dir, f"{os.path.splitext(file_name)[0]}.txt")
        log_file = open(log_path, 'w', encoding='utf-8')

        text_det_boxes = processor._get_all_text_boxes_from_text_det(text_det_results['results'])
        sorted_text_det_boxes = processor.sort_boxes_by_y(text_det_boxes)
        if not processor._has_dominant_xmin_bin(sorted_text_det_boxes):
            log_file.write("skip discord logic: no dominant x_min bin\n")
            log_file.close()
            continue

        sorted_box, _ = processor.format_conversation(
            layout_det_results['results'],
            text_det_results['results'],
            padding,
            image_sizes,
            ratios=ratios,
            app_type=DISCORD,
            log_file=log_file
        )
        nickname_box = processor.get_nickname_box_from_text_det_boxes(
            text_det_results['results'],
            padding,
            image_sizes,
            ratios,
            DISCORD,
            log_file=log_file
        )
        nickname = None
        if nickname_box:
            nickname = get_text_from_rec_model(nickname_box, image)
            if nickname.endswith('>'):
                nickname = nickname[:-1]
        if nickname:
            print(f"talker nickname:{nickname}", file=log_file)

        layout_text_boxes = []
        new_speaker_group_flag = False
        current_speaker = None
        for box in sorted_box:
            if box.layout_det == 'avatar':
                new_speaker_group_flag = True
                continue
            if box.layout_det == 'nickname':
                if not new_speaker_group_flag:
                    continue
                speaker_name = get_text_from_rec_model(box, image)
                print(f"******************************detect name:{speaker_name}", file=log_file)
                if nickname and speaker_name.startswith(nickname):
                    current_speaker = 'A'
                    print(f"talker {nickname} say:", file=log_file)
                else:
                    current_speaker = 'B'
                    print("user (yourself) say:", file=log_file)
                continue
            if box.layout_det == 'text':
                if current_speaker is None:
                    continue
                box.speaker = current_speaker
                layout_text_boxes.append(box)
                text_value = get_text_from_rec_model(box, image)
                print(text_value, file=log_file)
        log_file.close()

        layout_drawn = draw_boxes_by_speaker(image.copy(), layout_text_boxes)
        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_layout_det.png"
        )
        cv2.imwrite(output_path, cv2.cvtColor(layout_drawn, cv2.COLOR_RGB2BGR))
        print(f"{file_name}: output -> {output_path}")


if __name__ == "__main__":
    main()
