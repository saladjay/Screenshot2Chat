from screenshotanalysis.experience_formula import *
from screenshotanalysis.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM, ImageLoader, letterbox
import os
import pytest
import cv2
from screenshotanalysis import ChatLayoutAnalyzer, ChatMessageProcessor
import numpy as np
import random

text_box_test_data = [
    [171, 728, 279, 748], 
    [66, 657, 214, 677], 
    [67, 635, 183, 655], 
    [67, 593, 219, 614], 
    [68, 571, 203, 585], 
    [67, 549, 136, 567], 
    [68, 507, 234, 524], 
    [67, 484, 184, 504], 
    [68, 443, 195, 460], 
    [65, 421, 137, 442], 
    [67, 381, 155, 398], 
    [67, 358, 184, 378], 
    [66, 316, 243, 333], 
    [64, 289, 175, 312], 
    [66, 271, 136, 289],
    [66, 228, 224, 247], 
    [19, 55, 55, 96], 
    [100, 54, 190, 81], 
    [30, 12, 75, 30 ],
]

update_box_test_data = [
    [171, 728, 279, 748],
    [67, 671, 220, 690], 
    [68, 646, 205, 663], 
    [67, 625, 135, 643], 
    [67, 582, 234, 601], 
    [69, 562, 183, 579], 
    [67, 519, 194, 536], 
    [67, 499, 135, 517], 
    [68, 457, 155, 475], 
    [68, 436, 183, 453], 
    [66, 394, 243, 411], 
    [64, 366, 176, 388], 
    [66, 347, 136, 365], 
    [67, 304, 224, 323], 
    [66, 115, 185, 138], 
    [18, 54, 55, 97], 
    [100, 54, 190, 81], 
    [30, 12, 75, 31,]
]


class TestExperienceFormula:
    def test_reinit_experience_layout_feature(self):
        output_path = 'test_experience_text'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        text_det_analyzer.load_model()
        layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        layout_det_analyzer.load_model()

        for app_type in [DISCORD, INSTAGRAM, TELEGRAM, WHATSAPP]:
            for file in os.listdir('test_images'):
                if (file.endswith('.png') or file.endswith('.jpg')) and file.find(app_type) != -1:
                    image_path = os.path.join('test_images', file)
                    log_file = None
                    text_det_results = text_det_analyzer.analyze_chat_screenshot(image_path)
                    layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image_path)

                    assert text_det_results['success']
                    assert layout_det_results['success']
                    assert len(text_det_results['padding']) == len(layout_det_results['padding'])
                    assert all([v1==v2 for (v1, v2) in zip(text_det_results['padding'], layout_det_results['padding'])])
                    assert len(text_det_results['image_size']) == len(layout_det_results['image_size'])
                    assert all([v1==v2 for (v1, v2) in zip(text_det_results['image_size'], layout_det_results['image_size'])])

                    padding = text_det_results['padding']
                    image_sizes = text_det_results['image_size']
                    layout_postprocess = ChatMessageProcessor()
                    reinit_data(app_type)
                    layout_det_boxes, text_det_boxes = layout_postprocess.format_conversation(layout_det_results['results'], text_det_results['results'], padding, image_sizes, app_type = app_type, log_file = log_file)
                    update_data(padding, text_det_boxes, image_sizes, app_type)

        for app_type in [DISCORD, INSTAGRAM, TELEGRAM, WHATSAPP]:
            for file in os.listdir('test_images'):
                if (file.endswith('.png') or file.endswith('.jpg')) and file.find(app_type) != -1:
                    image_path = os.path.join('test_images', file)
                    log_file = open(os.path.join(output_path, file+'.txt'), 'w', encoding='utf-8')
                    text_det_results = text_det_analyzer.analyze_chat_screenshot(image_path)
                    layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image_path)
                    image = ImageLoader.load_image(image_path)
                    image = np.array(image)
                    image, _ = letterbox(image)
                    padding = text_det_results['padding']
                    image_sizes = text_det_results['image_size']
                    layout_postprocess = ChatMessageProcessor()
                    _, ratios = load_data(app_type)
                    text_boxes, _ = layout_postprocess.format_conversation(layout_det_results['results'], text_det_results['results'], padding, image_sizes, ratios = ratios, app_type = app_type, log_file = log_file)
                    for box in text_boxes:
                        print(type(box))
                        image = cv2.rectangle(image, 
                                    (box.x_min, box.y_min), 
                                    (box.x_max, box.y_max), 
                                    (0, 255, 255), 2)
                    cv2.imwrite(os.path.join(output_path, file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))