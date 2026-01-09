import pytest
import os
from PIL import Image
import numpy as np
import cv2
from screenshotanalysis import ChatLayoutAnalyzer, ChatTextRecognition
from screenshotanalysis.utils import ImageLoader, letterbox
from screenshotanalysis.processors import ChatMessageProcessor
OUTPUT_PATH = 'test_recognition'

class TestTextRecognition:
    def test_en_recognition(self):
        output_dir = f'{OUTPUT_PATH}/en/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
        text_rec.load_model()
        assert text_rec.model is not None
        log_file = open(f'{output_dir}/text_log.txt', 'w')
        image_path = './test_images/test_discord_2.png'
        text_det = ChatLayoutAnalyzer(model_name='PP-OCRv5_server_det')
        text_det.load_model()
        image = ImageLoader.load_image(image_path)
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        image = np.array(image)
        image, padding = letterbox(image)
        results = text_det.model.predict(image)
        layout_postprocess = ChatMessageProcessor(text_det.model_name)
        text_boxes = layout_postprocess._get_all_text_boxes_from_text_det(results)
        # image = text_det.current_image
        print(len(text_boxes), file=log_file)
        for box_index, text_box in enumerate(text_boxes):
            min_x, min_y, max_x, max_y = text_box.box.tolist()
            
            text_image = image[min_y:max_y, min_x:max_x, :]
            assert text_image is not None
            text_output = text_rec.predict_text(text_image)
            for res in text_output:
                score = res['rec_score']
                if score < 0.8:
                    continue 
                print(f'{min_x} {min_y} {max_x} {max_y}', file=log_file)
                cv2.imwrite(f"{output_dir}/{res['rec_text']}.jpg", text_image)
                print(res['rec_score'], file=log_file)
                # print(res['vis_font'], file=log_file)
                print(res['rec_text'], file=log_file)
                print('*'*20, file=log_file)
        log_file.close()
        

