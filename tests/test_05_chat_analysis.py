import pytest
import os
from PIL import Image
import numpy as np
import cv2
from chat_layout_analyzer import ChatLayoutAnalyzer, ChatTextRecognition
from chat_layout_analyzer.utils import ImageLoader, letterbox
from chat_layout_analyzer.processors import ChatMessageProcessor
from chat_layout_analyzer.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM
from chat_layout_analyzer.experience_formula import *
class TestLayoutAnalysis:
    def test_text_det(self):
        test_text_output = 'test_text_info'
        """测试图片预测功能"""
        if not os.path.exists('test_images'):
            pytest.skip("测试图片文件夹不存在，跳过图片预测测试")

        analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        analyzer.load_model()

        layout_postprocess = ChatMessageProcessor(analyzer.model_name)

        if not os.path.exists(test_text_output):
            os.makedirs(test_text_output)
        for app_type in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
            for file in os.listdir('test_images'):
                if (file.endswith('.png') or file.endswith('.jpg')) and file.find(app_type) != -1:
                    image_path = os.path.join('test_images', file)
                    image = ImageLoader.load_image(image_path)
                    if image.mode == 'RGBA':
                        image = image.convert("RGB")
                    image = np.array(image)
                    image, padding = letterbox(image)

                    result = analyzer.model.predict(image)
                    image_sizes = [image.shape[1], image.shape[0]]
                    _, ratios = load_data(app_type)

                    
                    layout_postprocess.draw_all_text_info_boxes(image.copy(), result, f'{test_text_output}/{file}', padding, image_sizes, ratios, app_type, True)

    def test_nickname_det(self):
        test_text_output = 'test_nickname'
        """测试图片预测功能"""
        if not os.path.exists('test_images'):
            pytest.skip("测试图片文件夹不存在，跳过图片预测测试")

        analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        analyzer.load_model()

        layout_postprocess = ChatMessageProcessor(analyzer.model_name)

        if not os.path.exists(test_text_output):
            os.makedirs(test_text_output)
        
        for app_type in [DISCORD, TELEGRAM, WHATSAPP, INSTAGRAM]:
            for file in os.listdir('test_images'):
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_path = os.path.join('test_images', file)
                    image = ImageLoader.load_image(image_path)
                    if image.mode == 'RGBA':
                        image = image.convert("RGB")
                    image = np.array(image)
                    image, padding = letterbox(image)

                    result = analyzer.model.predict(image)
                    image_sizes = [image.shape[1], image.shape[0]]
                    ratios = None
                    _, ratios = load_data(DISCORD)
      
                    if ratios is None:
                        continue
                    ratios = ratios.tolist()
                    layout_postprocess.draw_all_nickname_from_det_boxes(image.copy(), result, f'{test_text_output}/{file}', padding, image_sizes, ratios = ratios, app_type = app_type, enable_log = True)    

    def test_screenshot_text_box_analysis(self):
        test_output_folder = 'test_screenshot_text_analysis'
        if not os.path.exists(test_output_folder):
            os.makedirs(test_output_folder)
        
        text_det_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        text_det_analyzer.load_model()

        layout_det_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        layout_det_analyzer.load_model()

        text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
        text_rec.load_model()

        app_type = DISCORD
        layout_postprocess = ChatMessageProcessor()

        _, ratios = load_data(app_type)
        ratios = ratios.tolist()
        assert len(ratios) == 4, f'{app_type} 的ratios长度不等于4'

        def get_text_from_rec_model(text_box, image):
            min_x, min_y, max_x, max_y = text_box.box.tolist()
            text_image = image[min_y:max_y, min_x:max_x, :]
            assert text_image is not None
            text_output = text_rec.predict_text(text_image)
            return text_output[0]['rec_text']

        for file in os.listdir('test_images'):
            if (file.endswith('.png') or file.endswith('.jpg')) and file.find(app_type) != -1:
                image_path = os.path.join('test_images', file)
                log_file = open(os.path.join(test_output_folder, file+'.txt'), 'w', encoding='utf-8')

                text_det_results = text_det_analyzer.analyze_chat_screenshot(image_path)
                layout_det_results = layout_det_analyzer.analyze_chat_screenshot(image_path)
                padding = list(map(float, text_det_results['padding'])) 
                image_sizes = list(map(float, text_det_results['image_size'])) 
                sorted_box = layout_postprocess.format_conversation(layout_det_results['results'], text_det_results['results'], padding, image_sizes, ratios = ratios, app_type = DISCORD, log_file = log_file)
                nickname_box = layout_postprocess.get_nickname_box_from_text_det_boxes(text_det_results['results'], padding, image_sizes, ratios, app_type)
                nickname = None
                image = np.array(ImageLoader.load_image(image_path))
                image, _ = letterbox(image)
                if nickname_box:
                    nickname = get_text_from_rec_model(nickname_box, image)
                    if nickname.endswith('>'):
                        nickname = nickname[:-1]
                if nickname:
                    print(f"talker nickname:{nickname}", file=log_file)
                


                new_speaker_group_flag = False
                for box in sorted_box:
                    if box.layout_det == 'avatar':
                        new_speaker_group_flag = True
                        continue
                    if box.layout_det == 'nickname':
                        assert new_speaker_group_flag, '错误的头像和nickname排序，需要检查format_conversation的实现'
                        speaker_name = get_text_from_rec_model(box, image)
                        print(f"******************************detect name:{speaker_name}", file=log_file)
                        if speaker_name.startswith(nickname):
                            print(f'talker {nickname} say:', file=log_file)
                        else:
                            print(f'user (yourself) say:', file=log_file)
                        continue
                    if box.layout_det == 'text':
                        text = get_text_from_rec_model(box, image)
                        print(f'{text}', file=log_file)


                        
                    

                # nickname 