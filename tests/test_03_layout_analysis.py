import pytest
import os
from PIL import Image
import numpy as np
import cv2
from chat_layout_analyzer import ChatLayoutAnalyzer
from chat_layout_analyzer.utils import ImageLoader, letterbox
from chat_layout_analyzer.processors import ChatMessageProcessor
from chat_layout_analyzer.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM
from chat_layout_analyzer.experience_formula import *
class TestLayoutAnalysis:
    def test_text_det(self):
        test_layout_output = 'test_text_det'
        """测试图片预测功能"""
        if not os.path.exists('test_images'):
            pytest.skip("测试图片文件夹不存在，跳过图片预测测试")
        if not os.path.exists('test_output'):
            os.makedirs('test_output')
        analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        analyzer.load_model()

        layout_postprocess = ChatMessageProcessor(analyzer.model_name)

        if not os.path.exists(test_layout_output):
            os.makedirs(test_layout_output)
        
        for file in os.listdir('test_images'):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join('test_images', file)
                image = ImageLoader.load_image(image_path)
                if image.mode == 'RGBA':
                    image = image.convert("RGB")
                image = np.array(image)
                image, padding = letterbox(image)
                result = analyzer.model.predict(image)
                layout_postprocess.draw_all_text_boxes(image, result, f"{test_layout_output}/{file}", True)

    def test_layout_det(self):
        test_layout_output = 'test_layout_det_text'
        test_layout_image_output = 'test_layout_det_image'
        test_layout_other_output = 'test_layout_det_other'
        test_layout_avatar_output = 'test_layout_det_avatar'
        """测试图片预测功能"""
        if not os.path.exists('test_images'):
            pytest.skip("测试图片文件夹不存在，跳过图片预测测试")
        def check_path(p):
            if not os.path.exists(p):
                os.makedirs(p)
        for p in [test_layout_output, test_layout_image_output, test_layout_other_output, test_layout_avatar_output]:
            check_path(p)
        analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        analyzer.load_model()

        layout_postprocess = ChatMessageProcessor(analyzer.model_name)

        if not os.path.exists(test_layout_output):
            os.makedirs(test_layout_output)
        
        for file in os.listdir('test_images'):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join('test_images', file)
                image = ImageLoader.load_image(image_path)
                if image.mode == 'RGBA':
                    image = image.convert("RGB")
                image = np.array(image)
                image, padding = letterbox(image)
                result = analyzer.model.predict(image)
                layout_postprocess.draw_all_text_boxes(image.copy(), result, f"{test_layout_output}/{file}", True)
                layout_postprocess.draw_all_image_boxes(image.copy(), result, f"{test_layout_image_output}/{file}", True)
                layout_postprocess.draw_all_other_boxes(image.copy(), result, f"{test_layout_other_output}/{file}", True)
                if file.find('discord') != -1: 
                    layout_postprocess.draw_all_avatar_boxes(image.copy(), result, f'{test_layout_avatar_output}/{file}', True)



                

                
