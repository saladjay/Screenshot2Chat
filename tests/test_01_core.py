import pytest
import os
from PIL import Image
import numpy as np
import cv2
from chat_layout_analyzer import ChatLayoutAnalyzer
from chat_layout_analyzer.utils import ImageLoader



class TestChatLayoutAnalyzer:
    
    def test_initialization(self):
        """测试分析器初始化"""
        analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        assert analyzer.model_name == "PP-DocLayoutV2"
        assert analyzer.model is None
        analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        assert analyzer.model_name == "PP-OCRv5_server_det"
        assert analyzer.model is None

    def test_image_predict(self):
        """测试图片预测功能"""
        if not os.path.exists('test_images'):
            pytest.skip("测试图片文件夹不存在，跳过图片预测测试")
        if not os.path.exists('test_output'):
            os.makedirs('test_output')
        analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        analyzer.load_model()
        f = open('test_output/dummy.txt', 'w') 
        for file in os.listdir('test_images'):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join('test_images', file)
                image = ImageLoader.load_image(image_path)
                if image.mode == 'RGBA':
                    image = image.convert("RGB")
                image = np.array(image)

                result = analyzer.model.predict(image)
                for element in result:
                    for box in element['boxes']:
                        assert 'label' in box
                        assert 'score' in box
                        assert 'coordinate' in box
                        print(f'{file} - {box["label"]}: {box["score"]} at {box["coordinate"]}', file=f)
                        image = cv2.rectangle(image, 
                                              (int(box['coordinate'][0]), int(box['coordinate'][1])), 
                                              (int(box['coordinate'][2]), int(box['coordinate'][3])), 
                                              (0, 255, 0), 2)
                        image = cv2.putText(image, 
                                            f"{box['label']}:{box['score']:.2f}", 
                                            (int(box['coordinate'][0]), int(box['coordinate'][1]) - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, 
                                            (255, 0, 0), 2)
                cv2.imwrite(f"test_output/{file}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        f.close()

    def test_model_loading(self):
        """测试模型加载"""
        analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
        analyzer.load_model()
        assert analyzer.model is not None
        analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
        analyzer.load_model()
        assert analyzer.model is not None
        
    

if __name__ == "__main__":
    pytest.main()