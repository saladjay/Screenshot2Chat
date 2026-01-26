import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from paddlex import create_model
from paddleocr import LayoutDetection, TextDetection, PaddleOCR, TextRecognition
import logging
from screenshotanalysis.utils import ImageLoader, letterbox

PADDLE_MODEL_DIR = os.getenv('PADDLE_MODEL_DIR', '')
if PADDLE_MODEL_DIR == "":
    PADDLE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print("PADDLE_MODEL_DIR", PADDLE_MODEL_DIR)
class ChatTextRecognition:
    def __init__(self, model_name: str="", lang:str = "", **kwargs):

        self.lang = 'multi' if lang == '' else lang
        self.model_name = 'PaddleOCR' if model_name == '' else model_name
        self.model = None
        self.predict_kwargs = kwargs
        

    def load_model(self):
        if self.model_name == 'PaddleOCR':
            self.model = PaddleOCR(self.lang)
        if self.model_name == 'PP-OCRv5_server_rec':
            self.model = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir=os.path.join(PADDLE_MODEL_DIR, 'models/PP-OCRv5_server_rec/'))

    def predict_text(self, image:np.ndarray):
        if self.model_name != 'PaddleOCR':
            text = self.model.predict(image)
        return text


class ChatLayoutAnalyzer:
    """基于PP-DocLayoutV2的聊天内容定位分析器"""
    
    def __init__(self, model_name: str = "PP-DocLayout-L", **kwargs):
        """
        初始化聊天内容定位分析器
        
        Args:
            model_name: 模型名称，可选 PP-DocLayout-L/M/S
            **kwargs: 模型预测参数
        """
        self.model_name = model_name
        self.model = None
        self.predict_kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.current_image = None

    def load_model(self):
        """加载PP-DocLayoutV2模型"""
        if self.model is not None:
            self.logger.info(f"{self.model_name} 模型已加载，跳过")
            return self.model
            
        self.logger.info(f"正在加载模型 {self.model_name}...")
        try:
            if self.model_name == 'PP-DocLayoutV2':
                threshold_by_id = {
                    0 : 0.9, # abstract
                    1 : 0.6, # algorithm
                    2 : 0.6, # aside_text
                    3 : 0.6, # chart
                    4 : 0.6, # content
                    5 : 0.6, # display_formula
                    6 : 0.6, # doc_title
                    7 : 0.6, # figure_title
                    8 : 0.6, # footer
                    9 : 0.6, # footer_image
                    10: 0.6, # footnote
                    11: 0.6, # formula_number
                    12: 0.6, # header
                    13: 0.6, # header_image
                    14: 0.5, # image
                    15: 0.6, # inline_formula
                    16: 0.6, # number 
                    17: 0.4, # paragraph_title
                    18: 0.6, # reference
                    19: 0.6, # reference_content
                    20: 0.6, # seal
                    21: 0.6, # table
                    22: 0.4, # text
                    23: 0.6, # vertical_text
                    24: 0.6, # vision_footnote
                }
                model_dir = os.path.join(PADDLE_MODEL_DIR, 'models/PP-DocLayoutV2')
                self.logger.info(f"模型目录: {model_dir}")
                self.logger.info("开始创建 LayoutDetection 实例...")
                self.model = LayoutDetection(model_name=self.model_name, model_dir=model_dir, threshold=threshold_by_id)
                self.logger.info("LayoutDetection 实例创建完成")
            elif self.model_name == 'PP-DocLayout-L':
                raise NotImplementedError("PP-DocLayout-L模型加载未实现")
            elif self.model_name == "PP-OCRv5_server_det":
                model_dir = os.path.join(PADDLE_MODEL_DIR, 'models/PP-OCRv5_server_det/')
                self.logger.info(f"模型目录: {model_dir}")
                self.model = TextDetection(model_name=self.model_name, model_dir=model_dir)
            
            self.logger.info(f"{self.model_name} 模型加载成功")
        except BaseException as e:
            self.logger.error(f"模型加载失败 (load_model): {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise
        return self.model
    
    def analyze_chat_screenshot(self, image, **kwargs) -> Dict[str, Any]:
        """
        分析聊天截图的内容布局
        
        Args:
            image_path: 图片路径，支持本地文件路径或PIL.Image对象
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 加载模型
            self.logger.info("analyze_chat_screenshot: 开始加载模型...")
            # model = self.load_model()
            self.logger.info(f"analyze_chat_screenshot: 模型加载完成, model={self.model}")
            self.logger.info(f"image type:{type(image)}")
            if not isinstance(image, np.ndarray):
                # 预处理图像
                self.logger.info("开始加载图像...")
                image = ImageLoader.load_image(image)
                if image is None:
                    raise ValueError("图像加载失败，请检查图像路径或格式")
                self.logger.info(f"图像加载完成, mode={image.mode}")
                if image.mode == 'RGBA':
                    image = image.convert("RGB")

                image = np.array(image)

            self.logger.info(f"image type:{type(image)}, shape:{image.shape}")
            
            if kwargs.get("letterbox", None) is None:
                self.logger.info("开始 letterbox 处理...")
                image, padding = letterbox(image)
            else:
                padding = kwargs.get("padding")
                self.logger.info(f"使用自定义 padding: {padding}")
            self.logger.info(f"letterbox 完成, padding={padding}")

            # 执行版面分析
            self.logger.info("开始版面分析...")
            results = self.model.predict(image, **self.predict_kwargs)
            self.logger.info("版面分析完成")
            self.current_image = image
            return {
                'success': True,
                'image_size': [image.shape[1], image.shape[0]], # w, h
                'padding': [float(p) for p in padding],  # 确保 padding 是浮点数列表
                'results': results
            }
            
        except Exception as e:
            import traceback
            self.logger.error(f"分析过程出错: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_chat_elements(self, layout_results) -> List[Dict[str, Any]]:
        """提取与聊天内容相关的版面元素"""
        chat_elements = []
        
        for result in layout_results:
            # 转换结果为标准格式
            element = {
                'category': result.category,
                'bbox': result.bbox,  # [x1, y1, x2, y2]
                'confidence': result.score,
                'area': (result.bbox[2] - result.bbox[0]) * (result.bbox[3] - result.bbox[1])
            }
            
            # 过滤聊天相关元素
            if self._is_chat_related(element):
                chat_elements.append(element)
                
        # 按垂直位置排序（聊天消息通常从上到下排列）
        chat_elements.sort(key=lambda x: x['bbox'][1])
        
        return chat_elements
    
    def _is_chat_related(self, element: Dict[str, Any]) -> bool:
        """判断元素是否与聊天内容相关"""
        category = element['category']
        
        # 过滤掉页眉、页脚等非聊天元素
        non_chat_categories = ['page_header', 'page_footer', 'header_image', 'footer_image']
        if category in non_chat_categories:
            return False
            
        # 面积过小的元素可能是噪声
        if element['area'] < 100:  # 小于100像素可能是噪声
            return False
            
        return True
    
    def _group_chat_messages(self, elements: List[Dict[str, Any]]) -> List[List[Dict]]:
        """将元素分组为聊天消息"""
        if not elements:
            return []
            
        message_groups = []
        current_group = [elements[0]]
        
        for i in range(1, len(elements)):
            current_element = elements[i]
            last_element = current_group[-1]
            
            # 判断是否属于同一消息（基于垂直位置和重叠）
            if self._should_group_together(last_element, current_element):
                current_group.append(current_element)
            else:
                message_groups.append(current_group)
                current_group = [current_element]
                
        if current_group:
            message_groups.append(current_group)
            
        return message_groups
    
    def _should_group_together(self, elem1: Dict, elem2: Dict) -> bool:
        """判断两个元素是否应该分组到同一消息"""
        y1_center = (elem1['bbox'][1] + elem1['bbox'][3]) / 2
        y2_center = (elem2['bbox'][1] + elem2['bbox'][3]) / 2
        vertical_gap = y2_center - y1_center
        
        # 如果垂直间距较小，可能是同一消息的不同部分
        return vertical_gap < 100  # 可调整的阈值
    
    def _generate_summary(self, elements: List[Dict[str, Any]]) -> Dict[str, int]:
        """生成分析摘要"""
        category_count = {}
        for element in elements:
            category = element['category']
            category_count[category] = category_count.get(category, 0) + 1
            
        return {
            'total_elements': len(elements),
            'categories_found': category_count,
            'estimated_messages': len(self._group_chat_messages(elements))
        }

    def analyze_chat_session(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        分析整个聊天会话的多张截图
        
        Args:
            image_paths: 多张聊天截图路径列表
            
        Returns:
            会话分析结果
        """
        session_results = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"分析第 {i+1}/{len(image_paths)} 张截图...")
            result = self.analyze_chat_screenshot(image_path)
            result['image_index'] = i
            session_results.append(result)
            
        return {
            'session_summary': {
                'total_images': len(image_paths),
                'successful_analyses': sum(1 for r in session_results if r['success']),
                'total_messages': sum(r.get('total_messages', 0) for r in session_results if r['success'])
            },
            'detailed_results': session_results
        }