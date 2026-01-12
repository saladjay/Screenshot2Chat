import os
import sys
import logging
from pathlib import Path
from typing import List, Union, Dict
import requests
import base64
from PIL import Image
from io import BytesIO
import re
import validators
import cv2
image_suffixes = ['.png', '.jpg', '.jpeg', '.tiff']

import cv2
import numpy as np

DISCORD = 'discord'
WHATSAPP = 'whatsapp'
INSTAGRAM = 'instagram'
TELEGRAM = 'telegram'

def letterbox(img, new_shape=(800, 800), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    YOLO风格的图像缩放和填充预处理
    
    参数:
    - img: 输入图像 (numpy array)
    - new_shape: 目标尺寸 (width, height)
    - color: 填充颜色 (B, G, R)
    - auto: 是否自动调整填充以满足stride约束
    - scaleFill: 是否直接拉伸填充（不保持比例）
    - scaleup: 是否允许放大图像
    - stride: 模型下采样倍数，通常为32
    
    返回:
    - img: 处理后的图像
    - (left, top, right, bottom): 四边的填充大小
    """
    # 确保图像是3通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # 获取原始图像尺寸
    shape = img.shape[:2]  # [height, width]
    
    # 如果new_shape是整数，转换为正方形
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小不放大
        r = min(r, 1.0)
    
    # 计算缩放后的新尺寸（未填充）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要填充的宽高
    
    if auto:  # 最小矩形，确保能被stride整除
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # 直接拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    # 将填充均分到两侧
    dw /= 2
    dh /= 2
    
    # 调整图像大小
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 计算填充位置
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 添加填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, (left, top, right, bottom)

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chat_layout_analysis.log')
        ]
    )

def ensure_directory(path: str) -> None:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_image_path(image_path: str) -> bool:
    """验证图像路径是否有效"""
    return os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_image_files(directory: str) -> List[str]:
    """获取目录中的所有图像文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
            
    return sorted(image_files)


class ImageLoader:
    """图片加载工具类"""
    
    @staticmethod
    def from_url(url: str) -> Image.Image:
        """从URL加载图片"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            return image
        except Exception as e:
            print(f"URL图片加载失败: {e}")
            return None
    
    @staticmethod
    def from_base64(base64_str: str) -> Image.Image:
        """从Base64加载图片"""
        try:
            # 清理Base64字符串
            if base64_str.startswith('data:image'):
                base64_str = re.sub('^data:image/.+;base64,', '', base64_str)
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            return image
        except Exception as e:
            print(f"Base64图片加载失败: {e}")
            return None
    
    @staticmethod
    def to_base64(pil_image: Image.Image, format: str = 'jpg') -> str:
        """将PIL图片转换为Base64"""
        try:
            buffer = BytesIO()
            pil_image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Base64转换失败: {e}")
            return None

    @staticmethod
    def from_path(path_str:str) -> Image.Image:
        try:
            image = Image.open(path_str)
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            return image
        except Exception as e:
            print(f"路径图片加载失败 {e}")
            return None

    @staticmethod
    def _is_url(url_string:str):
        return validators.url(url_string)


    @staticmethod
    def _is_path(path_string:str):
        return os.path.exists(path_string)

    @staticmethod
    def _is_base64(base64_string:str):
        """
        使用正则表达式检查字符串是否符合Base64格式。
        """
        if not isinstance(s, str):
            return False
        # Base64正则模式：允许字母、数字、'+'、'/'，以及最多两个'='填充符
        pattern = r'^[A-Za-z0-9+/]*={0,2}$'
        # 检查长度是否为4的倍数且符合字符集
        return len(s) % 4 == 0 and re.match(pattern, s) is not None

    @staticmethod
    def load_image(input_value):
        if isinstance(input_value, Path):
            input_value = str(input_value)

        if ImageLoader._is_path(input_value):
            if Path(input_value).suffix.lower() in image_suffixes:
                return ImageLoader.from_path(input_value)
            else:
                print('未实现文件夹加载图片的方法')
                return None
        elif ImageLoader._is_url(input_value):
            return ImageLoader.from_url(input_value)

        elif ImageLoader._is_base64(input_value):
            return ImageLoader.from_base64(input_value)
        elif isinstance(input_value, Image.Image):
            if input_value.mode == 'RGBA':
                input_value = input_value.convert("RGB")
            return input_value
        else:
            raise ValueError(f"未实现的加载数据类型{type(input_value)}")