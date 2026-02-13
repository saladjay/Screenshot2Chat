"""文本检测器

包装现有的 ChatTextRecognition 和 ChatLayoutAnalyzer 实现，
提供符合 BaseDetector 接口的文本检测功能。
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from ..core.base_detector import BaseDetector
from ..core.data_models import DetectionResult
from ..core.exceptions import DetectionError, ModelLoadError, DataError
from ..logging import StructuredLogger


class TextDetector(BaseDetector):
    """文本检测器
    
    包装 PaddleOCR 的文本检测功能，支持多种后端和语言。
    
    Attributes:
        backend: 后端类型，支持 "paddleocr" 和 "PP-OCRv5_server_det"
        lang: 语言代码，如 "en", "zh", "multi" 等
        model_dir: 模型目录路径
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化文本检测器
        
        Args:
            config: 配置参数，支持以下键：
                - backend: 后端类型，默认 "PP-OCRv5_server_det"
                - lang: 语言代码，默认 "multi"
                - model_dir: 模型目录，默认自动检测
                - auto_load: 是否自动加载模型，默认 False
        """
        super().__init__(config)
        
        self.backend = self.get_config("backend", "PP-OCRv5_server_det")
        self.lang = self.get_config("lang", "multi")
        self.model_dir = self.get_config("model_dir", self._get_default_model_dir())
        self.logger = StructuredLogger(__name__)
        
        # Set context for all logs from this detector
        self.logger.set_context(
            detector_type="TextDetector",
            backend=self.backend,
            lang=self.lang
        )
        
        # 如果配置了自动加载，立即加载模型
        if self.get_config("auto_load", False):
            self.load_model()
    
    def _get_default_model_dir(self) -> str:
        """获取默认模型目录
        
        Returns:
            模型目录的绝对路径
        """
        # 尝试从环境变量获取
        paddle_model_dir = os.getenv('PADDLE_MODEL_DIR', '')
        if paddle_model_dir:
            return paddle_model_dir
        
        # 否则使用相对于当前文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    def load_model(self) -> None:
        """加载 PaddleOCR 文本检测模型
        
        根据配置的 backend 加载相应的模型。
        
        Raises:
            ModelLoadError: 如果模型加载失败
            ValueError: 如果指定的后端不支持
        """
        if self.is_model_loaded():
            self.logger.info("Model already loaded, skipping")
            return
        
        try:
            self.logger.info("Starting model load", backend=self.backend)
            
            if self.backend == "PP-OCRv5_server_det":
                from paddleocr import TextDetection
                
                model_path = os.path.join(self.model_dir, 'models/PP-OCRv5_server_det/')
                
                if not os.path.exists(model_path):
                    error_msg = f"Model directory not found: {model_path}"
                    self.logger.error(error_msg, model_path=model_path)
                    raise ModelLoadError(
                        f"{error_msg}. Please ensure the model is downloaded. "
                        f"Recovery suggestion: Download the model to {model_path}"
                    )
                
                self.logger.info("Loading PP-OCRv5_server_det", model_path=model_path)
                
                self.model = TextDetection(
                    model_name="PP-OCRv5_server_det",
                    model_dir=model_path
                )
                
            elif self.backend == "paddleocr":
                from paddleocr import PaddleOCR
                
                self.logger.info("Loading PaddleOCR", lang=self.lang)
                self.model = PaddleOCR(lang=self.lang, use_angle_cls=True, use_gpu=False)
                
            else:
                error_msg = f"Unsupported backend: {self.backend}"
                self.logger.error(error_msg, backend=self.backend)
                raise ValueError(
                    f"{error_msg}. "
                    f"Supported backends: 'paddleocr', 'PP-OCRv5_server_det'. "
                    f"Recovery suggestion: Set backend to one of the supported values."
                )
            
            self.logger.info("Model loaded successfully", backend=self.backend)
            
        except ImportError as e:
            error_msg = "PaddleOCR is not installed"
            self.logger.error(error_msg, error=str(e))
            raise ModelLoadError(
                f"{error_msg}. Please install it with: "
                "pip install paddleocr paddlepaddle. "
                "Recovery suggestion: Run the installation command and try again."
            ) from e
        except Exception as e:
            self.logger.error(
                "Failed to load model",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise ModelLoadError(f"Failed to load model: {e}") from e
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """执行文本检测
        
        Args:
            image: 输入图像，numpy array 格式 (H, W, C)
        
        Returns:
            检测结果列表，每个结果包含文本框的位置和置信度
            
        Raises:
            DetectionError: 如果检测失败
            DataError: 如果输入图像无效
        """
        # 确保模型已加载
        try:
            self.ensure_model_loaded()
        except Exception as e:
            raise DetectionError(f"Cannot perform detection: {e}") from e
        
        try:
            # 验证输入
            if image is None:
                raise DataError(
                    "Input image is None. "
                    "Recovery suggestion: Provide a valid numpy array image."
                )
            
            if not isinstance(image, np.ndarray):
                raise DataError(
                    f"Input must be a numpy array, got {type(image).__name__}. "
                    "Recovery suggestion: Convert your image to numpy array format."
                )
            
            self.logger.info(
                "Starting text detection",
                image_shape=image.shape,
                image_dtype=str(image.dtype)
            )
            
            # 预处理图像
            processed_image = self.preprocess(image)
            
            # 执行检测
            if self.backend == "PP-OCRv5_server_det":
                raw_results = self.model.predict(processed_image)
            elif self.backend == "paddleocr":
                # PaddleOCR 返回 [detection_results, recognition_results]
                # 我们只需要检测结果
                ocr_results = self.model.ocr(processed_image, rec=False)
                raw_results = ocr_results[0] if ocr_results else []
            else:
                raise DetectionError(f"Unknown backend: {self.backend}")
            
            # 后处理结果
            detection_results = self.postprocess(raw_results)
            
            self.logger.info(
                "Text detection completed",
                num_detections=len(detection_results)
            )
            
            return detection_results
            
        except (DataError, DetectionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "Text detection failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise DetectionError(
                f"Text detection failed: {e}. "
                f"Recovery suggestion: Check image format and model configuration."
            ) from e
    
    def postprocess(self, raw_results: Any) -> List[DetectionResult]:
        """后处理原始检测结果
        
        将 PaddleOCR 的原始输出转换为标准的 DetectionResult 格式。
        
        Args:
            raw_results: PaddleOCR 的原始输出
        
        Returns:
            标准化的检测结果列表
        """
        detection_results = []
        
        if not raw_results:
            return detection_results
        
        try:
            if self.backend == "PP-OCRv5_server_det":
                # PP-OCRv5_server_det 返回的格式
                for result in raw_results:
                    # result 应该有 bbox 和 score 属性
                    if hasattr(result, 'bbox') and hasattr(result, 'score'):
                        bbox = result.bbox
                        score = result.score
                    else:
                        # 如果是字典格式
                        bbox = result.get('bbox', result.get('box'))
                        score = result.get('score', result.get('confidence', 1.0))
                    
                    # 确保 bbox 是 [x_min, y_min, x_max, y_max] 格式
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        detection_results.append(
                            DetectionResult(
                                bbox=list(bbox),
                                score=float(score),
                                category="text",
                                metadata={"backend": self.backend}
                            )
                        )
                    elif isinstance(bbox, np.ndarray):
                        # 如果是 numpy array，可能是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 格式
                        if bbox.shape == (4, 2):
                            # 转换为 [x_min, y_min, x_max, y_max]
                            x_coords = bbox[:, 0]
                            y_coords = bbox[:, 1]
                            bbox_rect = [
                                float(x_coords.min()),
                                float(y_coords.min()),
                                float(x_coords.max()),
                                float(y_coords.max())
                            ]
                            detection_results.append(
                                DetectionResult(
                                    bbox=bbox_rect,
                                    score=float(score),
                                    category="text",
                                    metadata={"backend": self.backend}
                                )
                            )
                
            elif self.backend == "paddleocr":
                # PaddleOCR 返回的格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], confidence]
                for item in raw_results:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        # 提取坐标点
                        points = item[0] if isinstance(item[0], (list, np.ndarray)) else item
                        
                        # 转换为 numpy array 以便处理
                        if not isinstance(points, np.ndarray):
                            points = np.array(points)
                        
                        # 提取 x 和 y 坐标
                        if points.shape == (4, 2):
                            x_coords = points[:, 0]
                            y_coords = points[:, 1]
                            
                            bbox = [
                                float(x_coords.min()),
                                float(y_coords.min()),
                                float(x_coords.max()),
                                float(y_coords.max())
                            ]
                            
                            # 置信度（如果有的话）
                            score = 1.0
                            if len(item) > 1 and isinstance(item[1], (int, float)):
                                score = float(item[1])
                            
                            detection_results.append(
                                DetectionResult(
                                    bbox=bbox,
                                    score=score,
                                    category="text",
                                    metadata={"backend": self.backend}
                                )
                            )
            
        except Exception as e:
            self.logger.error(f"Failed to postprocess results: {e}")
            raise
        
        return detection_results
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像
        
        Args:
            image: 输入图像
        
        Returns:
            预处理后的图像
            
        Raises:
            DataError: 如果图像格式无效
        """
        try:
            # 检查图像格式
            if not isinstance(image, np.ndarray):
                raise DataError(
                    f"Image must be a numpy array, got {type(image).__name__}. "
                    "Recovery suggestion: Convert image to numpy array."
                )
            
            # 检查图像维度
            if image.ndim not in [2, 3]:
                raise DataError(
                    f"Image must be 2D or 3D, got {image.ndim}D with shape {image.shape}. "
                    "Recovery suggestion: Ensure image is in correct format (H, W) or (H, W, C)."
                )
            
            # 如果是灰度图，转换为 RGB
            if image.ndim == 2:
                self.logger.debug("Converting grayscale image to RGB")
                image = np.stack([image] * 3, axis=-1)
            
            # 如果是 RGBA，转换为 RGB
            if image.shape[2] == 4:
                self.logger.debug("Converting RGBA image to RGB")
                image = image[:, :, :3]
            
            return image
            
        except DataError:
            raise
        except Exception as e:
            self.logger.error(
                "Image preprocessing failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise DataError(f"Image preprocessing failed: {e}") from e

