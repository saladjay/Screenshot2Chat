"""昵称提取器

包装现有的昵称提取算法，提供统一的提取器接口。
"""

from typing import List, Dict, Any, Optional

import numpy as np

from ..core.base_extractor import BaseExtractor
from ..core.data_models import DetectionResult, ExtractionResult
from ..core.exceptions import ExtractionError, DataError
from ..logging import StructuredLogger
from screenshotanalysis.basemodel import TextBox
from screenshotanalysis.nickname_extractor import extract_nicknames_from_text_boxes


class NicknameExtractor(BaseExtractor):
    """昵称提取器
    
    从文本检测结果中提取昵称候选。使用综合评分系统，包括：
    - 位置评分：优先选择靠近屏幕顶部和中心的文本框
    - 尺寸评分：考虑文本框的宽度和高度
    - 文本特征评分：基于文本内容的特征
    - Y排名评分：基于垂直位置的排名
    
    Attributes:
        config: 配置参数，包括：
            - top_k: 返回前K个候选，默认3
            - min_top_margin_ratio: 最小顶部边距比例，默认0.05
            - top_region_ratio: 顶部区域比例，默认0.2
            - processor: ChatMessageProcessor实例（必需）
            - text_rec: 可选的OCR模型实例
            - ocr_reader: 可选的OCR读取函数
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化昵称提取器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 从配置中获取参数
        self.top_k = self.get_config('top_k', 3)
        self.min_top_margin_ratio = self.get_config('min_top_margin_ratio', 0.05)
        self.top_region_ratio = self.get_config('top_region_ratio', 0.2)
        
        # Initialize logger
        self.logger = StructuredLogger(__name__)
        self.logger.set_context(
            extractor_type="NicknameExtractor",
            top_k=self.top_k
        )
        
        # 获取必需的processor
        self.processor = self.get_config('processor')
        if self.processor is None:
            self.logger.warning("NicknameExtractor initialized without processor. "
                         "You must provide processor in config or pass it to extract().")
        
        # 可选的OCR模型和读取器
        self.text_rec = self.get_config('text_rec')
        self.ocr_reader = self.get_config('ocr_reader')
    
    def extract(
        self, 
        detection_results: List[DetectionResult], 
        image: Optional[np.ndarray] = None
    ) -> ExtractionResult:
        """从检测结果中提取昵称候选
        
        Args:
            detection_results: 文本检测结果列表
            image: 原始图像（letterboxed），用于OCR识别
        
        Returns:
            提取结果，包含昵称候选列表和置信度
            
        Raises:
            ExtractionError: 如果提取失败
            DataError: 如果输入数据无效
        """
        try:
            # 验证必需参数
            processor = self.processor or self.get_config('processor')
            if processor is None:
                error_msg = "processor is required for NicknameExtractor"
                self.logger.error(error_msg)
                raise DataError(
                    f"{error_msg}. Please provide it in config. "
                    "Recovery suggestion: Initialize NicknameExtractor with processor in config."
                )
            
            if image is None:
                error_msg = "image is required for NicknameExtractor to perform OCR"
                self.logger.error(error_msg)
                raise DataError(
                    f"{error_msg}. "
                    "Recovery suggestion: Provide the original image to the extract method."
                )
            
            # Validate image format
            if not isinstance(image, np.ndarray):
                raise DataError(
                    f"Image must be a numpy array, got {type(image).__name__}. "
                    "Recovery suggestion: Convert image to numpy array format."
                )
            
            self.logger.info(
                "Starting nickname extraction",
                num_detections=len(detection_results),
                image_shape=image.shape
            )
            
            # 将DetectionResult转换为TextBox
            text_boxes = self._convert_to_textboxes(detection_results)
            
            if not text_boxes:
                self.logger.warning("No text boxes to extract nicknames from")
                return ExtractionResult(
                    data={'nicknames': []},
                    confidence=0.0,
                    metadata={'reason': 'no_text_boxes'}
                )
            
            # 调用现有的昵称提取函数
            candidates = extract_nicknames_from_text_boxes(
                text_boxes=text_boxes,
                image=image,
                processor=processor,
                text_rec=self.text_rec,
                ocr_reader=self.ocr_reader,
                draw_results=False,
                top_k=self.top_k,
                min_top_margin_ratio=self.min_top_margin_ratio,
                top_region_ratio=self.top_region_ratio
            )
            
            # 计算整体置信度（使用第一个候选的得分）
            confidence = 0.0
            if candidates:
                # 将0-100的得分归一化到0-1
                confidence = candidates[0]['nickname_score'] / 100.0
            
            self.logger.info(
                "Nickname extraction completed",
                num_candidates=len(candidates),
                top_confidence=confidence
            )
            
            return ExtractionResult(
                data={'nicknames': candidates},
                confidence=confidence,
                metadata={
                    'num_candidates': len(candidates),
                    'top_k': self.top_k
                }
            )
            
        except (DataError, ExtractionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "Nickname extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise ExtractionError(
                f"Failed to extract nicknames: {e}. "
                f"Recovery suggestion: Check text detection results and image format."
            ) from e
    
    def _convert_to_textboxes(
        self, 
        detection_results: List[DetectionResult]
    ) -> List[TextBox]:
        """将DetectionResult转换为TextBox对象
        
        Args:
            detection_results: 检测结果列表
        
        Returns:
            TextBox对象列表
        """
        text_boxes = []
        
        for result in detection_results:
            # 只处理文本类别的检测结果
            if result.category != 'text':
                continue
            
            # 创建TextBox对象
            text_box = TextBox(
                box=result.bbox,
                score=result.score
            )
            
            # 如果metadata中有额外信息，设置到TextBox
            if 'text' in result.metadata:
                text_box.text = result.metadata['text']
            if 'text_type' in result.metadata:
                text_box.text_type = result.metadata['text_type']
            if 'source' in result.metadata:
                text_box.source = result.metadata['source']
            
            text_boxes.append(text_box)
        
        return text_boxes
    
    def validate(self, result: ExtractionResult) -> bool:
        """验证提取结果的有效性
        
        Args:
            result: 提取结果
        
        Returns:
            True 如果结果有效，否则 False
        """
        # 检查是否包含nicknames字段
        if 'nicknames' not in result.data:
            return False
        
        # 检查nicknames是否为列表
        nicknames = result.data['nicknames']
        if not isinstance(nicknames, list):
            return False
        
        # 检查每个候选是否包含必需字段
        for candidate in nicknames:
            if not isinstance(candidate, dict):
                return False
            
            required_fields = ['text', 'nickname_score', 'box']
            if not all(field in candidate for field in required_fields):
                return False
        
        return True
    
    def get_top_nickname(self, result: ExtractionResult) -> Optional[Dict[str, Any]]:
        """获取得分最高的昵称候选
        
        Args:
            result: 提取结果
        
        Returns:
            得分最高的昵称候选，如果没有候选则返回None
        """
        nicknames = result.data.get('nicknames', [])
        if not nicknames:
            return None
        
        return nicknames[0]
    
    def get_nickname_text(self, result: ExtractionResult) -> Optional[str]:
        """获取得分最高的昵称文本
        
        Args:
            result: 提取结果
        
        Returns:
            昵称文本，如果没有候选则返回None
        """
        top_candidate = self.get_top_nickname(result)
        if top_candidate is None:
            return None
        
        return top_candidate.get('text')
