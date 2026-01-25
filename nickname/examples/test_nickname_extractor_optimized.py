#!/usr/bin/env python3
"""
测试优化后的昵称提取模块

改进点：
1. 减少模型调用次数：复用text_rec模型实例
2. 使用全局统一的logger
3. 模块化设计，易于集成到其他项目
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from screenshotanalysis import ChatLayoutAnalyzer, ChatMessageProcessor, ChatTextRecognition
from screenshotanalysis import extract_nicknames_smart


# 配置全局logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    test_images_dir = "test_images"
    output_dir = "test_output/smart_nicknames_optimized"
    
    if not os.path.exists(test_images_dir):
        logger.error(f"目录不存在: {test_images_dir}")
        return 1
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not image_files:
        logger.error(f"{test_images_dir} 目录下没有图片")
        return 1
    
    logger.info("正在初始化模型...")
    
    # 初始化检测器（只初始化一次）
    text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
    text_analyzer.load_model()
    
    processor = ChatMessageProcessor()
    
    # 初始化文本识别模型（只初始化一次，所有图片复用）
    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()
    logger.info("模型初始化完成")
    
    logger.info(f"\n找到 {len(image_files)} 张图片")
    logger.info(f"结果将保存到: {output_dir}")
    
    # 处理每张图片
    all_results = {}
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        
        try:
            # 传入text_rec实例，避免重复加载模型
            results = extract_nicknames_smart(
                image_path,
                text_analyzer,
                processor,
                text_rec=text_rec,  # 复用模型实例
                draw_results=True,
                output_dir=output_dir
            )
            all_results[filename] = results
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            all_results[filename] = None
    
    # 输出汇总
    logger.info(f"\n\n{'='*80}")
    logger.info("汇总结果（智能评分 - 包含Y-rank评分）")
    logger.info(f"{'='*80}\n")
    
    for filename, results in all_results.items():
        logger.info(f"[图片] {filename}")
        if results:
            logger.info(f"   [OK] 检测到 {len(results)} 个可能的昵称:")
            for r in results:
                breakdown_str = ', '.join([
                    f"{k}={v:.1f}" for k, v in r['score_breakdown'].items()
                ])
                logger.info(
                    f"      - '{r['text']}' "
                    f"(得分: {r['nickname_score']:.1f}/100, "
                    f"Y排名: {r.get('y_rank', 'N/A')})"
                )
                logger.info(f"        细项: {breakdown_str}")
        else:
            logger.warning(f"   [WARN] 未检测到昵称")
        logger.info("")
    
    logger.info(f"{'='*80}")
    logger.info(f"所有结果已保存到: {output_dir}")
    logger.info(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
