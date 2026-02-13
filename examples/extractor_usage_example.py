"""提取器使用示例

演示如何使用新的提取器架构来提取昵称、说话者和布局信息。
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.screenshot2chat.core.data_models import DetectionResult, ExtractionResult
from src.screenshot2chat.extractors import (
    NicknameExtractor, 
    SpeakerExtractor, 
    LayoutExtractor
)


def create_sample_detection_results():
    """创建示例检测结果（模拟文本检测器的输出）"""
    # 模拟双列布局的聊天截图
    # 左列：说话者A的消息
    # 右列：说话者B的消息
    
    detection_results = [
        # 顶部昵称区域
        DetectionResult(
            bbox=[50, 20, 150, 45], 
            score=0.95, 
            category='text',
            metadata={'text': 'Alice', 'text_type': 'nickname'}
        ),
        DetectionResult(
            bbox=[570, 20, 670, 45], 
            score=0.93, 
            category='text',
            metadata={'text': 'Bob', 'text_type': 'nickname'}
        ),
        
        # 左列消息（说话者A）
        DetectionResult(
            bbox=[50, 100, 200, 130], 
            score=0.92, 
            category='text',
            metadata={'text': 'Hello!', 'text_type': 'message'}
        ),
        DetectionResult(
            bbox=[50, 150, 220, 180], 
            score=0.91, 
            category='text',
            metadata={'text': 'How are you?', 'text_type': 'message'}
        ),
        
        # 右列消息（说话者B）
        DetectionResult(
            bbox=[500, 200, 670, 230], 
            score=0.90, 
            category='text',
            metadata={'text': 'Hi there!', 'text_type': 'message'}
        ),
        DetectionResult(
            bbox=[520, 250, 670, 280], 
            score=0.89, 
            category='text',
            metadata={'text': "I'm good, thanks!", 'text_type': 'message'}
        ),
        
        # 左列消息（说话者A）
        DetectionResult(
            bbox=[50, 300, 240, 330], 
            score=0.88, 
            category='text',
            metadata={'text': 'Great to hear!', 'text_type': 'message'}
        ),
    ]
    
    return detection_results


def example_layout_extraction():
    """示例1：布局类型提取"""
    print("=" * 80)
    print("示例1：布局类型提取")
    print("=" * 80)
    
    # 创建布局提取器
    layout_extractor = LayoutExtractor(config={
        'screen_width': 720,
        'min_separation_ratio': 0.18
    })
    
    # 获取检测结果
    detection_results = create_sample_detection_results()
    
    # 提取布局信息
    result = layout_extractor.extract(detection_results)
    
    # 打印结果
    print(f"\n布局类型: {layout_extractor.get_layout_type(result)}")
    print(f"是否单列: {layout_extractor.is_single_column(result)}")
    print(f"是否双列: {layout_extractor.is_double_column(result)}")
    print(f"列数: {result.data['num_columns']}")
    print(f"置信度: {result.confidence:.2f}")
    
    print(f"\n左列文本框索引: {layout_extractor.get_column_boxes(result, 'left')}")
    print(f"右列文本框索引: {layout_extractor.get_column_boxes(result, 'right')}")
    
    # 打印列统计信息
    left_stats = layout_extractor.get_column_stats(result, 'left')
    right_stats = layout_extractor.get_column_stats(result, 'right')
    
    print(f"\n左列统计:")
    print(f"  中心位置: {left_stats['center']:.1f}px ({left_stats['center_normalized']:.2f})")
    print(f"  平均宽度: {left_stats['width']:.1f}px ({left_stats['width_normalized']:.2f})")
    print(f"  文本框数: {left_stats['count']}")
    
    print(f"\n右列统计:")
    print(f"  中心位置: {right_stats['center']:.1f}px ({right_stats['center_normalized']:.2f})")
    print(f"  平均宽度: {right_stats['width']:.1f}px ({right_stats['width_normalized']:.2f})")
    print(f"  文本框数: {right_stats['count']}")
    
    print(f"\n元数据: {result.metadata}")


def example_speaker_extraction():
    """示例2：说话者识别"""
    print("\n" + "=" * 80)
    print("示例2：说话者识别")
    print("=" * 80)
    
    # 创建说话者提取器
    speaker_extractor = SpeakerExtractor(config={
        'screen_width': 720,
        'memory_path': None  # 不持久化记忆（仅用于演示）
    })
    
    # 获取检测结果
    detection_results = create_sample_detection_results()
    
    # 提取说话者信息
    result = speaker_extractor.extract(detection_results)
    
    # 打印结果
    print(f"\n布局类型: {speaker_extractor.get_layout_type(result)}")
    print(f"是否双列: {speaker_extractor.is_double_column(result)}")
    print(f"置信度: {result.confidence:.2f}")
    
    print(f"\n说话者A的文本框索引: {speaker_extractor.get_speaker_boxes(result, 'A')}")
    print(f"说话者B的文本框索引: {speaker_extractor.get_speaker_boxes(result, 'B')}")
    
    print(f"\n说话者A消息数: {result.data['num_A']}")
    print(f"说话者B消息数: {result.data['num_B']}")
    
    # 打印每个文本框的说话者
    print("\n文本框说话者分配:")
    for i, det_result in enumerate(detection_results):
        speaker = speaker_extractor.get_speaker_for_box(result, i)
        text = det_result.metadata.get('text', 'N/A')
        print(f"  [{i}] {text:20s} -> Speaker {speaker}")
    
    # 打印记忆状态
    memory_state = speaker_extractor.get_memory_state()
    print(f"\n记忆状态:")
    print(f"  Speaker A: {memory_state['A']}")
    print(f"  Speaker B: {memory_state['B']}")
    print(f"  已处理帧数: {memory_state['frame_count']}")


def example_nickname_extraction():
    """示例3：昵称提取（需要processor和image）"""
    print("\n" + "=" * 80)
    print("示例3：昵称提取")
    print("=" * 80)
    
    print("\n注意：昵称提取需要processor和image参数。")
    print("这个示例展示如何初始化和配置NicknameExtractor。")
    
    # 创建昵称提取器
    nickname_extractor = NicknameExtractor(config={
        'top_k': 3,
        'min_top_margin_ratio': 0.05,
        'top_region_ratio': 0.2,
        # 'processor': processor,  # 需要提供ChatMessageProcessor实例
        # 'text_rec': text_rec,    # 可选的OCR模型
    })
    
    print(f"\n配置参数:")
    print(f"  top_k: {nickname_extractor.top_k}")
    print(f"  min_top_margin_ratio: {nickname_extractor.min_top_margin_ratio}")
    print(f"  top_region_ratio: {nickname_extractor.top_region_ratio}")
    
    print("\n要使用昵称提取器，需要:")
    print("  1. 提供ChatMessageProcessor实例（包含评分逻辑）")
    print("  2. 提供letterboxed图像用于OCR识别")
    print("  3. 可选：提供预加载的OCR模型以提高性能")
    
    print("\n使用示例代码:")
    print("""
    from screenshotanalysis.processors import ChatMessageProcessor
    from screenshotanalysis.core import ChatTextRecognition
    
    # 初始化processor和OCR模型
    processor = ChatMessageProcessor()
    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    text_rec.load_model()
    
    # 配置提取器
    nickname_extractor = NicknameExtractor(config={
        'processor': processor,
        'text_rec': text_rec,
        'top_k': 3
    })
    
    # 提取昵称
    result = nickname_extractor.extract(detection_results, image=letterboxed_image)
    
    # 获取结果
    nicknames = result.data['nicknames']
    top_nickname = nickname_extractor.get_top_nickname(result)
    nickname_text = nickname_extractor.get_nickname_text(result)
    """)


def example_combined_extraction():
    """示例4：组合使用多个提取器"""
    print("\n" + "=" * 80)
    print("示例4：组合使用多个提取器")
    print("=" * 80)
    
    # 创建提取器
    layout_extractor = LayoutExtractor(config={'screen_width': 720})
    speaker_extractor = SpeakerExtractor(config={'screen_width': 720})
    
    # 获取检测结果
    detection_results = create_sample_detection_results()
    
    # 1. 首先提取布局信息
    layout_result = layout_extractor.extract(detection_results)
    print(f"\n步骤1：布局分析")
    print(f"  布局类型: {layout_result.data['layout_type']}")
    print(f"  置信度: {layout_result.confidence:.2f}")
    
    # 2. 然后提取说话者信息
    speaker_result = speaker_extractor.extract(detection_results)
    print(f"\n步骤2：说话者识别")
    print(f"  说话者A消息数: {speaker_result.data['num_A']}")
    print(f"  说话者B消息数: {speaker_result.data['num_B']}")
    print(f"  置信度: {speaker_result.confidence:.2f}")
    
    # 3. 组合结果生成完整的对话结构
    print(f"\n步骤3：生成对话结构")
    
    dialog = {
        'layout': layout_result.data['layout_type'],
        'messages': []
    }
    
    for i, det_result in enumerate(detection_results):
        speaker = speaker_extractor.get_speaker_for_box(speaker_result, i)
        text = det_result.metadata.get('text', '')
        text_type = det_result.metadata.get('text_type', 'unknown')
        
        message = {
            'index': i,
            'speaker': speaker,
            'text': text,
            'type': text_type,
            'bbox': det_result.bbox,
            'score': det_result.score
        }
        
        dialog['messages'].append(message)
    
    # 打印对话结构
    print(f"\n对话结构:")
    print(f"  布局: {dialog['layout']}")
    print(f"  消息数: {len(dialog['messages'])}")
    
    print(f"\n消息列表:")
    for msg in dialog['messages']:
        print(f"  [{msg['index']}] Speaker {msg['speaker']:7s} | "
              f"{msg['type']:10s} | {msg['text']}")
    
    # 验证结果
    print(f"\n验证:")
    print(f"  布局结果有效: {layout_extractor.validate(layout_result)}")
    print(f"  说话者结果有效: {speaker_extractor.validate(speaker_result)}")


def example_json_export():
    """示例5：导出为JSON格式"""
    print("\n" + "=" * 80)
    print("示例5：导出为JSON格式")
    print("=" * 80)
    
    # 创建提取器
    layout_extractor = LayoutExtractor(config={'screen_width': 720})
    speaker_extractor = SpeakerExtractor(config={'screen_width': 720})
    
    # 获取检测结果
    detection_results = create_sample_detection_results()
    
    # 提取信息
    layout_result = layout_extractor.extract(detection_results)
    speaker_result = speaker_extractor.extract(detection_results)
    
    # 导出为JSON
    layout_json = layout_extractor.to_json(layout_result)
    speaker_json = speaker_extractor.to_json(speaker_result)
    
    print("\n布局结果JSON:")
    import json
    print(json.dumps(layout_json, indent=2, ensure_ascii=False))
    
    print("\n说话者结果JSON:")
    print(json.dumps(speaker_json, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("提取器使用示例")
    print("=" * 80)
    
    # 运行所有示例
    example_layout_extraction()
    example_speaker_extraction()
    example_nickname_extraction()
    example_combined_extraction()
    example_json_export()
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
