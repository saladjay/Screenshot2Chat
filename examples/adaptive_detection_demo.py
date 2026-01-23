"""
演示如何使用新的自适应聊天布局检测器

这个示例展示了如何使用ChatMessageProcessor的新方法来检测聊天布局，
无需指定应用类型。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from screenshotanalysis.processors import ChatMessageProcessor, TextBox


def create_sample_double_column_boxes():
    """创建示例双列布局的文本框"""
    # 左列（通常是对话者）
    left_boxes = [
        TextBox(box=[100, 100, 300, 150], score=0.95),
        TextBox(box=[110, 200, 310, 250], score=0.93),
        TextBox(box=[105, 300, 305, 350], score=0.94),
        TextBox(box=[100, 400, 300, 450], score=0.96),
    ]
    
    # 右列（通常是用户）
    right_boxes = [
        TextBox(box=[450, 150, 650, 200], score=0.92),
        TextBox(box=[460, 250, 660, 300], score=0.94),
        TextBox(box=[455, 350, 655, 400], score=0.93),
    ]
    
    return left_boxes + right_boxes


def create_sample_single_column_boxes():
    """创建示例单列布局的文本框"""
    boxes = [
        TextBox(box=[100, 100, 300, 150], score=0.95),
        TextBox(box=[110, 200, 310, 250], score=0.93),
        TextBox(box=[105, 300, 305, 350], score=0.94),
        TextBox(box=[100, 400, 300, 450], score=0.96),
    ]
    return boxes


def demo_basic_detection():
    """演示基本的自适应检测"""
    print("=" * 60)
    print("示例 1: 基本的自适应检测")
    print("=" * 60)
    
    processor = ChatMessageProcessor()
    boxes = create_sample_double_column_boxes()
    
    # 使用新的自适应检测方法
    result = processor.detect_chat_layout_adaptive(
        text_boxes=boxes,
        screen_width=720
    )
    
    print(f"\n检测结果:")
    print(f"  布局类型: {result['layout']}")
    print(f"  说话者A的消息数: {len(result['A'])}")
    print(f"  说话者B的消息数: {len(result['B'])}")
    print(f"  置信度: {result['metadata'].get('confidence', 'N/A')}")
    print(f"  帧计数: {result['metadata'].get('frame_count', 0)}")
    
    # 显示每个说话者的文本框位置
    print(f"\n说话者A的文本框:")
    for i, box in enumerate(result['A'], 1):
        print(f"  {i}. 位置: ({box.x_min:.0f}, {box.y_min:.0f}) - ({box.x_max:.0f}, {box.y_max:.0f})")
    
    print(f"\n说话者B的文本框:")
    for i, box in enumerate(result['B'], 1):
        print(f"  {i}. 位置: ({box.x_min:.0f}, {box.y_min:.0f}) - ({box.x_max:.0f}, {box.y_max:.0f})")


def demo_format_conversation():
    """演示格式化对话"""
    print("\n" + "=" * 60)
    print("示例 2: 格式化对话")
    print("=" * 60)
    
    processor = ChatMessageProcessor()
    boxes = create_sample_double_column_boxes()
    
    # 使用format_conversation_adaptive方法
    sorted_boxes, metadata = processor.format_conversation_adaptive(
        text_boxes=boxes,
        screen_width=720
    )
    
    print(f"\n元数据:")
    print(f"  布局类型: {metadata['layout']}")
    print(f"  说话者A消息数: {metadata['speaker_A_count']}")
    print(f"  说话者B消息数: {metadata['speaker_B_count']}")
    print(f"  置信度: {metadata['confidence']:.2f}")
    
    print(f"\n按时间顺序排列的消息:")
    for i, box in enumerate(sorted_boxes, 1):
        speaker = box.speaker
        print(f"  {i}. [{speaker}] 位置: ({box.x_min:.0f}, {box.y_min:.0f})")


def demo_single_column():
    """演示单列布局检测"""
    print("\n" + "=" * 60)
    print("示例 3: 单列布局检测")
    print("=" * 60)
    
    processor = ChatMessageProcessor()
    boxes = create_sample_single_column_boxes()
    
    result = processor.detect_chat_layout_adaptive(
        text_boxes=boxes,
        screen_width=720
    )
    
    print(f"\n检测结果:")
    print(f"  布局类型: {result['layout']}")
    print(f"  说话者A的消息数: {len(result['A'])}")
    print(f"  说话者B的消息数: {len(result['B'])}")
    
    if result['layout'] == 'single':
        print(f"\n✓ 正确识别为单列布局")
        print(f"  所有消息都分配给了说话者A")


def demo_with_memory():
    """演示使用记忆持久化"""
    print("\n" + "=" * 60)
    print("示例 4: 使用记忆持久化")
    print("=" * 60)
    
    import tempfile
    
    processor = ChatMessageProcessor()
    
    # 创建临时文件用于存储记忆
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        memory_path = f.name
    
    try:
        # 第一帧
        boxes1 = create_sample_double_column_boxes()
        result1 = processor.detect_chat_layout_adaptive(
            text_boxes=boxes1,
            screen_width=720,
            memory_path=memory_path
        )
        
        print(f"\n第一帧:")
        print(f"  布局: {result1['layout']}")
        print(f"  帧计数: {result1['metadata']['frame_count']}")
        
        # 第二帧（使用相同的记忆路径）
        boxes2 = create_sample_double_column_boxes()
        result2 = processor.detect_chat_layout_adaptive(
            text_boxes=boxes2,
            screen_width=720,
            memory_path=memory_path
        )
        
        print(f"\n第二帧:")
        print(f"  布局: {result2['layout']}")
        print(f"  帧计数: {result2['metadata']['frame_count']}")
        
        print(f"\n✓ 记忆已在多帧之间保持")
        print(f"  记忆文件: {memory_path}")
        
    finally:
        # 清理临时文件
        if os.path.exists(memory_path):
            os.unlink(memory_path)
            print(f"  已清理临时文件")


def demo_comparison():
    """演示新旧方法的对比"""
    print("\n" + "=" * 60)
    print("示例 5: 新旧方法对比")
    print("=" * 60)
    
    processor = ChatMessageProcessor()
    boxes = create_sample_double_column_boxes()
    
    print("\n旧方法特点:")
    print("  ✗ 需要指定app_type参数（Discord、WhatsApp等）")
    print("  ✗ 依赖YAML配置文件")
    print("  ✗ 使用硬编码的阈值")
    print("  ✗ 无法跨截图学习")
    
    print("\n新方法特点:")
    print("  ✓ 完全应用无关，无需app_type")
    print("  ✓ 不依赖配置文件")
    print("  ✓ 自适应几何学习")
    print("  ✓ 跨截图记忆和学习")
    
    # 使用新方法
    result = processor.detect_chat_layout_adaptive(
        text_boxes=boxes,
        screen_width=720
    )
    
    print(f"\n新方法检测结果:")
    print(f"  布局: {result['layout']}")
    print(f"  说话者A: {len(result['A'])} 条消息")
    print(f"  说话者B: {len(result['B'])} 条消息")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("自适应聊天布局检测器演示")
    print("=" * 60)
    
    # 运行所有演示
    demo_basic_detection()
    demo_format_conversation()
    demo_single_column()
    demo_with_memory()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
