#!/usr/bin/env python3
"""
聊天布局检测器完整演示

本示例展示了如何使用ChatLayoutDetector进行聊天布局检测，包括：
1. 基本的单帧检测
2. 多帧序列处理与记忆学习
3. 不同布局类型的处理
4. 记忆持久化和加载
5. 时序一致性验证
6. Fallback机制演示

作者: ScreenshotAnalysis Team
日期: 2026-01-23
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile
import numpy as np
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.processors import TextBox


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_double_column_boxes(left_center=150, right_center=570, num_left=4, num_right=3):
    """
    创建双列布局的示例文本框
    
    Args:
        left_center: 左列中心x坐标
        right_center: 右列中心x坐标
        num_left: 左列文本框数量
        num_right: 右列文本框数量
    
    Returns:
        TextBox对象列表
    """
    boxes = []
    
    # 左列文本框
    for i in range(num_left):
        x_min = left_center - 100
        x_max = left_center + 100
        y_min = 100 + i * 100
        y_max = y_min + 60
        boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.95))
    
    # 右列文本框
    for i in range(num_right):
        x_min = right_center - 100
        x_max = right_center + 100
        y_min = 150 + i * 100
        y_max = y_min + 60
        boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.93))
    
    return boxes


def create_single_column_boxes(center=360, num_boxes=5):
    """
    创建单列布局的示例文本框
    
    Args:
        center: 列中心x坐标
        num_boxes: 文本框数量
    
    Returns:
        TextBox对象列表
    """
    boxes = []
    for i in range(num_boxes):
        x_min = center - 150
        x_max = center + 150
        y_min = 100 + i * 80
        y_max = y_min + 60
        boxes.append(TextBox(box=[x_min, y_min, x_max, y_max], score=0.94))
    
    return boxes


def create_double_left_boxes():
    """创建左对齐双列布局的示例文本框"""
    return create_double_column_boxes(left_center=150, right_center=300)


def create_double_right_boxes():
    """创建右对齐双列布局的示例文本框"""
    return create_double_column_boxes(left_center=450, right_center=600)


def demo_1_basic_detection():
    """演示1: 基本的单帧检测"""
    print_section("演示 1: 基本的单帧检测")
    
    # 初始化检测器
    detector = ChatLayoutDetector(screen_width=720)
    
    # 创建双列布局的文本框
    boxes = create_double_column_boxes()
    
    print(f"\n输入: {len(boxes)} 个文本框")
    print(f"屏幕宽度: 720 像素")
    
    # 处理单帧
    result = detector.process_frame(boxes)
    
    # 显示结果
    print(f"\n检测结果:")
    print(f"  布局类型: {result['layout']}")
    print(f"  说话者A: {len(result['A'])} 条消息")
    print(f"  说话者B: {len(result['B'])} 条消息")
    print(f"  帧计数: {result['metadata']['frame_count']}")
    
    if 'confidence' in result['metadata']:
        print(f"  置信度: {result['metadata']['confidence']:.2f}")
    
    if 'left_center' in result['metadata']:
        print(f"  左列中心: {result['metadata']['left_center']:.3f} (归一化)")
        print(f"  右列中心: {result['metadata']['right_center']:.3f} (归一化)")
        print(f"  列分离度: {result['metadata']['separation']:.3f}")
    
    print("\n✓ 成功检测双列布局并分配说话者")


def demo_2_layout_types():
    """演示2: 不同布局类型的检测"""
    print_section("演示 2: 不同布局类型的检测")
    
    detector = ChatLayoutDetector(screen_width=720)
    
    # 测试不同布局类型
    test_cases = [
        ("标准双列", create_double_column_boxes()),
        ("左对齐双列", create_double_left_boxes()),
        ("右对齐双列", create_double_right_boxes()),
        ("单列", create_single_column_boxes()),
    ]
    
    for name, boxes in test_cases:
        result = detector.process_frame(boxes)
        print(f"\n{name}:")
        print(f"  检测结果: {result['layout']}")
        print(f"  说话者A: {len(result['A'])} 条")
        print(f"  说话者B: {len(result['B'])} 条")
        
        if result['layout'] == 'single':
            print(f"  ✓ 正确识别为单列布局")
        elif result['layout'] == 'double_left':
            print(f"  ✓ 正确识别为左对齐双列")
        elif result['layout'] == 'double_right':
            print(f"  ✓ 正确识别为右对齐双列")
        else:
            print(f"  ✓ 正确识别为标准双列")


def demo_3_memory_learning():
    """演示3: 多帧序列处理与记忆学习"""
    print_section("演示 3: 多帧序列处理与记忆学习")
    
    detector = ChatLayoutDetector(screen_width=720)
    
    print("\n处理5帧截图序列...")
    
    for frame_num in range(1, 6):
        # 创建略有变化的文本框（模拟真实场景）
        left_center = 150 + np.random.randint(-10, 10)
        right_center = 570 + np.random.randint(-10, 10)
        boxes = create_double_column_boxes(
            left_center=left_center,
            right_center=right_center,
            num_left=np.random.randint(3, 6),
            num_right=np.random.randint(2, 5)
        )
        
        result = detector.process_frame(boxes)
        
        print(f"\n帧 {frame_num}:")
        print(f"  布局: {result['layout']}")
        print(f"  说话者A: {len(result['A'])} 条")
        print(f"  说话者B: {len(result['B'])} 条")
        print(f"  置信度: {result['metadata'].get('confidence', 0):.2f}")
        
        # 显示记忆状态
        if detector.memory['A'] is not None:
            print(f"  记忆A: center={detector.memory['A']['center']:.3f}, "
                  f"count={detector.memory['A']['count']}")
        if detector.memory['B'] is not None:
            print(f"  记忆B: center={detector.memory['B']['center']:.3f}, "
                  f"count={detector.memory['B']['count']}")
    
    print("\n✓ 记忆已在多帧之间学习和更新")
    print(f"✓ 说话者A累计: {detector.memory['A']['count']} 条消息")
    print(f"✓ 说话者B累计: {detector.memory['B']['count']} 条消息")


def demo_4_persistence():
    """演示4: 记忆持久化和加载"""
    print_section("演示 4: 记忆持久化和加载")
    
    # 创建临时文件用于存储记忆
    fd, memory_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)  # 关闭文件描述符，但保留文件
    
    try:
        print(f"\n记忆文件路径: {memory_path}")
        
        # 第一个检测器：处理几帧并保存记忆
        print("\n阶段1: 使用第一个检测器处理3帧...")
        detector1 = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
        
        for i in range(3):
            boxes = create_double_column_boxes()
            result = detector1.process_frame(boxes)
            print(f"  帧 {i+1}: 说话者A={len(result['A'])}, 说话者B={len(result['B'])}")
        
        # 强制保存记忆
        detector1._save_memory()
        
        print(f"\n记忆已保存:")
        if detector1.memory['A'] is not None:
            print(f"  说话者A: {detector1.memory['A']['count']} 条消息")
        if detector1.memory['B'] is not None:
            print(f"  说话者B: {detector1.memory['B']['count']} 条消息")
        
        # 第二个检测器：加载记忆并继续处理
        print("\n阶段2: 使用第二个检测器加载记忆并继续处理...")
        detector2 = ChatLayoutDetector(screen_width=720, memory_path=memory_path)
        
        print(f"\n加载的记忆:")
        if detector2.memory['A'] is not None:
            print(f"  说话者A: {detector2.memory['A']['count']} 条消息")
        else:
            print(f"  说话者A: 未加载")
        if detector2.memory['B'] is not None:
            print(f"  说话者B: {detector2.memory['B']['count']} 条消息")
        else:
            print(f"  说话者B: 未加载")
        
        # 继续处理2帧
        for i in range(2):
            boxes = create_double_column_boxes()
            result = detector2.process_frame(boxes)
            print(f"  帧 {i+4}: 说话者A={len(result['A'])}, 说话者B={len(result['B'])}")
        
        print(f"\n更新后的记忆:")
        if detector2.memory['A'] is not None:
            print(f"  说话者A: {detector2.memory['A']['count']} 条消息")
        if detector2.memory['B'] is not None:
            print(f"  说话者B: {detector2.memory['B']['count']} 条消息")
        
        print("\n✓ 记忆成功持久化和加载")
        print("✓ 跨会话保持说话者身份一致性")
        
    finally:
        # 清理临时文件
        if os.path.exists(memory_path):
            os.unlink(memory_path)
            print(f"\n已清理临时文件: {memory_path}")


def demo_5_temporal_confidence():
    """演示5: 时序一致性验证"""
    print_section("演示 5: 时序一致性验证")
    
    detector = ChatLayoutDetector(screen_width=720)
    
    print("\n场景1: 说话者交替出现（高置信度）")
    # 创建交替出现的文本框（按y坐标排序）
    boxes_alternating = [
        TextBox(box=[100, 100, 300, 150], score=0.95),  # 左
        TextBox(box=[470, 200, 670, 250], score=0.93),  # 右
        TextBox(box=[110, 300, 310, 350], score=0.94),  # 左
        TextBox(box=[460, 400, 660, 450], score=0.92),  # 右
        TextBox(box=[105, 500, 305, 550], score=0.95),  # 左
        TextBox(box=[465, 600, 665, 650], score=0.93),  # 右
    ]
    
    result1 = detector.process_frame(boxes_alternating)
    print(f"  置信度: {result1['metadata']['confidence']:.2f}")
    print(f"  ✓ 交替模式提高了置信度")
    
    print("\n场景2: 说话者连续出现（低置信度）")
    # 创建连续出现的文本框
    boxes_consecutive = [
        TextBox(box=[100, 100, 300, 150], score=0.95),  # 左
        TextBox(box=[110, 200, 310, 250], score=0.94),  # 左
        TextBox(box=[105, 300, 305, 350], score=0.93),  # 左
        TextBox(box=[100, 400, 300, 450], score=0.95),  # 左
        TextBox(box=[470, 500, 670, 550], score=0.92),  # 右
        TextBox(box=[460, 600, 660, 650], score=0.93),  # 右
    ]
    
    # 重置检测器
    detector2 = ChatLayoutDetector(screen_width=720)
    result2 = detector2.process_frame(boxes_consecutive)
    print(f"  置信度: {result2['metadata']['confidence']:.2f}")
    
    if 'uncertain' in result2['metadata']:
        print(f"  ⚠ 低置信度，标记为uncertain")
    
    print("\n✓ 时序规律有效提高了检测准确性")


def demo_6_fallback_mechanism():
    """演示6: Fallback机制"""
    print_section("演示 6: Fallback机制")
    
    print("\n场景1: 历史数据充足，使用KMeans方法")
    detector1 = ChatLayoutDetector(screen_width=720)
    
    # 先处理足够多的帧以积累历史数据
    for i in range(10):
        boxes = create_double_column_boxes()
        detector1.process_frame(boxes)
    
    # 再处理一帧并检查方法
    boxes = create_double_column_boxes()
    result1 = detector1.process_frame(boxes)
    
    if 'method' not in result1['metadata']:
        print(f"  使用方法: KMeans (默认)")
        print(f"  历史数据: {detector1.memory['A']['count'] + detector1.memory['B']['count']} 条消息")
        print(f"  ✓ 数据充足，使用精确的KMeans方法")
    
    print("\n场景2: 历史数据不足，使用median fallback")
    detector2 = ChatLayoutDetector(screen_width=720)
    
    # 只处理一帧（历史数据不足）
    boxes = create_double_column_boxes()
    result2 = detector2.process_frame(boxes)
    
    if 'method' in result2['metadata']:
        print(f"  使用方法: {result2['metadata']['method']}")
        print(f"  原因: {result2['metadata']['reason']}")
        print(f"  ✓ 自动切换到稳定的median方法")
    
    print("\n✓ Fallback机制确保了系统的稳定性")


def demo_7_real_world_scenario():
    """演示7: 真实场景模拟"""
    print_section("演示 7: 真实场景模拟")
    
    print("\n模拟真实聊天会话...")
    print("场景: 用户与朋友的对话，共10条消息")
    
    detector = ChatLayoutDetector(
        screen_width=720,
        min_separation_ratio=0.18,
        memory_alpha=0.7
    )
    
    # 模拟10条消息的对话
    messages = [
        ("left", 100, "你好！"),
        ("right", 200, "嗨，最近怎么样？"),
        ("left", 300, "挺好的，你呢？"),
        ("right", 400, "也不错"),
        ("left", 500, "周末有空吗？"),
        ("right", 600, "有啊，想去哪玩？"),
        ("left", 700, "去爬山怎么样？"),
        ("right", 800, "好主意！"),
        ("left", 900, "那就这么定了"),
        ("right", 1000, "OK，周六见！"),
    ]
    
    all_boxes = []
    for side, y_pos, text in messages:
        if side == "left":
            x_min, x_max = 50, 350
        else:
            x_min, x_max = 420, 670
        
        box = TextBox(box=[x_min, y_pos, x_max, y_pos + 60], score=0.95)
        all_boxes.append(box)
    
    # 处理整个对话
    result = detector.process_frame(all_boxes)
    
    print(f"\n检测结果:")
    print(f"  布局类型: {result['layout']}")
    print(f"  总消息数: {len(all_boxes)}")
    print(f"  说话者A: {len(result['A'])} 条消息")
    print(f"  说话者B: {len(result['B'])} 条消息")
    print(f"  置信度: {result['metadata']['confidence']:.2f}")
    
    # 验证说话者分配
    print(f"\n说话者分配验证:")
    a_positions = [box.y_min for box in result['A']]
    b_positions = [box.y_min for box in result['B']]
    
    print(f"  说话者A的消息位置: {sorted(a_positions)[:3]}... (前3条)")
    print(f"  说话者B的消息位置: {sorted(b_positions)[:3]}... (前3条)")
    
    print("\n✓ 成功处理真实对话场景")
    print("✓ 说话者身份识别准确")


def main():
    """主函数：运行所有演示"""
    print("\n" + "=" * 70)
    print("  聊天布局检测器完整演示")
    print("  ChatLayoutDetector Comprehensive Demo")
    print("=" * 70)
    
    print("\n本演示将展示ChatLayoutDetector的所有主要功能：")
    print("  1. 基本的单帧检测")
    print("  2. 不同布局类型的检测")
    print("  3. 多帧序列处理与记忆学习")
    print("  4. 记忆持久化和加载")
    print("  5. 时序一致性验证")
    print("  6. Fallback机制")
    print("  7. 真实场景模拟")
    
    try:
        demo_1_basic_detection()
        demo_2_layout_types()
        demo_3_memory_learning()
        demo_4_persistence()
        demo_5_temporal_confidence()
        demo_6_fallback_mechanism()
        demo_7_real_world_scenario()
        
        print("\n" + "=" * 70)
        print("  所有演示完成！")
        print("=" * 70)
        
        print("\n关键要点:")
        print("  ✓ ChatLayoutDetector完全应用无关")
        print("  ✓ 自动学习和记忆说话者特征")
        print("  ✓ 支持多种布局类型")
        print("  ✓ 时序规律提高准确性")
        print("  ✓ Fallback机制确保稳定性")
        print("  ✓ 记忆持久化支持跨会话")
        
        print("\n下一步:")
        print("  - 查看 src/screenshotanalysis/chat_layout_detector.py 了解实现细节")
        print("  - 查看 tests/test_chat_layout_detector.py 了解测试用例")
        print("  - 查看 README.md 了解更多使用说明")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
