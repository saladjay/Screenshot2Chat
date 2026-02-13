"""示例：如何使用 TextDetector 和 BubbleDetector

这个示例展示了如何使用新的检测器架构来分析聊天截图。
"""

import numpy as np
from PIL import Image
from pathlib import Path

def example_text_detection():
    """示例：使用 TextDetector 检测文本框"""
    print("=" * 60)
    print("示例 1: 文本检测")
    print("=" * 60)
    
    from screenshot2chat.detectors import TextDetector
    
    # 创建文本检测器
    detector = TextDetector(config={
        "backend": "PP-OCRv5_server_det",  # 或 "paddleocr"
        "lang": "multi",
        "auto_load": False  # 设置为 True 可以自动加载模型
    })
    
    print(f"✓ TextDetector 已创建")
    print(f"  - Backend: {detector.backend}")
    print(f"  - Language: {detector.lang}")
    
    # 加载图像
    image_path = Path("test_images/test_whatsapp.png")
    if image_path.exists():
        image = Image.open(image_path)
        image_array = np.array(image)
        print(f"✓ 图像已加载: {image_array.shape}")
        
        # 预处理图像
        processed = detector.preprocess(image_array)
        print(f"✓ 图像预处理完成: {processed.shape}")
        
        # 注意：实际检测需要加载模型
        # detector.load_model()
        # results = detector.detect(processed)
        # print(f"✓ 检测到 {len(results)} 个文本框")
    else:
        print(f"⚠ 测试图像未找到: {image_path}")
    
    print()

def example_bubble_detection():
    """示例：使用 BubbleDetector 检测聊天气泡"""
    print("=" * 60)
    print("示例 2: 气泡检测")
    print("=" * 60)
    
    from screenshot2chat.detectors import BubbleDetector
    from screenshotanalysis.basemodel import TextBox
    
    # 创建气泡检测器
    detector = BubbleDetector(config={
        "screen_width": 720,
        "memory_path": "chat_memory.json",  # 可选：保存跨截图记忆
        "auto_load": True
    })
    
    print(f"✓ BubbleDetector 已创建")
    print(f"  - Screen width: {detector.screen_width}")
    print(f"  - Memory path: {detector.memory_path}")
    
    # 创建模拟的文本框（实际使用中，这些来自 TextDetector）
    text_boxes = [
        TextBox(box=[50, 100, 300, 150], score=0.9),   # 左列
        TextBox(box=[420, 150, 670, 200], score=0.9),  # 右列
        TextBox(box=[50, 200, 300, 250], score=0.9),   # 左列
        TextBox(box=[420, 250, 670, 300], score=0.9),  # 右列
    ]
    
    # 创建虚拟图像
    dummy_image = np.zeros((800, 720, 3), dtype=np.uint8)
    
    # 执行检测
    results = detector.detect(dummy_image, text_boxes=text_boxes)
    
    print(f"✓ 检测到 {len(results)} 个气泡")
    for i, result in enumerate(results):
        print(f"  气泡 {i+1}:")
        print(f"    - 说话者: {result.metadata.get('speaker')}")
        print(f"    - 布局: {result.metadata.get('layout')}")
        print(f"    - 位置: {result.bbox}")
    
    # 查看记忆状态
    memory = detector.get_memory_state()
    print(f"✓ 记忆状态:")
    print(f"  - 已处理帧数: {memory['frame_count']}")
    print(f"  - Speaker A 记忆: {memory['A']}")
    print(f"  - Speaker B 记忆: {memory['B']}")
    
    print()

def example_pipeline():
    """示例：组合使用 TextDetector 和 BubbleDetector"""
    print("=" * 60)
    print("示例 3: 完整流水线")
    print("=" * 60)
    
    from screenshot2chat.detectors import TextDetector, BubbleDetector
    from screenshot2chat.core import DetectionResult
    
    # 步骤 1: 创建检测器
    text_detector = TextDetector(config={"auto_load": False})
    bubble_detector = BubbleDetector(config={
        "screen_width": 720,
        "auto_load": True
    })
    
    print("✓ 检测器已创建")
    
    # 步骤 2: 模拟文本检测结果
    # 实际使用中: text_results = text_detector.detect(image)
    text_results = [
        DetectionResult(
            bbox=[50, 100, 300, 150],
            score=0.9,
            category="text",
            metadata={"text": "Hello!"}
        ),
        DetectionResult(
            bbox=[420, 150, 670, 200],
            score=0.9,
            category="text",
            metadata={"text": "Hi there!"}
        ),
    ]
    
    print(f"✓ 文本检测: {len(text_results)} 个文本框")
    
    # 步骤 3: 气泡检测
    dummy_image = np.zeros((800, 720, 3), dtype=np.uint8)
    bubble_results = bubble_detector.detect(dummy_image, text_boxes=text_results)
    
    print(f"✓ 气泡检测: {len(bubble_results)} 个气泡")
    
    # 步骤 4: 处理结果
    for i, bubble in enumerate(bubble_results):
        print(f"  消息 {i+1}:")
        print(f"    - 说话者: {bubble.metadata.get('speaker')}")
        print(f"    - 文本: {bubble.metadata.get('text', 'N/A')}")
        print(f"    - 置信度: {bubble.score:.2f}")
    
    print()

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("检测器使用示例")
    print("=" * 60 + "\n")
    
    example_text_detection()
    example_bubble_detection()
    example_pipeline()
    
    print("=" * 60)
    print("所有示例完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
