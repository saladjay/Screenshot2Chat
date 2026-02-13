"""迁移示例：从旧 API 迁移到新 API

这个示例展示了如何将使用旧版 ChatLayoutDetector API 的代码
迁移到新的模块化架构。每个示例都提供了旧版和新版的并排对比。

Requirements: 15.5
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from screenshotanalysis.basemodel import TextBox


def example_1_basic_usage():
    """示例 1: 基本使用 - 检测聊天气泡"""
    print("=" * 80)
    print("示例 1: 基本使用 - 检测聊天气泡")
    print("=" * 80)
    
    # 创建测试数据
    text_boxes = [
        TextBox(box=[50, 100, 300, 150], score=0.9),   # 左列
        TextBox(box=[420, 150, 670, 200], score=0.9),  # 右列
        TextBox(box=[50, 200, 300, 250], score=0.9),   # 左列
        TextBox(box=[420, 250, 670, 300], score=0.9),  # 右列
    ]
    
    print("\n【旧版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

# 初始化检测器
detector = ChatLayoutDetector(screen_width=720)

# 处理文本框
result = detector.process_frame(text_boxes)

# 使用结果
print(f"Layout: {result['layout']}")
print(f"Speaker A: {len(result['A'])} messages")
print(f"Speaker B: {len(result['B'])} messages")
    """)
    
    # 执行旧版代码
    try:
        from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
        
        detector = ChatLayoutDetector(screen_width=720)
        result = detector.process_frame(text_boxes)
        
        print("输出:")
        print(f"  Layout: {result['layout']}")
        print(f"  Speaker A: {len(result['A'])} messages")
        print(f"  Speaker B: {len(result['B'])} messages")
    except Exception as e:
        print(f"  执行出错: {e}")
    
    print("\n【新版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshot2chat import BubbleDetector
import numpy as np

# 初始化检测器
detector = BubbleDetector(config={
    "screen_width": 720,
    "auto_load": True
})

# 处理文本框（需要提供图像）
dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
detection_results = detector.detect(dummy_image, text_boxes=text_boxes)

# 使用结果
speaker_a = [r for r in detection_results if r.metadata.get("speaker") == "A"]
speaker_b = [r for r in detection_results if r.metadata.get("speaker") == "B"]
layout = detection_results[0].metadata.get("layout") if detection_results else "unknown"

print(f"Layout: {layout}")
print(f"Speaker A: {len(speaker_a)} messages")
print(f"Speaker B: {len(speaker_b)} messages")
    """)
    
    # 执行新版代码
    try:
        from src.screenshot2chat.detectors import BubbleDetector
        
        detector = BubbleDetector(config={
            "screen_width": 720,
            "auto_load": True
        })
        
        dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
        detection_results = detector.detect(dummy_image, text_boxes=text_boxes)
        
        speaker_a = [r for r in detection_results if r.metadata.get("speaker") == "A"]
        speaker_b = [r for r in detection_results if r.metadata.get("speaker") == "B"]
        layout = detection_results[0].metadata.get("layout") if detection_results else "unknown"
        
        print("输出:")
        print(f"  Layout: {layout}")
        print(f"  Speaker A: {len(speaker_a)} messages")
        print(f"  Speaker B: {len(speaker_b)} messages")
    except Exception as e:
        print(f"  执行出错: {e}")
    
    print("\n关键变化:")
    print("  ✓ 导入路径: screenshotanalysis.chat_layout_detector → screenshot2chat")
    print("  ✓ 类名: ChatLayoutDetector → BubbleDetector")
    print("  ✓ 初始化: 位置参数 → 配置字典")
    print("  ✓ 方法: process_frame() → detect()")
    print("  ✓ 参数: 需要提供 image 参数")
    print("  ✓ 返回值: 字典 → List[DetectionResult]")
    
    print()


def example_2_with_memory():
    """示例 2: 使用记忆功能"""
    print("=" * 80)
    print("示例 2: 使用记忆功能")
    print("=" * 80)
    
    # 创建多帧测试数据
    frames = [
        [
            TextBox(box=[50, 100, 300, 150], score=0.9),
            TextBox(box=[420, 150, 670, 200], score=0.9),
        ],
        [
            TextBox(box=[50, 200, 300, 250], score=0.9),
            TextBox(box=[420, 250, 670, 300], score=0.9),
        ],
    ]
    
    print("\n【旧版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

# 初始化检测器（带记忆）
detector = ChatLayoutDetector(
    screen_width=720,
    memory_path="chat_memory.json"
)

# 处理多帧
for frame_boxes in frames:
    result = detector.process_frame(frame_boxes)
    # 记忆会自动更新

# 访问记忆
print(f"Memory A: {detector.memory['A']}")
print(f"Memory B: {detector.memory['B']}")
print(f"Frame count: {detector.frame_count}")
    """)
    
    # 执行旧版代码
    try:
        from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
        
        detector = ChatLayoutDetector(
            screen_width=720,
            memory_path="chat_memory_old.json"
        )
        
        for frame_boxes in frames:
            result = detector.process_frame(frame_boxes)
        
        print("输出:")
        print(f"  Memory A: {detector.memory['A']}")
        print(f"  Memory B: {detector.memory['B']}")
        print(f"  Frame count: {detector.frame_count}")
    except Exception as e:
        print(f"  执行出错: {e}")
    
    print("\n【新版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshot2chat import BubbleDetector
import numpy as np

# 初始化检测器（带记忆）
detector = BubbleDetector(config={
    "screen_width": 720,
    "memory_path": "chat_memory.json",
    "auto_load": True
})

# 处理多帧
dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
for frame_boxes in frames:
    results = detector.detect(dummy_image, text_boxes=frame_boxes)
    # 记忆会自动更新

# 访问记忆
memory_state = detector.get_memory_state()
print(f"Memory A: {memory_state['A']}")
print(f"Memory B: {memory_state['B']}")
print(f"Frame count: {memory_state['frame_count']}")
    """)
    
    # 执行新版代码
    try:
        from src.screenshot2chat.detectors import BubbleDetector
        
        detector = BubbleDetector(config={
            "screen_width": 720,
            "memory_path": "chat_memory_new.json",
            "auto_load": True
        })
        
        dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
        for frame_boxes in frames:
            results = detector.detect(dummy_image, text_boxes=frame_boxes)
        
        memory_state = detector.get_memory_state()
        print("输出:")
        print(f"  Memory A: {memory_state['A']}")
        print(f"  Memory B: {memory_state['B']}")
        print(f"  Frame count: {memory_state['frame_count']}")
    except Exception as e:
        print(f"  执行出错: {e}")
    
    print("\n关键变化:")
    print("  ✓ 记忆访问: detector.memory → detector.get_memory_state()")
    print("  ✓ 帧计数: detector.frame_count → memory_state['frame_count']")
    print("  ✓ 其他功能保持一致")
    
    print()


def example_3_configuration():
    """示例 3: 配置参数"""
    print("=" * 80)
    print("示例 3: 配置参数")
    print("=" * 80)
    
    print("\n【旧版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

# 使用位置参数和关键字参数
detector = ChatLayoutDetector(
    screen_width=720,
    min_separation_ratio=0.18,
    memory_alpha=0.7,
    memory_path="chat_memory.json",
    save_interval=10
)
    """)
    
    print("\n【新版 API】")
    print("-" * 60)
    print("代码:")
    print("""
from screenshot2chat import BubbleDetector

# 使用配置字典
detector = BubbleDetector(config={
    "screen_width": 720,
    "min_separation_ratio": 0.18,
    "memory_alpha": 0.7,
    "memory_path": "chat_memory.json",
    "save_interval": 10,
    "auto_load": True  # 新增：自动加载模型
})
    """)
    
    print("\n关键变化:")
    print("  ✓ 参数传递: 位置/关键字参数 → 配置字典")
    print("  ✓ 新增参数: auto_load 控制是否自动加载")
    print("  ✓ 更灵活: 可以轻松添加新配置项")
    
    print()


def example_4_using_pipeline():
    """示例 4: 使用 Pipeline（新功能）"""
    print("=" * 80)
    print("示例 4: 使用 Pipeline（新功能）")
    print("=" * 80)
    
    print("\n【旧版 API】")
    print("-" * 60)
    print("说明:")
    print("  旧版 API 不支持流水线功能。")
    print("  需要手动组合多个处理步骤。")
    print()
    print("代码:")
    print("""
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.core import ChatTextRecognition

# 手动创建和组合组件
text_rec = ChatTextRecognition()
text_rec.load_model()

detector = ChatLayoutDetector(screen_width=720)

# 手动执行每个步骤
text_boxes = text_rec.recognize(image)
result = detector.process_frame(text_boxes)

# 手动处理结果
# ...
    """)
    
    print("\n【新版 API】")
    print("-" * 60)
    print("说明:")
    print("  新版 API 支持流水线，可以自动组合和执行多个步骤。")
    print()
    print("代码:")
    print("""
from screenshot2chat import Pipeline, TextDetector, BubbleDetector
from screenshot2chat import LayoutExtractor, SpeakerExtractor

# 创建流水线
pipeline = Pipeline(name="chat_analysis")

# 添加文本检测步骤
text_detector = TextDetector(config={"auto_load": True})
pipeline.add_detector("text_detection", text_detector)

# 添加气泡检测步骤
bubble_detector = BubbleDetector(config={
    "screen_width": 720,
    "auto_load": True
})
pipeline.add_detector("bubble_detection", bubble_detector, 
                     depends_on=["text_detection"])

# 添加提取器步骤
layout_extractor = LayoutExtractor()
pipeline.add_extractor("layout_extraction", layout_extractor,
                       depends_on=["text_detection"])

speaker_extractor = SpeakerExtractor()
pipeline.add_extractor("speaker_extraction", speaker_extractor,
                       depends_on=["bubble_detection"])

# 一次性执行所有步骤
results = pipeline.execute(image)

# 获取各步骤的结果
text_results = results["text_detection"]
bubble_results = results["bubble_detection"]
layout_result = results["layout_extraction"]
speaker_result = results["speaker_extraction"]
    """)
    
    print("\n新功能优势:")
    print("  ✓ 自动管理步骤依赖关系")
    print("  ✓ 统一的执行接口")
    print("  ✓ 可以保存和加载流水线配置")
    print("  ✓ 支持条件分支和并行执行（未来）")
    
    print()


def example_5_compatibility_layer():
    """示例 5: 使用兼容层"""
    print("=" * 80)
    print("示例 5: 使用兼容层（过渡方案）")
    print("=" * 80)
    
    print("\n说明:")
    print("  如果您暂时无法迁移，可以使用兼容层继续使用旧 API。")
    print("  兼容层会发出弃用警告，但功能完全兼容。")
    
    print("\n【使用兼容层】")
    print("-" * 60)
    print("代码:")
    print("""
# 方式 1: 使用兼容包装器（推荐）
from screenshot2chat.compat import ChatLayoutDetector

# 使用方式与旧版完全相同
detector = ChatLayoutDetector(screen_width=720)
result = detector.process_frame(text_boxes)

# 会收到弃用警告，但功能正常
    """)
    
    print("\n或者:")
    print("""
# 方式 2: 临时抑制警告（不推荐）
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
detector = ChatLayoutDetector(screen_width=720)
    """)
    
    # 演示兼容层
    try:
        from src.screenshot2chat.compat import ChatLayoutDetector
        
        print("\n执行兼容层代码:")
        text_boxes = [
            TextBox(box=[50, 100, 300, 150], score=0.9),
            TextBox(box=[420, 150, 670, 200], score=0.9),
        ]
        
        detector = ChatLayoutDetector(screen_width=720)
        result = detector.process_frame(text_boxes)
        
        print(f"  ✓ 兼容层工作正常")
        print(f"  Layout: {result['layout']}")
        print(f"  注意: 会收到 DeprecationWarning")
    except Exception as e:
        print(f"  执行出错: {e}")
    
    print("\n建议:")
    print("  ✓ 兼容层仅用于过渡期")
    print("  ✓ 尽快迁移到新 API")
    print("  ✓ 兼容层将在 v1.0.0 移除")
    
    print()


def example_6_result_conversion():
    """示例 6: 结果格式转换"""
    print("=" * 80)
    print("示例 6: 结果格式转换")
    print("=" * 80)
    
    print("\n说明:")
    print("  旧版返回字典格式，新版返回 DetectionResult 列表。")
    print("  如果需要旧版格式，可以使用转换函数。")
    
    print("\n【转换函数】")
    print("-" * 60)
    print("代码:")
    print("""
def convert_to_legacy_format(detection_results):
    '''将新版 DetectionResult 转换为旧版字典格式'''
    result = {
        "layout": "unknown",
        "A": [],
        "B": [],
        "metadata": {}
    }
    
    for dr in detection_results:
        speaker = dr.metadata.get("speaker")
        result["layout"] = dr.metadata.get("layout", "unknown")
        
        if speaker == "A":
            result["A"].append(dr)
        elif speaker == "B":
            result["B"].append(dr)
    
    return result

# 使用示例
detection_results = detector.detect(image, text_boxes=text_boxes)
legacy_result = convert_to_legacy_format(detection_results)

# 现在可以像旧版一样使用
print(f"Layout: {legacy_result['layout']}")
print(f"Speaker A: {len(legacy_result['A'])} messages")
    """)
    
    print("\n建议:")
    print("  ✓ 仅在必要时使用转换函数")
    print("  ✓ 新代码应直接使用 DetectionResult")
    print("  ✓ DetectionResult 提供更丰富的信息")
    
    print()


def migration_checklist():
    """迁移检查清单"""
    print("=" * 80)
    print("迁移检查清单")
    print("=" * 80)
    
    checklist = [
        ("更新导入语句", "screenshotanalysis → screenshot2chat"),
        ("更新类名", "ChatLayoutDetector → BubbleDetector"),
        ("更新初始化", "位置参数 → 配置字典"),
        ("更新方法调用", "process_frame() → detect()"),
        ("添加 image 参数", "detect() 需要 image 参数"),
        ("更新记忆访问", "detector.memory → get_memory_state()"),
        ("处理返回值", "字典 → List[DetectionResult]"),
        ("运行测试", "确保功能正常"),
        ("处理警告", "解决 DeprecationWarning"),
        ("考虑使用 Pipeline", "利用新功能（可选）"),
    ]
    
    print("\n请按照以下步骤进行迁移:\n")
    for i, (task, description) in enumerate(checklist, 1):
        print(f"  [ ] {i}. {task}")
        print(f"      {description}")
        print()
    
    print("完成所有步骤后，您的代码就成功迁移到新 API 了！")
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("迁移示例：从旧 API 迁移到新 API")
    print("=" * 80 + "\n")
    
    try:
        example_1_basic_usage()
        example_2_with_memory()
        example_3_configuration()
        example_4_using_pipeline()
        example_5_compatibility_layer()
        example_6_result_conversion()
        migration_checklist()
        
        print("=" * 80)
        print("✅ 所有迁移示例完成！")
        print("=" * 80)
        print("\n更多资源:")
        print("  📖 迁移指南: docs/MIGRATION_GUIDE.md")
        print("  📝 基本示例: examples/basic_pipeline_example.py")
        print("  🔧 高级用法: examples/pipeline_usage_example.py")
        print("  🧪 测试示例: tests/test_backward_compat.py")
        print()
        
    except Exception as e:
        print(f"\n❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
