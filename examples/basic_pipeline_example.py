"""基本流水线使用示例

这个示例展示了如何使用新的 Pipeline API 来分析聊天截图。
包括：
1. 创建和配置检测器
2. 创建和配置提取器
3. 构建完整的处理流水线
4. 执行流水线并处理结果
5. 保存和加载流水线配置

Requirements: 14.5
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from src.screenshot2chat.detectors import TextDetector, BubbleDetector
from src.screenshot2chat.extractors import NicknameExtractor, SpeakerExtractor, LayoutExtractor


def example_1_simple_pipeline():
    """示例 1: 创建一个简单的流水线"""
    print("=" * 80)
    print("示例 1: 创建简单流水线")
    print("=" * 80)
    
    # 步骤 1: 创建流水线
    pipeline = Pipeline(name="simple_chat_analysis")
    print("✓ 创建流水线: simple_chat_analysis")
    
    # 步骤 2: 添加文本检测器
    text_detector = TextDetector(config={
        'backend': 'paddleocr',
        'auto_load': False  # 演示模式，不加载模型
    })
    
    text_step = PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector,
        config={'backend': 'paddleocr'}
    )
    pipeline.add_step(text_step)
    print("✓ 添加步骤: text_detection (TextDetector)")
    
    # 步骤 3: 添加布局提取器
    layout_extractor = LayoutExtractor(config={
        'screen_width': 720
    })
    
    layout_step = PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=['text_detection']
    )
    pipeline.add_step(layout_step)
    print("✓ 添加步骤: layout_extraction (LayoutExtractor)")
    
    # 步骤 4: 验证流水线
    try:
        pipeline.validate()
        print("✓ 流水线验证通过")
    except ValueError as e:
        print(f"✗ 流水线验证失败: {e}")
        return
    
    # 步骤 5: 显示流水线信息
    print(f"\n流水线信息:")
    print(f"  名称: {pipeline.name}")
    print(f"  步骤数: {len(pipeline.steps)}")
    print(f"  步骤列表: {[step.name for step in pipeline.steps]}")
    
    execution_order = pipeline._get_execution_order()
    print(f"  执行顺序: {[step.name for step in execution_order]}")
    
    print()


def example_2_full_pipeline():
    """示例 2: 创建完整的聊天分析流水线"""
    print("=" * 80)
    print("示例 2: 完整的聊天分析流水线")
    print("=" * 80)
    
    # 创建流水线
    pipeline = Pipeline(name="full_chat_analysis")
    print("✓ 创建流水线: full_chat_analysis")
    
    # 步骤 1: 文本检测
    text_detector = TextDetector(config={
        'backend': 'paddleocr',
        'auto_load': False
    })
    pipeline.add_step(PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector
    ))
    print("✓ 添加步骤 1: text_detection")
    
    # 步骤 2: 气泡检测（依赖文本检测）
    bubble_detector = BubbleDetector(config={
        'screen_width': 720,
        'auto_load': True
    })
    pipeline.add_step(PipelineStep(
        name="bubble_detection",
        step_type=StepType.DETECTOR,
        component=bubble_detector,
        depends_on=["text_detection"]
    ))
    print("✓ 添加步骤 2: bubble_detection (依赖: text_detection)")
    
    # 步骤 3: 布局提取（依赖文本检测）
    layout_extractor = LayoutExtractor(config={
        'screen_width': 720
    })
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=["text_detection"]
    ))
    print("✓ 添加步骤 3: layout_extraction (依赖: text_detection)")
    
    # 步骤 4: 说话者提取（依赖气泡检测）
    speaker_extractor = SpeakerExtractor(config={
        'screen_width': 720
    })
    pipeline.add_step(PipelineStep(
        name="speaker_extraction",
        step_type=StepType.EXTRACTOR,
        component=speaker_extractor,
        config={'source': 'bubble_detection'},
        depends_on=["bubble_detection"]
    ))
    print("✓ 添加步骤 4: speaker_extraction (依赖: bubble_detection)")
    
    # 验证流水线
    try:
        pipeline.validate()
        print("✓ 流水线验证通过")
    except ValueError as e:
        print(f"✗ 流水线验证失败: {e}")
        return
    
    # 显示执行顺序
    execution_order = pipeline._get_execution_order()
    print(f"\n执行顺序:")
    for i, step in enumerate(execution_order, 1):
        deps = step.depends_on if step.depends_on else []
        print(f"  {i}. {step.name} (依赖: {deps if deps else '无'})")
    
    print()


def example_3_execute_pipeline():
    """示例 3: 执行流水线（使用模拟数据）"""
    print("=" * 80)
    print("示例 3: 执行流水线")
    print("=" * 80)
    
    # 创建简单流水线
    pipeline = Pipeline(name="demo_pipeline")
    
    # 只添加布局提取器（不需要实际的模型）
    layout_extractor = LayoutExtractor(config={
        'screen_width': 720
    })
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor
    ))
    
    print("✓ 创建演示流水线")
    
    # 创建模拟图像
    dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
    print(f"✓ 创建模拟图像: {dummy_image.shape}")
    
    # 注意：实际执行需要先有文本检测结果
    # 这里只是演示流水线的结构
    print("\n注意: 完整执行需要:")
    print("  1. 加载真实图像")
    print("  2. 加载 OCR 模型")
    print("  3. 执行文本检测")
    print("  4. 然后执行提取器")
    
    print("\n完整执行示例代码:")
    print("""
    # 加载图像
    image = Image.open("test_images/test_whatsapp.png")
    image_array = np.array(image)
    
    # 执行流水线
    results = pipeline.execute(image_array)
    
    # 获取结果
    layout_result = results.get("layout_extraction")
    if layout_result:
        print(f"布局类型: {layout_result.data['layout_type']}")
        print(f"置信度: {layout_result.confidence}")
    """)
    
    print()


def example_4_save_load_config():
    """示例 4: 保存和加载流水线配置"""
    print("=" * 80)
    print("示例 4: 保存和加载流水线配置")
    print("=" * 80)
    
    # 创建流水线
    pipeline = Pipeline(name="configurable_pipeline")
    
    # 添加步骤
    text_detector = TextDetector(config={
        'backend': 'paddleocr',
        'model_dir': 'models/PP-OCRv5_server_det/'
    })
    pipeline.add_step(PipelineStep(
        name="text_detection",
        step_type=StepType.DETECTOR,
        component=text_detector,
        config={'backend': 'paddleocr'}
    ))
    
    layout_extractor = LayoutExtractor(config={
        'screen_width': 720,
        'min_separation_ratio': 0.18
    })
    pipeline.add_step(PipelineStep(
        name="layout_extraction",
        step_type=StepType.EXTRACTOR,
        component=layout_extractor,
        config={'source': 'text_detection'},
        depends_on=['text_detection']
    ))
    
    print("✓ 创建流水线并添加步骤")
    
    # 保存配置
    config_path = "pipeline_basic_example.yaml"
    pipeline.save(config_path)
    print(f"✓ 保存配置到: {config_path}")
    
    # 显示保存的配置
    print("\n保存的配置内容:")
    print("-" * 60)
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    print("-" * 60)
    
    # 说明如何加载
    print("\n加载配置:")
    print("  pipeline = Pipeline.load('pipeline_basic_example.yaml')")
    print("  # 注意: 加载功能需要实现组件的反序列化")
    
    print()


def example_5_configure_components():
    """示例 5: 配置检测器和提取器"""
    print("=" * 80)
    print("示例 5: 配置检测器和提取器")
    print("=" * 80)
    
    print("\n1. 配置 TextDetector:")
    print("-" * 60)
    text_config = {
        'backend': 'paddleocr',           # OCR 后端
        'model_dir': 'models/PP-OCRv5_server_det/',  # 模型目录
        'lang': 'en',                     # 语言（en, ch, 等）
        'auto_load': False                # 演示模式，不自动加载模型
    }
    print("  配置参数:")
    for key, value in text_config.items():
        print(f"    {key}: {value}")
    
    text_detector = TextDetector(config=text_config)
    print(f"  ✓ TextDetector 已创建")
    print(f"  注意: auto_load=False，模型未加载（演示模式）")
    
    print("\n2. 配置 BubbleDetector:")
    print("-" * 60)
    bubble_config = {
        'screen_width': 720,              # 屏幕宽度
        'min_separation_ratio': 0.18,     # 最小分离比例
        'memory_alpha': 0.7,              # 记忆更新系数
        'memory_path': 'chat_memory.json', # 记忆文件路径
        'save_interval': 10,              # 保存间隔
        'auto_load': False                # 演示模式，不自动加载
    }
    print("  配置参数:")
    for key, value in bubble_config.items():
        print(f"    {key}: {value}")
    
    bubble_detector = BubbleDetector(config=bubble_config)
    print(f"  ✓ BubbleDetector 已创建")
    print(f"  注意: auto_load=False，不加载内部检测器（演示模式）")
    
    print("\n3. 配置 LayoutExtractor:")
    print("-" * 60)
    layout_config = {
        'screen_width': 720,              # 屏幕宽度
        'min_separation_ratio': 0.18      # 最小分离比例
    }
    print("  配置参数:")
    for key, value in layout_config.items():
        print(f"    {key}: {value}")
    
    layout_extractor = LayoutExtractor(config=layout_config)
    print(f"  ✓ LayoutExtractor 已创建")
    
    print("\n4. 配置 SpeakerExtractor:")
    print("-" * 60)
    speaker_config = {
        'screen_width': 720,              # 屏幕宽度
        'memory_path': None               # 不持久化记忆
    }
    print("  配置参数:")
    for key, value in speaker_config.items():
        print(f"    {key}: {value}")
    
    speaker_extractor = SpeakerExtractor(config=speaker_config)
    print(f"  ✓ SpeakerExtractor 已创建")
    
    print("\n5. 配置 NicknameExtractor:")
    print("-" * 60)
    nickname_config = {
        'top_k': 3,                       # 返回前 K 个候选
        'min_top_margin_ratio': 0.05,     # 最小顶部边距比例
        'top_region_ratio': 0.2           # 顶部区域比例
    }
    print("  配置参数:")
    for key, value in nickname_config.items():
        print(f"    {key}: {value}")
    
    print("  注意: NicknameExtractor 还需要 processor 和 text_rec 参数")
    print("  这些参数需要在实际使用时提供")
    
    print()


def example_6_error_handling():
    """示例 6: 错误处理"""
    print("=" * 80)
    print("示例 6: 错误处理")
    print("=" * 80)
    
    print("\n测试 1: 缺少依赖")
    print("-" * 60)
    try:
        pipeline = Pipeline(name="invalid_pipeline")
        
        # 添加一个依赖不存在的步骤
        extractor = LayoutExtractor()
        pipeline.add_step(PipelineStep(
            name="layout_extraction",
            step_type=StepType.EXTRACTOR,
            component=extractor,
            depends_on=["nonexistent_step"]
        ))
        
        # 验证会失败
        pipeline.validate()
        print("✗ 应该抛出错误")
    except ValueError as e:
        print(f"✓ 捕获到预期错误: {e}")
    
    print("\n测试 2: 重复的步骤名称")
    print("-" * 60)
    try:
        pipeline = Pipeline(name="duplicate_pipeline")
        
        # 添加第一个步骤
        detector1 = TextDetector()
        pipeline.add_step(PipelineStep(
            name="detector",
            step_type=StepType.DETECTOR,
            component=detector1
        ))
        
        # 尝试添加同名步骤
        detector2 = TextDetector()
        pipeline.add_step(PipelineStep(
            name="detector",  # 重复的名称
            step_type=StepType.DETECTOR,
            component=detector2
        ))
        
        print("✗ 应该抛出错误")
    except ValueError as e:
        print(f"✓ 捕获到预期错误: {e}")
    
    print("\n测试 3: 循环依赖")
    print("-" * 60)
    try:
        pipeline = Pipeline(name="circular_pipeline")
        
        # 创建循环依赖
        detector1 = TextDetector()
        pipeline.add_step(PipelineStep(
            name="step1",
            step_type=StepType.DETECTOR,
            component=detector1,
            depends_on=["step2"]
        ))
        
        detector2 = TextDetector()
        pipeline.add_step(PipelineStep(
            name="step2",
            step_type=StepType.DETECTOR,
            component=detector2,
            depends_on=["step1"]
        ))
        
        # 验证会失败
        pipeline.validate()
        print("✗ 应该抛出错误")
    except ValueError as e:
        print(f"✓ 捕获到预期错误: {e}")
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("基本流水线使用示例")
    print("=" * 80 + "\n")
    
    try:
        example_1_simple_pipeline()
        example_2_full_pipeline()
        example_3_execute_pipeline()
        example_4_save_load_config()
        example_5_configure_components()
        example_6_error_handling()
        
        print("=" * 80)
        print("✅ 所有示例完成！")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 查看 examples/migration_example.py 了解如何从旧 API 迁移")
        print("  2. 查看 docs/MIGRATION_GUIDE.md 获取详细的迁移指南")
        print("  3. 查看 examples/pipeline_usage_example.py 了解更多高级用法")
        print()
        
    except Exception as e:
        print(f"\n❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
