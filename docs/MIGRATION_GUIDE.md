# Migration Guide: 从旧版 API 迁移到新架构

本指南帮助您将现有代码从旧版 `ChatLayoutDetector` API 迁移到新的模块化架构。

## 目录

- [快速开始](#快速开始)
- [为什么要迁移？](#为什么要迁移)
- [向后兼容性](#向后兼容性)
- [迁移步骤](#迁移步骤)
- [API 对照表](#api-对照表)
- [示例代码对比](#示例代码对比)
- [实用工具函数](#实用工具函数)
- [常见问题](#常见问题)
- [迁移检查清单](#迁移检查清单)

## 快速开始

如果您想快速了解如何迁移，这里是最简单的示例：

### 最小迁移示例

**旧版代码:**
```python
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

detector = ChatLayoutDetector(screen_width=720)
result = detector.process_frame(text_boxes)
```

**新版代码:**
```python
from screenshot2chat import BubbleDetector
import numpy as np

detector = BubbleDetector(config={"screen_width": 720, "auto_load": True})
dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
detection_results = detector.detect(dummy_image, text_boxes=text_boxes)
```

### 运行示例代码

我们提供了完整的示例代码帮助您理解迁移过程：

```bash
# 查看基本用法
python examples/basic_pipeline_example.py

# 查看迁移示例（新旧对比）
python examples/migration_example.py

# 查看高级用法
python examples/pipeline_usage_example.py
```

### 三步快速迁移

1. **更新导入**: `ChatLayoutDetector` → `BubbleDetector`
2. **使用配置字典**: 参数改为 `config={...}`
3. **更新方法**: `process_frame()` → `detect(image, text_boxes=...)`

详细步骤请继续阅读下面的内容。

## 为什么要迁移？

新架构提供了以下优势：

1. **模块化设计**: 清晰的职责分离，更易于维护和扩展
2. **灵活的流水线**: 通过配置文件自由组合处理步骤
3. **统一的接口**: 所有检测器和提取器遵循相同的接口规范
4. **更好的测试**: 支持属性测试和单元测试的双重保障
5. **扩展性**: 轻松添加新的检测器和提取器

## 向后兼容性

**好消息**: 旧版 API 仍然可以使用！

我们提供了完整的向后兼容层，您的现有代码可以继续运行而无需修改。但是：

- 使用旧版 API 时会收到 `DeprecationWarning` 警告
- 旧版 API 将在 **v1.0.0** 版本中移除
- 建议尽快迁移到新 API 以享受新功能

## 迁移步骤

### 步骤 1: 评估现有代码

首先，识别代码中使用旧版 API 的位置：

```python
# 旧版导入
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

# 旧版使用
detector = ChatLayoutDetector(screen_width=720)
result = detector.process_frame(text_boxes)
```

### 步骤 2: 更新导入语句

将导入语句更新为新的模块化 API：

```python
# 新版导入
from screenshot2chat import BubbleDetector
```

### 步骤 3: 更新初始化代码

旧版使用位置参数，新版使用配置字典：

```python
# 旧版
detector = ChatLayoutDetector(
    screen_width=720,
    min_separation_ratio=0.18,
    memory_alpha=0.7,
    memory_path="chat_memory.json"
)

# 新版
detector = BubbleDetector(config={
    "screen_width": 720,
    "min_separation_ratio": 0.18,
    "memory_alpha": 0.7,
    "memory_path": "chat_memory.json",
    "auto_load": True  # 自动加载模型
})
```

### 步骤 4: 更新方法调用

新版 API 使用标准的 `detect()` 方法：

```python
# 旧版
result = detector.process_frame(text_boxes)

# 新版
# 注意：需要提供 image 参数（即使不使用）
import numpy as np
dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
detection_results = detector.detect(dummy_image, text_boxes=text_boxes)

# 如果需要旧版格式的结果，可以转换
result = convert_to_legacy_format(detection_results)
```

### 步骤 5: 测试迁移后的代码

运行测试确保功能正常：

```bash
pytest tests/
```

## API 对照表

### 类和方法对照

| 旧版 API | 新版 API | 说明 |
|---------|---------|------|
| `ChatLayoutDetector` | `BubbleDetector` | 气泡检测器 |
| `process_frame(boxes)` | `detect(image, text_boxes)` | 执行检测 |
| `split_columns(boxes)` | 内部方法，不直接调用 | 列分割 |
| `infer_speaker_in_frame(left, right)` | 内部方法，不直接调用 | 说话者推断 |
| `update_memory(assigned)` | 内部方法，不直接调用 | 记忆更新 |
| `detector.memory` | `detector.get_memory_state()` | 访问记忆 |
| `detector.frame_count` | `detector.get_memory_state()["frame_count"]` | 获取帧数 |

### 配置参数对照

| 旧版参数 | 新版配置键 | 说明 |
|---------|-----------|------|
| `screen_width` | `config["screen_width"]` | 屏幕宽度 |
| `min_separation_ratio` | `config["min_separation_ratio"]` | 最小分离比例 |
| `memory_alpha` | `config["memory_alpha"]` | 记忆更新系数 |
| `memory_path` | `config["memory_path"]` | 记忆文件路径 |
| `save_interval` | `config["save_interval"]` | 保存间隔 |

## 示例代码对比

### 示例 1: 基本使用

**旧版代码:**

```python
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.basemodel import TextBox

# 初始化
detector = ChatLayoutDetector(screen_width=720)

# 创建文本框
boxes = [
    TextBox(box=[100, 100, 200, 150], score=0.9),
    TextBox(box=[500, 200, 600, 250], score=0.85),
]

# 处理
result = detector.process_frame(boxes)

# 使用结果
print(f"Layout: {result['layout']}")
print(f"Speaker A: {len(result['A'])} messages")
print(f"Speaker B: {len(result['B'])} messages")
```

**新版代码:**

```python
from screenshot2chat import BubbleDetector
from screenshotanalysis.basemodel import TextBox
import numpy as np

# 初始化
detector = BubbleDetector(config={
    "screen_width": 720,
    "auto_load": True
})

# 创建文本框
boxes = [
    TextBox(box=[100, 100, 200, 150], score=0.9),
    TextBox(box=[500, 200, 600, 250], score=0.85),
]

# 处理（需要提供图像）
dummy_image = np.zeros((1080, 720, 3), dtype=np.uint8)
detection_results = detector.detect(dummy_image, text_boxes=boxes)

# 使用结果
speaker_a = [r for r in detection_results if r.metadata["speaker"] == "A"]
speaker_b = [r for r in detection_results if r.metadata["speaker"] == "B"]
layout = detection_results[0].metadata["layout"] if detection_results else "unknown"

print(f"Layout: {layout}")
print(f"Speaker A: {len(speaker_a)} messages")
print(f"Speaker B: {len(speaker_b)} messages")
```

### 示例 2: 使用记忆功能

**旧版代码:**

```python
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
print(f"Frame count: {detector.frame_count}")
```

**新版代码:**

```python
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
print(f"Frame count: {memory_state['frame_count']}")
```

### 示例 3: 使用 Pipeline（推荐）

新架构的最大优势是可以使用 Pipeline 组合多个处理步骤：

```python
from screenshot2chat import Pipeline, TextDetector, BubbleDetector
import numpy as np

# 创建流水线
pipeline = Pipeline(name="chat_analysis")

# 添加文本检测步骤
text_detector = TextDetector(config={
    "backend": "paddleocr",
    "auto_load": True
})
pipeline.add_detector("text_detection", text_detector)

# 添加气泡检测步骤
bubble_detector = BubbleDetector(config={
    "screen_width": 720,
    "auto_load": True
})
pipeline.add_detector("bubble_detection", bubble_detector, depends_on=["text_detection"])

# 执行流水线
image = np.array(...)  # 加载图像
results = pipeline.execute(image)

# 使用结果
text_results = results["text_detection"]
bubble_results = results["bubble_detection"]
```

## 常见问题

### Q1: 我必须立即迁移吗？

**A**: 不必须。旧版 API 在 v1.0.0 之前都会保持可用。但我们建议尽早迁移以享受新功能。

### Q2: 迁移会破坏我的现有代码吗？

**A**: 不会。我们提供了完整的向后兼容层。您可以逐步迁移，新旧 API 可以共存。

### Q3: 新版 API 的性能如何？

**A**: 新版 API 内部使用相同的算法，性能基本一致。Pipeline 模式可能会有轻微的额外开销（<5%），但提供了更好的灵活性。

### Q4: 如何处理 DeprecationWarning？

**A**: 有三种方式：

1. **迁移到新 API**（推荐）
2. **临时抑制警告**:
   ```python
   import warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)
   ```
3. **使用兼容包装器**:
   ```python
   from screenshot2chat.compat import ChatLayoutDetector
   ```

### Q5: 新版 API 支持哪些新功能？

**A**: 新版 API 支持：

- 流水线编排（Pipeline）
- 配置管理（ConfigManager）
- 多种提取器（NicknameExtractor, SpeakerExtractor, LayoutExtractor）
- 统一的数据模型（DetectionResult, ExtractionResult）
- 更好的错误处理和日志

### Q6: 我可以混合使用新旧 API 吗？

**A**: 可以，但不推荐。建议在同一个项目中统一使用新 API。

### Q7: 迁移后如何获取旧版格式的结果？

**A**: 可以使用转换函数：

```python
def convert_to_legacy_format(detection_results):
    """将新版 DetectionResult 转换为旧版格式"""
    result = {
        "layout": "unknown",
        "A": [],
        "B": [],
        "metadata": {}
    }
    
    for dr in detection_results:
        speaker = dr.metadata.get("speaker")
        result["layout"] = dr.metadata.get("layout", "unknown")
        
        # 转换回 TextBox（如果需要）
        # ...
        
        if speaker == "A":
            result["A"].append(dr)
        elif speaker == "B":
            result["B"].append(dr)
    
    return result
```

### Q8: 在哪里可以找到更多示例？

**A**: 查看以下资源：

- **基本用法**: `examples/basic_pipeline_example.py` - 展示如何使用新 API
- **迁移示例**: `examples/migration_example.py` - 新旧 API 并排对比
- **高级用法**: `examples/pipeline_usage_example.py` - Pipeline 高级功能
- **检测器示例**: `examples/detector_usage_example.py` - 检测器详细用法
- **提取器示例**: `examples/extractor_usage_example.py` - 提取器详细用法
- **测试用例**: `tests/test_backward_compat.py` - 向后兼容性测试
- **配置示例**: `examples/config_manager_demo.py` - 配置管理
- API 参考文档: `docs/API_REFERENCE.md`（如果可用）
- 架构文档: `docs/ARCHITECTURE.md`（如果可用）

### Q9: 新 API 有哪些最佳实践？

**A**: 推荐的最佳实践：

1. **使用配置字典**: 便于管理和传递配置
   ```python
   config = {
       "screen_width": 720,
       "auto_load": True
   }
   detector = BubbleDetector(config=config)
   ```

2. **使用 Pipeline**: 对于复杂流程，使用 Pipeline 管理
   ```python
   pipeline = Pipeline(name="my_pipeline")
   pipeline.add_detector("text", text_detector)
   pipeline.add_detector("bubble", bubble_detector, depends_on=["text"])
   ```

3. **保存配置**: 将 Pipeline 配置保存为 YAML 文件
   ```python
   pipeline.save("my_pipeline.yaml")
   # 之后可以加载
   pipeline = Pipeline.load("my_pipeline.yaml")
   ```

4. **错误处理**: 使用 try-except 处理可能的错误
   ```python
   try:
       results = detector.detect(image, text_boxes=boxes)
   except DetectionError as e:
       logger.error(f"Detection failed: {e}")
   ```

5. **记忆管理**: 定期保存记忆状态
   ```python
   detector = BubbleDetector(config={
       "memory_path": "memory.json",
       "save_interval": 10  # 每 10 帧保存一次
   })
   ```

### Q10: 如何逐步迁移大型项目？

**A**: 推荐的迁移策略：

1. **阶段 1: 评估**
   - 识别所有使用旧 API 的位置
   - 评估迁移工作量
   - 制定迁移计划

2. **阶段 2: 准备**
   - 确保有完整的测试覆盖
   - 创建迁移分支
   - 备份代码

3. **阶段 3: 逐模块迁移**
   - 从最简单的模块开始
   - 一次迁移一个模块
   - 每次迁移后运行测试

4. **阶段 4: 使用兼容层过渡**
   - 对于复杂模块，先使用兼容层
   - 逐步替换为新 API
   - 保持功能稳定

5. **阶段 5: 验证和优化**
   - 运行完整测试套件
   - 性能测试
   - 考虑使用新功能（如 Pipeline）

6. **阶段 6: 清理**
   - 移除兼容层代码
   - 更新文档
   - 代码审查

## 实用工具函数

### 结果格式转换

如果您需要在新旧代码之间转换结果格式，可以使用以下工具函数：

```python
def convert_to_legacy_format(detection_results):
    """将新版 DetectionResult 转换为旧版字典格式
    
    Args:
        detection_results: List[DetectionResult] from new API
        
    Returns:
        dict: Legacy format result
    """
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


def convert_from_legacy_format(legacy_result):
    """将旧版字典格式转换为新版 DetectionResult
    
    Args:
        legacy_result: dict from old API
        
    Returns:
        List[DetectionResult]: New format results
    """
    from screenshot2chat.core import DetectionResult
    
    detection_results = []
    layout = legacy_result.get("layout", "unknown")
    
    for speaker, boxes in [("A", legacy_result.get("A", [])), 
                           ("B", legacy_result.get("B", []))]:
        for box in boxes:
            dr = DetectionResult(
                bbox=box.box.tolist() if hasattr(box, 'box') else box,
                score=box.score if hasattr(box, 'score') else 1.0,
                category="bubble",
                metadata={
                    "speaker": speaker,
                    "layout": layout
                }
            )
            detection_results.append(dr)
    
    return detection_results
```

### 批量迁移脚本

如果您有大量代码需要迁移，可以使用以下脚本辅助：

```python
#!/usr/bin/env python3
"""批量迁移脚本

自动将旧版 API 调用替换为新版 API。
使用前请备份代码！
"""

import re
import sys
from pathlib import Path


def migrate_file(file_path):
    """迁移单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 替换导入语句
    content = re.sub(
        r'from screenshotanalysis\.chat_layout_detector import ChatLayoutDetector',
        'from screenshot2chat import BubbleDetector',
        content
    )
    
    # 替换类名
    content = re.sub(
        r'ChatLayoutDetector\(',
        'BubbleDetector(config={',
        content
    )
    
    # 替换方法调用
    content = re.sub(
        r'\.process_frame\(([^)]+)\)',
        r'.detect(dummy_image, text_boxes=\1)',
        content
    )
    
    # 替换记忆访问
    content = re.sub(
        r'detector\.memory\[',
        'detector.get_memory_state()[',
        content
    )
    
    content = re.sub(
        r'detector\.frame_count',
        'detector.get_memory_state()["frame_count"]',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def main():
    """批量迁移目录中的所有 Python 文件"""
    if len(sys.argv) < 2:
        print("Usage: python migrate_script.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    print(f"Migrating files in {directory}...")
    print("=" * 60)
    
    migrated_count = 0
    for py_file in directory.rglob("*.py"):
        if migrate_file(py_file):
            print(f"✓ Migrated: {py_file}")
            migrated_count += 1
    
    print("=" * 60)
    print(f"Migration complete! {migrated_count} files updated.")
    print("\nIMPORTANT: Please review the changes and run tests!")


if __name__ == '__main__':
    main()
```

**警告**: 自动迁移脚本可能无法处理所有情况。请务必：
1. 备份代码
2. 仔细审查更改
3. 运行完整的测试套件
4. 手动处理复杂情况

## 获取帮助

如果在迁移过程中遇到问题：

1. 查看示例代码: `examples/migration_example.py`
2. 查看基本用法: `examples/basic_pipeline_example.py`
3. 查看测试用例: `tests/test_backward_compat.py`
4. 查看 [API 参考文档](API_REFERENCE.md)（如果可用）
5. 提交 Issue 到 GitHub
6. 联系维护团队

## 迁移检查清单

使用此检查清单确保完整迁移：

### 代码更新
- [ ] 更新所有导入语句
  - `from screenshotanalysis.chat_layout_detector import ChatLayoutDetector`
  - → `from screenshot2chat import BubbleDetector`
- [ ] 更新类名
  - `ChatLayoutDetector` → `BubbleDetector`
- [ ] 更新初始化代码（使用配置字典）
  - `ChatLayoutDetector(screen_width=720)` 
  - → `BubbleDetector(config={"screen_width": 720})`
- [ ] 更新方法调用
  - `process_frame(boxes)` → `detect(image, text_boxes=boxes)`
- [ ] 添加 image 参数
  - `detect()` 方法需要 image 参数
- [ ] 更新记忆访问方式
  - `detector.memory` → `detector.get_memory_state()`
  - `detector.frame_count` → `detector.get_memory_state()["frame_count"]`
- [ ] 更新结果处理
  - 字典格式 → `List[DetectionResult]`
  - 或使用转换函数保持兼容

### 测试和验证
- [ ] 运行所有单元测试
- [ ] 运行集成测试
- [ ] 验证功能正确性
- [ ] 性能测试（确保无明显下降）
- [ ] 检查是否有 DeprecationWarning

### 文档和清理
- [ ] 更新代码注释
- [ ] 更新 README 和文档
- [ ] 移除或更新弃用警告的处理
- [ ] 代码审查

### 可选优化
- [ ] 考虑使用 Pipeline 模式
- [ ] 使用 ConfigManager 管理配置
- [ ] 使用新的提取器（LayoutExtractor, SpeakerExtractor）
- [ ] 保存 Pipeline 配置为 YAML 文件

### 示例代码参考
- [ ] 查看 `examples/basic_pipeline_example.py`
- [ ] 查看 `examples/migration_example.py`
- [ ] 查看 `examples/pipeline_usage_example.py`
- [ ] 运行示例代码确保理解

## 总结

迁移到新 API 虽然需要一些工作，但会带来更好的代码结构和更强的扩展性。我们提供了完整的向后兼容支持，让您可以按照自己的节奏进行迁移。

如有任何问题，欢迎随时联系我们！
