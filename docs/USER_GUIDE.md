# Screenshot2Chat 用户指南

## 欢迎

欢迎使用Screenshot2Chat！这是一个强大的聊天截图分析库，可以帮助您从聊天应用截图中提取结构化信息。

本指南将带您快速上手，并介绍常见的使用场景。

## 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [基础概念](#基础概念)
- [常见使用场景](#常见使用场景)
- [配置指南](#配置指南)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)
- [FAQ](#faq)

---

## 快速开始

### 5分钟入门

让我们从一个最简单的例子开始：

```python
from screenshot2chat.pipeline import Pipeline
import cv2

# 1. 从配置文件创建流水线
pipeline = Pipeline.from_config("config/basic_pipeline.yaml")

# 2. 加载图像
image = cv2.imread("my_screenshot.png")

# 3. 执行分析
results = pipeline.execute(image)

# 4. 查看结果
print(f"检测到 {len(results['text_detection'])} 个文本框")
print(f"提取的昵称: {results['nickname_extraction']['data']['nicknames']}")
```

就这么简单！您已经完成了第一次聊天截图分析。

---

## 安装

### 系统要求

- Python 3.8 或更高版本
- 操作系统: Windows, macOS, Linux
- 推荐: CUDA支持的GPU（可选，用于加速）

### 使用pip安装

```bash
pip install screenshot2chat
```

### 从源码安装

```bash
git clone https://github.com/your-org/screenshot2chat.git
cd screenshot2chat
pip install -e .
```

### 安装依赖

基础依赖会自动安装。如果需要特定功能，可以安装额外的依赖：

```bash
# OCR支持
pip install paddlepaddle paddleocr

# GPU加速
pip install paddlepaddle-gpu

# 云端API支持
pip install openai anthropic
```

### 验证安装

```python
import screenshot2chat
print(screenshot2chat.__version__)
```

---

## 基础概念

### 核心组件

Screenshot2Chat由以下核心组件组成：

#### 1. Detector (检测器)

检测器负责在图像中识别特定元素：

- **TextDetector**: 检测文本框
- **BubbleDetector**: 检测聊天气泡
- **AvatarDetector**: 检测头像（计划中）

#### 2. Extractor (提取器)

提取器从检测结果中提取结构化信息：

- **NicknameExtractor**: 提取昵称
- **SpeakerExtractor**: 识别说话者
- **LayoutExtractor**: 分析布局

#### 3. Pipeline (流水线)

流水线将多个检测器和提取器组合成完整的处理流程。

### 数据流

```
输入图像
  ↓
TextDetector (检测文本)
  ↓
BubbleDetector (检测气泡)
  ↓
NicknameExtractor (提取昵称)
  ↓
SpeakerExtractor (识别说话者)
  ↓
输出结构化数据
```

---

## 常见使用场景

### 场景1: 基础文本提取

从聊天截图中提取所有文本内容。

```python
from screenshot2chat.detectors import TextDetector
import cv2

# 创建文本检测器
detector = TextDetector(backend="paddleocr")
detector.load_model()

# 加载图像
image = cv2.imread("chat_screenshot.png")

# 检测文本
results = detector.detect(image)

# 打印所有文本
for result in results:
    if result.metadata.get("text"):
        print(result.metadata["text"])
```

### 场景2: 昵称识别

识别聊天截图中的用户昵称。

```python
from screenshot2chat.pipeline import Pipeline
import cv2

# 使用预配置的流水线
pipeline = Pipeline.from_config("config/nickname_pipeline.yaml")

# 处理图像
image = cv2.imread("chat_screenshot.png")
results = pipeline.execute(image)

# 获取昵称
nicknames = results["nickname_extraction"]["data"]["nicknames"]

print("识别到的昵称:")
for nick in nicknames:
    print(f"  - {nick['text']} (置信度: {nick['nickname_score']:.1f})")
```

### 场景3: 完整对话分析

分析完整的聊天对话，包括说话者识别和布局分析。

```python
from screenshot2chat.pipeline import Pipeline
import cv2
import json

# 创建完整分析流水线
pipeline = Pipeline.from_config("config/full_analysis.yaml")

# 处理图像
image = cv2.imread("chat_screenshot.png")
results = pipeline.execute(image)

# 构建对话结构
dialog = {
    "layout": results["layout_extraction"]["data"]["layout_type"],
    "speakers": results["speaker_extraction"]["data"]["speakers"],
    "messages": []
}

# 提取消息
for bubble in results["bubble_detection"]:
    message = {
        "speaker": bubble.metadata.get("speaker", "unknown"),
        "text": bubble.metadata.get("text", ""),
        "bbox": bubble.bbox
    }
    dialog["messages"].append(message)

# 保存结果
with open("dialog_output.json", "w", encoding="utf-8") as f:
    json.dump(dialog, f, ensure_ascii=False, indent=2)

print(f"分析完成！共 {len(dialog['messages'])} 条消息")
```

### 场景4: 批量处理

批量处理多张聊天截图。

```python
from screenshot2chat.pipeline import Pipeline
from pathlib import Path
import cv2
from tqdm import tqdm

# 创建流水线
pipeline = Pipeline.from_config("config/basic_pipeline.yaml")

# 获取所有图像文件
image_dir = Path("screenshots")
image_files = list(image_dir.glob("*.png"))

# 批量处理
results_list = []
for image_path in tqdm(image_files, desc="Processing"):
    image = cv2.imread(str(image_path))
    results = pipeline.execute(image)
    results_list.append({
        "file": image_path.name,
        "results": results
    })

print(f"处理完成！共处理 {len(results_list)} 张图像")
```

### 场景5: 自定义流水线

创建自定义的处理流水线。

```python
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors import TextDetector, BubbleDetector
from screenshot2chat.extractors import NicknameExtractor, LayoutExtractor

# 创建组件
text_detector = TextDetector(backend="paddleocr")
bubble_detector = BubbleDetector(config={"screen_width": 1080})
nickname_extractor = NicknameExtractor(config={"top_k": 5})
layout_extractor = LayoutExtractor()

# 构建流水线
pipeline = Pipeline(name="my_custom_pipeline")

# 添加步骤
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=text_detector,
    config={"use_gpu": True}
))

pipeline.add_step(PipelineStep(
    name="bubble_detection",
    step_type=StepType.DETECTOR,
    component=bubble_detector,
    depends_on=["text_detection"]
))

pipeline.add_step(PipelineStep(
    name="nickname_extraction",
    step_type=StepType.EXTRACTOR,
    component=nickname_extractor,
    config={"source": "text_detection"}
))

pipeline.add_step(PipelineStep(
    name="layout_extraction",
    step_type=StepType.EXTRACTOR,
    component=layout_extractor,
    config={"source": "bubble_detection"}
))

# 验证流水线
if pipeline.validate():
    print("流水线配置有效！")
    
# 执行
image = cv2.imread("screenshot.png")
results = pipeline.execute(image)
```

### 场景6: 性能监控

监控处理性能，优化流水线。

```python
from screenshot2chat.pipeline import Pipeline
from screenshot2chat.monitoring import PerformanceMonitor
import cv2

# 创建流水线和监控器
pipeline = Pipeline.from_config("config/basic_pipeline.yaml")
monitor = PerformanceMonitor()

# 处理多张图像
images = [cv2.imread(f"screenshot_{i}.png") for i in range(10)]

for i, image in enumerate(images):
    monitor.start_timer(f"image_{i}")
    results = pipeline.execute(image)
    monitor.stop_timer(f"image_{i}")
    
    # 记录内存
    monitor.record_memory(f"after_image_{i}")

# 生成性能报告
report = monitor.generate_report()
print(report)

# 获取统计信息
stats = monitor.get_stats("image_0")
print(f"平均处理时间: {stats['mean']:.3f}秒")
print(f"最快: {stats['min']:.3f}秒")
print(f"最慢: {stats['max']:.3f}秒")
```

---

## 配置指南

### 配置文件结构

Screenshot2Chat使用YAML或JSON格式的配置文件。

#### 基础配置示例 (basic_pipeline.yaml)

```yaml
name: "basic_chat_analysis"
version: "1.0"

steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    config:
      backend: "paddleocr"
      model_dir: "models/PP-OCRv5_server_det/"
      use_gpu: true
    enabled: true

  - name: "bubble_detection"
    type: "detector"
    class: "BubbleDetector"
    config:
      screen_width: 720
      memory_path: "chat_memory.json"
    depends_on: ["text_detection"]
    enabled: true

  - name: "nickname_extraction"
    type: "extractor"
    class: "NicknameExtractor"
    config:
      source: "text_detection"
      top_k: 3
      min_top_margin_ratio: 0.05
    enabled: true

output:
  format: "json"
  include_metadata: true
```

### 配置管理

使用ConfigManager管理配置：

```python
from screenshot2chat.config import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 加载默认配置
config.load("config/default.yaml", layer="default")

# 加载用户配置（会覆盖默认配置）
config.load("config/user.yaml", layer="user")

# 获取配置值
backend = config.get("detector.text.backend")
threshold = config.get("detector.text.threshold", default=0.5)

# 运行时修改配置
config.set("detector.text.use_gpu", True, layer="runtime")

# 保存用户配置
config.save("config/user.yaml", layer="user")
```

### 配置优先级

配置采用三层结构，优先级从高到低：

1. **Runtime** (运行时): 程序运行时动态设置的配置
2. **User** (用户): 用户自定义配置
3. **Default** (默认): 系统默认配置

---

## 最佳实践

### 1. 选择合适的后端

根据您的需求选择OCR后端：

- **PaddleOCR**: 推荐用于中文和多语言场景，准确率高
- **Tesseract**: 适合英文场景，速度快
- **EasyOCR**: 支持多种语言，易于使用

```python
# 中文场景
detector = TextDetector(backend="paddleocr")

# 英文场景
detector = TextDetector(backend="tesseract")
```

### 2. GPU加速

如果有GPU，启用GPU加速可以显著提升性能：

```yaml
config:
  use_gpu: true
  gpu_mem: 2000  # MB
```

### 3. 批量处理优化

批量处理时，复用流水线实例：

```python
# 好的做法
pipeline = Pipeline.from_config("config.yaml")
for image in images:
    results = pipeline.execute(image)

# 不好的做法（每次都创建新实例）
for image in images:
    pipeline = Pipeline.from_config("config.yaml")  # 避免这样做
    results = pipeline.execute(image)
```

### 4. 错误处理

始终添加适当的错误处理：

```python
from screenshot2chat.core.exceptions import DetectionError, ModelLoadError

try:
    detector.load_model()
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
    # 使用备用方案
    
try:
    results = detector.detect(image)
except DetectionError as e:
    print(f"检测失败: {e}")
    # 记录错误并继续
```

### 5. 日志记录

使用结构化日志记录重要事件：

```python
from screenshot2chat.logging import StructuredLogger

logger = StructuredLogger("my_app")
logger.set_context(user_id="12345")

logger.info("开始处理图像", image_size=image.shape)
logger.warning("检测置信度较低", confidence=0.3)
```

---

## 故障排除

### 常见问题

#### 问题1: 模型加载失败

**错误信息**: `ModelLoadError: Failed to load model`

**解决方案**:
1. 检查模型文件是否存在
2. 验证模型路径配置是否正确
3. 确保有足够的内存

```python
# 检查模型路径
import os
model_path = "models/PP-OCRv5_server_det/"
if not os.path.exists(model_path):
    print(f"模型路径不存在: {model_path}")
```

#### 问题2: 检测结果为空

**可能原因**:
- 图像质量太低
- 检测阈值设置过高
- 图像格式不支持

**解决方案**:
```python
# 降低检测阈值
config = {
    "det_db_thresh": 0.2,  # 降低阈值
    "det_db_box_thresh": 0.3
}
detector = TextDetector(config=config)
```

#### 问题3: 内存不足

**解决方案**:
1. 减小批处理大小
2. 降低图像分辨率
3. 使用CPU模式

```python
# 调整图像大小
import cv2
image = cv2.imread("large_image.png")
image = cv2.resize(image, (720, 1280))  # 缩小图像
```

#### 问题4: 处理速度慢

**优化建议**:
1. 启用GPU加速
2. 使用更快的后端
3. 减少流水线步骤
4. 批量处理

```python
# 启用GPU
config = {"use_gpu": True}

# 或使用更快的后端
detector = TextDetector(backend="tesseract")
```

---

## FAQ

### Q: Screenshot2Chat支持哪些聊天应用？

A: Screenshot2Chat是应用无关的，可以处理任何聊天应用的截图，包括微信、WhatsApp、Telegram、Discord等。

### Q: 可以处理视频吗？

A: 目前主要支持静态图像。如果需要处理视频，可以先提取关键帧，然后逐帧处理。

### Q: 支持哪些语言？

A: 取决于您选择的OCR后端。PaddleOCR支持80+种语言，包括中文、英文、日文、韩文等。

### Q: 如何提高识别准确率？

A: 
1. 使用高质量的截图
2. 选择合适的OCR后端
3. 调整检测阈值
4. 使用GPU加速

### Q: 可以在生产环境使用吗？

A: 可以。Screenshot2Chat设计时考虑了生产环境的需求，包括性能监控、错误处理、日志记录等。

### Q: 如何贡献代码？

A: 欢迎贡献！请查看项目的CONTRIBUTING.md文件了解详情。

### Q: 有商业支持吗？

A: 请联系项目维护者了解商业支持选项。

---

## 下一步

现在您已经掌握了Screenshot2Chat的基础使用，可以：

1. 查看[API参考文档](API_REFERENCE.md)了解详细的API说明
2. 阅读[架构文档](ARCHITECTURE.md)了解系统设计
3. 参考[示例代码](../examples/)获取更多灵感
4. 加入社区讨论，分享您的使用经验

## 获取帮助

如果遇到问题：

1. 查看本文档的[故障排除](#故障排除)部分
2. 搜索[GitHub Issues](https://github.com/your-org/screenshot2chat/issues)
3. 提交新的Issue
4. 加入社区讨论

---

**祝您使用愉快！** 🎉
