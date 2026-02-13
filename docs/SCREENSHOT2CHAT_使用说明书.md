# Screenshot2Chat 详细使用说明书

## 📖 目录

- [1. 概述](#1-概述)
- [2. 系统架构](#2-系统架构)
- [3. 安装与配置](#3-安装与配置)
- [4. 核心组件详解](#4-核心组件详解)
- [5. 快速入门](#5-快速入门)
- [6. 高级使用](#6-高级使用)
- [7. 配置管理](#7-配置管理)
- [8. 性能优化](#8-性能优化)
- [9. 故障排除](#9-故障排除)
- [10. API 参考](#10-api-参考)
- [11. 最佳实践](#11-最佳实践)
- [12. 常见问题](#12-常见问题)

---

## 1. 概述

### 1.1 什么是 Screenshot2Chat？

Screenshot2Chat 是一个强大的、模块化的聊天截图分析框架，能够从各种聊天应用的截图中自动提取结构化信息。该系统采用先进的计算机视觉和自然语言处理技术，支持：

- ✅ **应用无关**: 支持微信、WhatsApp、Telegram、Discord、Instagram 等任何聊天应用
- ✅ **智能检测**: 自动识别文本框、聊天气泡、头像、表情等元素
- ✅ **布局分析**: 自动识别单列/双列布局，区分不同说话者
- ✅ **跨截图学习**: 支持记忆功能，保持说话者身份的一致性
- ✅ **高度可扩展**: 模块化设计，易于添加新功能
- ✅ **性能监控**: 内置性能监控和日志系统

### 1.2 主要特性

#### 核心功能
- **文本检测**: 基于 PaddleOCR 的高精度文本检测和识别
- **气泡检测**: 智能识别聊天气泡和说话者身份
- **昵称提取**: 自动提取聊天界面中的用户昵称
- **说话者识别**: 区分不同说话者的消息
- **布局分析**: 识别聊天界面的布局类型

#### 技术特性
- **流水线架构**: 灵活的流水线系统，支持自定义处理流程
- **并行处理**: 支持多步骤并行执行，提升处理速度
- **条件执行**: 支持基于条件的步骤执行
- **配置管理**: 三层配置系统（默认/用户/运行时）
- **性能监控**: 实时监控处理性能和资源使用
- **结构化日志**: 完整的日志记录系统

### 1.3 适用场景


- 📱 **社交媒体分析**: 分析聊天记录，提取对话内容
- 🔍 **内容审核**: 自动检测和分类聊天内容
- 📊 **数据挖掘**: 从聊天截图中提取结构化数据
- 🤖 **聊天机器人训练**: 收集对话数据用于训练
- 📝 **文档生成**: 将聊天记录转换为文档格式
- 🎓 **研究分析**: 学术研究中的对话分析

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Screenshot2Chat                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Detectors   │  │  Extractors  │  │  Pipeline    │  │
│  │  (检测器)     │  │  (提取器)     │  │  (流水线)     │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤  │
│  │ TextDetector │  │ Nickname     │  │ Step         │  │
│  │ BubbleDetect │  │ Speaker      │  │ Execution    │  │
│  │ AvatarDetect │  │ Layout       │  │ Dependency   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Config     │  │  Monitoring  │  │   Logging    │  │
│  │  (配置管理)   │  │  (性能监控)   │  │  (日志系统)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Core (核心抽象层)                      │   │
│  │  BaseDetector | BaseExtractor | DataModels      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 模块说明

#### 核心模块 (core/)
- **BaseDetector**: 检测器基类，定义检测器接口
- **BaseExtractor**: 提取器基类，定义提取器接口
- **DataModels**: 数据模型（DetectionResult, ExtractionResult）
- **Exceptions**: 自定义异常类

#### 检测器模块 (detectors/)
- **TextDetector**: 文本检测器，基于 PaddleOCR
- **BubbleDetector**: 聊天气泡检测器，识别说话者

#### 提取器模块 (extractors/)
- **NicknameExtractor**: 昵称提取器
- **SpeakerExtractor**: 说话者识别器
- **LayoutExtractor**: 布局分析器

#### 流水线模块 (pipeline/)
- **Pipeline**: 流水线管理器
- **PipelineStep**: 流水线步骤定义
- **StepType**: 步骤类型枚举

#### 配置模块 (config/)
- **ConfigManager**: 配置管理器，支持三层配置

#### 监控模块 (monitoring/)
- **PerformanceMonitor**: 性能监控器

#### 日志模块 (logging/)
- **StructuredLogger**: 结构化日志记录器

#### 模型管理 (models/)
- **ModelManager**: 模型管理器，统一管理各种模型

---

## 3. 安装与配置

### 3.1 系统要求


**硬件要求**:
- CPU: 2核心以上（推荐4核心）
- 内存: 4GB 以上（推荐8GB）
- 硬盘: 2GB 可用空间（用于模型存储）
- GPU: 可选，CUDA 支持的 NVIDIA GPU（用于加速）

**软件要求**:
- Python: 3.8 或更高版本
- 操作系统: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- PaddlePaddle: 2.4.0 或更高版本

### 3.2 安装步骤

#### 方法1: 使用 pip 安装（推荐）

```bash
# 基础安装
pip install screenshot2chat

# 安装 PaddleOCR 依赖
pip install paddlepaddle paddleocr

# GPU 版本（如果有 NVIDIA GPU）
pip install paddlepaddle-gpu
```

#### 方法2: 从源码安装

```bash
# 克隆仓库
git clone https://github.com/your-org/screenshot2chat.git
cd screenshot2chat

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 3.3 模型下载

Screenshot2Chat 需要下载 OCR 模型才能正常工作。

#### 自动下载（推荐）

```python
from screenshot2chat.models import ModelManager

# 创建模型管理器
model_manager = ModelManager()

# 下载默认模型
model_manager.download_model("PP-OCRv5_server_det")
model_manager.download_model("PP-OCRv5_server_rec")

print("模型下载完成！")
```

#### 手动下载

1. 访问 [PaddleOCR 模型库](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md)
2. 下载所需模型
3. 解压到 `models/` 目录

```
models/
├── PP-OCRv5_server_det/
│   ├── inference.pdmodel
│   ├── inference.pdiparams
│   └── inference.pdiparams.info
└── PP-OCRv5_server_rec/
    ├── inference.pdmodel
    ├── inference.pdiparams
    └── inference.pdiparams.info
```

### 3.4 验证安装

```python
import screenshot2chat
import numpy as np

# 检查版本
print(f"Screenshot2Chat 版本: {screenshot2chat.__version__}")

# 测试导入
from screenshot2chat.detectors import TextDetector
from screenshot2chat.pipeline import Pipeline

# 创建测试检测器
detector = TextDetector(config={"auto_load": False})
print("✓ 安装验证成功！")
```

### 3.5 环境变量配置

可选的环境变量配置：

```bash
# 设置模型目录
export PADDLE_MODEL_DIR=/path/to/models

# 设置日志级别
export SCREENSHOT2CHAT_LOG_LEVEL=INFO

# 启用 GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## 4. 核心组件详解

### 4.1 检测器 (Detectors)

检测器负责在图像中识别特定元素。

#### 4.1.1 TextDetector (文本检测器)

**功能**: 检测图像中的文本区域

**支持的后端**:
- `paddleocr`: PaddleOCR 标准版
- `PP-OCRv5_server_det`: PaddleOCR 服务器版（推荐）

**基本用法**:

```python
from screenshot2chat.detectors import TextDetector
import cv2

# 创建检测器
detector = TextDetector(config={
    "backend": "PP-OCRv5_server_det",
    "lang": "multi",  # 支持多语言
    "model_dir": "models/",
    "auto_load": True  # 自动加载模型
})

# 加载图像
image = cv2.imread("chat_screenshot.png")

# 执行检测
results = detector.detect(image)

# 处理结果
for result in results:
    print(f"文本框: {result.bbox}")
    print(f"置信度: {result.score}")
    print(f"类别: {result.category}")
```

**配置参数**:


| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | str | "PP-OCRv5_server_det" | OCR 后端类型 |
| `lang` | str | "multi" | 语言代码 (en, zh, multi 等) |
| `model_dir` | str | 自动检测 | 模型目录路径 |
| `auto_load` | bool | False | 是否自动加载模型 |
| `use_gpu` | bool | False | 是否使用 GPU |
| `det_db_thresh` | float | 0.3 | 检测阈值 |
| `det_db_box_thresh` | float | 0.5 | 边界框阈值 |

**返回值**: `List[DetectionResult]`

每个 `DetectionResult` 包含:
- `bbox`: 边界框坐标 [x_min, y_min, x_max, y_max]
- `score`: 置信度分数 (0.0-1.0)
- `category`: 类别标签 ("text")
- `metadata`: 额外信息（如识别的文本内容）

#### 4.1.2 BubbleDetector (气泡检测器)

**功能**: 检测聊天气泡并识别说话者

**特性**:
- 自动识别单列/双列布局
- 跨截图记忆学习
- 说话者身份跟踪
- 支持记忆持久化

**基本用法**:

```python
from screenshot2chat.detectors import BubbleDetector
import cv2

# 创建检测器
detector = BubbleDetector(config={
    "screen_width": 720,
    "memory_path": "chat_memory.json",  # 保存记忆
    "memory_alpha": 0.7,  # 记忆更新系数
    "auto_load": True
})

# 需要先有文本检测结果
text_results = text_detector.detect(image)

# 执行气泡检测
bubble_results = detector.detect(image, text_boxes=text_results)

# 查看结果
for bubble in bubble_results:
    speaker = bubble.metadata.get("speaker")  # "A" 或 "B"
    layout = bubble.metadata.get("layout")    # 布局类型
    print(f"说话者 {speaker}: {bubble.bbox}")

# 查看记忆状态
memory = detector.get_memory_state()
print(f"已处理帧数: {memory['frame_count']}")
```

**配置参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `screen_width` | int | 720 | 屏幕宽度（像素） |
| `min_separation_ratio` | float | 0.18 | 最小列分离比例 |
| `memory_alpha` | float | 0.7 | 记忆更新的滑动平均系数 |
| `memory_path` | str | None | 记忆持久化文件路径 |
| `save_interval` | int | 10 | 自动保存间隔（帧数） |
| `auto_load` | bool | False | 是否自动加载 |

**记忆管理**:

```python
# 重置记忆
detector.reset_memory()

# 手动保存记忆
detector.save_memory()

# 获取记忆状态
memory = detector.get_memory_state()
```

### 4.2 提取器 (Extractors)

提取器从检测结果中提取结构化信息。

#### 4.2.1 LayoutExtractor (布局提取器)

**功能**: 分析聊天界面的布局类型

**支持的布局类型**:
- `single`: 单列布局
- `double`: 标准双列布局
- `double_left`: 左对齐双列
- `double_right`: 右对齐双列

**基本用法**:

```python
from screenshot2chat.extractors import LayoutExtractor

# 创建提取器
extractor = LayoutExtractor(config={
    "screen_width": 720,
    "min_separation_ratio": 0.18
})

# 从文本检测结果提取布局
layout_result = extractor.extract(text_results, image)

# 获取布局信息
layout_type = layout_result.data["layout_type"]
is_double = layout_result.data["is_double_column"]
confidence = layout_result.confidence

print(f"布局类型: {layout_type}")
print(f"置信度: {confidence:.2f}")
```

**返回数据结构**:

```python
{
    "layout_type": "double",        # 布局类型
    "is_single_column": False,      # 是否单列
    "is_double_column": True,       # 是否双列
    "num_columns": 2,               # 列数
    "left_boxes": [0, 2, 4],        # 左列文本框索引
    "right_boxes": [1, 3, 5],       # 右列文本框索引
    "left_stats": {                 # 左列统计信息
        "center": 180.5,
        "center_normalized": 0.25,
        "width": 250.0,
        "count": 3
    },
    "right_stats": {                # 右列统计信息
        "center": 540.5,
        "center_normalized": 0.75,
        "width": 250.0,
        "count": 3
    }
}
```

#### 4.2.2 SpeakerExtractor (说话者提取器)

**功能**: 识别和区分不同的说话者

**基本用法**:

```python
from screenshot2chat.extractors import SpeakerExtractor

# 创建提取器
extractor = SpeakerExtractor(config={
    "screen_width": 720
})

# 从气泡检测结果提取说话者
speaker_result = extractor.extract(bubble_results, image)

# 获取说话者信息
speakers = speaker_result.data["speakers"]
for speaker_id, info in speakers.items():
    print(f"说话者 {speaker_id}:")
    print(f"  消息数: {info['message_count']}")
    print(f"  位置特征: {info['position']}")
```

#### 4.2.3 NicknameExtractor (昵称提取器)

**功能**: 从聊天界面提取用户昵称

**基本用法**:

```python
from screenshot2chat.extractors import NicknameExtractor

# 创建提取器
extractor = NicknameExtractor(config={
    "top_k": 3,                    # 返回前3个候选
    "min_top_margin_ratio": 0.05,  # 最小顶部边距
    "top_region_ratio": 0.2        # 顶部区域比例
})

# 提取昵称
nickname_result = extractor.extract(text_results, image)

# 获取昵称列表
nicknames = nickname_result.data["nicknames"]
for nick in nicknames:
    print(f"昵称: {nick['text']}")
    print(f"置信度: {nick['nickname_score']:.2f}")
```

### 4.3 流水线 (Pipeline)

流水线系统用于组合多个检测器和提取器，构建完整的处理流程。

#### 4.3.1 创建流水线

**方法1: 代码方式创建**

```python
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors import TextDetector, BubbleDetector
from screenshot2chat.extractors import LayoutExtractor

# 创建流水线
pipeline = Pipeline(name="chat_analysis")

# 添加文本检测步骤
text_detector = TextDetector(config={"auto_load": True})
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=text_detector
))

# 添加气泡检测步骤（依赖文本检测）
bubble_detector = BubbleDetector(config={"auto_load": True})
pipeline.add_step(PipelineStep(
    name="bubble_detection",
    step_type=StepType.DETECTOR,
    component=bubble_detector,
    depends_on=["text_detection"]
))

# 添加布局提取步骤
layout_extractor = LayoutExtractor()
pipeline.add_step(PipelineStep(
    name="layout_extraction",
    step_type=StepType.EXTRACTOR,
    component=layout_extractor,
    config={"source": "text_detection"},
    depends_on=["text_detection"]
))

# 验证流水线
pipeline.validate()
```

**方法2: 配置文件方式创建**


```yaml
# pipeline_config.yaml
name: "chat_analysis_pipeline"
enable_monitoring: true
parallel_executor: "thread"
max_workers: 4

steps:
  - name: "text_detection"
    step_type: "detector"
    component: "TextDetector"
    config:
      backend: "PP-OCRv5_server_det"
      auto_load: true
    enabled: true

  - name: "bubble_detection"
    step_type: "detector"
    component: "BubbleDetector"
    config:
      screen_width: 720
      memory_path: "chat_memory.json"
    depends_on: ["text_detection"]
    enabled: true

  - name: "layout_extraction"
    step_type: "extractor"
    component: "LayoutExtractor"
    config:
      source: "text_detection"
      screen_width: 720
    depends_on: ["text_detection"]
    enabled: true
```

```python
# 从配置文件加载
pipeline = Pipeline.from_config("pipeline_config.yaml")
```

#### 4.3.2 执行流水线

```python
import cv2

# 加载图像
image = cv2.imread("chat_screenshot.png")

# 执行流水线
results = pipeline.execute(image)

# 获取各步骤的结果
text_results = results.get("text_detection")
bubble_results = results.get("bubble_detection")
layout_results = results.get("layout_extraction")

# 处理结果
print(f"检测到 {len(text_results)} 个文本框")
print(f"检测到 {len(bubble_results)} 个气泡")
print(f"布局类型: {layout_results.data['layout_type']}")
```

#### 4.3.3 条件执行

流水线支持基于条件的步骤执行：

```python
# 添加条件步骤
pipeline.add_step(PipelineStep(
    name="advanced_analysis",
    step_type=StepType.EXTRACTOR,
    component=advanced_extractor,
    condition="len(result.text_detection) > 10",  # 只在文本框超过10个时执行
    depends_on=["text_detection"]
))
```

**支持的条件表达式**:
- `len(result.step_name) > value`: 检查结果数量
- `result.step_name.field == value`: 检查字段值
- `result.step_name is not None`: 检查结果是否存在

#### 4.3.4 并行执行

流水线支持并行执行独立的步骤：

```python
# 创建并行组
pipeline.add_step(PipelineStep(
    name="nickname_extraction",
    step_type=StepType.EXTRACTOR,
    component=nickname_extractor,
    depends_on=["text_detection"],
    parallel_group="group1"  # 并行组ID
))

pipeline.add_step(PipelineStep(
    name="layout_extraction",
    step_type=StepType.EXTRACTOR,
    component=layout_extractor,
    depends_on=["text_detection"],
    parallel_group="group1"  # 同一并行组
))

# 这两个步骤会并行执行
```

---

## 5. 快速入门

### 5.1 第一个示例：文本检测

```python
from screenshot2chat.detectors import TextDetector
import cv2

# 1. 创建检测器
detector = TextDetector(config={
    "backend": "paddleocr",
    "auto_load": True
})

# 2. 加载图像
image = cv2.imread("chat_screenshot.png")

# 3. 执行检测
results = detector.detect(image)

# 4. 显示结果
print(f"检测到 {len(results)} 个文本框")
for i, result in enumerate(results):
    print(f"文本框 {i+1}:")
    print(f"  位置: {result.bbox}")
    print(f"  置信度: {result.score:.2f}")
```

### 5.2 第二个示例：完整分析流水线

```python
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors import TextDetector, BubbleDetector
from screenshot2chat.extractors import LayoutExtractor, NicknameExtractor
import cv2

# 1. 创建流水线
pipeline = Pipeline(name="complete_analysis")

# 2. 添加步骤
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=TextDetector(config={"auto_load": True})
))

pipeline.add_step(PipelineStep(
    name="bubble_detection",
    step_type=StepType.DETECTOR,
    component=BubbleDetector(config={"auto_load": True}),
    depends_on=["text_detection"]
))

pipeline.add_step(PipelineStep(
    name="layout_extraction",
    step_type=StepType.EXTRACTOR,
    component=LayoutExtractor(),
    config={"source": "text_detection"},
    depends_on=["text_detection"]
))

# 3. 验证流水线
if pipeline.validate():
    print("✓ 流水线配置有效")

# 4. 执行分析
image = cv2.imread("chat_screenshot.png")
results = pipeline.execute(image)

# 5. 处理结果
layout = results["layout_extraction"]
print(f"布局类型: {layout.data['layout_type']}")
print(f"置信度: {layout.confidence:.2f}")

bubbles = results["bubble_detection"]
print(f"检测到 {len(bubbles)} 个聊天气泡")
```

### 5.3 第三个示例：批量处理

```python
from screenshot2chat.pipeline import Pipeline
from pathlib import Path
import cv2
from tqdm import tqdm

# 1. 创建流水线
pipeline = Pipeline.from_config("config/basic_pipeline.yaml")

# 2. 获取所有图像
image_dir = Path("screenshots")
image_files = list(image_dir.glob("*.png"))

# 3. 批量处理
results_list = []
for image_path in tqdm(image_files, desc="处理中"):
    # 加载图像
    image = cv2.imread(str(image_path))
    
    # 执行分析
    results = pipeline.execute(image)
    
    # 保存结果
    results_list.append({
        "file": image_path.name,
        "layout": results["layout_extraction"].data["layout_type"],
        "num_bubbles": len(results["bubble_detection"])
    })

# 4. 输出统计
print(f"处理完成！共处理 {len(results_list)} 张图像")
```

---

## 6. 高级使用

### 6.1 自定义检测器

创建自定义检测器需要继承 `BaseDetector` 类：

```python
from screenshot2chat.core import BaseDetector, DetectionResult
import numpy as np
from typing import List, Dict, Any, Optional

class CustomDetector(BaseDetector):
    """自定义检测器示例"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 初始化自定义参数
        self.threshold = self.get_config("threshold", 0.5)
    
    def load_model(self) -> None:
        """加载模型"""
        # 实现模型加载逻辑
        self.model = YourModel()
        print("模型加载完成")
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """执行检测"""
        # 确保模型已加载
        self.ensure_model_loaded()
        
        # 执行检测逻辑
        raw_results = self.model.predict(image)
        
        # 转换为标准格式
        detection_results = []
        for result in raw_results:
            detection_results.append(
                DetectionResult(
                    bbox=result["bbox"],
                    score=result["score"],
                    category="custom",
                    metadata={"custom_field": result["data"]}
                )
            )
        
        return detection_results
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 实现预处理逻辑
        return image
    
    def postprocess(self, raw_results: Any) -> List[DetectionResult]:
        """后处理结果"""
        # 实现后处理逻辑
        return []

# 使用自定义检测器
detector = CustomDetector(config={"threshold": 0.7})
detector.load_model()
results = detector.detect(image)
```

### 6.2 自定义提取器

