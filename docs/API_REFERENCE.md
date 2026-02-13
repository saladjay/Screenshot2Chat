# API 参考文档

## 概述

本文档提供Screenshot2Chat库的完整API参考。所有公共接口、类、方法和参数都在此详细说明。

## 目录

- [核心抽象类](#核心抽象类)
  - [BaseDetector](#basedetector)
  - [BaseExtractor](#baseextractor)
- [数据模型](#数据模型)
  - [DetectionResult](#detectionresult)
  - [ExtractionResult](#extractionresult)
  - [TextBox](#textbox)
  - [Bubble](#bubble)
  - [Dialog](#dialog)
- [检测器](#检测器)
  - [TextDetector](#textdetector)
  - [BubbleDetector](#bubbledetector)
- [提取器](#提取器)
  - [NicknameExtractor](#nicknameextractor)
  - [SpeakerExtractor](#speakerextractor)
  - [LayoutExtractor](#layoutextractor)
- [流水线](#流水线)
  - [Pipeline](#pipeline)
  - [PipelineStep](#pipelinestep)
- [配置管理](#配置管理)
  - [ConfigManager](#configmanager)
- [模型管理](#模型管理)
  - [ModelManager](#modelmanager)
  - [ModelMetadata](#modelmetadata)
- [性能监控](#性能监控)
  - [PerformanceMonitor](#performancemonitor)
- [日志系统](#日志系统)
  - [StructuredLogger](#structuredlogger)
- [异常](#异常)

---

## 核心抽象类

### BaseDetector

检测器抽象基类，所有检测器都应继承此类。

**模块**: `screenshot2chat.core.base_detector`

#### 类定义

```python
class BaseDetector(ABC):
    """检测器抽象基类"""
    
    def __init__(self, config: Dict[str, Any] = None)
```

#### 参数

- `config` (Dict[str, Any], optional): 检测器配置字典

#### 抽象方法

##### load_model()


加载模型到内存。

```python
@abstractmethod
def load_model(self) -> None
```

**返回**: None

**说明**: 子类必须实现此方法以加载特定的模型。

##### detect()

执行检测操作。

```python
@abstractmethod
def detect(self, image: np.ndarray) -> List[DetectionResult]
```

**参数**:
- `image` (np.ndarray): 输入图像，格式为numpy数组

**返回**: List[DetectionResult] - 检测结果列表

**说明**: 子类必须实现此方法以执行具体的检测逻辑。

#### 模板方法

##### preprocess()

预处理图像。

```python
def preprocess(self, image: np.ndarray) -> np.ndarray
```

**参数**:
- `image` (np.ndarray): 原始图像

**返回**: np.ndarray - 预处理后的图像

**说明**: 子类可以重写此方法以实现自定义预处理。

##### postprocess()

后处理检测结果。

```python
def postprocess(self, raw_results: Any) -> List[DetectionResult]
```

**参数**:
- `raw_results` (Any): 原始检测结果

**返回**: List[DetectionResult] - 标准化的检测结果

**说明**: 子类可以重写此方法以实现自定义后处理。

#### 示例

```python
from screenshot2chat.core.base_detector import BaseDetector
from screenshot2chat.core.data_models import DetectionResult
import numpy as np

class MyDetector(BaseDetector):
    def load_model(self):
        # 加载模型
        self.model = load_my_model()
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        # 预处理
        processed = self.preprocess(image)
        
        # 执行检测
        raw_results = self.model.predict(processed)
        
        # 后处理
        return self.postprocess(raw_results)
```

---

### BaseExtractor

提取器抽象基类，所有提取器都应继承此类。

**模块**: `screenshot2chat.core.base_extractor`

#### 类定义

```python
class BaseExtractor(ABC):
    """提取器抽象基类"""
    
    def __init__(self, config: Dict[str, Any] = None)
```

#### 参数

- `config` (Dict[str, Any], optional): 提取器配置字典

#### 抽象方法

##### extract()

从检测结果中提取信息。

```python
@abstractmethod
def extract(self, detection_results: List[DetectionResult], 
            image: np.ndarray = None) -> ExtractionResult
```

**参数**:
- `detection_results` (List[DetectionResult]): 检测结果列表
- `image` (np.ndarray, optional): 原始图像（某些提取器可能需要）

**返回**: ExtractionResult - 提取结果

**说明**: 子类必须实现此方法以执行具体的提取逻辑。

#### 方法

##### validate()

验证提取结果的有效性。

```python
def validate(self, result: ExtractionResult) -> bool
```

**参数**:
- `result` (ExtractionResult): 提取结果

**返回**: bool - 结果是否有效

**说明**: 子类可以重写此方法以实现自定义验证逻辑。

#### 示例

```python
from screenshot2chat.core.base_extractor import BaseExtractor
from screenshot2chat.core.data_models import ExtractionResult

class MyExtractor(BaseExtractor):
    def extract(self, detection_results, image=None):
        # 提取逻辑
        data = self._process_detections(detection_results)
        
        result = ExtractionResult(
            data=data,
            confidence=0.95
        )
        
        # 验证结果
        if self.validate(result):
            return result
        else:
            raise ValueError("Invalid extraction result")
```

---

## 数据模型

### DetectionResult

检测结果数据类。

**模块**: `screenshot2chat.core.data_models`

#### 类定义

```python
@dataclass
class DetectionResult:
    bbox: List[float]      # [x_min, y_min, x_max, y_max]
    score: float           # 置信度分数 (0-1)
    category: str          # 类别名称
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 属性

- `bbox` (List[float]): 边界框坐标 [x_min, y_min, x_max, y_max]
- `score` (float): 检测置信度，范围0-1
- `category` (str): 检测对象的类别
- `metadata` (Dict[str, Any]): 额外的元数据

#### 示例

```python
from screenshot2chat.core.data_models import DetectionResult

result = DetectionResult(
    bbox=[100, 200, 300, 400],
    score=0.95,
    category="text",
    metadata={"text": "Hello World"}
)
```

---

### ExtractionResult

提取结果数据类。

**模块**: `screenshot2chat.core.data_models`

#### 类定义

```python
@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    confidence: float = 1.0
```

#### 属性

- `data` (Dict[str, Any]): 提取的数据
- `confidence` (float): 提取置信度，范围0-1

#### 方法

##### to_json()

转换为JSON格式。

```python
def to_json(self) -> Dict[str, Any]
```

**返回**: Dict[str, Any] - JSON格式的数据

#### 示例

```python
from screenshot2chat.core.data_models import ExtractionResult

result = ExtractionResult(
    data={"nicknames": ["Alice", "Bob"]},
    confidence=0.9
)

json_data = result.to_json()
# {'data': {'nicknames': ['Alice', 'Bob']}, 'confidence': 0.9}
```

---

### TextBox

文本框数据模型（向后兼容）。

**模块**: `screenshot2chat.core.data_models`


#### 类定义

```python
@dataclass
class TextBox:
    box: np.ndarray           # [x_min, y_min, x_max, y_max]
    score: float
    text: Optional[str] = None
    text_type: Optional[str] = None
    source: Optional[str] = None
    speaker: Optional[str] = None
    layout_det: Optional[Any] = None
    related_layout_boxes: List[Any] = field(default_factory=list)
```

#### 属性

- `box` (np.ndarray): 边界框坐标
- `score` (float): 置信度分数
- `text` (str, optional): 识别的文本内容
- `text_type` (str, optional): 文本类型
- `source` (str, optional): 来源标识
- `speaker` (str, optional): 说话者标识
- `layout_det` (Any, optional): 布局检测信息
- `related_layout_boxes` (List[Any]): 相关的布局框

#### 属性方法

- `x_min`, `y_min`, `x_max`, `y_max`: 边界框坐标
- `center_x`, `center_y`: 中心点坐标
- `width`, `height`: 宽度和高度

---

## 检测器

### TextDetector

文本检测器，支持多种OCR后端。

**模块**: `screenshot2chat.detectors.text_detector`

#### 类定义

```python
class TextDetector(BaseDetector):
    def __init__(self, backend: str = "paddleocr", config: Dict[str, Any] = None)
```

#### 参数

- `backend` (str): OCR后端，支持 "paddleocr", "tesseract", "easyocr"
- `config` (Dict[str, Any], optional): 配置字典

#### 配置选项

```python
{
    "model_dir": "models/PP-OCRv5_server_det/",  # 模型目录
    "det_db_thresh": 0.3,                         # 检测阈值
    "det_db_box_thresh": 0.5,                     # 框阈值
    "use_gpu": True                               # 是否使用GPU
}
```

#### 方法

##### detect()

检测图像中的文本框。

```python
def detect(self, image: np.ndarray) -> List[DetectionResult]
```

**参数**:
- `image` (np.ndarray): 输入图像

**返回**: List[DetectionResult] - 文本框检测结果

#### 示例

```python
from screenshot2chat.detectors.text_detector import TextDetector
import cv2

# 创建检测器
detector = TextDetector(backend="paddleocr")
detector.load_model()

# 加载图像
image = cv2.imread("screenshot.png")

# 执行检测
results = detector.detect(image)

for result in results:
    print(f"Text box at {result.bbox} with score {result.score}")
```

---

### BubbleDetector

聊天气泡检测器。

**模块**: `screenshot2chat.detectors.bubble_detector`


#### 类定义

```python
class BubbleDetector(BaseDetector):
    def __init__(self, config: Dict[str, Any] = None)
```

#### 配置选项

```python
{
    "screen_width": 720,                    # 屏幕宽度
    "memory_path": "chat_memory.json",      # 记忆文件路径
    "min_cluster_size": 3                   # 最小聚类大小
}
```

#### 方法

##### detect()

检测聊天气泡。

```python
def detect(self, image: np.ndarray) -> List[DetectionResult]
```

**参数**:
- `image` (np.ndarray): 输入图像

**返回**: List[DetectionResult] - 气泡检测结果

**说明**: 需要先运行TextDetector获取文本框，然后通过Pipeline传递给BubbleDetector。

#### 示例

```python
from screenshot2chat.detectors.bubble_detector import BubbleDetector

detector = BubbleDetector(config={"screen_width": 1080})
detector.load_model()

# 通常在Pipeline中使用
# results = detector.detect(image)
```

---

## 提取器

### NicknameExtractor

昵称提取器。

**模块**: `screenshot2chat.extractors.nickname_extractor`

#### 类定义

```python
class NicknameExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any] = None)
```

#### 配置选项

```python
{
    "top_k": 3,                      # 返回前K个候选
    "min_top_margin_ratio": 0.05,    # 最小顶部边距比例
    "top_region_ratio": 0.2          # 顶部区域比例
}
```

#### 方法

##### extract()

从文本框中提取昵称候选。

```python
def extract(self, detection_results: List[DetectionResult], 
            image: np.ndarray = None) -> ExtractionResult
```

**参数**:
- `detection_results` (List[DetectionResult]): 文本检测结果
- `image` (np.ndarray, optional): 原始图像

**返回**: ExtractionResult - 包含昵称候选列表

**返回数据格式**:
```python
{
    "nicknames": [
        {
            "text": "Alice",
            "nickname_score": 95.5,
            "bbox": [10, 20, 100, 50]
        },
        ...
    ]
}
```

#### 示例

```python
from screenshot2chat.extractors.nickname_extractor import NicknameExtractor

extractor = NicknameExtractor(config={"top_k": 5})
result = extractor.extract(text_detection_results, image)

for nickname in result.data["nicknames"]:
    print(f"Nickname: {nickname['text']}, Score: {nickname['nickname_score']}")
```

---

### SpeakerExtractor

说话者识别提取器。

**模块**: `screenshot2chat.extractors.speaker_extractor`

#### 类定义

```python
class SpeakerExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any] = None)
```

#### 方法

##### extract()

识别每个气泡的说话者。

```python
def extract(self, detection_results: List[DetectionResult], 
            image: np.ndarray = None) -> ExtractionResult
```

**返回数据格式**:
```python
{
    "speakers": {
        "bubble_0": "left",
        "bubble_1": "right",
        ...
    }
}
```

---

### LayoutExtractor

布局分析提取器。

**模块**: `screenshot2chat.extractors.layout_extractor`


#### 类定义

```python
class LayoutExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any] = None)
```

#### 方法

##### extract()

分析聊天界面布局类型。

```python
def extract(self, detection_results: List[DetectionResult], 
            image: np.ndarray = None) -> ExtractionResult
```

**返回数据格式**:
```python
{
    "layout_type": "double",  # "single", "double", "double_left", "double_right"
    "column_info": {
        "left_column": [...],
        "right_column": [...]
    }
}
```

---

## 流水线

### Pipeline

处理流水线，管理多个处理步骤的执行。

**模块**: `screenshot2chat.pipeline.pipeline`

#### 类定义

```python
class Pipeline:
    def __init__(self, name: str = "default")
```

#### 参数

- `name` (str): 流水线名称

#### 方法

##### add_step()

添加处理步骤。

```python
def add_step(self, step: PipelineStep) -> 'Pipeline'
```

**参数**:
- `step` (PipelineStep): 流水线步骤

**返回**: Pipeline - 返回自身以支持链式调用

##### execute()

执行流水线。

```python
def execute(self, image: np.ndarray, **kwargs) -> Dict[str, Any]
```

**参数**:
- `image` (np.ndarray): 输入图像
- `**kwargs`: 额外的执行参数

**返回**: Dict[str, Any] - 包含所有步骤结果的字典

##### validate()

验证流水线配置。

```python
def validate(self) -> bool
```

**返回**: bool - 配置是否有效

##### from_config()

从配置文件创建流水线。

```python
@classmethod
def from_config(cls, config: Union[str, Dict[str, Any]]) -> 'Pipeline'
```

**参数**:
- `config` (str | Dict): 配置文件路径或配置字典

**返回**: Pipeline - 配置好的流水线实例

#### 示例

```python
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors import TextDetector
import cv2

# 方式1: 手动构建
pipeline = Pipeline(name="my_pipeline")
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=TextDetector(),
    config={}
))

# 方式2: 从配置文件加载
pipeline = Pipeline.from_config("config/pipeline.yaml")

# 执行
image = cv2.imread("screenshot.png")
results = pipeline.execute(image)

print(results)
```

---

### PipelineStep

流水线步骤定义。

**模块**: `screenshot2chat.pipeline.pipeline`

#### 类定义

```python
@dataclass
class PipelineStep:
    name: str
    step_type: StepType
    component: Any
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
```

#### 属性

- `name` (str): 步骤名称
- `step_type` (StepType): 步骤类型（DETECTOR/EXTRACTOR/PROCESSOR）
- `component` (Any): 组件实例
- `config` (Dict): 步骤配置
- `enabled` (bool): 是否启用
- `depends_on` (List[str]): 依赖的步骤名称列表
- `condition` (str, optional): 条件表达式

---

## 配置管理

### ConfigManager

配置管理器，支持分层配置。

**模块**: `screenshot2chat.config.config_manager`


#### 类定义

```python
class ConfigManager:
    def __init__(self)
```

#### 方法

##### load()

加载配置文件。

```python
def load(self, config_path: str, layer: str = 'user') -> None
```

**参数**:
- `config_path` (str): 配置文件路径（支持.yaml和.json）
- `layer` (str): 配置层级，可选 "default", "user", "runtime"

##### get()

获取配置值。

```python
def get(self, key: str, default: Any = None) -> Any
```

**参数**:
- `key` (str): 配置键，支持点号分隔的嵌套键（如 "detector.text.backend"）
- `default` (Any): 默认值

**返回**: Any - 配置值

**优先级**: runtime > user > default

##### set()

设置配置值。

```python
def set(self, key: str, value: Any, layer: str = 'runtime') -> None
```

**参数**:
- `key` (str): 配置键
- `value` (Any): 配置值
- `layer` (str): 配置层级

##### save()

保存配置到文件。

```python
def save(self, config_path: str, layer: str = 'user') -> None
```

**参数**:
- `config_path` (str): 保存路径
- `layer` (str): 要保存的配置层级

##### validate()

验证配置有效性。

```python
def validate(self, schema: Dict[str, Any]) -> bool
```

**参数**:
- `schema` (Dict): 验证模式

**返回**: bool - 是否有效

#### 示例

```python
from screenshot2chat.config import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 加载配置文件
config.load("config/default.yaml", layer="default")
config.load("config/user.yaml", layer="user")

# 获取配置
backend = config.get("detector.text.backend", default="paddleocr")
threshold = config.get("detector.text.threshold", default=0.5)

# 设置运行时配置
config.set("detector.text.use_gpu", True, layer="runtime")

# 保存用户配置
config.save("config/user.yaml", layer="user")
```

---

## 模型管理

### ModelManager

模型管理器，支持模型版本管理和缓存。

**模块**: `screenshot2chat.models.model_manager`

#### 类定义

```python
class ModelManager:
    def __init__(self, model_dir: str = "models")
```

#### 参数

- `model_dir` (str): 模型存储目录

#### 方法

##### register()

注册模型。

```python
def register(self, metadata: ModelMetadata, model_path: str) -> None
```

**参数**:
- `metadata` (ModelMetadata): 模型元数据
- `model_path` (str): 模型文件路径

##### load()

加载模型。

```python
def load(self, name: str, version: str = "latest") -> Any
```

**参数**:
- `name` (str): 模型名称
- `version` (str): 模型版本，默认"latest"

**返回**: Any - 加载的模型对象

##### list_versions()

列出模型的所有版本。

```python
def list_versions(self, name: str) -> List[str]
```

**参数**:
- `name` (str): 模型名称

**返回**: List[str] - 版本列表

##### get_metadata()

获取模型元数据。

```python
def get_metadata(self, name: str, version: str = "latest") -> ModelMetadata
```

**返回**: ModelMetadata - 模型元数据

#### 示例

```python
from screenshot2chat.models import ModelManager, ModelMetadata

# 创建模型管理器
manager = ModelManager(model_dir="models")

# 注册模型
metadata = ModelMetadata(
    name="text_detector",
    version="1.0.0",
    model_type="detection",
    framework="paddleocr"
)
manager.register(metadata, "models/text_detector_v1.pth")

# 加载模型
model = manager.load("text_detector", version="1.0.0")

# 列出版本
versions = manager.list_versions("text_detector")
print(versions)  # ['1.0.0', '1.1.0', ...]
```

---

### ModelMetadata

模型元数据。

**模块**: `screenshot2chat.models.model_manager`


#### 类定义

```python
@dataclass
class ModelMetadata:
    name: str
    version: str
    model_type: str
    framework: str
    created_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
```

#### 属性

- `name` (str): 模型名称
- `version` (str): 版本号
- `model_type` (str): 模型类型（如 "detection", "recognition"）
- `framework` (str): 框架名称（如 "paddleocr", "pytorch"）
- `created_at` (datetime): 创建时间
- `metrics` (Dict[str, float]): 性能指标
- `tags` (List[str]): 标签列表
- `description` (str): 描述信息

---

## 性能监控

### PerformanceMonitor

性能监控器，追踪执行时间和资源使用。

**模块**: `screenshot2chat.monitoring.performance_monitor`

#### 类定义

```python
class PerformanceMonitor:
    def __init__(self)
```

#### 方法

##### start_timer()

开始计时。

```python
def start_timer(self, name: str) -> None
```

**参数**:
- `name` (str): 计时器名称

##### stop_timer()

停止计时并记录。

```python
def stop_timer(self, name: str) -> float
```

**参数**:
- `name` (str): 计时器名称

**返回**: float - 经过的时间（秒）

##### get_stats()

获取统计信息。

```python
def get_stats(self, name: str) -> Dict[str, float]
```

**参数**:
- `name` (str): 计时器名称

**返回**: Dict[str, float] - 统计信息（mean, std, min, max, count）

##### generate_report()

生成性能报告。

```python
def generate_report(self) -> str
```

**返回**: str - 格式化的性能报告

##### record_memory()

记录内存使用。

```python
def record_memory(self, name: str) -> None
```

**参数**:
- `name` (str): 记录点名称

#### 示例

```python
from screenshot2chat.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

# 开始计时
monitor.start_timer("text_detection")

# 执行操作
results = detector.detect(image)

# 停止计时
elapsed = monitor.stop_timer("text_detection")
print(f"Detection took {elapsed:.3f}s")

# 获取统计
stats = monitor.get_stats("text_detection")
print(f"Average: {stats['mean']:.3f}s")

# 生成报告
report = monitor.generate_report()
print(report)
```

---

## 日志系统

### StructuredLogger

结构化日志记录器。

**模块**: `screenshot2chat.logging.structured_logger`

#### 类定义

```python
class StructuredLogger:
    def __init__(self, name: str)
```

#### 参数

- `name` (str): 日志记录器名称

#### 方法

##### set_context()

设置日志上下文。

```python
def set_context(self, **kwargs) -> None
```

**参数**:
- `**kwargs`: 上下文键值对

##### info()

记录INFO级别日志。

```python
def info(self, message: str, **kwargs) -> None
```

**参数**:
- `message` (str): 日志消息
- `**kwargs`: 额外的上下文信息

##### warning()

记录WARNING级别日志。

```python
def warning(self, message: str, **kwargs) -> None
```

##### error()

记录ERROR级别日志。

```python
def error(self, message: str, exc_info: bool = True, **kwargs) -> None
```

**参数**:
- `message` (str): 错误消息
- `exc_info` (bool): 是否包含异常信息
- `**kwargs`: 额外的上下文信息

##### debug()

记录DEBUG级别日志。

```python
def debug(self, message: str, **kwargs) -> None
```

#### 示例

```python
from screenshot2chat.logging import StructuredLogger

logger = StructuredLogger("my_module")

# 设置上下文
logger.set_context(user_id="12345", session_id="abc")

# 记录日志
logger.info("Processing image", image_size=(1080, 1920))
logger.warning("Low confidence detection", confidence=0.3)

try:
    # 某些操作
    pass
except Exception as e:
    logger.error("Detection failed", exc_info=True)
```

---

## 异常

### 异常层次结构

**模块**: `screenshot2chat.core.exceptions`

```python
ScreenshotAnalysisError          # 基础异常
├── ConfigurationError           # 配置错误
├── ModelError                   # 模型错误
│   ├── ModelLoadError          # 模型加载失败
│   └── ModelNotFoundError      # 模型未找到
├── DetectionError              # 检测错误
├── ExtractionError             # 提取错误
├── PipelineError               # 流水线错误
├── ValidationError             # 验证错误
└── DataError                   # 数据错误
```

#### 使用示例

```python
from screenshot2chat.core.exceptions import (
    ModelLoadError,
    DetectionError,
    ConfigurationError
)

try:
    detector.load_model()
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    
try:
    results = detector.detect(image)
except DetectionError as e:
    logger.error(f"Detection failed: {e}")
```

---

## 完整示例

### 基本使用

```python
from screenshot2chat.pipeline import Pipeline
import cv2

# 从配置文件创建流水线
pipeline = Pipeline.from_config("config/chat_analysis.yaml")

# 加载图像
image = cv2.imread("screenshot.png")

# 执行流水线
results = pipeline.execute(image)

# 访问结果
text_boxes = results["text_detection"]
bubbles = results["bubble_detection"]
nicknames = results["nickname_extraction"]["data"]["nicknames"]

print(f"Found {len(text_boxes)} text boxes")
print(f"Found {len(bubbles)} bubbles")
print(f"Top nickname: {nicknames[0]['text']}")
```

### 自定义流水线

```python
from screenshot2chat.pipeline import Pipeline, PipelineStep, StepType
from screenshot2chat.detectors import TextDetector, BubbleDetector
from screenshot2chat.extractors import NicknameExtractor
from screenshot2chat.monitoring import PerformanceMonitor

# 创建组件
text_detector = TextDetector(backend="paddleocr")
bubble_detector = BubbleDetector()
nickname_extractor = NicknameExtractor(config={"top_k": 5})

# 构建流水线
pipeline = Pipeline(name="custom_pipeline")
pipeline.add_step(PipelineStep(
    name="text_detection",
    step_type=StepType.DETECTOR,
    component=text_detector
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

# 添加性能监控
monitor = PerformanceMonitor()

# 执行
image = cv2.imread("screenshot.png")
results = pipeline.execute(image)

# 查看性能报告
print(monitor.generate_report())
```

---

## 版本信息

- **当前版本**: 1.0.0
- **最后更新**: 2024
- **兼容性**: Python 3.8+

## 相关文档

- [架构文档](ARCHITECTURE.md)
- [用户指南](USER_GUIDE.md)
- [迁移指南](MIGRATION_GUIDE.md)
- [配置管理](CONFIG_MANAGER.md)
- [性能监控](PERFORMANCE_MONITORING.md)

---

**注意**: 本文档描述的是重构后的新API。如果您正在使用旧版API，请参考[迁移指南](MIGRATION_GUIDE.md)了解如何升级。
