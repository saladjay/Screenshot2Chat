# 聊天截图分析库架构文档

## 概述

本文档描述了聊天截图分析库（Screenshot2Chat）的系统架构设计。该库是一个通用的、可扩展的聊天截图分析框架，支持从数据标注到模型部署的完整工作流。

### 设计目标

1. **模块化架构**: 将功能划分为独立、可替换的模块
2. **技术路径灵活性**: 支持OCR、深度学习、传统CV、云端API等多种技术
3. **完整工作流**: 覆盖数据标注、模型训练、模型部署全流程
4. **向后兼容**: 保持现有API的兼容性
5. **可扩展性**: 通过插件机制支持第三方扩展

### 核心特性

- 🔧 **灵活的流水线系统**: 通过配置文件定义处理流程
- 🎯 **统一的抽象接口**: BaseDetector和BaseExtractor提供一致的编程模型
- 📊 **性能监控**: 内置性能追踪和分析工具
- 🔄 **向后兼容**: 完整支持旧版API
- 🚀 **多后端支持**: 支持PaddleOCR、云端API等多种技术栈

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                          用户层 (User Layer)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Public API  │  │  CLI Tools   │  │  Config Files (YAML) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        核心层 (Core Layer)                        │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │    Pipeline      │  │    Config    │  │     Model       │   │
│  │   Orchestrator   │  │    Manager   │  │    Manager      │   │
│  └──────────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       处理层 (Processing Layer)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Detectors   │  │  Extractors  │  │    Processors        │  │
│  │  (检测器)     │  │  (提取器)     │  │    (处理器)           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 (Data Layer)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Data Models  │  │ Data Manager │  │  Annotation Tool     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        模型层 (Model Layer)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │Local Models  │  │  Cloud APIs  │  │  Training Pipeline   │  │
│  │(PaddleOCR等) │  │(GPT-4V等)    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   基础设施层 (Infrastructure Layer)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Logging    │  │  Performance │  │  Storage Backend     │  │
│  │   System     │  │   Monitor    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. 核心层 (Core Layer)

#### Pipeline Orchestrator (流水线编排器)

**职责**: 管理处理步骤的执行顺序和数据流

**核心功能**:
- 顺序执行: 按照定义的顺序执行处理步骤
- 并行执行: 支持独立步骤的并行处理
- 条件分支: 根据中间结果选择执行路径
- 依赖管理: 自动解析和验证步骤间的依赖关系

**关键接口**:
```python
class Pipeline:
    def add_step(self, step: PipelineStep) -> 'Pipeline'
    def execute(self, image: np.ndarray) -> Dict[str, Any]
    def validate(self) -> bool
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Pipeline'
```

**配置示例**:
```yaml
name: "chat_analysis_pipeline"
steps:
  - name: "text_detection"
    type: "detector"
    class: "TextDetector"
    config:
      backend: "paddleocr"
  - name: "bubble_detection"
    type: "detector"
    class: "BubbleDetector"
    depends_on: ["text_detection"]
```

#### Config Manager (配置管理器)

**职责**: 统一管理所有配置参数

**核心功能**:
- 分层配置: 支持default/user/runtime三层配置
- 配置继承: 高优先级配置覆盖低优先级配置
- 配置验证: 检查参数类型和有效性
- 配置持久化: 支持YAML和JSON格式

**配置优先级**:
```
runtime (最高) > user > default (最低)
```

**关键接口**:
```python
class ConfigManager:
    def load(self, config_path: str, layer: str = 'user') -> None
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any, layer: str = 'runtime') -> None
    def save(self, config_path: str, layer: str = 'user') -> None
```

#### Model Manager (模型管理器)

**职责**: 管理模型的加载、缓存、版本控制

**核心功能**:
- 模型注册: 记录模型元信息和路径
- 版本管理: 支持多版本模型共存
- 模型缓存: 避免重复加载
- 性能追踪: 记录模型推理性能

**关键接口**:
```python
class ModelManager:
    def register(self, metadata: ModelMetadata, model_path: str) -> None
    def load(self, name: str, version: str = "latest") -> Any
    def list_versions(self, name: str) -> List[str]
```

### 2. 处理层 (Processing Layer)

#### Detectors (检测器)

**职责**: 检测图像中的特定元素

**抽象基类**:
```python
class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> None
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray
    def postprocess(self, raw_results: Any) -> List[DetectionResult]
```

**实现类**:
- **TextDetector**: 文本框检测（支持PaddleOCR、Tesseract等）
- **BubbleDetector**: 聊天气泡检测（基于ChatLayoutDetector）
- **AvatarDetector**: 头像检测（计划中）
- **EmojiDetector**: 表情检测（计划中）

**统一输出格式**:
```python
@dataclass
class DetectionResult:
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    score: float
    category: str
    metadata: Dict[str, Any]
```

#### Extractors (提取器)

**职责**: 从检测结果中提取结构化信息

**抽象基类**:
```python
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult
    
    def validate(self, result: ExtractionResult) -> bool
```

**实现类**:
- **NicknameExtractor**: 昵称提取（基于综合评分系统）
- **SpeakerExtractor**: 说话者识别
- **LayoutExtractor**: 布局分析（单列/双列）
- **DialogExtractor**: 对话结构提取（计划中）

**统一输出格式**:
```python
@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    confidence: float
    
    def to_json(self) -> Dict[str, Any]
```

### 3. 数据层 (Data Layer)

#### Data Models (数据模型)

**核心数据结构**:

```python
# 文本框（向后兼容）
@dataclass
class TextBox:
    box: np.ndarray
    score: float
    text: Optional[str]
    speaker: Optional[str]

# 头像
@dataclass
class Avatar:
    bbox: List[float]
    score: float
    speaker_id: Optional[str]

# 聊天气泡
@dataclass
class Bubble:
    bbox: List[float]
    text_boxes: List[TextBox]
    avatar: Optional[Avatar]
    speaker: Optional[str]

# 对话
@dataclass
class Dialog:
    bubbles: List[Bubble]
    layout_type: str
    speakers: Dict[str, Any]
    
    def to_json(self) -> Dict[str, Any]
```

### 4. 基础设施层 (Infrastructure Layer)

#### Logging System (日志系统)

**结构化日志**:
```python
class StructuredLogger:
    def set_context(self, **kwargs) -> None
    def info(self, message: str, **kwargs) -> None
    def error(self, message: str, exc_info: bool = True, **kwargs) -> None
    def warning(self, message: str, **kwargs) -> None
```

**日志级别**:
- DEBUG: 详细的调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息

#### Performance Monitor (性能监控)

**功能**:
- 执行时间追踪
- 内存使用监控
- 性能统计分析
- 报告生成

**使用示例**:
```python
monitor = PerformanceMonitor()
monitor.start_timer("text_detection")
# ... 执行检测 ...
elapsed = monitor.stop_timer("text_detection")
report = monitor.generate_report()
```

## 数据流

### 典型处理流程

```
1. 输入图像
   ↓
2. TextDetector 检测文本框
   ↓
3. BubbleDetector 检测聊天气泡
   ↓
4. NicknameExtractor 提取昵称
   ↓
5. SpeakerExtractor 识别说话者
   ↓
6. LayoutExtractor 分析布局
   ↓
7. 输出结构化结果 (JSON)
```

### 数据转换

```
原始图像 (np.ndarray)
   ↓
DetectionResult (检测器输出)
   ↓
ExtractionResult (提取器输出)
   ↓
Dialog (最终结构化数据)
   ↓
JSON (序列化输出)
```

## 扩展机制

### 添加新检测器

1. 继承BaseDetector
2. 实现load_model()和detect()方法
3. 在配置文件中注册

```python
class MyDetector(BaseDetector):
    def load_model(self) -> None:
        # 加载模型
        pass
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        # 执行检测
        pass
```

### 添加新提取器

1. 继承BaseExtractor
2. 实现extract()方法
3. 在配置文件中注册

```python
class MyExtractor(BaseExtractor):
    def extract(self, detection_results: List[DetectionResult], 
                image: np.ndarray = None) -> ExtractionResult:
        # 执行提取
        pass
```

## 部署架构

### 服务器部署

```
┌─────────────────────────────────────────┐
│         Load Balancer (Nginx)           │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼────────┐
│  API Server 1  │    │  API Server 2   │
│  (FastAPI)     │    │  (FastAPI)      │
└───────┬────────┘    └────────┬────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Model Manager       │
        │   (Shared Cache)      │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Storage Backend     │
        │   (S3/MinIO)          │
        └───────────────────────┘
```

### 边缘设备部署

```
┌─────────────────────────────────────────┐
│         Mobile Application              │
│  ┌─────────────────────────────────┐   │
│  │   Screenshot Analysis SDK       │   │
│  │  ┌──────────────────────────┐   │   │
│  │  │  Quantized Models        │   │   │
│  │  │  (TFLite/NCNN/MNN)       │   │   │
│  │  └──────────────────────────┘   │   │
│  │  ┌──────────────────────────┐   │   │
│  │  │  Local Processing        │   │   │
│  │  └──────────────────────────┘   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## 性能目标

### 延迟目标

- 文本检测: <200ms (单张图片, GPU)
- 气泡检测: <100ms (基于已有文本框)
- 昵称提取: <50ms
- 完整流水线: <500ms (GPU), <2s (CPU)

### 吞吐量目标

- 批量处理: >100 images/minute (GPU)
- 并行处理: 线性扩展到8核
- API响应: >1000 requests/minute

### 资源目标

- 内存占用: <2GB (单个模型)
- 模型大小: <500MB (服务器), <50MB (移动端)
- 磁盘空间: <10GB (包含所有模型)

## 安全考虑

### 数据隐私

1. **本地处理优先**: 默认在本地处理图像，不上传到云端
2. **数据脱敏**: 在使用云端API前自动检测和过滤敏感信息
3. **加密存储**: 支持对标注数据和模型进行加密存储
4. **访问控制**: 记录所有数据访问日志

### API安全

1. **认证**: 支持API Key和OAuth 2.0
2. **速率限制**: 防止API滥用
3. **输入验证**: 严格验证所有输入参数
4. **HTTPS**: 强制使用HTTPS传输

## 向后兼容性

### 兼容层设计

系统提供完整的向后兼容层，确保旧代码无需修改即可运行：

```python
# 旧API（仍然可用）
from screenshotanalysis import ChatLayoutDetector
detector = ChatLayoutDetector()

# 新API（推荐使用）
from screenshot2chat.pipeline import Pipeline
pipeline = Pipeline.from_config("config.yaml")
```

### 弃用策略

- 旧API会发出弃用警告
- 至少保持2个主要版本的兼容性
- 提供详细的迁移指南

## 未来规划

### 短期 (3-6个月)

- 更多检测器: 表情检测、头像检测、时间戳检测
- 更多提取器: 对话结构提取、情感分析
- Web UI: 基于浏览器的标注工具
- 更多云端API: Google Gemini, Azure Vision

### 中期 (6-12个月)

- 自动标注: 使用大模型辅助标注
- 主动学习: 智能选择需要标注的样本
- 联邦学习: 支持分布式训练
- 实时处理: 支持视频流分析

### 长期 (12+个月)

- 多模态分析: 结合文本、图像、音频
- 跨平台统一: 支持更多聊天应用
- 智能推荐: 基于历史数据推荐最佳配置
- 自动优化: 自动调优模型和参数

## 参考资料

- [配置管理文档](CONFIG_MANAGER.md)
- [性能监控文档](PERFORMANCE_MONITORING.md)
- [条件并行流水线文档](CONDITIONAL_PARALLEL_PIPELINE.md)
- [迁移指南](MIGRATION_GUIDE.md)
- [API参考文档](API_REFERENCE.md)
- [用户指南](USER_GUIDE.md)

## 总结

Screenshot2Chat采用清晰的分层架构，通过抽象接口和配置驱动的设计，实现了高度的模块化和可扩展性。系统支持从数据标注到模型部署的完整工作流，能够满足研究、开发和生产环境的各种需求。

通过向后兼容层和渐进式迁移策略，现有用户可以平滑地过渡到新架构，同时享受新功能带来的便利。
