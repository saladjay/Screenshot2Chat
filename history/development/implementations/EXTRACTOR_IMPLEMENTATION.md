# 提取器实现文档

## 概述

本文档描述了三个新实现的提取器，它们遵循统一的 `BaseExtractor` 接口，用于从检测结果中提取结构化信息。

## 已实现的提取器

### 1. NicknameExtractor（昵称提取器）

**位置**: `src/screenshot2chat/extractors/nickname_extractor.py`

**功能**: 从文本检测结果中提取昵称候选，使用综合评分系统。

**配置参数**:
- `top_k`: 返回前K个候选，默认3
- `min_top_margin_ratio`: 最小顶部边距比例，默认0.05
- `top_region_ratio`: 顶部区域比例，默认0.2
- `processor`: ChatMessageProcessor实例（必需）
- `text_rec`: 可选的OCR模型实例
- `ocr_reader`: 可选的OCR读取函数

**评分系统**:
- 位置评分：优先选择靠近屏幕顶部和中心的文本框
- 尺寸评分：考虑文本框的宽度和高度
- 文本特征评分：基于文本内容的特征
- Y排名评分：基于垂直位置的排名（第1名20分，第2名15分，第3名10分）

**使用示例**:
```python
from screenshotanalysis.processors import ChatMessageProcessor
from src.screenshot2chat.extractors import NicknameExtractor

# 初始化
processor = ChatMessageProcessor()
extractor = NicknameExtractor(config={
    'processor': processor,
    'top_k': 3
})

# 提取昵称
result = extractor.extract(detection_results, image=letterboxed_image)

# 获取结果
nicknames = result.data['nicknames']
top_nickname = extractor.get_top_nickname(result)
nickname_text = extractor.get_nickname_text(result)
```

**输出格式**:
```json
{
  "data": {
    "nicknames": [
      {
        "text": "Alice",
        "ocr_score": 0.95,
        "nickname_score": 85.5,
        "score_breakdown": {
          "position": 30.0,
          "size": 25.5,
          "text": 10.0,
          "y_rank": 20.0
        },
        "box": [50, 20, 150, 45],
        "center_x": 100.0,
        "y_min": 20.0,
        "y_rank": 1
      }
    ]
  },
  "confidence": 0.855,
  "metadata": {
    "num_candidates": 3,
    "top_k": 3
  }
}
```

### 2. SpeakerExtractor（说话者识别提取器）

**位置**: `src/screenshot2chat/extractors/speaker_extractor.py`

**功能**: 从气泡检测结果中识别说话者身份（Speaker A和Speaker B）。

**配置参数**:
- `screen_width`: 屏幕宽度（像素），默认720
- `min_separation_ratio`: 最小列分离比例，默认0.18
- `memory_alpha`: 记忆更新的滑动平均系数，默认0.7
- `memory_path`: 记忆数据持久化路径，可选
- `save_interval`: 自动保存间隔（帧数），默认10
- `layout_detector`: 可选的ChatLayoutDetector实例

**工作原理**:
1. 使用ChatLayoutDetector进行列分割
2. 基于几何学习方法识别说话者
3. 使用跨截图记忆保持说话者身份的一致性
4. 计算基于时序规律的置信度

**使用示例**:
```python
from src.screenshot2chat.extractors import SpeakerExtractor

# 初始化
extractor = SpeakerExtractor(config={
    'screen_width': 720,
    'memory_path': 'chat_memory.json'
})

# 提取说话者
result = extractor.extract(detection_results)

# 获取结果
layout = extractor.get_layout_type(result)
speaker_A_boxes = extractor.get_speaker_boxes(result, 'A')
speaker_B_boxes = extractor.get_speaker_boxes(result, 'B')

# 获取特定文本框的说话者
speaker = extractor.get_speaker_for_box(result, box_index)
```

**输出格式**:
```json
{
  "data": {
    "layout": "double",
    "speakers": {
      "0": "A",
      "1": "B",
      "2": "A"
    },
    "speaker_A": [0, 2],
    "speaker_B": [1],
    "num_A": 2,
    "num_B": 1
  },
  "confidence": 0.67,
  "metadata": {
    "frame_count": 1,
    "left_center": 0.18,
    "right_center": 0.83,
    "separation": 0.65,
    "confidence": 0.67
  }
}
```

### 3. LayoutExtractor（布局类型提取器）

**位置**: `src/screenshot2chat/extractors/layout_extractor.py`

**功能**: 检测聊天界面的布局类型（单列/双列/左对齐/右对齐）。

**配置参数**:
- `screen_width`: 屏幕宽度（像素），默认720
- `min_separation_ratio`: 最小列分离比例，默认0.18
- `layout_detector`: 可选的ChatLayoutDetector实例

**支持的布局类型**:
- `single`: 单列布局（所有消息在同一列）
- `double`: 标准双列布局（左右两列，分别代表不同说话者）
- `double_left`: 左对齐双列布局（两列都在屏幕左侧）
- `double_right`: 右对齐双列布局（两列都在屏幕右侧）

**使用示例**:
```python
from src.screenshot2chat.extractors import LayoutExtractor

# 初始化
extractor = LayoutExtractor(config={
    'screen_width': 720
})

# 提取布局
result = extractor.extract(detection_results)

# 获取结果
layout_type = extractor.get_layout_type(result)
is_single = extractor.is_single_column(result)
is_double = extractor.is_double_column(result)

# 获取列信息
left_boxes = extractor.get_column_boxes(result, 'left')
right_boxes = extractor.get_column_boxes(result, 'right')
left_stats = extractor.get_column_stats(result, 'left')
```

**输出格式**:
```json
{
  "data": {
    "layout_type": "double",
    "is_single_column": false,
    "is_double_column": true,
    "num_columns": 2,
    "left_boxes": [0, 2, 3],
    "right_boxes": [1, 4, 5],
    "left_stats": {
      "center": 126.25,
      "center_normalized": 0.18,
      "width": 152.5,
      "width_normalized": 0.21,
      "count": 3
    },
    "right_stats": {
      "center": 600.0,
      "center_normalized": 0.83,
      "width": 140.0,
      "width_normalized": 0.19,
      "count": 3
    }
  },
  "confidence": 0.66,
  "metadata": {
    "num_text_boxes": 6,
    "num_left_boxes": 3,
    "num_right_boxes": 3
  }
}
```

## 统一接口

所有提取器都继承自 `BaseExtractor` 并实现以下接口：

### 核心方法

```python
def extract(
    self, 
    detection_results: List[DetectionResult], 
    image: Optional[np.ndarray] = None
) -> ExtractionResult:
    """从检测结果中提取信息"""
    pass
```

### 辅助方法

```python
def validate(self, result: ExtractionResult) -> bool:
    """验证提取结果的有效性"""
    pass

def to_json(self, result: ExtractionResult) -> Dict[str, Any]:
    """将提取结果转换为JSON格式"""
    pass

def get_config(self, key: str, default: Any = None) -> Any:
    """获取配置值"""
    pass

def set_config(self, key: str, value: Any) -> None:
    """设置配置值"""
    pass
```

## 数据流

```
DetectionResult (from detector)
    ↓
BaseExtractor.extract()
    ↓
ExtractionResult
    ↓
to_json() → JSON output
```

## 组合使用示例

```python
from src.screenshot2chat.extractors import (
    LayoutExtractor,
    SpeakerExtractor,
    NicknameExtractor
)

# 1. 布局分析
layout_extractor = LayoutExtractor(config={'screen_width': 720})
layout_result = layout_extractor.extract(detection_results)

# 2. 说话者识别
speaker_extractor = SpeakerExtractor(config={'screen_width': 720})
speaker_result = speaker_extractor.extract(detection_results)

# 3. 昵称提取（需要processor和image）
nickname_extractor = NicknameExtractor(config={
    'processor': processor,
    'text_rec': text_rec
})
nickname_result = nickname_extractor.extract(detection_results, image)

# 4. 组合结果
dialog = {
    'layout': layout_result.data['layout_type'],
    'speakers': speaker_result.data['speakers'],
    'nicknames': nickname_result.data['nicknames'],
    'messages': []
}
```

## 测试

### 基本测试

运行基本功能测试：
```bash
python test_extractors_basic.py
```

测试覆盖：
- 初始化测试
- 空输入处理
- 单列布局处理
- 双列布局处理
- 验证功能
- 辅助方法

### 使用示例

运行完整的使用示例：
```bash
python examples/extractor_usage_example.py
```

示例包括：
1. 布局类型提取
2. 说话者识别
3. 昵称提取配置
4. 组合使用多个提取器
5. JSON导出

## 性能特点

### NicknameExtractor
- 依赖OCR识别，性能取决于OCR模型
- 建议复用OCR模型实例以提高性能
- 处理时间：~50-200ms（取决于候选框数量）

### SpeakerExtractor
- 轻量级几何计算
- 支持跨截图记忆学习
- 处理时间：<100ms

### LayoutExtractor
- 使用KMeans聚类或median fallback
- 首次调用可能有JIT编译延迟（已优化）
- 处理时间：<100ms

## 向后兼容性

所有提取器都包装了现有的实现：
- `NicknameExtractor` 包装 `extract_nicknames_from_text_boxes()`
- `SpeakerExtractor` 包装 `ChatLayoutDetector.process_frame()`
- `LayoutExtractor` 包装 `ChatLayoutDetector.split_columns()`

这确保了与现有代码的兼容性，同时提供了统一的新接口。

## 下一步

1. 实现 `DialogExtractor` 用于对话结构提取
2. 添加更多单元测试和属性测试
3. 优化性能（批量处理、缓存等）
4. 添加更多配置选项和自定义能力
5. 完善文档和示例

## 参考

- 设计文档: `.kiro/specs/screenshot-analysis-library-refactor/design.md`
- 需求文档: `.kiro/specs/screenshot-analysis-library-refactor/requirements.md`
- 任务列表: `.kiro/specs/screenshot-analysis-library-refactor/tasks.md`
- BaseExtractor: `src/screenshot2chat/core/base_extractor.py`
- 数据模型: `src/screenshot2chat/core/data_models.py`
