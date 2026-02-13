# ConfigManager 使用文档

## 概述

`ConfigManager` 是一个强大的配置管理系统，支持分层配置、多种文件格式、配置验证等功能。它是聊天截图分析库重构的核心组件之一。

## 特性

- **分层配置**: 支持 default、user、runtime 三层配置，自动按优先级合并
- **多格式支持**: 支持 YAML 和 JSON 配置文件
- **嵌套键访问**: 使用点号分隔符访问嵌套配置（如 `model.text_detector.backend`）
- **配置验证**: 支持类型检查、范围验证、必需字段检查
- **配置历史**: 自动保存配置变更历史
- **灵活的API**: 简洁易用的 get/set 接口

## 快速开始

### 基本使用

```python
from src.screenshot2chat.config import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 设置配置
config.set('app.name', 'Screenshot Analysis')
config.set('app.version', '2.0.0')
config.set('app.debug', False)

# 获取配置
app_name = config.get('app.name')  # 'Screenshot Analysis'
debug = config.get('app.debug')    # False

# 获取不存在的键，返回默认值
timeout = config.get('app.timeout', 30)  # 30
```

### 分层配置

ConfigManager 支持三层配置结构，优先级从高到低：

1. **runtime**: 运行时配置（最高优先级）
2. **user**: 用户配置
3. **default**: 默认配置（最低优先级）

```python
config = ConfigManager()

# 在不同层级设置配置
config.set('model.backend', 'paddleocr', layer='default')
config.set('model.backend', 'tesseract', layer='user')
config.set('model.backend', 'easyocr', layer='runtime')

# 获取配置（runtime 层优先）
backend = config.get('model.backend')  # 'easyocr'

# 清除 runtime 层
config.clear('runtime')
backend = config.get('model.backend')  # 'tesseract'
```

### 文件操作

#### 保存配置

```python
config = ConfigManager()

# 设置一些配置
config.set('pipeline.name', 'chat_analysis')
config.set('detector.text.backend', 'paddleocr')
config.set('detector.bubble.screen_width', 720)

# 保存为 JSON
config.save('config.json', layer='runtime')

# 保存为 YAML
config.save('config.yaml', layer='runtime')
```

#### 加载配置

```python
config = ConfigManager()

# 从 JSON 加载
config.load('config.json', layer='user')

# 从 YAML 加载
config.load('config.yaml', layer='user')

# 访问加载的配置
pipeline_name = config.get('pipeline.name')
```

### 配置验证

ConfigManager 支持使用 schema 验证配置的有效性：

```python
config = ConfigManager()

# 设置配置
config.set('server.port', 8080)
config.set('server.host', 'localhost')
config.set('server.workers', 4)

# 定义验证 schema
schema = {
    'server.port': {
        'type': int,
        'required': True,
        'min': 1,
        'max': 65535
    },
    'server.host': {
        'type': str,
        'required': True
    },
    'server.workers': {
        'type': int,
        'required': True,
        'min': 1,
        'max': 16
    }
}

# 验证配置
if config.validate(schema):
    print("配置有效")
else:
    print("配置无效")
```

## API 参考

### ConfigManager 类

#### `__init__()`

创建一个新的 ConfigManager 实例。

```python
config = ConfigManager()
```

#### `get(key: str, default: Any = None) -> Any`

获取配置值。

**参数:**
- `key`: 配置键，支持点号分隔的嵌套键
- `default`: 默认值，当键不存在时返回

**返回:** 配置值或默认值

**示例:**
```python
value = config.get('model.backend')
timeout = config.get('timeout', 30)
```

#### `set(key: str, value: Any, layer: str = 'runtime') -> None`

设置配置值。

**参数:**
- `key`: 配置键，支持点号分隔的嵌套键
- `value`: 配置值
- `layer`: 配置层级，可选值: 'default', 'user', 'runtime'

**示例:**
```python
config.set('model.backend', 'paddleocr')
config.set('debug', True, layer='user')
```

#### `load(config_path: str, layer: str = 'user') -> None`

从文件加载配置。

**参数:**
- `config_path`: 配置文件路径（支持 .json, .yaml, .yml）
- `layer`: 要加载到的配置层级

**异常:**
- `ValueError`: 如果文件格式不支持或层级无效
- `FileNotFoundError`: 如果文件不存在

**示例:**
```python
config.load('config.yaml', layer='user')
config.load('override.json', layer='runtime')
```

#### `save(config_path: str, layer: str = 'user') -> None`

保存配置到文件。

**参数:**
- `config_path`: 配置文件路径（支持 .json, .yaml, .yml）
- `layer`: 要保存的配置层级

**异常:**
- `ValueError`: 如果文件格式不支持或层级无效

**示例:**
```python
config.save('config.yaml', layer='user')
config.save('runtime.json', layer='runtime')
```

#### `validate(schema: Optional[Dict[str, Any]] = None) -> bool`

验证配置的有效性。

**参数:**
- `schema`: 可选的验证模式

**返回:** True 如果配置有效，否则 False

**Schema 格式:**
```python
schema = {
    'key.path': {
        'type': int/str/float/bool/list/dict,
        'required': bool,
        'min': number,      # 仅用于数值类型
        'max': number,      # 仅用于数值类型
        'choices': list     # 枚举值
    }
}
```

**示例:**
```python
schema = {
    'port': {'type': int, 'required': True, 'min': 1, 'max': 65535},
    'host': {'type': str, 'required': True},
    'debug': {'type': bool, 'required': False}
}
is_valid = config.validate(schema)
```

#### `get_all(layer: Optional[str] = None) -> Dict[str, Any]`

获取完整的配置字典。

**参数:**
- `layer`: 可选的层级名称。如果为 None，返回合并后的配置

**返回:** 配置字典

**示例:**
```python
# 获取合并后的配置
all_config = config.get_all()

# 获取特定层级的配置
user_config = config.get_all('user')
```

#### `clear(layer: Optional[str] = None) -> None`

清空配置。

**参数:**
- `layer`: 可选的层级名称。如果为 None，清空所有层级

**示例:**
```python
# 清空特定层级
config.clear('runtime')

# 清空所有层级
config.clear()
```

#### `get_history() -> List[Dict[str, Any]]`

获取配置变更历史。

**返回:** 配置历史列表

**示例:**
```python
history = config.get_history()
for entry in history:
    print(f"Layer: {entry['layer']}")
    print(f"Config: {entry['config']}")
```

## 使用场景

### 场景 1: 流水线配置

```python
config = ConfigManager()

# 默认配置
config.set('pipeline.timeout', 60, layer='default')
config.set('detector.text.backend', 'paddleocr', layer='default')

# 用户自定义配置
config.set('pipeline.name', 'my_pipeline', layer='user')
config.set('detector.text.lang', 'chi_sim+eng', layer='user')

# 运行时覆盖
config.set('pipeline.timeout', 120, layer='runtime')
config.set('debug', True, layer='runtime')

# 获取最终配置
pipeline_config = {
    'name': config.get('pipeline.name'),
    'timeout': config.get('pipeline.timeout'),
    'detector': {
        'text': {
            'backend': config.get('detector.text.backend'),
            'lang': config.get('detector.text.lang')
        }
    },
    'debug': config.get('debug')
}
```

### 场景 2: 配置文件管理

```python
# 加载默认配置
config = ConfigManager()
config.load('config/default.yaml', layer='default')

# 加载用户配置（覆盖部分默认值）
config.load('config/user.yaml', layer='user')

# 运行时修改
config.set('debug', True, layer='runtime')

# 保存用户配置
config.save('config/user.yaml', layer='user')
```

### 场景 3: 配置验证

```python
config = ConfigManager()
config.load('config.yaml')

# 定义验证规则
schema = {
    'model.text_detector.backend': {
        'type': str,
        'required': True,
        'choices': ['paddleocr', 'tesseract', 'easyocr']
    },
    'model.text_detector.confidence_threshold': {
        'type': float,
        'required': True,
        'min': 0.0,
        'max': 1.0
    },
    'pipeline.max_workers': {
        'type': int,
        'required': False,
        'min': 1,
        'max': 16
    }
}

# 验证配置
if not config.validate(schema):
    raise ValueError("配置验证失败")
```

## 配置文件示例

### YAML 格式

```yaml
# config.yaml
pipeline:
  name: chat_analysis
  timeout: 60
  steps:
    - text_detection
    - bubble_detection
    - nickname_extraction

detectors:
  text:
    backend: paddleocr
    model_dir: models/PP-OCRv5_server_det/
    confidence_threshold: 0.5
  
  bubble:
    screen_width: 720
    memory_path: chat_memory.json

extractors:
  nickname:
    top_k: 3
    min_score: 0.5
```

### JSON 格式

```json
{
  "pipeline": {
    "name": "chat_analysis",
    "timeout": 60,
    "steps": [
      "text_detection",
      "bubble_detection",
      "nickname_extraction"
    ]
  },
  "detectors": {
    "text": {
      "backend": "paddleocr",
      "model_dir": "models/PP-OCRv5_server_det/",
      "confidence_threshold": 0.5
    },
    "bubble": {
      "screen_width": 720,
      "memory_path": "chat_memory.json"
    }
  },
  "extractors": {
    "nickname": {
      "top_k": 3,
      "min_score": 0.5
    }
  }
}
```

## 最佳实践

### 1. 使用分层配置

将配置分为三层，便于管理和覆盖：

- **default**: 存放系统默认配置
- **user**: 存放用户自定义配置
- **runtime**: 存放临时运行时配置

### 2. 使用配置验证

在加载配置后立即验证，确保配置的正确性：

```python
config.load('config.yaml')
if not config.validate(schema):
    raise ValueError("配置无效")
```

### 3. 使用嵌套键

使用点号分隔符组织配置，使结构更清晰：

```python
config.set('model.text_detector.backend', 'paddleocr')
config.set('model.text_detector.confidence', 0.5)
```

### 4. 保存配置历史

在修改重要配置前，可以先保存当前配置：

```python
# 保存当前配置
config.save('backup.yaml', layer='user')

# 修改配置
config.set('important.setting', new_value)
```

### 5. 使用环境特定配置

为不同环境（开发、测试、生产）使用不同的配置文件：

```python
import os

env = os.getenv('ENV', 'development')
config.load(f'config/{env}.yaml', layer='user')
```

## 常见问题

### Q: 如何重置配置到默认值？

```python
# 清除 user 和 runtime 层
config.clear('user')
config.clear('runtime')
```

### Q: 如何查看当前生效的完整配置？

```python
full_config = config.get_all()
print(full_config)
```

### Q: 如何处理配置文件不存在的情况？

```python
from pathlib import Path

config_path = 'config.yaml'
if Path(config_path).exists():
    config.load(config_path)
else:
    print(f"配置文件不存在: {config_path}")
    # 使用默认配置
```

### Q: 如何在多个模块间共享配置？

创建一个全局配置实例：

```python
# config_instance.py
from src.screenshot2chat.config import ConfigManager

global_config = ConfigManager()
global_config.load('config.yaml')

# 在其他模块中使用
from config_instance import global_config

backend = global_config.get('model.backend')
```

## 相关文档

- [设计文档](../design.md)
- [需求文档](../requirements.md)
- [API 参考](API_REFERENCE.md)

## 更新日志

### v1.0.0 (2024)
- 初始版本
- 支持分层配置
- 支持 YAML 和 JSON 格式
- 支持配置验证
- 支持配置历史
