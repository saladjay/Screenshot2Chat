# Nickname提取逻辑说明

## 概述

当前系统使用**三层回退策略**来提取聊天截图中的沟通对象昵称（nickname）。这是一个完全**应用无关**的方法，仅依赖几何属性和位置关系。

## 主入口函数

```python
ChatMessageProcessor.extract_nicknames_adaptive(
    layout_det_results,  # PP-DocLayoutV2的检测结果
    text_det_results,    # PP-OCRv5_server_det的检测结果
    image,               # 原始图像数组
    screen_width,        # 屏幕宽度
    memory_path=None,    # 可选：说话者记忆持久化路径
    log_file=None        # 可选：日志文件
)
```

### 返回值结构

```python
{
    'speaker_A': {
        'nickname': str or None,      # 提取的昵称文本
        'box': TextBox or None,       # 昵称所在的文本框
        'method': str                 # 使用的检测方法
    },
    'speaker_B': {
        'nickname': str or None,
        'box': TextBox or None,
        'method': str
    },
    'metadata': {
        'layout': str,                # 布局类型（double, double_left等）
        'confidence': float,          # 置信度
        'frame_count': int            # 帧计数
    }
}
```

## 三层回退策略

### ⚠️ PP-DocLayoutV2支持的标签类型

PP-DocLayoutV2是一个**文档布局分析模型**，支持以下25种标签：

```
0: abstract          - 摘要
1: algorithm         - 算法
2: aside_text        - 旁注文本
3: chart             - 图表
4: content           - 内容
5: display_formula   - 显示公式
6: doc_title         - 文档标题
7: figure_title      - 图标题
8: footer            - 页脚
9: footer_image      - 页脚图片
10: footnote         - 脚注
11: formula_number   - 公式编号
12: header           - 页眉
13: header_image     - 页眉图片
14: image            - 图片
15: inline_formula   - 行内公式
16: number           - 数字
17: paragraph_title  - 段落标题
18: reference        - 参考文献
19: reference_content- 参考文献内容
20: seal             - 印章
21: table            - 表格
22: text             - 文本
23: vertical_text    - 竖排文本
24: vision_footnote  - 视觉脚注
```

**重要**: 这些标签都是为**文档分析**设计的，**不包含**聊天应用特有的标签如：
- ❌ nickname（昵称）
- ❌ avatar（头像）
- ❌ chat_bubble（聊天气泡）
- ❌ message（消息）

因此，方法1和方法2在当前模型下**无法正常工作**。

---

### 方法1: Layout Det直接检测 (`_extract_from_layout_det`)

**原理**: 直接从PP-DocLayoutV2的检测结果中查找标记为'nickname'的框

**步骤**:
1. 筛选所有`layout_det == 'nickname'`的框
2. 按照`speaker`属性分组（A或B）
3. 为每个说话者选择第一个nickname框

**⚠️ 重要说明**:
PP-DocLayoutV2是一个**文档布局分析模型**，它只能识别文档相关的标签类型：
- text, image, table, header, footer, chart等
- **不包含**聊天应用特有的'nickname'或'avatar'类型

**当前状态**: ❌ **此方法实际上不可用**
- PP-DocLayoutV2不支持'nickname'标签
- 这个方法是为未来可能的自定义模型预留的接口

---

### 方法2: Avatar邻近搜索 (`_extract_from_avatar_neighbor`)

**原理**: 在头像（avatar）附近查找文本框作为昵称

**步骤**:
1. 获取所有标记为'avatar'的layout_det框
2. 对每个avatar框：
   - 查找在其**上方或右侧**的text_det框
   - 应用尺寸过滤（最小高度10px，最小宽度20px）
   - 计算距离，选择最近的文本框
3. 为找到的文本框分配对应的speaker

**⚠️ 重要说明**:
PP-DocLayoutV2支持'image'标签，但这是指**文档中的图片**（如图表、插图），不是聊天应用的头像。
- 在聊天截图中，PP-DocLayoutV2可能会将头像识别为'image'
- 但这个识别不够可靠，因为聊天气泡中的图片消息也会被识别为'image'
- 需要额外的逻辑来区分头像和其他图片

**位置判断逻辑**:
```python
def _is_above_or_right(text_box, avatar_box):
    # 上方: text_box的底部在avatar顶部之上
    # 右侧: text_box的左边在avatar右边之右
    return (text_box.y_max < avatar_box.y_min or 
            text_box.x_min > avatar_box.x_max)
```

**距离计算**:
```python
def _calculate_distance(text_box, avatar_box):
    # 使用中心点的欧几里得距离
    dx = text_box.center_x - avatar_box.center_x
    dy = text_box.center_y - avatar_box.center_y
    return sqrt(dx*dx + dy*dy)
```

**当前状态**: ❌ **此方法基本不可用**
- 代码中查找`layout_det == 'avatar'`的框，但PP-DocLayoutV2不支持此标签
- 即使改为查找'image'，也难以区分头像和其他图片
- 需要额外的头像检测逻辑（如基于位置、大小、形状等）

---

### 方法3: 顶部区域搜索 (`_extract_from_top_region`)

**原理**: 在屏幕顶部10%区域查找文本框，按左右位置分配给不同说话者

**步骤**:

#### 1. 筛选顶部区域
```python
top_region_boundary = screen_height * 0.1
top_boxes = [box for box in text_det_boxes if box.y_max < top_region_boundary]
```

#### 2. 应用尺寸过滤
```python
# 最小尺寸
min_height = 10 pixels
min_width = 20 pixels

# 最大宽度（排除页眉）
max_width = screen_width * 0.4  # 40%屏幕宽度
```

#### 3. 按水平位置分组
```python
screen_center = screen_width * 0.5

left_boxes = [box for box in filtered if box.center_x < screen_center]
right_boxes = [box for box in filtered if box.center_x >= screen_center]
```

#### 4. 选择最顶部的框
```python
# 对每一侧，选择y_min最小的框（最靠近顶部）
left_nickname = min(left_boxes, key=lambda b: b.y_min) if left_boxes else None
right_nickname = min(right_boxes, key=lambda b: b.y_min) if right_boxes else None
```

#### 5. 映射到Speaker A/B
```python
# 分析layout_det框的speaker分布
# 统计A和B分别在左侧和右侧的数量
# 确定哪个speaker主要在左侧

if a_left_count + b_right_count > b_left_count + a_right_count:
    left_is_speaker_A = True
else:
    left_is_speaker_A = False

# 根据判断结果分配
if left_is_speaker_A:
    result['A'] = left_nickname
    result['B'] = right_nickname
else:
    result['A'] = right_nickname
    result['B'] = left_nickname
```

**优点**:
- 不依赖特定的layout_det类型
- 适用于大多数聊天应用（昵称通常在顶部）
- 回退策略的最后保障

**缺点**:
- 可能误识别时间戳或其他顶部文本为昵称
- 对于没有明显左右分布的布局可能不准确

**当前状态**: ✅ **所有测试图片都使用此方法**（20次成功）

---

## OCR文本提取

找到nickname框后，使用OCR提取文本：

```python
def _run_ocr_on_nickname(nickname_box, image, log_file):
    # 1. 裁剪图像到nickname框区域
    cropped = image[y_min:y_max, x_min:x_max]
    
    # 2. 使用PP-OCRv5_server_rec进行文本识别
    text_rec = ChatTextRecognition(model_name='PP-OCRv5_server_rec', lang='en')
    ocr_result = text_rec.predict_text(cropped)
    
    # 3. 清理结果（去除尾部特殊字符）
    cleaned_text = text.rstrip('>< |\t\n\r')
    
    return cleaned_text
```

---

## 当前测试结果分析

### 测试统计
- **总图片数**: 14张
- **成功提取**: 14张（100%）
- **使用方法**:
  - Method 1 (layout_det): 0次
  - Method 2 (avatar_neighbor): 0次
  - Method 3 (top_region): 20次（某些图片提取了2个speaker）

### 检测到的"昵称"示例

| 图片 | 检测结果 | 布局类型 | 问题 |
|------|---------|---------|------|
| test_whatsapp_2.png | "20:03", "GG" | double | "20:03"是时间戳，"GG"是真实昵称 |
| test_whatsapp_3.png | "20:04", "GG" | double | "20:04"是时间戳，"GG"是真实昵称 |
| test_discord.png | "16:52" | double_left | 时间戳 |
| test_instagram.png | "16:47" | double | 时间戳 |
| test_bumble (2).jpg | "V", "0.5/0" | double | 可能是UI元素 |

### 主要问题

1. **时间戳误识别**: 大部分检测结果是屏幕顶部的时间（如"20:03"、"16:52"）
2. **缺少真实昵称**: 很多图片的真实用户昵称没有被检测到
3. **方法1和2不可用**: 
   - **方法1**: PP-DocLayoutV2是文档布局模型，不支持'nickname'标签
   - **方法2**: PP-DocLayoutV2不支持'avatar'标签，即使用'image'也难以区分头像和其他图片
4. **只能依赖方法3**: 目前只有顶部区域搜索可用，但容易误识别时间戳

---

## 改进建议

### 短期改进

1. **增强顶部区域过滤**:
   - 排除时间格式的文本（正则匹配 `\d{1,2}:\d{2}`）
   - 增加对特定UI元素的过滤

2. **调整区域范围**:
   - 扩大搜索区域到顶部15-20%
   - 考虑搜索特定的"标题栏"区域

3. **添加文本验证**:
   - 检查提取的文本是否像昵称（长度、字符类型等）
   - 排除纯数字、纯符号的结果

### 长期改进

1. **使用专门的聊天应用模型**:
   - PP-DocLayoutV2是为文档设计的，不适合聊天应用
   - 需要训练或使用专门识别聊天UI元素的模型（nickname, avatar, chat_bubble等）
   - 或者使用通用目标检测模型（YOLO, Faster R-CNN）并在聊天截图上微调

2. **改进头像检测**:
   - 基于'image'标签 + 位置/大小/形状启发式规则
   - 头像通常是小的、正方形的、位于消息左侧或右侧的图片
   - 可以通过聚类分析区分头像和消息图片

3. **添加第四层回退**:
   - 基于聊天气泡的位置推断昵称位置
   - 使用更复杂的启发式规则

3. **多帧分析**:
   - 如果是视频或多张截图，综合多帧信息
   - 提高准确性和鲁棒性

---

## 使用示例

```python
from screenshotanalysis import ChatLayoutAnalyzer
from screenshotanalysis.processors import ChatMessageProcessor
from screenshotanalysis.utils import ImageLoader, letterbox
import numpy as np

# 初始化
text_analyzer = ChatLayoutAnalyzer(model_name="PP-OCRv5_server_det")
text_analyzer.load_model()

layout_analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayoutV2")
layout_analyzer.load_model()

processor = ChatMessageProcessor()

# 加载图片
image = ImageLoader.load_image("test.png")
image_array = np.array(image)
processed_image, padding = letterbox(image_array)

# 检测
text_det_results = text_analyzer.model.predict(processed_image)
layout_det_results = layout_analyzer.model.predict(processed_image)

# 提取nickname
result = processor.extract_nicknames_adaptive(
    layout_det_results=layout_det_results,
    text_det_results=text_det_results,
    image=processed_image,
    screen_width=processed_image.shape[1]
)

# 获取结果
nickname_a = result['speaker_A']['nickname']
nickname_b = result['speaker_B']['nickname']
method_a = result['speaker_A']['method']
method_b = result['speaker_B']['method']

print(f"Speaker A: {nickname_a} (method: {method_a})")
print(f"Speaker B: {nickname_b} (method: {method_b})")
```

---

## 相关文件

- **主实现**: `src/screenshotanalysis/processors.py`
  - `extract_nicknames_adaptive()` (行1580-1820)
  - `_extract_from_layout_det()` (行976-1081)
  - `_extract_from_avatar_neighbor()` (行1082-1225)
  - `_extract_from_top_region()` (行1226-1480)
  - `_run_ocr_on_nickname()` (行1481-1579)

- **示例脚本**:
  - `examples/extract_nicknames_demo.py` - 基础版本
  - `examples/extract_nicknames_detailed.py` - 详细日志版本
  - `examples/show_all_nicknames.py` - 简洁输出版本

- **测试日志**: `test_nickname_extraction/*.log`
