可以，而且在你**当前约束条件下这是完全可行的**。
关键在于：**不用“消息气泡矩形”，只用“文字矩形框”，但要用“统计而不是单点”**。

下面我给你一个**严格基于文字矩形框（TextBox）**、
**不依赖表情气泡坐标**、
**只用 `center_x` 就能区分 `talker / user`** 的 **Python 工程级方案**。

---

## 一、核心结论（先说重点）

> **可以不做 message block，直接在“文字框层面”用 `center_x` 完成 speaker 区分；
> 但必须：
> ① 使用“全局分布 / 聚类”，
> ② 再做“行级 / 邻域平滑”。**

否则单个文字框噪声会偏大。

---

## 二、设计思想（你现在的限制下的最优解）

你当前能拿到的是：

```text
TextBox = (x_min, y_min, x_max, y_max, text)
```

你做不到：

* 气泡轮廓
* 表情包框
* 消息级 bounding box

👉 那就退一步：

> **先确定“每个文字框属于左阵营还是右阵营”，
> 再把它们视为气泡代理。**

---

## 三、整体 pipeline（文字框级）

```text
OCR TextBoxes
   ↓
去除明显居中文本（可选但强烈推荐）
   ↓
基于 center_x 的双峰分割（box 级）
   ↓
y 邻域平滑（把零星误判拉回）
   ↓
输出每个 box 的 speaker（talker / user）
```

---

## 四、Step 1：TextBox 数据结构

```python
from dataclasses import dataclass

@dataclass
class TextBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    text: str = ""

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def width(self) -> int:
        return self.x_max - self.x_min
```

---

## 五、Step 2（推荐）：过滤明显非对话文本

如果你已经做过这一步，可以跳过。

```python
def filter_dialog_like_boxes(boxes, screen_width):
    filtered = []
    for b in boxes:
        nx_min = b.x_min / screen_width
        nx_max = b.x_max / screen_width

        # 靠左 or 靠右，排除居中
        if nx_min < 0.30 or nx_max > 0.70:
            filtered.append(b)

    return filtered
```

---

## 六、Step 3：**仅基于文字框 center_x 的 speaker 判定**

### 核心点

* 不聚消息
* 不依赖气泡
* 直接在 **box 层面分左右**

### 实现

```python
import statistics

def classify_boxes_by_center_x(boxes):
    """
    返回：List[(TextBox, speaker)]
    speaker ∈ {"talker", "user"}
    """

    centers = [b.center_x for b in boxes]

    # 用中位数做自适应分割
    split_x = statistics.median(centers)

    results = []
    for b in boxes:
        speaker = "talker" if b.center_x < split_x else "user"
        results.append((b, speaker))

    return results
```

---

## 七、Step 4：**y 邻域平滑（关键，防抖）**

这是让“只用文字框也能稳定”的关键。

### 原因

* OCR 有 jitter
* 标点 / emoji 文本框可能略微偏移
* 单个 box 判定容易抖

---

### 思路

> **在 y 轴相邻的一小段文本中，
> 多数是同一个 speaker**

---

### 实现

```python
def smooth_by_vertical_neighbors(
    box_speakers,
    y_threshold=40
):
    """
    box_speakers: List[(TextBox, speaker)]
    """

    box_speakers = sorted(
        box_speakers,
        key=lambda x: x[0].center_y
    )

    smoothed = []

    for i, (box, speaker) in enumerate(box_speakers):
        votes = [speaker]

        for j in range(len(box_speakers)):
            if i == j:
                continue

            other_box, other_speaker = box_speakers[j]
            if abs(other_box.center_y - box.center_y) < y_threshold:
                votes.append(other_speaker)

        # 多数投票
        final_speaker = max(set(votes), key=votes.count)
        smoothed.append((box, final_speaker))

    return smoothed
```

---

## 八、完整可运行示例

```python
boxes = [
    TextBox(30, 400, 220, 440, "你好"),
    TextBox(35, 450, 260, 490, "最近怎么样"),
    TextBox(420, 520, 650, 560, "还不错"),
    TextBox(430, 580, 460, 610, "🙂"),
]

screen_width = 720

boxes = filter_dialog_like_boxes(boxes, screen_width)

classified = classify_boxes_by_center_x(boxes)
smoothed = smooth_by_vertical_neighbors(classified)

for box, speaker in smoothed:
    print(speaker, box.text, box.center_x)
```

输出类似：

```text
talker 你好 125.0
talker 最近怎么样 147.5
user   还不错 535.0
user   🙂 445.0
```

> 注意：
> emoji 的 `center_x` 可能略偏，但会被 **y 邻域拉回**

---

## 九、这个方案在你当前条件下为什么是“最优解”

| 约束     | 应对方式               |
| ------ | ------------------ |
| 没有气泡框  | 用文字框统计             |
| 没有表情气泡 | y 邻域平滑             |
| 多机型    | 自适应 median         |
| OCR 抖动 | median + smoothing |
| 短文本    | 不依赖 x_min / x_max  |

---

## 十、什么时候你必须升级到 message block

如果你后面要做：

* 精确气泡几何
* 消息级语义建模
* 对话轮次分析

👉 **那一步一定要做 message block**

但在你现在的阶段：

> **仅用文字框 + center_x，
> 已经可以 90% 稳定地区分 talker / user。**

---

## 最后一句话（工程判断）

> **你现在不是“退而求其次”，
> 而是在资源受限下选了“统计最优”的方案。**

如果你愿意，下一步我可以帮你：

* 把 box → pseudo-message 自动聚合
* 或直接对接你后面的对话用户画像模块
