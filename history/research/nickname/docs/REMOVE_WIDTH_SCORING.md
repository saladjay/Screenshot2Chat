# 移除宽度评分改进

## 问题

之前的评分系统中，Size得分（0-15分）完全依靠文本框的宽度（width）来评分。这导致了不公平的情况：

**问题案例：test_bumble (2).jpg**
- "你的Opening Move" - Size得分 15.0（宽度合适）
- "天天" - Size得分 11.7（宽度较窄，被扣分）

但实际上，nickname的长度差异很大是正常的：
- 短nickname："天天"、"Kai"、"Jt"
- 长nickname："你的Opening Move"、"Sophon Admin"

**用宽度评分的问题：**
- 短nickname会被不公平地扣分
- 宽度不是判断nickname的可靠特征
- 与字体高度（Height）的作用重复

## 解决方案

**移除Size评分，重新分配权重：**

### 改进前的评分系统（总分100分）
1. Position（位置）：30分
2. **Size（宽度）：15分** ❌ 移除
3. Text（文本类型）：25分
4. Y_position（Y位置）：10分
5. Height（字体高度）：20分

### 改进后的评分系统（总分100分）
1. **Position（位置）：35分** ⬆️ +5分
2. **Text（文本类型）：30分** ⬆️ +5分
3. **Y_position（Y位置）：15分** ⬆️ +5分
4. Height（字体高度）：20分（保持不变）

**权重分配逻辑：**
- Position最重要，增加到35分（nickname通常在中心）
- Text很重要，增加到30分（过滤系统文本）
- Y_position重要，增加到15分（nickname在顶部）
- Height保持20分（字体大小是关键特征）

## 测试结果对比

### test_bumble (2).jpg

**改进前：**
```
1. '你的Opening Move' (score: 80.7)
   Breakdown: Position=29.0, Size=15.0, Text=25.0, Y=10.0, Height=1.7

2. '天天' (score: 70.7)
   Breakdown: Pos=21.0, Size=11.7, Text=25.0, Y=10.0, Height=2.9
```

**改进后：**
```
1. '你的Opening Move' (score: 80.5)
   Breakdown: Position=33.8, Text=30.0, Y=15.0, Height=1.7

2. '天天' (score: 72.4)
   Breakdown: Pos=24.5, Text=30.0, Y=15.0, Height=2.9
```

**分析：**
- "你的Opening Move" 仍然是第一名（位置最佳）
- "天天" 的得分提高了（70.7 → 72.4），不再因为宽度短而被过度惩罚
- 两者的得分差距略微缩小（10.0 → 8.1），更加公平

### test_whatsapp_2.png

**改进前：**
```
1. 'Gg Gg' (score: 83.4)
   Breakdown: Position=29.8, Size=14.8, Text=25.0, Y=10.0, Height=3.8
```

**改进后：**
```
1. 'Gg Gg' (score: 83.5)
   Breakdown: Position=34.7, Text=30.0, Y=15.0, Height=3.8
```

**分析：**
- 得分略微提高（83.4 → 83.5）
- 各项得分更加均衡

### test.jpg

**改进前：**
```
1. '王涛' (score: 76.4)
   Breakdown: Position=29.8, Size=9.1, Text=25.0, Y=10.0, Height=2.5
```

**改进后：**
```
1. '王涛' (score: 82.2)
   Breakdown: Position=34.7, Text=30.0, Y=15.0, Height=2.5
```

**分析：**
- 得分显著提高（76.4 → 82.2）
- 不再因为宽度略窄而被扣分

## 改进效果

### 1. 更公平
- 短nickname（"天天"、"Jt"）不再被不公平地扣分
- 长nickname（"你的Opening Move"）也不会因为宽度而获得不当优势

### 2. 更简洁
- 评分因素从5个减少到4个
- 移除了不可靠的宽度特征
- 评分逻辑更清晰

### 3. 更合理
- Position权重增加（35分）：位置是最重要的特征
- Text权重增加（30分）：过滤系统文本很关键
- Y_position权重增加（15分）：顶部位置很重要
- Height保持（20分）：字体大小是关键特征

### 4. 保持高准确率
- 所有测试图片仍然正确识别nickname
- 得分分布更加合理

## 评分因素说明

| 因素 | 满分 | 含义 | 评分依据 |
|------|------|------|----------|
| Position | 35 | 水平位置 | 越靠近屏幕中心得分越高 |
| Text | 30 | 文本类型 | 不是系统文本（时间、状态等）得满分 |
| Y_position | 15 | 垂直位置 | 在顶部区域（5%-15%屏幕高度）得满分 |
| Height | 20 | 字体高度 | 字体高度在2%-8%屏幕高度之间，越大越好 |

## 代码改动

### 1. 移除Size评分计算
```python
# 删除了这段代码：
ideal_width_min = screen_width * 0.15
ideal_width_max = screen_width * 0.50

if ideal_width_min <= box.width <= ideal_width_max:
    size_score = 15
elif box.width < ideal_width_min:
    size_score = (box.width / ideal_width_min) * 15
else:
    size_score = (ideal_width_max / box.width) * 15
```

### 2. 调整其他因素权重
```python
position_score = (1 - normalized_distance) * 35  # 30 → 35
text_score = 30 if not system_text else 0       # 25 → 30
y_score = 15 if in_top_region else 0            # 10 → 15
height_score = ...                               # 保持 20
```

### 3. 更新输出格式
- 移除了得分细项中的 "Size" 显示
- 只显示：Position, Text, Y, Height

## 结论

移除宽度评分使得nickname检测更加公平和合理。短nickname和长nickname都能得到公正的评分，评分系统更加关注真正重要的特征：位置、文本类型、Y位置和字体高度。

这个改进解决了用户提出的问题：不应该依靠width来评分，因为nickname的长度差异很大是正常现象。
