# 得分细项显示功能

## 功能说明

为了更好地理解nickname检测的评分逻辑，现在在打印候选得分时会显示详细的得分细项。

## 实现改进

### 1. 修改 `_calculate_nickname_score` 方法

**返回值改变：**
- 原来：返回单个 `float` 值（总分）
- 现在：返回 `tuple` (总分, 得分细项字典)

**得分细项字典包含：**
```python
{
    'position': float,      # 位置得分 (0-30)
    'size': float,          # 尺寸得分 (0-15)
    'text': float,          # 文本得分 (0-25)
    'y_position': float,    # Y位置得分 (0-10)
    'height': float,        # 字体高度得分 (0-20)
    'total': float          # 总分 (0-100)
}
```

### 2. 修改 `extract_nicknames_smart` 方法

- 接收并存储得分细项
- 在候选字典中添加 `score_breakdown` 字段
- 在日志中打印得分细项

### 3. 修改 demo 脚本

在 `examples/nickname_detection_demo.py` 中显示得分细项：

```python
print(f"  Breakdown: Position={bd['position']:.1f}, Size={bd['size']:.1f}, "
      f"Text={bd['text']:.1f}, Y={bd['y_position']:.1f}, Height={bd['height']:.1f}")
```

## 输出示例

### 示例 1: test_whatsapp_2.png

```
Detected Nickname: 'Gg Gg'
  Score: 83.4/100
  Breakdown: Position=29.8, Size=14.8, Text=25.0, Y=10.0, Height=3.8
  Position: (190, 45)

Other candidates:
  2. 'https://t.me/sophon_share_bot?' (score: 78.0)
     Breakdown: Pos=29.5, Size=11.9, Text=25.0, Y=10.0, Height=1.7
  3. 'online' (score: 58.9)
     Breakdown: Pos=29.8, Size=14.8, Text=0.0, Y=10.0, Height=14.2
```

**分析：**
- "Gg Gg" 总分最高 (83.4)
- "online" 虽然 Height 得分很高 (14.2)，但 Text 得分为 0（被识别为状态文本）
- 这清楚地解释了为什么 "Gg Gg" 被选中

### 示例 2: test.jpg

```
Detected Nickname: '王涛'
  Score: 76.4/100
  Breakdown: Position=29.8, Size=9.1, Text=25.0, Y=10.0, Height=2.5
  Position: (190, 50)

Other candidates:
  2. '没找到' (score: 69.9)
     Breakdown: Pos=12.0, Size=15.0, Text=25.0, Y=10.0, Height=7.9
  3. 'Cursor交流10群' (score: 60.9)
     Breakdown: Pos=20.5, Size=15.0, Text=25.0, Y=0.0, Height=0.4
```

**分析：**
- "王涛" 的 Position 得分最高 (29.8)，最靠近中心
- "没找到" 虽然 Height 得分更高 (7.9)，但 Position 得分较低 (12.0)
- "Cursor交流10群" 的 Y 得分为 0（不在顶部区域）

### 示例 3: test_bumble (4).jpg

```
Detected Nickname: 'Jt的Opening Move'
  Score: 67.7/100
  Breakdown: Position=14.8, Size=15.0, Text=25.0, Y=10.0, Height=2.9
  Position: (94, 108)

Other candidates:
  2. 'HR/r' (score: 65.4)
     Breakdown: Pos=26.2, Size=4.2, Text=25.0, Y=5.0, Height=5.0
  3. 'Jt' (score: 63.5)
     Breakdown: Pos=19.5, Size=6.5, Text=25.0, Y=10.0, Height=2.5
```

**分析：**
- "Jt的Opening Move" 虽然 Position 得分不是最高，但综合得分最高
- "HR/r" 的 Size 得分很低 (4.2)，太窄
- "Jt" 的各项得分都比较平均，但总分略低

## 得分细项的作用

### 1. 调试和优化
- 清楚地看到每个候选框在各个维度的表现
- 帮助理解为什么某个候选被选中或被排除
- 便于调整评分权重

### 2. 问题诊断
- 当检测结果不理想时，可以通过得分细项找出原因
- 例如：如果 Text 得分为 0，说明被误判为系统文本

### 3. 透明度
- 让用户了解评分逻辑
- 增加系统的可解释性

## 得分细项含义

| 细项 | 满分 | 含义 | 高分条件 |
|------|------|------|----------|
| Position | 30 | 水平位置 | 越靠近屏幕中心得分越高 |
| Size | 15 | 宽度 | 宽度在15%-50%屏幕宽度之间 |
| Text | 25 | 文本类型 | 不是系统文本（时间、状态等） |
| Y | 10 | 垂直位置 | 在顶部区域但不是极端顶部 |
| Height | 20 | 字体高度 | 字体高度在2%-8%屏幕高度之间，越大越好 |

## 使用建议

### 查看详细日志
如果需要更详细的调试信息，可以传入 `log_file` 参数：

```python
with open('nickname_detection.log', 'w', encoding='utf-8') as log:
    result = processor.extract_nicknames_smart(
        text_det_results, 
        image, 
        log_file=log
    )
```

### 分析得分细项
当检测结果不符合预期时：
1. 查看 Top 候选的得分细项
2. 查看其他候选的得分细项
3. 对比各项得分，找出差异
4. 根据实际情况调整评分权重或过滤规则

## 总结

得分细项显示功能提供了完整的评分透明度，帮助理解和优化nickname检测逻辑。通过查看各个维度的得分，可以清楚地知道为什么某个候选被选中，以及如何改进检测效果。
