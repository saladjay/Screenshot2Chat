# 更新 test_nicknames_smart.py 以使用新的 Y-rank 评分系统

## 问题

`examples/test_nicknames_smart.py` 文件使用了本地定义的旧版评分函数，没有包含新的 Y-rank 评分（0-20分）。

## 原因

该文件定义了自己的 `calculate_nickname_score` 函数，而不是使用 `processor._calculate_nickname_score` 方法。这导致它使用的是旧的评分系统：
- 位置得分：0-35分（旧）
- 文本得分：0-30分
- Y位置得分：0-15分
- 高度得分：0-20分
- **缺少 Y-rank 得分：0-20分**

## 解决方案

### 1. 更新 `extract_nicknames_smart` 函数

修改函数以使用 `processor._calculate_nickname_score` 方法：

```python
# 按Y位置排序以计算排名
sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)
box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}

# 获取Y排名
y_rank = box_to_rank.get(id(box), None)

# 使用processor的新评分方法（包含Y-rank得分）
nickname_score, score_breakdown = processor._calculate_nickname_score(
    box, cleaned_text, screen_width, screen_height, y_rank=y_rank
)
```

### 2. 废弃本地函数

将本地定义的辅助函数标记为已废弃：
- `calculate_nickname_score` → `calculate_nickname_score_OLD`
- `is_extreme_edge_box` → `is_extreme_edge_box_OLD`
- `is_likely_system_text` → `is_likely_system_text_OLD`

现在使用 processor 的方法：
- `processor._calculate_nickname_score()`
- `processor._is_extreme_edge_box()`
- `processor._is_likely_system_text()`

### 3. 更新输出格式

添加 Y-rank 信息到输出：

```python
print(f"'{cleaned_text}' -> 得分: {nickname_score:.1f}/100 (Y排名: {y_rank})")
print(f"  细项: {breakdown_str}")  # 现在包含 y_rank
```

## 测试结果

运行更新后的脚本，成功显示 Y-rank 得分：

```
[图片] test_whatsapp.png
   [OK] 检测到 3 个可能的昵称:
      - 'Gg Gg (你)' (得分: 78.8/100, Y排名: 1)
        细项: position=11.7, text=30.0, y_position=15.0, height=2.1, y_rank=20.0, total=78.8
      - '给自己发消息' (得分: 66.9/100, Y排名: 3)
        细项: position=11.5, text=30.0, y_position=15.0, height=0.4, y_rank=10.0, total=66.9
      - '00。' (得分: 54.5/100, Y排名: 4)
        细项: position=2.6, text=30.0, y_position=15.0, height=6.9, y_rank=0.0, total=54.5
```

✓ Y排名 1 获得 20.0 分
✓ Y排名 3 获得 10.0 分
✓ Y排名 4+ 获得 0.0 分

## 新的评分系统（总分100分）

1. **位置得分（0-15分）**：水平位置靠近屏幕中心
2. **文本得分（0-30分）**：不是系统UI文本
3. **Y位置得分（0-15分）**：在顶部区域但不是极端顶部
4. **高度得分（0-20分）**：字体大小（昵称通常较大）
5. **Y-rank 得分（0-20分）**：Y方向排名
   - 第1名：20分
   - 第2名：15分
   - 第3名：10分
   - 第4名及以后：0分

## 修改的文件

- `examples/test_nicknames_smart.py` - 更新为使用新的 Y-rank 评分系统

## 日期

2026年1月23日
