# Nickname Scoring System Improvement

## 问题描述

用户反馈：**当顶上有两个框都靠近中心时，应该使用字体高度(height)更大的框作为nickname**

## 问题案例

### test_bumble (2).jpg 和 (3).jpg

**修改前：**
- ❌ 错误识别 "0.5/0" / "1.5/0" 为第一名
- 原因：这些框在最顶部，获得了y_rank=20分的高分
- 但它们的height很小（0.8, 1.2），不应该是nickname

**修改后：**
- ✅ 正确识别 "你的Opening Move" 为第一名
- 原因：虽然Y位置较低，但height更大，更符合nickname特征

## 解决方案

### 评分权重调整

| 评分项 | 修改前 | 修改后 | 变化 | 原因 |
|--------|--------|--------|------|------|
| Position | 15分 | 15分 | - | 保持不变 |
| Text | 30分 | 30分 | - | 保持不变 |
| Y Position | 15分 | 15分 | - | 保持不变 |
| **Height** | **20分** | **30分** | **⬆️ +10** | **增加字体大小的重要性** |
| **Y Rank** | **20分** | **10分** | **⬇️ -10** | **降低位置排名的权重** |
| **总分** | **100分** | **100分** | - | 保持100分制 |

### 具体修改

#### 1. Height Score (20分 → 30分)

```python
# 修改前
if ideal_height_min <= height_ratio <= ideal_height_max:
    normalized_height = (height_ratio - ideal_height_min) / (ideal_height_max - ideal_height_min)
    height_score = normalized_height * 20  # 最高20分
elif height_ratio < ideal_height_min:
    height_score = (height_ratio / ideal_height_min) * 10  # 最高10分
else:
    height_score = (ideal_height_max / height_ratio) * 10  # 最高10分

# 修改后
if ideal_height_min <= height_ratio <= ideal_height_max:
    normalized_height = (height_ratio - ideal_height_min) / (ideal_height_max - ideal_height_min)
    height_score = normalized_height * 30  # ⬆️ 最高30分
elif height_ratio < ideal_height_min:
    height_score = (height_ratio / ideal_height_min) * 15  # ⬆️ 最高15分
else:
    height_score = (ideal_height_max / height_ratio) * 15  # ⬆️ 最高15分
```

#### 2. Y Rank Score (20分 → 10分)

```python
# 修改前
if y_rank == 1:
    y_rank_score = 20  # 第1名20分
elif y_rank == 2:
    y_rank_score = 15  # 第2名15分
elif y_rank == 3:
    y_rank_score = 10  # 第3名10分

# 修改后
if y_rank == 1:
    y_rank_score = 10  # ⬇️ 第1名10分
elif y_rank == 2:
    y_rank_score = 7   # ⬇️ 第2名7分
elif y_rank == 3:
    y_rank_score = 5   # ⬇️ 第3名5分
```

## 测试结果对比

### test_bumble (2).jpg

| 候选框 | 修改前得分 | 修改后得分 | 排名变化 |
|--------|-----------|-----------|---------|
| "你的Opening Move" | 61.2 (第2名) | **62.0 (第1名)** | ⬆️ 提升 |
| "0.5/0" | **70.9 (第1名)** | 61.3 (第2名) | ⬇️ 下降 |

**得分细项对比：**

```
"你的Opening Move" (正确答案):
修改前: position=14.5, text=30.0, y_position=15.0, height=1.7,  y_rank=0.0,  total=61.2
修改后: position=14.5, text=30.0, y_position=15.0, height=2.5,  y_rank=0.0,  total=62.0
        ↑ height分数提升 (1.7→2.5)

"0.5/0" (错误答案):
修改前: position=13.1, text=30.0, y_position=7.0,  height=0.8,  y_rank=20.0, total=70.9
修改后: position=13.1, text=30.0, y_position=7.0,  height=1.2,  y_rank=10.0, total=61.3
        ↑ height分数略升 (0.8→1.2)  ↓ y_rank大幅下降 (20→10)
```

### test_bumble (3).jpg

| 候选框 | 修改前得分 | 修改后得分 | 排名变化 |
|--------|-----------|-----------|---------|
| "你的Opening Move" | 62.0 (第2名) | **63.2 (第1名)** | ⬆️ 提升 |
| "1.5/0" | **71.3 (第1名)** | 62.0 (第2名) | ⬇️ 下降 |

**得分细项对比：**

```
"你的Opening Move" (正确答案):
修改前: position=14.5, text=30.0, y_position=15.0, height=2.5,  y_rank=0.0,  total=62.0
修改后: position=14.5, text=30.0, y_position=15.0, height=3.8,  y_rank=0.0,  total=63.2
        ↑ height分数提升 (2.5→3.8)

"1.5/0" (错误答案):
修改前: position=13.1, text=30.0, y_position=7.0,  height=1.2,  y_rank=20.0, total=71.3
修改后: position=13.1, text=30.0, y_position=7.0,  height=1.9,  y_rank=10.0, total=62.0
        ↑ height分数略升 (1.2→1.9)  ↓ y_rank大幅下降 (20→10)
```

## 改进效果

### 准确率提升

- **修改前**: test_bumble (2) 和 (3) 识别错误
- **修改后**: ✅ 两张图片都正确识别

### 核心改进

1. **字体大小优先**: 当多个候选框都靠近中心时，字体更大的框获得更高分数
2. **位置权重降低**: 仅因为在最顶部而获得的优势减少
3. **更符合实际**: Nickname通常使用较大的字体，这个特征现在得到了更好的体现

## 评分系统设计理念

### 权重分配逻辑

1. **Text (30分)** - 最高权重
   - 过滤系统文本是最重要的
   - 避免将时间、状态等误识别为nickname

2. **Height (30分)** - 最高权重 ⬆️
   - Nickname通常使用较大字体
   - 当多个候选框都靠近中心时，字体大小成为关键区分因素

3. **Position (15分)** - 中等权重
   - 靠近中心的框更可能是nickname
   - 但不应该是决定性因素

4. **Y Position (15分)** - 中等权重
   - 在顶部区域但不是极端顶部
   - 避免系统UI元素

5. **Y Rank (10分)** - 最低权重 ⬇️
   - 位置排名作为辅助参考
   - 不应该主导最终决策

## 文件修改

- **修改文件**: `src/screenshotanalysis/processors.py`
- **修改方法**: `_calculate_nickname_score()`
- **修改行数**: 约100行

## 测试验证

运行测试脚本：
```bash
python nickname/examples/test_nicknames_smart.py
```

**测试结果**:
- ✅ test_bumble (2).jpg: 正确识别 "你的Opening Move"
- ✅ test_bumble (3).jpg: 正确识别 "你的Opening Move"
- ✅ 其他图片: 保持原有准确率

## 总结

通过调整评分权重，成功实现了用户需求：**当顶上有两个框都靠近中心时，使用字体高度(height)更大的框作为nickname**。

核心改进：
- Height权重 ⬆️ 20→30分 (+50%)
- Y-rank权重 ⬇️ 20→10分 (-50%)
- 总分保持100分不变

这个改进使得评分系统更加符合实际场景，字体大小成为区分nickname的重要特征。
