# 绘制Top3昵称检测结果功能

## 功能概述

为 `examples/test_nicknames_smart.py` 添加了可视化功能，将得分前三的昵称候选框绘制到图片上并保存到输出文件夹。

## 新增功能

### 1. 绘制函数 `draw_top3_results`

```python
def draw_top3_results(image_path, top_candidates, output_dir="test_output/smart_nicknames"):
    """
    绘制得分前三的候选框到图片上
    
    Args:
        image_path: 原始图片路径
        top_candidates: 前三名候选者列表
        output_dir: 输出目录
    """
```

**功能特点：**
- 为前三名候选框绘制不同颜色的边框：
  - 第1名：绿色
  - 第2名：橙色
  - 第3名：红色
- 显示排名、文本和得分
- 显示Y-rank排名
- 自动创建输出目录
- 保存为 `top3_原文件名` 格式

### 2. 更新 `extract_nicknames_smart` 函数

添加了两个新参数：
- `draw_results=False`: 是否绘制结果
- `output_dir="test_output/smart_nicknames"`: 输出目录

### 3. 简化输出

移除了 "候选框评分:" 部分的详细输出，只保留：
- 基本信息（图片名、尺寸、检测框数量）
- 最终选择的Top3结果

## 使用方法

### 运行脚本

```bash
python examples/test_nicknames_smart.py
```

### 输出结果

1. **控制台输出**：
   - 每张图片的处理进度
   - 最终选择的Top3昵称及其得分
   - 汇总结果

2. **可视化图片**：
   - 保存位置：`test_output/smart_nicknames/`
   - 文件命名：`top3_原文件名`
   - 包含：
     - 彩色边框标注（绿/橙/红）
     - 排名标签
     - 文本内容
     - 得分信息
     - Y-rank排名

## 示例输出

### 控制台输出

```
================================================================================
图片: test_whatsapp.png
屏幕尺寸: 384x800
检测到 45 个文本框
过滤掉 8 个极端边缘框
保留 37 个候选框
顶部区域候选框: 12 个

最终选择（按得分排序）:
  1. 'Gg Gg (你)' (得分: 78.8/100, Y排名: 1)
     细项: position=11.7, text=30.0, y_position=15.0, height=2.1, y_rank=20.0, total=78.8
  2. '给自己发消息' (得分: 66.9/100, Y排名: 3)
     细项: position=11.5, text=30.0, y_position=15.0, height=0.4, y_rank=10.0, total=66.9
  3. '00。' (得分: 54.5/100, Y排名: 4)
     细项: position=2.6, text=30.0, y_position=15.0, height=6.9, y_rank=0.0, total=54.5
================================================================================
结果已保存到: test_output/smart_nicknames/top3_test_whatsapp.png
```

### 可视化图片特点

- **第1名（绿色框）**：
  - 标签：`#1: Gg Gg (你) (78.8)`
  - 框内显示：`Y-Rank: 1`

- **第2名（橙色框）**：
  - 标签：`#2: 给自己发消息 (66.9)`
  - 框内显示：`Y-Rank: 3`

- **第3名（红色框）**：
  - 标签：`#3: 00。 (54.5)`
  - 框内显示：`Y-Rank: 4`

## 生成的文件

运行脚本后，在 `test_output/smart_nicknames/` 目录下生成了以下文件：

```
test_output/smart_nicknames/
├── top3_test_bumble (1).jpg
├── top3_test_bumble (2).jpg
├── top3_test_bumble (3).jpg
├── top3_test_bumble (4).jpg
├── top3_test_discord.png
├── top3_test_discord_2.png
├── top3_test_discord_3.png
├── top3_test_instagram.png
├── top3_test_instagram_2.png
├── top3_test_telegram1.png
├── top3_test_whatsapp.png
├── top3_test_whatsapp_2.png
└── top3_test_whatsapp_3.png
```

## 技术细节

### 颜色定义（BGR格式）

```python
colors = [
    (0, 255, 0),    # 绿色 - 第1名
    (0, 165, 255),  # 橙色 - 第2名
    (0, 0, 255),    # 红色 - 第3名
]
```

### 绘制元素

1. **矩形框**：3像素粗的彩色边框
2. **标签背景**：与边框同色的填充矩形
3. **标签文字**：白色文字，包含排名、文本和得分
4. **Y-rank标签**：框内显示Y排名信息

### 文本处理

- 字体：`cv2.FONT_HERSHEY_SIMPLEX`
- 标签字体大小：0.6
- Y-rank字体大小：0.5
- 文字粗细：2像素

## 改进点

1. **简化输出**：移除了中间过程的详细评分输出，只保留最终结果
2. **可视化**：添加了直观的图片标注，便于查看检测结果
3. **自动化**：自动创建输出目录，批量处理所有图片
4. **信息完整**：在图片上同时显示排名、文本、得分和Y-rank

## 修改的文件

- `examples/test_nicknames_smart.py` - 添加绘制功能和简化输出

## 日期

2026年1月23日
