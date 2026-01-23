"""手动测试split_columns方法"""
import sys
import numpy as np
sys.path.insert(0, 'src')

from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
from screenshotanalysis.processors import TextBox

# 创建检测器
detector = ChatLayoutDetector(screen_width=720)

# 测试1: 空列表
print("测试1: 空列表")
layout, left, right, fallback_meta = detector.split_columns([])
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "single" and len(left) == 0 and len(right) == 0

# 测试2: 少于4个文本框
print("\n测试2: 少于4个文本框")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([105, 300, 205, 350], 0.9)
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "single" and len(left) == 3 and len(right) == 0

# 测试3: 标准双列布局（左右分开）
print("\n测试3: 标准双列布局")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),  # 左列
    TextBox([110, 200, 210, 250], 0.9),  # 左列
    TextBox([500, 100, 600, 150], 0.9),  # 右列
    TextBox([510, 200, 610, 250], 0.9),  # 右列
    TextBox([105, 300, 205, 350], 0.9),  # 左列
    TextBox([505, 300, 605, 350], 0.9),  # 右列
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "double" and len(left) == 3 and len(right) == 3

# 测试4: 左对齐双列布局
print("\n测试4: 左对齐双列布局")
boxes = [
    TextBox([50, 100, 150, 150], 0.9),   # 左列
    TextBox([60, 200, 160, 250], 0.9),   # 左列
    TextBox([200, 100, 300, 150], 0.9),  # 右列
    TextBox([210, 200, 310, 250], 0.9),  # 右列
    TextBox([55, 300, 155, 350], 0.9),   # 左列
    TextBox([205, 300, 305, 350], 0.9),  # 右列
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "double_left" and len(left) == 3 and len(right) == 3

# 测试5: 右对齐双列布局
print("\n测试5: 右对齐双列布局")
boxes = [
    TextBox([420, 100, 520, 150], 0.9),  # 左列
    TextBox([430, 200, 530, 250], 0.9),  # 左列
    TextBox([570, 100, 670, 150], 0.9),  # 右列
    TextBox([580, 200, 680, 250], 0.9),  # 右列
    TextBox([425, 300, 525, 350], 0.9),  # 左列
    TextBox([575, 300, 675, 350], 0.9),  # 右列
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "double_right" and len(left) == 3 and len(right) == 3

# 测试6: 低分离度（应判定为单列）
print("\n测试6: 低分离度（应判定为单列）")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([120, 300, 220, 350], 0.9),
    TextBox([130, 400, 230, 450], 0.9),
    TextBox([140, 500, 240, 550], 0.9),
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert layout == "single" and len(left) == 5 and len(right) == 0

# 测试7: 列分配完整性
print("\n测试7: 列分配完整性")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([500, 100, 600, 150], 0.9),
    TextBox([510, 200, 610, 250], 0.9),
    TextBox([105, 300, 205, 350], 0.9),
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
if fallback_meta:
    print(f"  Fallback: {fallback_meta}")
assert len(left) + len(right) == len(boxes), "列分配完整性失败"

print("\n✅ 所有测试通过！")
