"""测试fallback机制"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector

class TextBox:
    def __init__(self, box, score):
        self.box = np.array(box)
        self.score = score
    
    @property
    def x_min(self): return self.box[0]
    @property
    def y_min(self): return self.box[1]
    @property
    def x_max(self): return self.box[2]
    @property
    def y_max(self): return self.box[3]
    @property
    def center_x(self): return (self.x_min + self.x_max) / 2
    @property
    def width(self): return self.x_max - self.x_min

# 测试1: 空memory时应该使用fallback
print("=" * 60)
print("测试1: 空memory时应该使用fallback")
print("=" * 60)
detector = ChatLayoutDetector(screen_width=720)
print(f"Memory A: {detector.memory['A']}")
print(f"Memory B: {detector.memory['B']}")
print(f"Should use fallback: {detector.should_use_fallback()}")
assert detector.should_use_fallback() == True, "空memory应该使用fallback"
print("✓ 通过")

# 测试2: 使用fallback方法分列
print("\n" + "=" * 60)
print("测试2: 使用fallback方法分列（双列布局）")
print("=" * 60)
boxes = [
    TextBox([105, 100, 205, 150], 0.9),  # 左列
    TextBox([115, 200, 215, 250], 0.9),  # 左列
    TextBox([505, 100, 605, 150], 0.9),  # 右列
    TextBox([515, 200, 615, 250], 0.9),  # 右列
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"布局: {layout}")
print(f"左列: {len(left)}, 右列: {len(right)}")
print(f"Fallback metadata: {fallback_meta}")
assert fallback_meta is not None, "应该有fallback metadata"
assert fallback_meta["method"] == "median_fallback", "方法应该是median_fallback"
assert "reason" in fallback_meta, "应该包含原因"
print("✓ 通过")

# 测试3: 单侧数据不强制分列
print("\n" + "=" * 60)
print("测试3: 单侧数据不强制分列")
print("=" * 60)
boxes = [
    TextBox([105, 100, 205, 150], 0.9),
    TextBox([115, 200, 215, 250], 0.9),
    TextBox([125, 300, 225, 350], 0.9),
    TextBox([135, 400, 235, 450], 0.9),
]
layout, left, right, fallback_meta = detector.split_columns(boxes)
print(f"布局: {layout}")
print(f"左列: {len(left)}, 右列: {len(right)}")
print(f"Fallback metadata: {fallback_meta}")
assert layout == "single", "单侧数据应该判定为单列"
assert fallback_meta is not None, "应该有fallback metadata"
assert fallback_meta["reason"] == "single_sided_data", "原因应该是single_sided_data"
print("✓ 通过")

# 测试4: 填充memory后不使用fallback
print("\n" + "=" * 60)
print("测试4: 填充memory后不使用fallback")
print("=" * 60)
detector2 = ChatLayoutDetector(screen_width=720)
# 手动填充memory
detector2.memory["A"] = {"center": 0.2, "width": 0.1, "count": 30}
detector2.memory["B"] = {"center": 0.8, "width": 0.1, "count": 25}
print(f"Memory A count: {detector2.memory['A']['count']}")
print(f"Memory B count: {detector2.memory['B']['count']}")
print(f"Total count: {detector2.memory['A']['count'] + detector2.memory['B']['count']}")
print(f"Should use fallback: {detector2.should_use_fallback()}")
assert detector2.should_use_fallback() == False, "有足够历史数据时不应该使用fallback"

boxes = [
    TextBox([105, 100, 205, 150], 0.9),  # 左列
    TextBox([115, 200, 215, 250], 0.9),  # 左列
    TextBox([505, 100, 605, 150], 0.9),  # 右列
    TextBox([515, 200, 615, 250], 0.9),  # 右列
]
layout, left, right, fallback_meta = detector2.split_columns(boxes)
print(f"布局: {layout}")
print(f"Fallback metadata: {fallback_meta}")
assert fallback_meta is None, "有足够历史数据时不应该有fallback metadata"
print("✓ 通过")

# 测试5: process_frame包含fallback信息
print("\n" + "=" * 60)
print("测试5: process_frame包含fallback信息")
print("=" * 60)
detector3 = ChatLayoutDetector(screen_width=720)
boxes = [
    TextBox([105, 100, 205, 150], 0.9),
    TextBox([115, 200, 215, 250], 0.9),
    TextBox([505, 100, 605, 150], 0.9),
    TextBox([515, 200, 615, 250], 0.9),
]
result = detector3.process_frame(boxes)
print(f"布局: {result['layout']}")
print(f"Metadata: {result['metadata']}")
assert "method" in result["metadata"], "metadata应该包含method字段"
assert result["metadata"]["method"] == "median_fallback", "方法应该是median_fallback"
assert "reason" in result["metadata"], "metadata应该包含reason字段"
print("✓ 通过")

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
