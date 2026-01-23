"""简单测试split_columns方法 - 不依赖完整环境"""
import numpy as np
from sklearn.cluster import KMeans

# 模拟TextBox类
class TextBox:
    def __init__(self, box, score):
        self.box = np.array(box) if isinstance(box, list) else box
        self.score = score
        self.x_min, self.y_min, self.x_max, self.y_max = self.box.tolist()
    
    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2
    
    @property
    def width(self):
        return self.x_max - self.x_min

# 简化的split_columns实现（直接复制）
def split_columns(boxes, screen_width=720, min_separation_ratio=0.18):
    """测试版本的split_columns"""
    if not boxes:
        return "single", [], []
    
    # 1. 提取并归一化center_x
    center_x_values = np.array([box.center_x for box in boxes])
    normalized_centers = center_x_values / screen_width
    
    # 2. 如果样本数<4，判定为单列
    if len(boxes) < 4:
        return "single", list(boxes), []
    
    # 3. 使用KMeans(n_clusters=2)聚类
    try:
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(normalized_centers.reshape(-1, 1))
        
        # 获取聚类中心并排序（左到右）
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())
        left_center, right_center = cluster_centers[0], cluster_centers[1]
        
        # 4. 计算分离度
        separation_ratio = right_center - left_center
        
        # 5. 如果分离度<min_separation_ratio，判定为单列
        if separation_ratio < min_separation_ratio:
            return "single", list(boxes), []
        
        # 6. 判定为双列，并根据聚类中心位置判断子类型
        if left_center < 0.5 and right_center < 0.5:
            layout_type = "double_left"
        elif left_center > 0.5 and right_center > 0.5:
            layout_type = "double_right"
        else:
            layout_type = "double"
        
        # 7. 将文本框分配到左列或右列
        left_boxes = []
        right_boxes = []
        
        for box, norm_center in zip(boxes, normalized_centers):
            dist_to_left = abs(norm_center - left_center)
            dist_to_right = abs(norm_center - right_center)
            
            if dist_to_left <= dist_to_right:
                left_boxes.append(box)
            else:
                right_boxes.append(box)
        
        return layout_type, left_boxes, right_boxes
        
    except Exception as e:
        print(f"KMeans聚类失败: {e}")
        return "single", list(boxes), []

# 运行测试
print("=" * 60)
print("测试split_columns方法")
print("=" * 60)

# 测试1: 空列表
print("\n测试1: 空列表")
layout, left, right = split_columns([])
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert layout == "single" and len(left) == 0 and len(right) == 0
print("  ✅ 通过")

# 测试2: 少于4个文本框
print("\n测试2: 少于4个文本框")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([105, 300, 205, 350], 0.9)
]
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert layout == "single" and len(left) == 3 and len(right) == 0
print("  ✅ 通过")

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
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
print(f"  左列center_x: {[b.center_x for b in left]}")
print(f"  右列center_x: {[b.center_x for b in right]}")
assert layout == "double" and len(left) == 3 and len(right) == 3
print("  ✅ 通过")

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
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert layout == "double_left" and len(left) == 3 and len(right) == 3
print("  ✅ 通过")

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
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert layout == "double_right" and len(left) == 3 and len(right) == 3
print("  ✅ 通过")

# 测试6: 低分离度（应判定为单列）
print("\n测试6: 低分离度（应判定为单列）")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([120, 300, 220, 350], 0.9),
    TextBox([130, 400, 230, 450], 0.9),
    TextBox([140, 500, 240, 550], 0.9),
]
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert layout == "single" and len(left) == 5 and len(right) == 0
print("  ✅ 通过")

# 测试7: 列分配完整性
print("\n测试7: 列分配完整性")
boxes = [
    TextBox([100, 100, 200, 150], 0.9),
    TextBox([110, 200, 210, 250], 0.9),
    TextBox([500, 100, 600, 150], 0.9),
    TextBox([510, 200, 610, 250], 0.9),
    TextBox([105, 300, 205, 350], 0.9),
]
layout, left, right = split_columns(boxes)
print(f"  布局: {layout}, 左列: {len(left)}, 右列: {len(right)}")
assert len(left) + len(right) == len(boxes), "列分配完整性失败"
print("  ✅ 通过")

# 测试8: center_x归一化范围
print("\n测试8: center_x归一化范围")
boxes = [
    TextBox([0, 100, 100, 150], 0.9),
    TextBox([620, 200, 720, 250], 0.9),
    TextBox([300, 300, 400, 350], 0.9),
    TextBox([50, 400, 150, 450], 0.9),
]
center_x_values = np.array([box.center_x for box in boxes])
normalized_centers = center_x_values / 720
print(f"  归一化center_x: {normalized_centers}")
assert np.all((normalized_centers >= 0) & (normalized_centers <= 1))
print("  ✅ 通过")

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
