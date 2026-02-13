# 通用的聊天框分析工具，不依赖特定app类型
import numpy as np
import os
from typing import Dict, List
from sklearn.cluster import KMeans
import joblib

OUTPUT_PATH = 'history'
DURING_TEST = False
def concat_data(group_data_dict: List[Dict]):
    """合并多组数据"""
    data_lists = []
    for group_data in group_data_dict:
        padding = group_data['padding']
        text_boxes = group_data['text_boxes']
        image_sizes = group_data['image_sizes']
        data = np.zeros((text_boxes.shape[0], 4+4+2))

        data[:,:4] = text_boxes
        for i in range(4):
            data[:,4+i] = padding[i]
        
        for i in range(2):
            data[:,8+i] = image_sizes[i]
        
        data_lists.append(data)

    return np.concatenate(data_lists)


class SpeakerPositionKMeans:
    """基于KMeans的说话者位置聚类器（应用无关）"""
    
    def __init__(self):
        self.model = None
        self.left_center = None
        self.right_center = None
        self.data = None

    def fit(self, center_x_history, update=False):
        """
        训练KMeans模型
        
        Args:
            center_x_history: List[float] - 文本框中心X坐标历史数据
            update: bool - 是否更新现有模型
        """
        if not update:
            X = np.array(center_x_history).reshape(-1, 1)
        else:
            old_X = self.load_data()
            if old_X is not None:
                new_X = np.array(center_x_history).reshape(-1, 1)
                X = np.vstack([old_X, new_X])
            else:
                X = np.array(center_x_history).reshape(-1, 1)
        
        self.data = X
        self.model = KMeans(
            n_clusters=2,
            n_init="auto",
            random_state=42
        )
        self.model.fit(X)

        centers = sorted(self.model.cluster_centers_.flatten())
        self.left_center = centers[0]
        self.right_center = centers[1]
        self.save()

    def predict(self, center_x):
        """
        预测说话者类型
        
        Args:
            center_x: float - 文本框中心X坐标
            
        Returns:
            str - "left" 或 "right"
        """
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError("KMeans model not fitted")

        # 用物理位置而不是 cluster id 判断
        if abs(center_x - self.left_center) < abs(center_x - self.right_center):
            return "left"
        else:
            return "right"
    
    def save(self):
        """保存模型和数据"""
        output = OUTPUT_PATH
        if not os.path.exists(output):
            os.makedirs(output)
        joblib.dump(self.model, f'{output}/kmeans_model.joblib')
        np.save(f'{output}/text_box_center.npy', self.data)

    def load_data(self):
        """加载历史数据"""
        data_path = f'{OUTPUT_PATH}/text_box_center.npy'
        data = None
        if os.path.exists(data_path):
            try:
                data = np.load(data_path)
            except Exception:
                return None
        return data

    def load_model(self):
        """加载已训练的模型"""
        model_path = f'{OUTPUT_PATH}/kmeans_model.joblib'
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                # 重新计算中心点
                if self.model is not None:
                    centers = sorted(self.model.cluster_centers_.flatten())
                    self.left_center = centers[0]
                    self.right_center = centers[1]
            except Exception:
                self.model = None
        else:
            self.model = None


def use_center_x_split_talker_and_user():
    """使用中心X坐标分割说话者（占位函数）"""
    pass