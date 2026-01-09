# 使用经验公式的方式处理discord， WhatsApp， Instagram， telegram这四个app聊天框的分析
import numpy as np
import yaml
import os
from functools import reduce
from sklearn.cluster import KMeans
import joblib
from screenshotanalysis.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM

USER_LEFT = 'user_box_left_filters'
USER_RIGHT = 'user_box_right_filters'
TALKER_LEFT = 'talker_box_left_filters'
TALKER_RIGHT = 'talker_box_right_filters'

OUTPUT_PATH = 'history'
DURING_TEST = False
def calculate_condition_by_yaml_config(left_ratio, right_ratio, conversation_app_type):
    with open('conversation_analysis_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")

    def use_filters(filters, ratio, filter_reuslts):
        for f in filters:
            if f == "none":
                continue
            f_type, f_value = f.split("_")
            if f_type == 'lt':
                filter_reuslts.append(ratio < float(f_value))
            elif f_type == 'gt':
                filter_reuslts.append(ratio > float(f_value))
            else:
                raise NotImplementedError(f'conversation_analysis_config.yaml里{conversation_app_type}中存在未实现的filter')

    filter_reuslts = []
    left_filters = config[conversation_app_type][USER_LEFT]
    right_filters = config[conversation_app_type][USER_RIGHT]
    use_filters(left_filters, left_ratio, filter_reuslts)
    use_filters(right_filters, right_ratio, filter_reuslts)

    user_condition = reduce(np.logical_and, filter_reuslts)
    user_left_start = left_ratio[user_condition]
    user_right_end = right_ratio[user_condition]

    filter_reuslts = []
    left_filters = config[conversation_app_type][TALKER_LEFT]
    right_filters = config[conversation_app_type][TALKER_RIGHT]
    use_filters(left_filters, left_ratio, filter_reuslts)
    use_filters(right_filters, right_ratio, filter_reuslts)

    talker_condition = reduce(np.logical_and, filter_reuslts)
    talker_left_start = left_ratio[talker_condition]
    talker_right_end = right_ratio[talker_condition]

    return np.median(user_left_start), np.median(user_right_end), np.median(talker_left_start), np.median(talker_right_end)

def reinit_data(conversation_app_type):
    data = np.zeros((0, 10))
    ratios = np.zeros((0, 4))
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")

    output = f'{OUTPUT_PATH}/{conversation_app_type}'
    if os.path.exists(output) == False:
        os.makedirs(output)
    np.save(f'{output}/data.npy', data)
    np.savetxt(f'{output}/ratios.txt', np.array(ratios), fmt='%.3f', delimiter=',')

def init_data(padding, text_boxes, image_sizes, conversation_app_type):
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")

    output = f'{OUTPUT_PATH}/{conversation_app_type}'
    if not os.path.exists(f'{output}'):
        os.makedirs(output)
    data = np.zeros((text_boxes.shape[0], 4+4+2)) # [min_x, min_y, max_x, max_y, l, t, r, b, w, h]
                                                #  0      1      2      3      4  5  6  7  8  9
    data[:,:4] = text_boxes
    for i in range(4):
        data[:,4+i] = padding[i]
    
    for i in range(2):
        data[:,8+i] = image_sizes[i]

    image_width_wo_padding = data[:, 8] - data[:, 4] - data[:, 6]
    text_box_left = data[:, 0] - data[:, 4]
    text_box_left_ratio = text_box_left / (image_width_wo_padding + 1e-9)

    text_box_right = data[:, 2] - data[:, 4]
    text_box_right_ratio = text_box_right / (image_width_wo_padding + 1e-9)

    user_left, user_right, talker_left, talker_right = calculate_condition_by_yaml_config(text_box_left_ratio, text_box_right_ratio, conversation_app_type)

    save_data(data, [user_left, user_right, talker_left, talker_right], conversation_app_type)
    
def load_data(conversation_app_type):
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")
    output = f'{OUTPUT_PATH}/{conversation_app_type}'
    data_path = f"{output}/data.npy"
    ratios_path = f'{output}/ratios.txt'
    if not all([os.path.exists(output), os.path.exists(data_path), os.path.exists(ratios_path)]) :
        raise FileNotFoundError(f"{output}, {data_path}或{ratios_path}路径不存在")
    try:
        data = np.load(data_path)
    except Exception as e:
        raise e

    try:
        ratios = np.loadtxt(ratios_path, delimiter=',')
    except Exception as e:
        raise e
    return data, ratios

def save_data(data, ratios, conversation_app_type):
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")

    output = f'{OUTPUT_PATH}/{conversation_app_type}'
    if os.path.exists(output) == False:
        os.makedirs(output)
    
    np.save(f'{output}/data.npy', data)
    np.savetxt(f'{output}/ratios.txt', np.array(ratios), fmt='%.3f', delimiter=',')

def update_data(padding, text_boxes, image_sizes, conversation_app_type): # l,t,r,b   [n,4]    w, h
    if conversation_app_type not in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
        raise NotImplementedError(f"未实现{conversation_app_type}的配置")
    data = np.zeros((text_boxes.shape[0], 4+4+2)) # [min_x, min_y, max_x, max_y, l, t, r, b, w, h]
                                                #  0      1      2      3      4  5  6  7  8  9
    data[:,:4] = text_boxes
    for i in range(4):
        data[:,4+i] = padding[i]
    
    for i in range(2):
        data[:,8+i] = image_sizes[i]

    old_data, _ = load_data(conversation_app_type)
    new_data = np.zeros((old_data.shape[0] + data.shape[0], 10))
    new_data[:old_data.shape[0],:] = old_data
    new_data[old_data.shape[0]:,:] = data

    image_width_wo_padding = new_data[:, 8] - new_data[:, 4] - new_data[:, 6]
    text_box_left = new_data[:, 0] - new_data[:, 4]
    text_box_left_ratio = text_box_left / (image_width_wo_padding + 1e-9)

    text_box_right = new_data[:, 2] - new_data[:, 4]
    text_box_right_ratio = text_box_right / (image_width_wo_padding + 1e-9)

    user_left, user_right, talker_left, talker_right = calculate_condition_by_yaml_config(text_box_left_ratio, text_box_right_ratio, conversation_app_type)

    save_data(new_data, [user_left, user_right, talker_left, talker_right], conversation_app_type)


class SpeakerPositionKMeans:
    def __init__(self):
        self.model = None
        self.left_center = None
        self.right_center = None
        self.data = None
        self.app_type = None

    def fit(self, center_x_history, conversation_app_type, update=False):
        """
        center_x_history: List[[center_x]]
        """
        if not update:
            X = np.array(center_x_history)
        else:
            old_X = self.load_data()
            if old_X is not None:
                new_X = np.array(center_x_history)
                X = np.vstack([old_X, new_X])
            else:
                X = np.array(center_x_history)
        self.data = X
        self.app_type = conversation_app_type
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
        输入单个 center_x，输出 talker / user
        """
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError("KMeans model not fitted")

        cluster = self.model.predict([[center_x]])[0]

        # 用物理位置而不是 cluster id 判断
        if abs(center_x - self.left_center) < abs(center_x - self.right_center):
            return "talker"
        else:
            return "user"
    
    def save(self):
        output = f'{OUTPUT_PATH}/{conversation_app_type}'
        if os.path.exists(output) == False:
            os.makedirs(output)
        joblib.dump(self.model, f'{output}/kmeans_model.joblib')
        old_data = np.save(f'{output}/text_box_center.npy', self.data)

    def load_data(self):
        output = f'{OUTPUT_PATH}/{conversation_app_type}'
        data_path = f'{output}/text_box_center.npy'
        data = None
        if os.path.exists(data_path):
            try:
                data = np.load(data_path)
            except Exception as e:
                return None
        return data


    def load_model(self):
        output = f'{OUTPUT_PATH}/{conversation_app_type}'
        model_path = f'{output}/kmeans_model.joblib'
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                self.model = None
        else:
            self.model = None

def use_center_x_split_talker_and_user():
    pass