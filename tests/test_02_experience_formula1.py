from chat_layout_analyzer.experience_formula import *
from chat_layout_analyzer.utils import DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM 
from chat_layout_analyzer import ChatLayoutAnalyzer, ChatMessageProcessor
import numpy as np
import random

text_box_test_data = [
    [171, 728, 279, 748], 
    [66, 657, 214, 677], 
    [67, 635, 183, 655], 
    [67, 593, 219, 614], 
    [68, 571, 203, 585], 
    [67, 549, 136, 567], 
    [68, 507, 234, 524], 
    [67, 484, 184, 504], 
    [68, 443, 195, 460], 
    [65, 421, 137, 442], 
    [67, 381, 155, 398], 
    [67, 358, 184, 378], 
    [66, 316, 243, 333], 
    [64, 289, 175, 312], 
    [66, 271, 136, 289],
    [66, 228, 224, 247], 
    [19, 55, 55, 96], 
    [100, 54, 190, 81], 
    [30, 12, 75, 30 ],
]

update_box_test_data = [
    [171, 728, 279, 748],
    [67, 671, 220, 690], 
    [68, 646, 205, 663], 
    [67, 625, 135, 643], 
    [67, 582, 234, 601], 
    [69, 562, 183, 579], 
    [67, 519, 194, 536], 
    [67, 499, 135, 517], 
    [68, 457, 155, 475], 
    [68, 436, 183, 453], 
    [66, 394, 243, 411], 
    [64, 366, 176, 388], 
    [66, 347, 136, 365], 
    [67, 304, 224, 323], 
    [66, 115, 185, 138], 
    [18, 54, 55, 97], 
    [100, 54, 190, 81], 
    [30, 12, 75, 31,]
]


class TestExperienceFormula:
    def test_init(self):
        global text_box_test_data
        for app in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
            padding = random.randint(1,32)
            padding = [padding, 0, padding, 0]
            image_sizes = [random.choice([448,480,512]), 800]
            text_box_test_data = np.array(text_box_test_data)
            init_data(padding, text_box_test_data, image_sizes, app)

            assert os.path.exists(f'{OUTPUT_PATH}/{app}/data.npy')
            assert os.path.exists(f'{OUTPUT_PATH}/{app}/ratios.txt')

    def test_load(self):
        for app in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
            loaded_data = load_data(app)
            assert isinstance(loaded_data, tuple)
            assert len(loaded_data) == 2
            assert loaded_data[0] is not None and isinstance(loaded_data[0], np.ndarray)
            assert loaded_data[1] is not None and isinstance(loaded_data[1], np.ndarray)

    def test_update(self):
        global update_box_test_data
        for app in [DISCORD, WHATSAPP, INSTAGRAM, TELEGRAM]:
            padding = random.randint(1,32)
            padding = [padding, 0, padding, 0]
            image_sizes = [random.choice([448,480,512]), 800]
            update_box_test_data = np.array(update_box_test_data)
            update_data(padding, update_box_test_data, image_sizes, app)
            self.test_load()

    
    