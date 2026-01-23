"""
集成兼容性测试

测试新的ChatLayoutDetector与现有代码的兼容性，确保：
1. 新旧方法可以共存
2. TextBox对象在新旧代码间传递
3. 现有测试用例仍然通过

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import pytest
import numpy as np
from screenshotanalysis.processors import ChatMessageProcessor, TextBox
from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
import tempfile
import os


class TestIntegrationCompatibility:
    """测试新旧方法的集成兼容性"""
    
    def test_new_and_old_methods_coexist(self):
        """
        测试新旧方法可以共存
        
        验证ChatMessageProcessor类同时拥有旧的方法和新的自适应方法，
        并且两者可以独立使用而不会相互干扰。
        
        Requirements: 7.1, 7.2
        """
        processor = ChatMessageProcessor()
        
        # 验证旧方法仍然存在
        assert hasattr(processor, 'format_conversation'), \
            "旧的format_conversation方法应该仍然存在"
        assert hasattr(processor, 'sort_boxes_by_y'), \
            "旧的sort_boxes_by_y方法应该仍然存在"
        assert hasattr(processor, 'estimate_main_value'), \
            "旧的estimate_main_value方法应该仍然存在"
        
        # 验证新方法已添加
        assert hasattr(processor, 'detect_chat_layout_adaptive'), \
            "新的detect_chat_layout_adaptive方法应该存在"
        assert hasattr(processor, 'format_conversation_adaptive'), \
            "新的format_conversation_adaptive方法应该存在"
        
        # 验证两个方法可以独立调用
        # 创建测试数据
        boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[500, 200, 600, 250], score=0.9),
            TextBox(box=[110, 300, 210, 350], score=0.9),
            TextBox(box=[510, 400, 610, 450], score=0.9),
        ]
        
        # 调用新方法
        result = processor.detect_chat_layout_adaptive(boxes, screen_width=720)
        assert 'layout' in result, "新方法应该返回layout字段"
        assert 'A' in result, "新方法应该返回A字段"
        assert 'B' in result, "新方法应该返回B字段"
        
        # 调用旧方法（sort_boxes_by_y）
        sorted_boxes = processor.sort_boxes_by_y(boxes)
        assert len(sorted_boxes) == len(boxes), "旧方法应该正常工作"
        assert sorted_boxes[0].y_min <= sorted_boxes[-1].y_min, \
            "旧方法应该按y坐标排序"
    
    def test_textbox_compatibility_between_old_and_new(self):
        """
        测试TextBox对象在新旧代码间传递
        
        验证TextBox对象可以在旧代码和新代码之间传递，
        并且所有必需的属性都能正确访问。
        
        Requirements: 7.3, 7.4
        """
        # 创建TextBox对象（使用旧代码的方式）
        box = TextBox(box=[100, 200, 300, 250], score=0.95)
        
        # 验证TextBox具有新代码需要的所有属性
        assert hasattr(box, 'center_x'), "TextBox应该有center_x属性"
        assert hasattr(box, 'width'), "TextBox应该有width属性"
        assert hasattr(box, 'height'), "TextBox应该有height属性"
        assert hasattr(box, 'box'), "TextBox应该有box属性"
        assert hasattr(box, 'score'), "TextBox应该有score属性"
        
        # 验证属性值正确
        assert box.center_x == 200.0, "center_x计算应该正确"
        assert box.width == 200.0, "width计算应该正确"
        assert box.height == 50.0, "height计算应该正确"
        
        # 将TextBox传递给新代码
        detector = ChatLayoutDetector(screen_width=720)
        boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[500, 200, 600, 250], score=0.9),
            TextBox(box=[110, 300, 210, 350], score=0.9),
            TextBox(box=[510, 400, 610, 450], score=0.9),
        ]
        
        # 新代码应该能够处理这些TextBox对象
        result = detector.process_frame(boxes)
        assert result is not None, "新代码应该能处理旧的TextBox对象"
        assert 'layout' in result, "新代码应该返回正确的结果格式"
        
        # 验证返回的TextBox对象仍然可以被旧代码使用
        processor = ChatMessageProcessor()
        returned_boxes = result['A'] + result['B']
        sorted_boxes = processor.sort_boxes_by_y(returned_boxes)
        assert len(sorted_boxes) == len(boxes), "旧代码应该能处理新代码返回的TextBox"
    
    def test_textbox_properties_preserved(self):
        """
        测试TextBox属性在新旧代码间传递时保持不变
        
        验证TextBox对象通过新代码处理后，其属性值不会被修改。
        
        Requirements: 7.3, 7.4
        """
        # 创建带有额外属性的TextBox
        box = TextBox(box=[100, 200, 300, 250], score=0.95)
        box.text_type = 'test_type'
        box.source = 'test_source'
        box.layout_det = 'text'
        
        # 记录原始值
        original_box_array = box.box.copy()
        original_score = box.score
        original_text_type = box.text_type
        original_source = box.source
        original_layout_det = box.layout_det
        
        # 通过新代码处理
        detector = ChatLayoutDetector(screen_width=720)
        boxes = [box]
        result = detector.process_frame(boxes)
        
        # 验证属性未被修改
        assert np.array_equal(box.box, original_box_array), \
            "box坐标不应该被修改"
        assert box.score == original_score, "score不应该被修改"
        assert box.text_type == original_text_type, "text_type不应该被修改"
        assert box.source == original_source, "source不应该被修改"
        assert box.layout_det == original_layout_det, "layout_det不应该被修改"
    
    def test_existing_processor_methods_still_work(self):
        """
        测试现有的ChatMessageProcessor方法仍然正常工作
        
        验证添加新方法后，旧的方法仍然能够正常执行。
        
        Requirements: 7.1, 7.2
        """
        processor = ChatMessageProcessor()
        
        # 测试sort_boxes_by_y
        boxes = [
            TextBox(box=[100, 300, 200, 350], score=0.9),
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[100, 200, 200, 250], score=0.9),
        ]
        sorted_boxes = processor.sort_boxes_by_y(boxes)
        assert sorted_boxes[0].y_min == 100, "第一个应该是y_min最小的"
        assert sorted_boxes[1].y_min == 200, "第二个应该是y_min中等的"
        assert sorted_boxes[2].y_min == 300, "第三个应该是y_min最大的"
        
        # 测试estimate_main_value
        boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),  # height=50
            TextBox(box=[100, 200, 200, 252], score=0.9),  # height=52
            TextBox(box=[100, 300, 200, 348], score=0.9),  # height=48
            TextBox(box=[100, 400, 200, 450], score=0.9),  # height=50
        ]
        main_height = processor.estimate_main_value(boxes, 'height', bin_size=2)
        assert main_height is not None, "应该能计算主高度"
        assert 48 <= main_height <= 52, "主高度应该在合理范围内"
    
    def test_adaptive_method_returns_compatible_format(self):
        """
        测试自适应方法返回与现有代码兼容的格式
        
        验证新的自适应方法返回的数据结构可以被现有代码使用。
        
        Requirements: 7.2, 7.3
        """
        processor = ChatMessageProcessor()
        
        # 创建测试数据
        boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[500, 200, 600, 250], score=0.9),
            TextBox(box=[110, 300, 210, 350], score=0.9),
            TextBox(box=[510, 400, 610, 450], score=0.9),
        ]
        
        # 调用自适应方法
        result = processor.detect_chat_layout_adaptive(boxes, screen_width=720)
        
        # 验证返回格式
        assert isinstance(result, dict), "应该返回字典"
        assert 'layout' in result, "应该包含layout字段"
        assert 'A' in result, "应该包含A字段"
        assert 'B' in result, "应该包含B字段"
        assert 'metadata' in result, "应该包含metadata字段"
        
        # 验证返回的TextBox列表可以被旧代码使用
        all_boxes = result['A'] + result['B']
        assert len(all_boxes) == len(boxes), "所有文本框都应该被分配"
        
        # 验证可以使用旧方法处理返回的boxes
        sorted_boxes = processor.sort_boxes_by_y(all_boxes)
        assert len(sorted_boxes) == len(all_boxes), "旧方法应该能处理新方法返回的boxes"
    
    def test_format_conversation_adaptive_compatibility(self):
        """
        测试format_conversation_adaptive与现有代码的兼容性
        
        验证新的format_conversation_adaptive方法返回的格式
        与旧的format_conversation方法兼容。
        
        Requirements: 7.2, 7.3
        """
        processor = ChatMessageProcessor()
        
        # 创建测试数据
        boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[500, 200, 600, 250], score=0.9),
            TextBox(box=[110, 300, 210, 350], score=0.9),
            TextBox(box=[510, 400, 610, 450], score=0.9),
        ]
        
        # 调用新的format_conversation_adaptive
        sorted_boxes, metadata = processor.format_conversation_adaptive(
            boxes, screen_width=720
        )
        
        # 验证返回格式
        assert isinstance(sorted_boxes, list), "应该返回列表"
        assert isinstance(metadata, dict), "应该返回元数据字典"
        assert len(sorted_boxes) == len(boxes), "所有文本框都应该被返回"
        
        # 验证文本框已按y坐标排序
        for i in range(len(sorted_boxes) - 1):
            assert sorted_boxes[i].y_min <= sorted_boxes[i+1].y_min, \
                "文本框应该按y坐标排序"
        
        # 验证每个文本框都有speaker标记
        for box in sorted_boxes:
            assert hasattr(box, 'speaker'), "每个文本框应该有speaker属性"
            assert box.speaker in ['A', 'B'], "speaker应该是A或B"
        
        # 验证元数据包含必要信息
        assert 'layout' in metadata, "元数据应该包含layout"
        assert 'speaker_A_count' in metadata, "元数据应该包含speaker_A_count"
        assert 'speaker_B_count' in metadata, "元数据应该包含speaker_B_count"
    
    def test_memory_persistence_does_not_affect_old_code(self):
        """
        测试记忆持久化不影响旧代码
        
        验证新代码的记忆持久化功能不会影响旧代码的执行。
        
        Requirements: 7.1, 7.2
        """
        processor = ChatMessageProcessor()
        
        # 创建临时记忆文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # 使用新方法处理数据（会创建记忆文件）
            boxes = [
                TextBox(box=[100, 100, 200, 150], score=0.9),
                TextBox(box=[500, 200, 600, 250], score=0.9),
            ]
            
            result1 = processor.detect_chat_layout_adaptive(
                boxes, screen_width=720, memory_path=temp_path
            )
            
            # 验证记忆文件被创建
            assert os.path.exists(temp_path), "记忆文件应该被创建"
            
            # 使用旧方法处理相同数据
            sorted_boxes = processor.sort_boxes_by_y(boxes)
            
            # 验证旧方法不受影响
            assert len(sorted_boxes) == len(boxes), "旧方法应该正常工作"
            
            # 再次使用新方法，验证记忆被加载
            result2 = processor.detect_chat_layout_adaptive(
                boxes, screen_width=720, memory_path=temp_path
            )
            
            # 验证两次结果一致（说明记忆正常工作）
            assert result2['layout'] == result1['layout'], \
                "使用记忆后结果应该一致"
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_textbox_with_numpy_array_box(self):
        """
        测试TextBox的box属性为numpy数组时的兼容性
        
        验证无论box是list还是numpy数组，新旧代码都能正确处理。
        
        Requirements: 7.3, 7.4
        """
        # 使用numpy数组创建TextBox
        box_array = np.array([100, 200, 300, 250])
        box1 = TextBox(box=box_array, score=0.9)
        
        # 使用list创建TextBox
        box_list = [100, 200, 300, 250]
        box2 = TextBox(box=box_list, score=0.9)
        
        # 验证两种方式创建的TextBox都能被新代码处理
        detector = ChatLayoutDetector(screen_width=720)
        
        result1 = detector.process_frame([box1])
        result2 = detector.process_frame([box2])
        
        assert result1 is not None, "numpy数组box应该能被处理"
        assert result2 is not None, "list box应该能被处理"
        
        # 验证两种方式创建的TextBox都能被旧代码处理
        processor = ChatMessageProcessor()
        
        sorted1 = processor.sort_boxes_by_y([box1])
        sorted2 = processor.sort_boxes_by_y([box2])
        
        assert len(sorted1) == 1, "numpy数组box应该能被旧代码处理"
        assert len(sorted2) == 1, "list box应该能被旧代码处理"
    
    def test_empty_input_compatibility(self):
        """
        测试空输入的兼容性
        
        验证新旧代码都能正确处理空输入。
        
        Requirements: 7.1, 7.2, 7.3
        """
        processor = ChatMessageProcessor()
        
        # 测试旧方法处理空输入
        empty_boxes = []
        sorted_boxes = processor.sort_boxes_by_y(empty_boxes)
        assert sorted_boxes == [], "旧方法应该能处理空输入"
        
        # 测试新方法处理空输入
        result = processor.detect_chat_layout_adaptive(empty_boxes, screen_width=720)
        assert result['layout'] == 'single', "空输入应该判定为单列"
        assert result['A'] == [], "空输入的A列应该为空"
        assert result['B'] == [], "空输入的B列应该为空"
    
    def test_single_box_compatibility(self):
        """
        测试单个文本框的兼容性
        
        验证新旧代码都能正确处理只有一个文本框的情况。
        
        Requirements: 7.1, 7.2, 7.3
        """
        processor = ChatMessageProcessor()
        
        single_box = [TextBox(box=[100, 100, 200, 150], score=0.9)]
        
        # 测试旧方法
        sorted_boxes = processor.sort_boxes_by_y(single_box)
        assert len(sorted_boxes) == 1, "旧方法应该能处理单个文本框"
        
        # 测试新方法
        result = processor.detect_chat_layout_adaptive(single_box, screen_width=720)
        assert result['layout'] == 'single', "单个文本框应该判定为单列"
        assert len(result['A']) + len(result['B']) == 1, \
            "单个文本框应该被分配到某一列"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
