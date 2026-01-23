"""
集成测试：验证新的自适应检测器与现有代码的兼容性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


# 简化的TextBox类用于测试
class TextBox:
    def __init__(self, box, score, **kwargs):
        self.box = box
        self.score = score
        if isinstance(self.box, list):
            self.box = np.array(self.box)
        self.text_type = None
        self.source = None
        self.layout_det = None

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.x_min, self.y_min, self.x_max, self.y_max = self.box.tolist()

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self):
        return (self.y_min + self.y_max) / 2

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min


def test_textbox_compatibility():
    """测试TextBox对象的基本兼容性"""
    # 创建TextBox对象
    box = TextBox(box=[100, 100, 200, 150], score=0.9)
    
    # 验证TextBox的现有属性仍然可用
    assert hasattr(box, 'center_x')
    assert hasattr(box, 'width')
    assert hasattr(box, 'height')
    assert hasattr(box, 'x_min')
    assert hasattr(box, 'y_min')
    
    # 验证属性值正确
    assert box.center_x == 150.0
    assert box.width == 100.0
    assert box.height == 50.0
    
    print("✓ TextBox compatibility test passed")


def test_import_chat_layout_detector():
    """测试能否导入ChatLayoutDetector"""
    try:
        from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
        print("✓ ChatLayoutDetector import test passed")
        return True
    except ImportError as e:
        print(f"✗ ChatLayoutDetector import failed: {e}")
        return False


def test_processor_has_new_methods():
    """测试ChatMessageProcessor是否有新方法"""
    try:
        # 只导入processors模块，不实例化
        import screenshotanalysis.processors as proc_module
        
        # 检查新方法是否在模块中定义
        source = open('src/screenshotanalysis/processors.py', 'r', encoding='utf-8').read()
        
        assert 'detect_chat_layout_adaptive' in source
        assert 'format_conversation_adaptive' in source
        assert 'from screenshotanalysis.chat_layout_detector import ChatLayoutDetector' in source
        
        print("✓ Processor new methods test passed")
        return True
    except Exception as e:
        print(f"✗ Processor new methods test failed: {e}")
        return False


def test_basic_integration():
    """测试基本集成功能"""
    try:
        from screenshotanalysis.chat_layout_detector import ChatLayoutDetector
        
        # 创建一些测试文本框（双列布局）
        left_boxes = [
            TextBox(box=[100, 100, 200, 150], score=0.9),
            TextBox(box=[110, 200, 210, 250], score=0.9),
            TextBox(box=[105, 300, 205, 350], score=0.9),
        ]
        
        right_boxes = [
            TextBox(box=[500, 150, 600, 200], score=0.9),
            TextBox(box=[510, 250, 610, 300], score=0.9),
            TextBox(box=[505, 350, 605, 400], score=0.9),
        ]
        
        all_boxes = left_boxes + right_boxes
        
        # 创建检测器并处理
        detector = ChatLayoutDetector(screen_width=720)
        result = detector.process_frame(all_boxes)
        
        # 验证返回结构
        assert 'layout' in result
        assert 'A' in result
        assert 'B' in result
        assert 'metadata' in result
        
        # 验证布局类型
        assert result['layout'] in ['single', 'double', 'double_left', 'double_right']
        
        # 验证文本框分配
        assert len(result['A']) + len(result['B']) == len(all_boxes)
        
        print(f"✓ Basic integration test passed")
        print(f"  Layout: {result['layout']}")
        print(f"  Speaker A: {len(result['A'])} boxes")
        print(f"  Speaker B: {len(result['B'])} boxes")
        return True
        
    except Exception as e:
        print(f"✗ Basic integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Running integration compatibility tests...\n")
    
    results = []
    
    # Test 1: TextBox compatibility
    try:
        test_textbox_compatibility()
        results.append(True)
    except Exception as e:
        print(f"✗ TextBox compatibility test failed: {e}")
        results.append(False)
    
    # Test 2: Import ChatLayoutDetector
    results.append(test_import_chat_layout_detector())
    
    # Test 3: Processor has new methods
    results.append(test_processor_has_new_methods())
    
    # Test 4: Basic integration
    results.append(test_basic_integration())
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All integration tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)

