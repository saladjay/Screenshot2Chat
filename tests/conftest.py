# conftest.py
def pytest_collection_modifyitems(session, config, items):
    """
    在所有测试用例被收集后，执行前调用。items参数包含了所有测试用例的列表。
    """
    # 定义你期望的模块执行优先级
    file_priority = {
        "test_core.py": 0,
        "test_experience_formula.py": 1,
        "test_layout_analysis.py": 2,
        "test_recognition.py": 3,
        "test_chat_analysis.py":4,
    }
    
    # 对测试用例进行排序
    items.sort(key=lambda item: file_priority.get(item.module.__name__, 999))