"""ConfigManager使用示例

演示如何使用ConfigManager进行配置管理
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.screenshot2chat.config import ConfigManager


def demo_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("1. 基本使用")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 设置配置
    config.set('app.name', 'Screenshot Analysis')
    config.set('app.version', '2.0.0')
    config.set('app.debug', False)
    
    # 获取配置
    print(f"应用名称: {config.get('app.name')}")
    print(f"应用版本: {config.get('app.version')}")
    print(f"调试模式: {config.get('app.debug')}")
    print(f"不存在的键: {config.get('app.nonexistent', '默认值')}")
    print()


def demo_layered_config():
    """分层配置示例"""
    print("=" * 60)
    print("2. 分层配置")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 在不同层级设置配置
    config.set('model.backend', 'paddleocr', layer='default')
    config.set('model.backend', 'tesseract', layer='user')
    config.set('model.backend', 'easyocr', layer='runtime')
    
    print(f"当前后端 (runtime优先): {config.get('model.backend')}")
    
    # 清除runtime层
    config.clear('runtime')
    print(f"清除runtime后 (user优先): {config.get('model.backend')}")
    
    # 清除user层
    config.clear('user')
    print(f"清除user后 (default优先): {config.get('model.backend')}")
    print()


def demo_file_operations():
    """文件操作示例"""
    print("=" * 60)
    print("3. 文件操作 (保存/加载)")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 设置一些配置
    config.set('pipeline.name', 'chat_analysis')
    config.set('pipeline.steps', ['text_detection', 'bubble_detection', 'nickname_extraction'])
    config.set('detector.text.backend', 'paddleocr')
    config.set('detector.text.model_dir', 'models/PP-OCRv5_server_det/')
    config.set('detector.bubble.screen_width', 720)
    
    # 保存为JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    config.save(json_path, layer='runtime')
    print(f"✓ 配置已保存到: {json_path}")
    
    # 保存为YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
    
    config.save(yaml_path, layer='runtime')
    print(f"✓ 配置已保存到: {yaml_path}")
    
    # 加载配置
    config2 = ConfigManager()
    config2.load(json_path, layer='user')
    print(f"✓ 从JSON加载: pipeline.name = {config2.get('pipeline.name')}")
    
    config3 = ConfigManager()
    config3.load(yaml_path, layer='user')
    print(f"✓ 从YAML加载: pipeline.name = {config3.get('pipeline.name')}")
    
    # 清理临时文件
    Path(json_path).unlink()
    Path(yaml_path).unlink()
    print()


def demo_validation():
    """配置验证示例"""
    print("=" * 60)
    print("4. 配置验证")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 设置配置
    config.set('server.port', 8080)
    config.set('server.host', 'localhost')
    config.set('server.workers', 4)
    config.set('server.timeout', 30)
    
    # 定义验证schema
    schema = {
        'server.port': {
            'type': int,
            'required': True,
            'min': 1,
            'max': 65535
        },
        'server.host': {
            'type': str,
            'required': True
        },
        'server.workers': {
            'type': int,
            'required': True,
            'min': 1,
            'max': 16
        },
        'server.timeout': {
            'type': int,
            'required': False,
            'min': 1,
            'max': 300
        }
    }
    
    # 验证配置
    if config.validate(schema):
        print("✓ 配置验证通过")
    else:
        print("✗ 配置验证失败")
    
    # 设置无效值
    config.set('server.port', 70000)  # 超出范围
    if config.validate(schema):
        print("✓ 配置验证通过")
    else:
        print("✗ 配置验证失败 (port超出范围)")
    
    # 修正值
    config.set('server.port', 8080)
    if config.validate(schema):
        print("✓ 配置验证通过 (修正后)")
    else:
        print("✗ 配置验证失败")
    print()


def demo_nested_config():
    """嵌套配置示例"""
    print("=" * 60)
    print("5. 嵌套配置")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 设置嵌套配置
    config.set('detectors.text.backend', 'paddleocr')
    config.set('detectors.text.config.model_dir', 'models/text/')
    config.set('detectors.text.config.use_gpu', True)
    
    config.set('detectors.bubble.backend', 'custom')
    config.set('detectors.bubble.config.screen_width', 720)
    config.set('detectors.bubble.config.memory_path', 'memory.json')
    
    config.set('extractors.nickname.top_k', 3)
    config.set('extractors.nickname.min_score', 0.5)
    
    # 获取完整配置
    full_config = config.get_all()
    
    print("完整配置结构:")
    import json
    print(json.dumps(full_config, indent=2, ensure_ascii=False))
    print()


def demo_config_history():
    """配置历史示例"""
    print("=" * 60)
    print("6. 配置历史")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 初始配置
    config.set('version', '1.0.0', layer='user')
    print(f"初始版本: {config.get('version')}")
    
    # 创建临时文件用于演示load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
        import json
        json.dump({'version': '2.0.0'}, f)
    
    # 加载新配置（会保存历史）
    config.load(json_path, layer='user')
    print(f"更新后版本: {config.get('version')}")
    
    # 查看历史
    history = config.get_history()
    print(f"配置历史记录数: {len(history)}")
    if history:
        print(f"上一个版本: {history[-1]['config'].get('version')}")
    
    # 清理
    Path(json_path).unlink()
    print()


def demo_real_world_example():
    """真实场景示例：流水线配置"""
    print("=" * 60)
    print("7. 真实场景：流水线配置")
    print("=" * 60)
    
    config = ConfigManager()
    
    # 默认配置
    config.set('pipeline.name', 'default_pipeline', layer='default')
    config.set('pipeline.timeout', 60, layer='default')
    config.set('detector.text.backend', 'paddleocr', layer='default')
    
    # 用户配置（覆盖部分默认值）
    config.set('pipeline.name', 'my_custom_pipeline', layer='user')
    config.set('detector.text.backend', 'tesseract', layer='user')
    config.set('detector.text.lang', 'chi_sim+eng', layer='user')
    
    # 运行时配置（临时覆盖）
    config.set('pipeline.timeout', 120, layer='runtime')
    config.set('debug', True, layer='runtime')
    
    print("最终生效的配置:")
    print(f"  pipeline.name: {config.get('pipeline.name')}")
    print(f"  pipeline.timeout: {config.get('pipeline.timeout')}")
    print(f"  detector.text.backend: {config.get('detector.text.backend')}")
    print(f"  detector.text.lang: {config.get('detector.text.lang')}")
    print(f"  debug: {config.get('debug')}")
    print()
    
    print("各层级配置:")
    print(f"  default层: {config.get_all('default')}")
    print(f"  user层: {config.get_all('user')}")
    print(f"  runtime层: {config.get_all('runtime')}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ConfigManager 使用示例")
    print("=" * 60 + "\n")
    
    demo_basic_usage()
    demo_layered_config()
    demo_file_operations()
    demo_validation()
    demo_nested_config()
    demo_config_history()
    demo_real_world_example()
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
