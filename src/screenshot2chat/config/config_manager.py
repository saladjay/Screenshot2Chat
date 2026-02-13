"""配置管理器模块

提供分层配置管理功能，支持默认配置、用户配置和运行时配置的三层结构。
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import yaml
import json
import copy


class ConfigManager:
    """配置管理器
    
    支持三层配置结构：
    - default: 默认配置
    - user: 用户配置
    - runtime: 运行时配置
    
    优先级: runtime > user > default
    """
    
    def __init__(self):
        """初始化配置管理器"""
        self.configs: Dict[str, Dict[str, Any]] = {
            'default': {},
            'user': {},
            'runtime': {}
        }
        self.history: List[Dict[str, Any]] = []
    
    def load(self, config_path: str, layer: str = 'user') -> None:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            layer: 配置层级 (default/user/runtime)
        
        Raises:
            ValueError: 如果配置层级无效或文件格式不支持
            FileNotFoundError: 如果配置文件不存在
        """
        if layer not in self.configs:
            raise ValueError(f"Invalid layer: {layer}. Must be one of: default, user, runtime")
        
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Supported: .yaml, .yml, .json")
        
        # 保存历史版本
        if self.configs[layer]:
            self.history.append({
                'layer': layer,
                'config': copy.deepcopy(self.configs[layer])
            })
        
        self.configs[layer] = config if config is not None else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        支持点号分隔的嵌套键，例如: "model.text_detector.backend"
        优先级: runtime > user > default
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值，当键不存在时返回
        
        Returns:
            配置值，如果不存在则返回default
        """
        keys = key.split('.')
        
        # 按优先级查找
        for layer in ['runtime', 'user', 'default']:
            value = self.configs[layer]
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                continue
        
        return default
    
    def set(self, key: str, value: Any, layer: str = 'runtime') -> None:
        """设置配置值
        
        支持点号分隔的嵌套键，会自动创建中间层级。
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
            layer: 配置层级 (default/user/runtime)
        
        Raises:
            ValueError: 如果配置层级无效
        """
        if layer not in self.configs:
            raise ValueError(f"Invalid layer: {layer}. Must be one of: default, user, runtime")
        
        keys = key.split('.')
        config = self.configs[layer]
        
        # 导航到最后一层的父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # 如果中间键不是字典，需要替换为字典
                config[k] = {}
            config = config[k]
        
        # 设置最终值
        config[keys[-1]] = value
    
    def save(self, config_path: str, layer: str = 'user') -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件路径
            layer: 要保存的配置层级 (default/user/runtime)
        
        Raises:
            ValueError: 如果配置层级无效或文件格式不支持
        """
        if layer not in self.configs:
            raise ValueError(f"Invalid layer: {layer}. Must be one of: default, user, runtime")
        
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.configs[layer]
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Supported: .yaml, .yml, .json")
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置的有效性
        
        Args:
            schema: 可选的验证模式，包含类型和范围约束
        
        Returns:
            True 如果配置有效，否则 False
        
        Note:
            如果未提供schema，则执行基本的结构验证
        """
        if schema is None:
            # 基本验证：确保所有层级都是字典
            for layer, config in self.configs.items():
                if not isinstance(config, dict):
                    return False
            return True
        
        # 使用提供的schema进行验证
        return self._validate_with_schema(schema)
    
    def _validate_with_schema(self, schema: Dict[str, Any]) -> bool:
        """使用schema验证配置
        
        Args:
            schema: 验证模式，格式为:
                {
                    'key.path': {
                        'type': str/int/float/bool/list/dict,
                        'required': bool,
                        'min': number (for numeric types),
                        'max': number (for numeric types),
                        'choices': list (for enum types)
                    }
                }
        
        Returns:
            True 如果所有配置都符合schema，否则 False
        """
        for key_path, constraints in schema.items():
            value = self.get(key_path)
            
            # 检查必需字段
            if constraints.get('required', False) and value is None:
                return False
            
            # 如果值为None且不是必需的，跳过其他检查
            if value is None:
                continue
            
            # 类型检查
            expected_type = constraints.get('type')
            if expected_type and not isinstance(value, expected_type):
                return False
            
            # 数值范围检查
            if isinstance(value, (int, float)):
                min_val = constraints.get('min')
                max_val = constraints.get('max')
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
            
            # 枚举值检查
            choices = constraints.get('choices')
            if choices and value not in choices:
                return False
        
        return True
    
    def get_all(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """获取完整的配置字典
        
        Args:
            layer: 可选的层级名称。如果提供，返回该层级的配置；
                   如果为None，返回合并后的配置（按优先级）
        
        Returns:
            配置字典
        """
        if layer is not None:
            if layer not in self.configs:
                raise ValueError(f"Invalid layer: {layer}")
            return copy.deepcopy(self.configs[layer])
        
        # 合并所有层级的配置
        merged = {}
        for layer in ['default', 'user', 'runtime']:
            merged = self._deep_merge(merged, self.configs[layer])
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
        
        Returns:
            合并后的字典
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def clear(self, layer: Optional[str] = None) -> None:
        """清空配置
        
        Args:
            layer: 可选的层级名称。如果提供，清空该层级；
                   如果为None，清空所有层级
        """
        if layer is not None:
            if layer not in self.configs:
                raise ValueError(f"Invalid layer: {layer}")
            self.configs[layer] = {}
        else:
            for layer in self.configs:
                self.configs[layer] = {}
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取配置历史
        
        Returns:
            配置历史列表
        """
        return copy.deepcopy(self.history)
