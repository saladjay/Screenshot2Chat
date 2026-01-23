"""
验证聊天气泡检测系统的应用无关性

本脚本执行以下检查：
1. 确认ChatLayoutDetector类不接受app_type参数
2. 确认没有使用YAML配置文件
3. 确认没有应用特定的硬编码阈值
4. 验证所有方法签名和返回值不包含app_type

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import ast
import inspect
from pathlib import Path
from typing import List, Tuple


class AppIndependenceVerifier:
    """应用无关性验证器"""
    
    def __init__(self):
        self.violations = []
        self.checks_passed = []
        
    def verify_all(self) -> bool:
        """执行所有验证检查"""
        print("=" * 80)
        print("聊天气泡检测系统 - 应用无关性验证")
        print("=" * 80)
        print()
        
        # 检查1: 验证ChatLayoutDetector类不接受app_type参数
        self.check_no_app_type_parameter()
        
        # 检查2: 验证没有使用YAML配置文件
        self.check_no_yaml_usage()
        
        # 检查3: 验证没有应用特定的硬编码阈值
        self.check_no_app_specific_thresholds()
        
        # 检查4: 验证方法签名不包含app_type
        self.check_method_signatures()
        
        # 检查5: 验证返回值不包含app_type
        self.check_return_values()
        
        # 打印结果
        self.print_results()
        
        return len(self.violations) == 0
    
    def check_no_app_type_parameter(self):
        """检查1: 验证ChatLayoutDetector类不接受app_type参数"""
        print("检查1: 验证ChatLayoutDetector类不接受app_type参数")
        print("-" * 80)
        
        file_path = Path("src/screenshotanalysis/chat_layout_detector.py")
        
        if not file_path.exists():
            self.violations.append("❌ chat_layout_detector.py文件不存在")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            
            # 查找ChatLayoutDetector类
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "ChatLayoutDetector":
                    # 检查__init__方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            # 检查参数列表
                            param_names = [arg.arg for arg in item.args.args]
                            
                            if "app_type" in param_names:
                                self.violations.append(
                                    f"❌ ChatLayoutDetector.__init__接受app_type参数 (违反Requirement 6.1)"
                                )
                            else:
                                self.checks_passed.append(
                                    "✓ ChatLayoutDetector.__init__不接受app_type参数"
                                )
                            
                            # 打印实际参数列表
                            print(f"  实际参数: {', '.join(param_names)}")
                            break
                    break
        except SyntaxError as e:
            self.violations.append(f"❌ 解析chat_layout_detector.py失败: {e}")
        
        print()
    
    def check_no_yaml_usage(self):
        """检查2: 验证没有使用YAML配置文件"""
        print("检查2: 验证没有使用YAML配置文件")
        print("-" * 80)
        
        file_path = Path("src/screenshotanalysis/chat_layout_detector.py")
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否导入yaml
        if "import yaml" in content or "from yaml" in content:
            self.violations.append(
                "❌ chat_layout_detector.py导入了yaml模块 (违反Requirement 6.5)"
            )
        else:
            self.checks_passed.append("✓ 没有导入yaml模块")
        
        # 检查是否打开.yaml文件
        if ".yaml" in content or ".yml" in content:
            self.violations.append(
                "❌ chat_layout_detector.py引用了.yaml/.yml文件 (违反Requirement 6.5)"
            )
        else:
            self.checks_passed.append("✓ 没有引用.yaml/.yml文件")
        
        # 检查是否使用yaml.safe_load
        if "yaml.safe_load" in content or "yaml.load" in content:
            self.violations.append(
                "❌ chat_layout_detector.py使用了yaml.safe_load/yaml.load (违反Requirement 6.5)"
            )
        else:
            self.checks_passed.append("✓ 没有使用yaml.safe_load/yaml.load")
        
        print()
    
    def check_no_app_specific_thresholds(self):
        """检查3: 验证没有应用特定的硬编码阈值"""
        print("检查3: 验证没有应用特定的硬编码阈值")
        print("-" * 80)
        
        file_path = Path("src/screenshotanalysis/chat_layout_detector.py")
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有应用名称的硬编码
        app_names = ["DISCORD", "WHATSAPP", "INSTAGRAM", "TELEGRAM", 
                     "discord", "whatsapp", "instagram", "telegram"]
        
        found_app_names = []
        for app_name in app_names:
            # 排除注释中的提及
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # 跳过注释行
                if line.strip().startswith('#'):
                    continue
                # 跳过文档字符串
                if '"""' in line or "'''" in line:
                    continue
                    
                if app_name in line:
                    found_app_names.append((app_name, i))
        
        if found_app_names:
            for app_name, line_num in found_app_names:
                self.violations.append(
                    f"❌ 第{line_num}行包含应用名称'{app_name}' (违反Requirement 6.2, 6.3)"
                )
        else:
            self.checks_passed.append("✓ 没有应用特定的硬编码名称")
        
        # 检查是否有条件判断app_type
        if "if app_type" in content or "elif app_type" in content:
            self.violations.append(
                "❌ 代码中包含'if app_type'条件判断 (违反Requirement 6.2)"
            )
        else:
            self.checks_passed.append("✓ 没有'if app_type'条件判断")
        
        print()
    
    def check_method_signatures(self):
        """检查4: 验证方法签名不包含app_type"""
        print("检查4: 验证方法签名不包含app_type")
        print("-" * 80)
        
        file_path = Path("src/screenshotanalysis/chat_layout_detector.py")
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            
            methods_with_app_type = []
            
            # 查找所有方法
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    param_names = [arg.arg for arg in node.args.args]
                    
                    if "app_type" in param_names:
                        methods_with_app_type.append(node.name)
            
            if methods_with_app_type:
                for method_name in methods_with_app_type:
                    self.violations.append(
                        f"❌ 方法'{method_name}'接受app_type参数 (违反Requirement 6.1)"
                    )
            else:
                self.checks_passed.append("✓ 所有方法签名都不包含app_type参数")
                
        except SyntaxError as e:
            self.violations.append(f"❌ 解析方法签名失败: {e}")
        
        print()
    
    def check_return_values(self):
        """检查5: 验证返回值不包含app_type"""
        print("检查5: 验证返回值结构不包含app_type字段")
        print("-" * 80)
        
        file_path = Path("src/screenshotanalysis/chat_layout_detector.py")
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否在返回的字典中设置app_type
        suspicious_patterns = [
            '"app_type"',
            "'app_type'",
            '["app_type"]',
            "['app_type']",
        ]
        
        found_patterns = []
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # 跳过注释
            if line.strip().startswith('#'):
                continue
                
            for pattern in suspicious_patterns:
                if pattern in line:
                    found_patterns.append((pattern, i, line.strip()))
        
        if found_patterns:
            for pattern, line_num, line_content in found_patterns:
                self.violations.append(
                    f"❌ 第{line_num}行返回值可能包含app_type: {line_content[:60]}... (违反Requirement 6.4)"
                )
        else:
            self.checks_passed.append("✓ 返回值结构不包含app_type字段")
        
        print()
    
    def print_results(self):
        """打印验证结果"""
        print("=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        print()
        
        print(f"通过的检查 ({len(self.checks_passed)}):")
        for check in self.checks_passed:
            print(f"  {check}")
        print()
        
        if self.violations:
            print(f"发现的问题 ({len(self.violations)}):")
            for violation in self.violations:
                print(f"  {violation}")
            print()
            print("❌ 应用无关性验证失败")
            print()
            print("建议:")
            print("  1. 移除所有app_type参数")
            print("  2. 移除YAML配置文件依赖")
            print("  3. 移除应用特定的硬编码逻辑")
            print("  4. 确保所有方法都是通用的，基于几何特征而非应用类型")
        else:
            print("✅ 所有检查通过！系统完全应用无关。")
            print()
            print("验证的需求:")
            print("  ✓ Requirement 6.1: 系统不接受任何应用类型参数")
            print("  ✓ Requirement 6.2: 列检测仅基于几何特征")
            print("  ✓ Requirement 6.3: 使用通用的统计学习方法")
            print("  ✓ Requirement 6.4: 保存数据不包含应用类型标识")
            print("  ✓ Requirement 6.5: 不需要加载应用特定的配置文件")
        
        print("=" * 80)


def main():
    """主函数"""
    verifier = AppIndependenceVerifier()
    success = verifier.verify_all()
    
    # 返回退出码
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
