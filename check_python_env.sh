#!/bin/bash
VENV_PATH="/root/project/chat-layout-analyzer/.venv"

echo "🔍 Python环境诊断工具"
echo "===================="

# 检查路径是否存在
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虚拟环境目录不存在: $VENV_PATH"
    exit 1
fi

# 检查虚拟环境
echo "1. 虚拟环境检查:"
if [ -f "$VENV_PATH/bin/python" ]; then
    echo "   ✅ python可执行文件存在"
    if [ -x "$VENV_PATH/bin/python" ]; then
        echo "   ✅ python文件有执行权限"
    else
        echo "   ❌ python文件无执行权限，正在修复..."
        chmod +x "$VENV_PATH/bin/python"
    fi
else
    echo "   ❌ python可执行文件不存在"
fi

# 测试运行
echo -e "\n2. 测试运行:"
$VENV_PATH/bin/python -c "import sys; print(f'Python版本: {sys.version}')"

echo -e "\n3. 快速修复:"
echo "   执行以下命令激活虚拟环境:"
echo "   source $VENV_PATH/bin/activate"
echo "   或使用别名:"
echo "   alias python='$VENV_PATH/bin/python'"
