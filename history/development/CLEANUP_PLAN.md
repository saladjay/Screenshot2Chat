# 代码清理计划

## 清理日期
2026年2月13日

## 清理目标
删除项目中的无用代码、重复文件和临时测试文件，保持代码库整洁。

## 待删除文件清单

### 1. 根目录临时测试文件（11个）
这些是开发过程中的临时验证文件，功能已被tests/目录中的正式测试覆盖：

- ✅ `test_backward_compat.py` - 已有 `tests/test_backward_compatibility_properties.py`
- ✅ `test_conditional_parallel_pipeline.py` - 临时验证文件
- ✅ `test_extractors_basic.py` - 已有 `tests/test_base_extractor_properties.py`
- ✅ `test_fallback_mechanism.py` - 已有 `tests/test_chat_layout_detector.py`
- ✅ `test_fallback_verification.py` - 临时验证文件
- ✅ `test_pipeline_basic.py` - 已有 `tests/test_pipeline_properties.py`
- ✅ `test_pipeline_integration.py` - 已有 `tests/test_checkpoint_phase3.py`
- ✅ `test_split_columns_manual.py` - 临时测试文件
- ✅ `test_split_columns_simple.py` - 临时测试文件
- ✅ `test_task8_verification.py` - 任务验证文件，已完成
- ✅ `verify_app_independence.py` - 验证脚本，已完成
- ✅ `verify_performance_monitoring.py` - 验证脚本，已完成

### 2. 旧配置文件（1个）
新系统已不再使用YAML配置：

- ✅ `config/analysis_config.yaml` - 旧配置系统，新系统使用ConfigManager
- ✅ `config/` 目录（如果为空）

### 3. 研究文档目录（1个）
早期研究文档，可以归档：

- ⚠️ `how to detect chat bubble/` - 早期研究文档（建议移到history/research/） 没问题

### 4. 示例配置文件（2个）
这些是示例文件，可以移到examples/目录：

- ⚠️ `pipeline_basic_example.yaml` - 移到 `examples/configs/` 没问题
- ⚠️ `pipeline_config_example.yaml` - 移到 `examples/configs/` 没问题
- ⚠️ `output_example.json` - 移到 `examples/outputs/` 没问题

### 5. Nickname子项目（1个目录）
如果nickname是独立子项目且不再使用：

- ❓ `nickname/` - 需要确认是否还在使用 不再使用

### 6. 测试输出目录（可选清理）
这些目录包含测试生成的临时文件：

- ⚠️ `test_output/` - 测试输出（可以清空但保留目录） 好
- ⚠️ `test_text_det/` - 测试检测结果（可以清空但保留目录）好

## 清理步骤

### 第一步：删除根目录临时测试文件
```bash
# 删除12个临时测试和验证文件
rm test_backward_compat.py
rm test_conditional_parallel_pipeline.py
rm test_extractors_basic.py
rm test_fallback_mechanism.py
rm test_fallback_verification.py
rm test_pipeline_basic.py
rm test_pipeline_integration.py
rm test_split_columns_manual.py
rm test_split_columns_simple.py
rm test_task8_verification.py
rm verify_app_independence.py
rm verify_performance_monitoring.py
```

### 第二步：删除旧配置文件
```bash
# 删除旧配置系统
rm config/analysis_config.yaml
rmdir config  # 如果目录为空
```

### 第三步：归档研究文档
```bash
# 移动研究文档到归档目录
mkdir -p history/research
mv "how to detect chat bubble" history/research/
```

### 第四步：整理示例文件
```bash
# 创建examples子目录
mkdir -p examples/configs
mkdir -p examples/outputs

# 移动示例文件
mv pipeline_basic_example.yaml examples/configs/
mv pipeline_config_example.yaml examples/configs/
mv output_example.json examples/outputs/
```

### 第五步：清理测试输出（可选）
```bash
# 清空测试输出目录但保留目录结构
rm -rf test_output/*
rm -rf test_text_det/*
```

## 预期效果

### 清理前
- 根目录文件：~40个
- 临时测试文件：12个
- 配置文件混乱

### 清理后
- 根目录文件：~25个
- 所有测试在tests/目录
- 示例文件在examples/目录
- 研究文档在history/目录
- 结构清晰，易于维护

## 风险评估

### 低风险（可以直接删除）
- ✅ 根目录的12个临时测试文件
- ✅ config/analysis_config.yaml（已被ConfigManager替代）

### 中风险（需要确认）
- ⚠️ nickname/ 目录（需要确认是否还在使用）
- ⚠️ 示例配置文件（移动而非删除）

### 无风险（仅移动）
- ✅ 研究文档移到history/
- ✅ 示例文件移到examples/

## 验证步骤

清理后需要验证：
1. 运行所有测试：`pytest tests/`
2. 检查导入：确保没有代码引用已删除的文件
3. 运行示例：确保examples/目录中的示例正常工作

## 备注

- 所有删除操作前建议先提交git，以便回滚
- nickname/目录需要进一步确认其用途
- 测试输出目录可以保留，但定期清理
