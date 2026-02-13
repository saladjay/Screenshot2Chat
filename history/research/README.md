# 研究文档归档

## 概述

本目录包含项目开发过程中的研究和原型代码。这些代码记录了功能探索和实验的过程，最终的功能已集成到主代码库中。

## 归档日期
2026年2月13日

## 目录结构

### how to detect chat bubble/
早期聊天气泡检测研究文档

**内容**:
- `advice1.md` - 研究建议1
- `advice2.md` - 研究建议2
- `advice3.md` - 研究建议3
- `advice4.md` - 研究建议4
- `advice5.md` - 研究建议5

**状态**: 研究完成，方法已应用到主系统

### nickname/
昵称提取功能研究项目

**结构**:
```
nickname/
├── docs/                        # 9个文档
│   ├── SMART_NICKNAME_DETECTION_SUMMARY.md
│   ├── HEIGHT_SCORING_IMPROVEMENT.md
│   ├── NICKNAME_EXTRACTION_LOGIC.md
│   ├── NICKNAME_EXTRACTION_STATUS.md
│   ├── REMOVE_WIDTH_SCORING.md
│   ├── SCORE_BREAKDOWN_FEATURE.md
│   ├── DRAW_TOP3_NICKNAMES_FEATURE.md
│   ├── UPDATE_TEST_NICKNAMES_SMART.md
│   └── Y_RANK_SCORING_IMPLEMENTATION.md
├── examples/                    # 8个示例
│   ├── extract_nicknames_demo.py
│   ├── extract_nicknames_detailed.py
│   ├── nickname_detection_demo.py
│   ├── show_all_nicknames.py
│   ├── test_nickname_extractor_optimized.py
│   ├── test_nicknames_smart.py
│   ├── test_top3_nicknames_filtered.py
│   └── test_top3_nicknames.py
└── tests/                       # 4个测试
    ├── test_extract_nicknames_adaptive.py
    ├── test_y_rank_demo.py
    ├── test_y_rank_scoring.py
    └── test_y_rank_simple.py
```

**研究成果**:
- 智能昵称检测算法
- 综合评分系统（位置、大小、文本、Y坐标）
- 系统UI文本过滤
- 边缘检测优化

**集成位置**: 
- `src/screenshot2chat/extractors/nickname_extractor.py` - 主要实现
- `src/screenshotanalysis/processors.py` - 处理器集成

**状态**: 研究完成，功能已完全集成到主代码库

## 研究价值

这些归档的研究代码具有以下价值：

1. **历史记录**: 记录了功能开发的探索过程
2. **设计决策**: 展示了为什么选择当前的实现方案
3. **学习资源**: 可以了解功能是如何从原型演进到生产代码的
4. **参考实现**: 包含了多种实现方案和测试用例

## 使用建议

### 查看研究过程
如果想了解某个功能的研究过程：
1. 查看对应目录的文档
2. 运行示例代码（可能需要调整路径）
3. 对比研究代码和当前实现

### 参考实现
如果需要类似功能的实现参考：
1. 研究代码展示了多种实现方案
2. 文档记录了各种方案的优缺点
3. 测试用例展示了功能的预期行为

## 注意事项

⚠️ **这些代码是历史归档，不应该在生产环境使用**

- 研究代码可能不完整或包含实验性功能
- 依赖关系可能已过时
- 功能已被主代码库的实现替代
- 仅供参考和学习使用

## 相关文档

### 当前实现
- `src/screenshot2chat/extractors/nickname_extractor.py` - 昵称提取器
- `src/screenshotanalysis/processors.py` - 消息处理器
- `docs/API_REFERENCE.md` - API文档

### 开发文档
- `history/development/` - 开发过程文档
- `.kiro/specs/screenshot-analysis-library-refactor/` - 规格文档

### 测试
- `tests/test_nickname_extractor_unit.py` - 单元测试
- `tests/test_nickname_extraction_properties.py` - 属性测试
- `tests/test_nickname_extraction_helpers.py` - 辅助函数测试

## 归档原因

这些研究代码被归档的原因：

1. **功能已集成**: 所有有价值的功能都已集成到主代码库
2. **避免混淆**: 防止开发者误用旧的研究代码
3. **保持整洁**: 保持项目根目录和主代码库的整洁
4. **保留历史**: 保留研究过程以供参考和学习

## 维护

这些归档的研究代码：
- ✅ 不需要维护或更新
- ✅ 不需要运行测试
- ✅ 仅作为历史记录保存
- ✅ 如有需要可以参考，但不应修改
