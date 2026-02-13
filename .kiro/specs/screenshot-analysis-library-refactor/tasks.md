# Implementation Plan: 聊天截图分析库重构

## Overview

本实施计划将设计文档中的架构转化为具体的编码任务。当前系统已有基础实现（ChatLayoutDetector、TextBox、昵称提取等），本计划专注于建立新的模块化架构，同时保持向后兼容性。

### Current Implementation Status

已实现的组件：
- ✅ TextBox 数据模型 (src/screenshot2chat/basemodel.py)
- ✅ ChatLayoutDetector 聊天布局检测器 (src/screenshot2chat/chat_layout_detector.py)
- ✅ ChatTextRecognition 文本识别 (src/screenshot2chat/core.py)
- ✅ ChatLayoutAnalyzer 版面分析 (src/screenshot2chat/core.py)
- ✅ NicknameExtractor 昵称提取 (src/screenshot2chat/nickname_extractor.py)
- ✅ 完整的测试套件 (tests/)

待实现的核心架构：
- ❌ BaseDetector/BaseExtractor 抽象基类
- ❌ Pipeline 流水线系统
- ❌ ConfigManager 配置管理
- ❌ ModelManager 模型管理
- ❌ 向后兼容层

## Tasks

### Phase 1: 核心抽象层 (Core Abstractions)

- [x] 1. 创建核心抽象类和数据模型
  - [x] 1.1 创建新模块目录结构
    - 创建 src/screenshot2chat/core/ 目录
    - 创建 src/screenshot2chat/detectors/ 目录
    - 创建 src/screenshot2chat/extractors/ 目录
    - 创建 src/screenshot2chat/pipeline/ 目录
    - 创建 src/screenshot2chat/config/ 目录
    - 创建 src/screenshot2chat/models/ 目录
    - _Requirements: 1.1_

  - [x] 1.2 实现 DetectionResult 和 ExtractionResult 数据类
    - 在 src/screenshot2chat/core/data_models.py 中定义 DetectionResult
    - 定义 ExtractionResult 类
    - 添加 to_json() 方法
    - _Requirements: 1.5, 6.6, 7.6_

  - [x]* 1.3 编写数据模型的单元测试
    - 测试 DetectionResult 序列化
    - 测试 ExtractionResult 序列化
    - _Requirements: 1.5_

  - [x] 1.4 实现 BaseDetector 抽象类
    - 在 src/screenshot2chat/core/base_detector.py 中定义 BaseDetector
    - 实现 load_model() 抽象方法
    - 实现 detect() 抽象方法
    - 实现 preprocess() 和 postprocess() 模板方法
    - _Requirements: 6.1_

  - [x]* 1.5 编写 BaseDetector 的属性测试
    - **Property 7: Detector Interface Conformance**
    - **Validates: Requirements 3.5, 6.6**

  - [x] 1.6 实现 BaseExtractor 抽象类
    - 在 src/screenshot2chat/core/base_extractor.py 中定义 BaseExtractor
    - 实现 extract() 抽象方法
    - 实现 validate() 方法
    - 实现 to_json() 方法
    - _Requirements: 7.1_

  - [x]* 1.7 编写 BaseExtractor 的属性测试
    - **Property 9: Extractor JSON Output Validity**
    - **Validates: Requirements 7.6**

### Phase 2: 包装现有实现 (Wrap Existing Components)

- [-] 2. 将现有检测器迁移到新架构
  - [x] 2.1 实现 TextDetector（包装 ChatTextRecognition）
    - 创建 src/screenshot2chat/detectors/text_detector.py
    - 继承 BaseDetector
    - 包装现有的 ChatTextRecognition 逻辑
    - 实现 detect() 方法，返回 List[DetectionResult]
    - 支持 PaddleOCR 后端
    - _Requirements: 6.2, 3.1_

  - [x]* 2.2 编写 TextDetector 的单元测试
    - 测试在真实图像上的检测
    - 测试结果格式转换
    - _Requirements: 6.2_

  - [x] 2.3 实现 BubbleDetector（包装 ChatLayoutDetector）
    - 创建 src/screenshot2chat/detectors/bubble_detector.py
    - 继承 BaseDetector
    - 包装现有的 ChatLayoutDetector 逻辑
    - 实现 detect() 方法
    - 保持跨截图记忆功能
    - _Requirements: 6.5_

  - [x]* 2.4 编写 BubbleDetector 的集成测试
    - 测试布局检测功能
    - 测试记忆更新机制
    - _Requirements: 6.5_

- [x] 3. 将现有提取器迁移到新架构
  - [x] 3.1 实现 NicknameExtractor（包装现有逻辑）
    - 创建 src/screenshot2chat/extractors/nickname_extractor.py
    - 继承 BaseExtractor
    - 包装现有的昵称提取算法
    - 实现 extract() 方法
    - 支持综合评分系统
    - _Requirements: 7.2_

  - [x]* 3.2 编写 NicknameExtractor 的单元测试
    - 测试昵称提取逻辑
    - 测试评分系统
    - _Requirements: 7.2_

  - [x] 3.3 实现 SpeakerExtractor
    - 创建 src/screenshot2chat/extractors/speaker_extractor.py
    - 基于 ChatLayoutDetector 的说话者推断
    - 实现 extract() 方法
    - _Requirements: 7.3_

  - [x] 3.4 实现 LayoutExtractor
    - 创建 src/screenshot2chat/extractors/layout_extractor.py
    - 检测布局类型（单列/双列）
    - 实现 extract() 方法
    - _Requirements: 7.4_

  - [x]* 3.5 编写提取器的属性测试
    - **Property 10: Extractor Chain Composition**
    - **Validates: Requirements 7.7**

- [x] 4. Checkpoint - 验证核心组件
  - 运行所有现有测试，确保无回归
  - 运行新的单元测试
  - 验证检测器和提取器的基本功能
  - 如有问题，询问用户

### Phase 3: 流水线和配置系统 (Pipeline & Configuration)

- [x] 5. 实现 Pipeline 流水线系统
  - [x] 5.1 实现 PipelineStep 和 Pipeline 基础类
    - 创建 src/screenshot2chat/pipeline/pipeline.py
    - 定义 StepType 枚举
    - 实现 PipelineStep 数据类
    - 实现 Pipeline 类的 add_step() 方法
    - 实现基本的 execute() 方法
    - _Requirements: 8.1_

  - [x] 5.2 实现流水线配置加载
    - 支持从 YAML/JSON 加载配置
    - 实现 from_config() 类方法
    - 解析步骤配置并创建组件实例
    - _Requirements: 8.1_

  - [x]* 5.3 编写流水线配置 round-trip 的属性测试
    - **Property 2: Pipeline Configuration Round-Trip**
    - **Validates: Requirements 8.7**

  - [x] 5.4 实现流水线执行顺序控制
    - 支持 depends_on 依赖声明
    - 按依赖关系排序步骤
    - 顺序执行步骤
    - _Requirements: 8.2_

  - [x]* 5.5 编写执行顺序的属性测试
    - **Property 11: Pipeline Execution Order Preservation**
    - **Validates: Requirements 8.2**

  - [x] 5.6 实现流水线验证
    - 检查步骤依赖关系
    - 检查配置完整性
    - 提供清晰的错误消息
    - _Requirements: 8.5_

  - [x]* 5.7 编写流水线验证的属性测试
    - **Property 14: Pipeline Validation Failure Detection**
    - **Validates: Requirements 8.5**

- [x] 6. 实现 ConfigManager
  - [x] 6.1 实现分层配置系统
    - 创建 src/screenshot2chat/config/config_manager.py
    - 支持 default/user/runtime 三层
    - 实现 get() 和 set() 方法
    - 支持点号分隔的嵌套键
    - _Requirements: 9.1_

  - [x]* 6.2 编写配置层级优先级的属性测试
    - **Property 16: Configuration Layer Priority**
    - **Validates: Requirements 9.1**

  - [x] 6.3 实现配置文件加载和保存
    - 支持 YAML 和 JSON 格式
    - 实现 load() 和 save() 方法
    - _Requirements: 9.5_

  - [x]* 6.4 编写配置 round-trip 的属性测试
    - **Property 21: Configuration Export-Import Round-Trip**
    - **Validates: Requirements 9.7**

  - [x] 6.5 实现配置验证
    - 添加 validate() 方法
    - 支持类型检查和范围验证
    - _Requirements: 9.4_

  - [x]* 6.6 编写配置验证的属性测试
    - **Property 19: Configuration Validation Rejection**
    - **Validates: Requirements 9.4**

- [x] 7. Checkpoint - 验证流水线和配置系统
  - 测试完整的流水线配置和执行
  - 验证配置管理功能
  - 如有问题，询问用户

### Phase 4: 向后兼容和集成 (Backward Compatibility & Integration)

- [x] 8. 实现向后兼容层
  - [x] 8.1 创建 ChatLayoutDetector 兼容包装器
    - 在 src/screenshot2chat/compat/ 目录创建兼容层
    - 包装新的 BubbleDetector
    - 保持旧接口不变
    - 添加弃用警告
    - _Requirements: 15.1, 15.3, 15.4_

  - [x]* 8.2 编写向后兼容性的测试
    - **Property 26: Backward Compatibility Preservation**
    - **Validates: Requirements 15.1, 15.2, 15.3**

  - [x]* 8.3 编写弃用警告的测试
    - **Property 27: Deprecation Warning Emission**
    - **Validates: Requirements 15.4**

  - [x] 8.4 更新 __init__.py 导出
    - 更新 src/screenshot2chat/__init__.py
    - 导出新的抽象类和实现
    - 保持旧 API 的导出
    - _Requirements: 15.1_

- [x] 9. 实现端到端示例
  - [x] 9.1 创建基本使用示例
    - 创建 examples/basic_pipeline_example.py
    - 展示如何使用新的 Pipeline API
    - 展示如何配置检测器和提取器
    - _Requirements: 14.5_

  - [x] 9.2 创建迁移示例
    - 创建 examples/migration_example.py
    - 展示如何从旧 API 迁移到新 API
    - 提供并排对比
    - _Requirements: 15.5_

  - [x] 9.3 编写迁移指南文档
    - 创建 docs/MIGRATION_GUIDE.md
    - 说明新旧 API 的对应关系
    - 提供迁移步骤
    - _Requirements: 15.5_

- [x] 10. Checkpoint - 完整系统验证
  - 运行所有单元测试
  - 运行所有属性测试
  - 运行端到端测试
  - 验证向后兼容性
  - 如有问题，询问用户

### Phase 5: 高级功能 (Advanced Features) - 可选

以下任务为可选的高级功能，可根据项目需求决定是否实施：

- [x] 11. 实现 ModelManager（可选）
  - [x] 11.1 实现 ModelMetadata 数据类
    - 创建 src/screenshot2chat/models/model_manager.py
    - 定义模型元信息字段
    - 实现序列化和反序列化
    - _Requirements: 10.1_

  - [x] 11.2 实现模型注册和加载功能
    - 实现 register() 方法
    - 实现 load() 方法
    - 支持按版本号和标签加载
    - _Requirements: 10.1, 10.3_

  - [x]* 11.3 编写模型管理的属性测试
    - **Property 22: Model Metadata Completeness**
    - **Property 23: Model Version Loading Correctness**
    - **Validates: Requirements 10.1, 10.3, 10.4**

- [x] 12. 实现性能监控（可选）
  - [x] 12.1 实现 PerformanceMonitor
    - 创建 src/screenshot2chat/monitoring/performance_monitor.py
    - 记录每个步骤的执行时间
    - 记录内存使用情况
    - _Requirements: 11.1, 11.4_

  - [x] 12.2 集成到 Pipeline
    - 在 Pipeline.execute() 中集成性能监控
    - 记录每个步骤的指标
    - _Requirements: 8.6, 11.1_

  - [x]* 12.3 编写性能监控的属性测试
    - **Property 15: Pipeline Metrics Recording**
    - **Validates: Requirements 8.6**

- [x] 13. 实现条件分支和并行执行（可选）
  - [x] 13.1 实现条件分支支持
    - 添加条件表达式解析
    - 根据中间结果选择分支
    - _Requirements: 8.3_

  - [x]* 13.2 编写条件分支的属性测试
    - **Property 12: Pipeline Conditional Branch Correctness**
    - **Validates: Requirements 8.3**

  - [x] 13.3 实现并行执行支持
    - 使用 ThreadPoolExecutor 或 ProcessPoolExecutor
    - 合并并行步骤的结果
    - _Requirements: 8.4_

  - [x] 13.4 编写并行执行的属性测试

    - **Property 13: Pipeline Parallel Execution Completeness**
    - **Validates: Requirements 8.4**

- [x] 14. 实现错误处理和日志系统（可选）
  - [x] 14.1 定义异常层次结构
    - 创建 src/screenshot2chat/core/exceptions.py
    - 创建所有异常类
    - 确保异常继承关系正确
    - _Requirements: 12.2_

  - [x] 14.2 实现 StructuredLogger
    - 创建 src/screenshot2chat/logging/structured_logger.py
    - 支持上下文信息
    - 支持多种日志级别
    - _Requirements: 12.1, 12.4, 12.5_

  - [x] 14.3 在关键位置添加错误处理
    - 在检测器和提取器中添加错误处理
    - 在流水线执行中添加错误处理
    - 提供恢复建议
    - _Requirements: 12.3, 12.7_

- [x] 15. 编写完整文档（可选）
  - [x] 15.1 编写架构设计文档
    - 创建 docs/ARCHITECTURE.md
    - 系统架构图
    - 模块说明
    - _Requirements: 14.1_

  - [x] 15.2 编写 API 参考文档
    - 创建 docs/API_REFERENCE.md
    - 所有公共接口的文档
    - 参数说明和示例
    - _Requirements: 14.2_

  - [x] 15.3 编写用户指南
    - 创建 docs/USER_GUIDE.md
    - 快速开始教程
    - 常见使用场景
    - _Requirements: 14.3_

## Notes

- 标记为 `*` 的任务是可选的测试任务，可以根据项目进度决定是否实施
- Phase 1-4 是核心功能，必须完成
- Phase 5 是高级功能，可根据需求选择性实施
- 每个 Checkpoint 任务都是验证点，确保在继续之前所有功能正常
- 属性测试使用 Hypothesis 库，每个测试至少运行 100 次迭代
- 单元测试使用 pytest 框架
- 所有任务都引用了对应的需求编号，确保可追溯性
- 建议按顺序执行任务，因为后续任务依赖前面任务的成果

## Implementation Priority

### 高优先级（必须完成）
1. Phase 1: 核心抽象层 - 建立新架构的基础
2. Phase 2: 包装现有实现 - 将现有功能迁移到新架构
3. Phase 3: 流水线和配置系统 - 实现灵活的编排能力
4. Phase 4: 向后兼容和集成 - 确保平滑迁移

### 中优先级（建议完成）
5. Phase 5 中的 ModelManager - 支持模型版本管理
6. Phase 5 中的性能监控 - 支持性能分析

### 低优先级（可选）
7. Phase 5 中的条件分支和并行执行 - 高级流水线功能
8. Phase 5 中的完整文档 - 可以逐步完善

## Success Criteria

重构成功的标准：
1. ✅ 所有现有测试通过（无回归）
2. ✅ 新的抽象类和实现可用
3. ✅ 至少一个完整的端到端示例可运行
4. ✅ 向后兼容层正常工作
5. ✅ 核心属性测试通过（至少 10 个属性）
6. ✅ 迁移指南文档完成

## Estimated Timeline

- Phase 1: 1-2 周
- Phase 2: 1-2 周
- Phase 3: 2-3 周
- Phase 4: 1 周
- Phase 5（可选）: 2-4 周

总计：5-8 周（核心功能）+ 2-4 周（可选高级功能）
