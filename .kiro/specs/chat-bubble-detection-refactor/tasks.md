# Implementation Plan: Chat Bubble Detection Refactor

## Overview

本实现计划将当前基于YAML配置的应用特定聊天气泡检测系统重构为通用的、自适应的几何学习系统。实现将分为以下几个阶段：核心检测器实现、记忆管理、持久化、时序一致性、fallback机制、测试和集成。每个任务都是增量式的，确保每一步都能验证功能正确性。

关键设计原则：
- **历史KMeans为主，median为fallback** - 充分利用历史数据的稳定性
- **时序规律作为极强信号** - 利用对话交替模式提高准确性
- **完全应用无关** - 不依赖任何应用类型或配置文件
- **跨截图一致性** - 通过记忆学习保持说话者身份稳定

补充说明：
- 单图场景易受样本不足、离群点影响，KMeans不稳定时优先使用median(center_x)分割。
- KMeans结果需按cluster_center的x值排序来确定left/right，避免依赖cluster_id。

## Tasks

- [x] 1. 创建核心ChatLayoutDetector类框架
  - 在src/screenshotanalysis/目录下创建新文件chat_layout_detector.py
  - 实现ChatLayoutDetector类的基本结构和初始化方法
  - 实现辅助函数calculate_column_stats和geometric_distance
  - _Requirements: 5.1, 8.1, 8.2, 8.3_

- [ ]* 1.1 为核心类编写单元测试
  - 创建tests/test_chat_layout_detector.py
  - 测试初始化参数验证
  - 测试辅助函数的正确性
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 2. 实现列分割和布局分类功能
  - [x] 2.1 实现split_columns方法
    - 实现center_x提取和归一化
    - 实现KMeans聚类逻辑
    - 实现分离度计算和布局类型判定（single/double/double_left/double_right）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 2.2 编写split_columns的属性测试

    - **Property 1: center_x归一化范围**
    - **Property 2: 少量样本判定为单列**
    - **Property 3: 低分离度判定为单列**
    - **Property 4: 高分离度判定为双列**
    - **Validates: Requirements 1.1, 1.2, 1.4, 1.5**

  - [x] 2.3 编写布局子类型的属性测试

    - **Property 5: 左对齐双列判定**
    - **Property 6: 右对齐双列判定**
    - **Property 7: 标准双列判定**
    - **Validates: Requirements 1.6, 1.7, 1.8**

  - [x] 2.4 编写列分配的属性测试

    - **Property 8: 单列布局右列为空**
    - **Property 9: 列分配完整性**
    - **Property 10: 最近聚类中心分配**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. 实现说话者推断和匹配功能
  - [x] 3.1 实现infer_speaker_in_frame方法
    - 实现列特征统计计算
    - 实现首次推断逻辑（无历史记忆时）
    - 实现基于历史记忆的最小代价匹配
    - _Requirements: 3.3, 4.1, 4.2, 4.4, 4.5_

  - [ ]* 3.2 编写说话者匹配的属性测试
    - **Property 15: 几何距离对称性**
    - **Property 16: 最小代价匹配**
    - **Property 17: 说话者分配互斥性**
    - **Validates: Requirements 4.2, 4.4, 4.5**

- [x] 4. 实现记忆管理功能
  - [x] 4.1 实现update_memory方法
    - 实现记忆初始化逻辑
    - 实现滑动平均更新算法
    - 实现count累加
    - _Requirements: 3.1, 3.2, 3.5_

  - [ ]* 4.2 编写记忆更新的属性测试
    - **Property 12: 特征提取完整性**
    - **Property 13: 滑动平均更新**
    - **Property 14: 记忆计数单调递增**
    - **Validates: Requirements 3.2, 3.5**

- [x] 5. 实现持久化功能
  - [x] 5.1 实现_save_memory和_load_memory方法
    - 实现JSON序列化和反序列化
    - 实现文件路径处理和目录创建
    - 实现错误处理（文件不存在、损坏、权限问题）
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 5.2 编写持久化的属性测试
    - **Property 21: 记忆持久化往返一致性**
    - **Property 22: 自动保存触发**
    - **Validates: Requirements 9.1, 9.2, 9.5**

  - [ ]* 5.3 编写持久化错误处理的单元测试
    - 测试文件不存在情况
    - 测试文件损坏情况
    - 测试权限不足情况
    - _Requirements: 9.3, 9.4_

- [x] 6. 实现统一接口process_frame
  - [x] 6.1 实现process_frame方法
    - 整合split_columns、infer_speaker_in_frame和update_memory
    - 实现帧计数器更新
    - 构建返回结果字典（包含layout、A、B、metadata）
    - _Requirements: 5.1, 5.2, 5.5, 5.6_

  - [x] 6.2 实现时序一致性验证
    - 实现calculate_temporal_confidence方法
    - 分析文本框y坐标时序，检测说话者交替模式
    - 计算置信度并在metadata中添加confidence字段
    - 当置信度低于阈值时标记为"uncertain"
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 6.3 编写process_frame的属性测试
    - **Property 11: 返回结果包含布局类型**
    - **Property 18: 双列布局非空性**
    - **Property 19: 帧计数器递增**
    - **Validates: Requirements 5.2, 5.4, 5.5**

  - [ ]* 6.4 编写时序一致性的属性测试
    - **Property 23: 时序交替提高置信度**
    - **Property 24: 置信度范围有效性**
    - **Property 25: 低置信度标记**
    - **Validates: Requirements 9.2, 9.3, 9.4, 9.5**

- [x] 7. 实现Fallback机制
  - [x] 7.1 实现fallback判断和median方法
    - 实现should_use_fallback方法检查历史数据量
    - 实现split_columns_median_fallback使用median分割
    - 在split_columns中集成fallback逻辑
    - 在metadata中添加方法标记和原因
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 7.2 编写fallback机制的属性测试
    - **Property 26: Fallback触发条件**
    - **Property 27: Fallback方法标记**
    - **Property 28: 单侧数据不强制分列**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

- [x] 8. 确保应用无关性
  - [x] 8.1 验证系统不依赖应用类型
    - 代码审查：确认没有app_type参数
    - 代码审查：确认没有使用YAML配置文件
    - 代码审查：确认没有应用特定的硬编码阈值
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [ ]* 8.2 编写应用无关性的属性测试
    - **Property 20: 应用类型不在保存数据中**
    - 验证process_frame方法签名
    - 验证返回结果不包含app_type
    - **Validates: Requirements 6.1, 6.4**

- [x] 9. Checkpoint - 核心功能验证
  - 运行所有单元测试和属性测试
  - 确保所有测试通过
  - 如有问题，询问用户

- [ ] 10. 集成测试和真实数据验证
  - [ ] 10.1 编写多帧序列集成测试
    - 测试跨3-5帧的说话者一致性
    - 测试布局变化时的适应性
    - 测试记忆收敛速度
    - _Requirements: 3.4, 4.4, 4.5_

  - [ ]* 10.2 使用真实截图进行验证
    - 使用test_images目录中的Discord截图测试
    - 使用test_images目录中的WhatsApp截图测试
    - 使用test_images目录中的Instagram截图测试
    - 使用test_images目录中的Telegram截图测试
    - 记录检测准确率
    - _Requirements: 6.2, 6.3_

- [x] 11. 与现有代码集成
  - [x] 11.1 更新ChatMessageProcessor类
    - 在processors.py中导入ChatLayoutDetector
    - 添加使用新检测器的可选方法
    - 保持现有方法不变以确保向后兼容
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 11.2 编写集成兼容性测试

    - 测试新旧方法可以共存
    - 测试TextBox对象在新旧代码间传递
    - 测试现有测试用例仍然通过
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 12. 性能优化和文档
  - [x] 12.1 性能测试和优化
    - 测试单帧处理时间（目标<100ms）
    - 测试内存占用
    - 如需要，优化KMeans参数或使用缓存
    - _Requirements: 所有_

  - [ ] 12.2 编写使用文档和示例

    - 在chat_layout_detector.py中添加详细的docstring
    - 创建examples/chat_detection_demo.py示例脚本
    - 更新README.md说明新功能
    - _Requirements: 所有_

- [ ] 13. Final Checkpoint - 完整系统验证
  - 运行完整测试套件（单元测试、属性测试、集成测试）
  - 验证测试覆盖率达标（代码行>90%，分支>85%）
  - 验证所有真实数据测试通过
  - 确保所有测试通过，询问用户是否有问题

## Notes

- 标记为`*`的任务是可选的，可以跳过以加快MVP开发
- 每个任务都引用了具体的需求条款以确保可追溯性
- Checkpoint任务确保增量验证
- 属性测试使用Hypothesis库，每个测试运行100次迭代
- 单元测试使用pytest框架
- 集成测试验证端到端流程和真实场景
- 系统设计为完全应用无关，不依赖任何YAML配置或应用类型标识
