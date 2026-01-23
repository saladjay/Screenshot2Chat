# Requirements Document

## Introduction

本文档定义了聊天气泡识别系统的重构需求。当前系统使用基于YAML配置的经验公式方法，针对不同应用（Discord、WhatsApp、Instagram、Telegram）进行硬编码处理。重构目标是建立一个通用的、自适应的聊天气泡检测系统，能够在不依赖应用类型先验知识的情况下，自动识别和分类聊天消息。

## Glossary

- **System**: 聊天气泡检测系统
- **TextBox**: 文本框对象，包含位置坐标(x_min, y_min, x_max, y_max)和相关属性
- **Column**: 聊天界面中的垂直列，通常左列和右列分别代表不同的说话者
- **Speaker**: 说话者，可以是用户(user)或对话者(talker)
- **Layout**: 聊天界面布局类型，包括单列(single)、双列(double)、左对齐双列(double_left)或右对齐双列(double_right)
- **center_x**: 文本框的水平中心坐标，计算公式为 (x_min + x_max) / 2
- **Historical_Data**: 跨多张截图收集的历史数据
- **Memory**: 系统维护的跨截图统计信息，用于学习说话者的几何特征
- **Separation_Ratio**: 两列中心点之间的归一化距离，用于判断是否为双列布局

## Requirements

### Requirement 1: 自动列检测与布局分类

**User Story:** 作为系统用户，我希望系统能够自动检测聊天界面的布局类型（单列、双列分开、双列都靠左、双列都靠右），这样我就不需要手动指定应用类型。

#### Acceptance Criteria

1. WHEN 系统接收到文本框列表和屏幕宽度 THEN THE System SHALL 计算所有文本框的center_x并进行归一化
2. WHEN center_x数量少于4个 THEN THE System SHALL 判定为单列布局
3. WHEN 系统使用KMeans(n_clusters=2)对归一化的center_x进行聚类 THEN THE System SHALL 计算两个聚类中心的分离度
4. WHEN 分离度小于最小分离阈值(默认0.18) THEN THE System SHALL 判定为单列布局
5. WHEN 分离度大于等于最小分离阈值 THEN THE System SHALL 判定为双列布局
6. WHEN 判定为双列布局且两个聚类中心都小于0.5(屏幕中线) THEN THE System SHALL 标记为左对齐双列(double_left)
7. WHEN 判定为双列布局且两个聚类中心都大于0.5 THEN THE System SHALL 标记为右对齐双列(double_right)
8. WHEN 判定为双列布局且两个聚类中心分别在0.5两侧 THEN THE System SHALL 标记为标准双列(double)

### Requirement 2: 文本框列分配

**User Story:** 作为系统用户，我希望系统能够将每个文本框正确分配到左列或右列，无论是哪种布局类型，这样我就能区分不同说话者的消息。

#### Acceptance Criteria

1. WHEN 系统判定为单列布局 THEN THE System SHALL 将所有文本框分配到左列，右列为空
2. WHEN 系统判定为双列布局(任何类型) THEN THE System SHALL 根据center_x与两个聚类中心的距离将文本框分配到左列或右列
3. WHEN 文本框的center_x更接近左侧聚类中心 THEN THE System SHALL 将该文本框分配到左列
4. WHEN 文本框的center_x更接近右侧聚类中心 THEN THE System SHALL 将该文本框分配到右列
5. WHEN 系统返回结果 THEN THE System SHALL 包含布局类型标识(single/double/double_left/double_right)

### Requirement 3: 历史数据学习

**User Story:** 作为系统用户，我希望系统能够从历史截图中学习说话者的位置模式，这样在处理新截图时能够保持一致的说话者识别。

#### Acceptance Criteria

1. WHEN 系统首次处理截图 THEN THE System SHALL 初始化空的Memory结构，包含Speaker A和Speaker B的几何特征
2. WHEN 系统收集到新的文本框数据 THEN THE System SHALL 提取center_x和width特征
3. WHEN Memory为空(首次处理) THEN THE System SHALL 默认将左列分配给Speaker A，右列分配给Speaker B
4. WHEN Memory已存在 THEN THE System SHALL 计算当前列特征与历史Speaker特征的几何距离
5. WHEN 系统完成说话者分配 THEN THE System SHALL 使用滑动平均(alpha=0.7)更新Memory中的center和width

### Requirement 4: 跨截图说话者匹配

**User Story:** 作为系统用户，我希望系统能够在多张截图之间保持说话者身份的一致性，即使布局略有变化。

#### Acceptance Criteria

1. WHEN 系统处理新截图的双列布局 THEN THE System SHALL 计算左列特征与Speaker A的距离d_LA和与Speaker B的距离d_LB
2. WHEN 系统计算距离 THEN THE System SHALL 使用公式 dist = |center差异|/屏幕宽度 + |width差异|/屏幕宽度
3. WHEN 系统计算右列特征与两个Speaker的距离d_RA和d_RB THEN THE System SHALL 使用相同的距离公式
4. WHEN d_LA + d_RB <= d_LB + d_RA THEN THE System SHALL 将左列分配给Speaker A，右列分配给Speaker B
5. WHEN d_LA + d_RB > d_LB + d_RA THEN THE System SHALL 将左列分配给Speaker B，右列分配给Speaker A

### Requirement 5: 统一接口

**User Story:** 作为系统开发者，我希望有一个统一的接口来处理截图，这样我就不需要为不同布局类型编写不同的代码。

#### Acceptance Criteria

1. WHEN 调用process_frame方法 THEN THE System SHALL 接受文本框列表作为输入
2. WHEN 系统处理完成 THEN THE System SHALL 返回包含layout类型(single/double/double_left/double_right)、Speaker A的文本框列表和Speaker B的文本框列表的字典
3. WHEN 返回结果中layout为"single" THEN THE System SHALL 确保Speaker B的列表为空
4. WHEN 返回结果中layout为任何双列类型 THEN THE System SHALL 确保Speaker A和Speaker B的列表都包含相应的文本框
5. WHEN 系统处理每一帧 THEN THE System SHALL 自动更新帧计数器
6. WHEN 返回结果 THEN THE System SHALL 包含布局子类型信息以便下游处理

### Requirement 6: 应用无关性

**User Story:** 作为系统用户，我希望系统完全不依赖应用类型信息，这样我就能处理任何聊天应用的截图而无需配置。

#### Acceptance Criteria

1. WHEN 系统处理截图 THEN THE System SHALL 不接受任何应用类型参数
2. WHEN 系统进行列检测 THEN THE System SHALL 仅基于几何特征而不使用应用特定的规则
3. WHEN 系统分配说话者 THEN THE System SHALL 使用通用的统计学习方法而不是应用特定的阈值
4. WHEN 系统保存历史数据 THEN THE System SHALL 不包含应用类型标识
5. WHEN 系统初始化 THEN THE System SHALL 不需要加载应用特定的配置文件

### Requirement 7: 向后兼容

**User Story:** 作为系统维护者，我希望新系统能够与现有代码兼容，这样我就能逐步迁移而不会破坏现有功能。

#### Acceptance Criteria

1. WHEN 新的ChatLayoutDetector类被实例化 THEN THE System SHALL 保持TextBox类的现有接口不变
2. WHEN 系统提供新的检测方法 THEN THE System SHALL 可以与现有的processors模块共存
3. WHEN 系统返回结果 THEN THE System SHALL 使用与现有代码兼容的数据结构
4. WHEN 系统处理TextBox对象 THEN THE System SHALL 使用现有的center_x、width、height等属性

### Requirement 8: 配置灵活性

**User Story:** 作为系统用户，我希望能够调整检测参数以适应不同的使用场景，这样我就能优化检测效果。

#### Acceptance Criteria

1. WHEN 实例化ChatLayoutDetector THEN THE System SHALL 接受screen_width作为必需参数
2. WHEN 调用split_columns方法 THEN THE System SHALL 接受min_separation_ratio作为可选参数，默认值为0.18
3. WHEN 更新Memory THEN THE System SHALL 使用可配置的滑动平均系数alpha，默认值为0.7
4. WHEN 系统判断列分离度 THEN THE System SHALL 允许用户自定义分离阈值

### Requirement 9: 时序一致性验证

**User Story:** 作为系统用户，我希望系统能够利用对话的时序规律来提高说话者识别的准确性，这样即使几何特征不明显时也能正确识别。

#### Acceptance Criteria

1. WHEN 系统完成初步的说话者分配 THEN THE System SHALL 分析文本框的y坐标时序
2. WHEN 检测到同一说话者连续出现超过阈值次数 THEN THE System SHALL 降低该分配的置信度
3. WHEN 检测到说话者交替出现的模式 THEN THE System SHALL 提高该分配的置信度
4. WHEN 系统返回结果 THEN THE System SHALL 包含置信度(confidence)字段，范围为0.0到1.0
5. WHEN 置信度低于阈值(默认0.5) THEN THE System SHALL 在metadata中标记为"uncertain"

### Requirement 11: Fallback机制

**User Story:** 作为系统用户，我希望系统在历史数据不足时能够使用更稳定的方法，这样即使是新用户也能获得合理的检测结果。

#### Acceptance Criteria

1. WHEN 历史数据中的文本框总数少于阈值(默认50个) THEN THE System SHALL 使用median方法而不是KMeans进行列分割
2. WHEN Memory为空或未稳定 THEN THE System SHALL 在metadata中标记使用的方法为"median_fallback"
3. WHEN 只有一侧有文本框 THEN THE System SHALL 判定为单列布局而不是强行分成两列
4. WHEN 系统使用fallback方法 THEN THE System SHALL 在metadata中包含fallback原因
5. WHEN 历史数据积累到足够数量 THEN THE System SHALL 自动切换回KMeans方法

### Requirement 12: 数据持久化

**User Story:** 作为系统用户，我希望系统能够保存和加载历史学习数据，这样我就不需要每次重启都重新学习。

#### Acceptance Criteria

1. WHEN 系统需要保存Memory数据 THEN THE System SHALL 将Memory序列化为JSON或pickle格式
2. WHEN 系统启动时 THEN THE System SHALL 尝试从指定路径加载历史Memory数据
3. WHEN 历史数据文件不存在 THEN THE System SHALL 初始化空Memory并正常运行
4. WHEN 历史数据文件损坏 THEN THE System SHALL 记录警告并初始化空Memory
5. WHEN 系统更新Memory THEN THE System SHALL 自动保存到持久化存储
