# AI Notes Index (ELEN)

本目录改为“按主题分类”维护，不再按时间顺序追加。

## 分类导航

- `architecture_and_flow.md`
  - 项目主流程、模块关系、两阶段训练链路
- `data_pipeline_navigation.md`
  - navigation 数据下载、转换、HDF5 字段、常见问题
- `training_and_scripts.md`
  - `scripts/` 训练与测试脚本约定、路径约定、快速命令
- `wm_critic_notes.md`
  - `src/wm/critic.py` 设计说明（含 AdaLN-Zero 融合）
- `backbone_notes.md`
  - `src/backbone/` 抽象接口与 `Qwen35Backbone` 状态

## 维护规则

- 新信息优先写入对应分类文件。
- 若需新增主题，创建新文件并在本索引登记。
- 避免重复记录同一信息；以“结构化摘要 + 文件路径”为主。
