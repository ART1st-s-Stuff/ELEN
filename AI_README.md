## 项目目标
开发一个高效的世界模型。
架构：LLM+LeWM.
- 初始时，LLM的输入为Text Prompt和当前图像经过LeWM Encoder之后的隐状态$z_t$
- LLM输出为预期的目标状态（使用文字描述）$target_t$和指定step的action list（输出action之前可选CoT）
- LeWM predictor根据action list和$z_t$预测执行action list之后的隐状态$z_{t+n_{steps}}$
- LeWM critic根据$target_t$的embedding和$z_{t+n_{steps}}$预测是否到达终点
- 若到达终点，输出action list.
- 若没有到达终点，把action list也添加到LLM input，重新生成新的action list.

## 项目进度
- [x] LeWM predictor组件
- [x] LeWM critic组件
- [ ] LLM主要模块
- [ ] LLM latent space and LeWM latent space alignment

当前已经在navigation环境中实现了LeWM部分。

## 开发行为准则
prompt/代码中没有清楚描述/你不理解的部分，征询人类开发者意见。
遇到可选超过一种方案的情况，征询人类开发者意见。
如果要进行大范围破坏性改动，征询人类开发者意见。

代码要求：
- 尽量复用现有代码
- 创建新代码时需要具备可复用性和可配置性，保证后续可持续发展
- 禁止硬编码参数，训练参数使用hydra等工具管理
- 保持代码逻辑简洁且易于理解
- 函数、变量的命名也要易于理解
- 与核心逻辑无关的内容（如库兼容性），将其单独放在utils/中

文件要求：
- 遵循项目结构
- 模块化，按职责分离

注释要求：
- python源码中的注释遵循python规范
- 简洁易懂，面向人类开发者
- 复杂逻辑额外添加注释说明

AI笔记：
- 由于对话历史可能会被清除，所以你需要在ai_notes/内保存所有关键信息
- 匹配当前代码库内容，每次修改后按需更新
- 尽量节约Token。例如对于简单阅读代码已经完全足够的情况，你可以保存信息的索引，甚至无需记录。
- 禁止包含用户个人信息。
- 尽量减少和当前运行环境有关的信息。

## 项目结构
- AI_README.md: 这个文件。由人类开发者编写。【非必要不修改】
- ai_notes/: 由你自己记录的开发笔记。
- datasets/: 下载的数据集
- lewm/: lewm原始代码。【非必要不修改】
- models/: 模型输出目录
- src/: 主要源码
  - env/: agent交互环境相关
    - navigation: navigation环境
  - wm/: world model相关
    - critic.py: 评分器
  - backbone/: 语言模型相关
    - qwen35.py
  - dev/: 开发用代码，用于测试项目各组件功能
  - train/: 用于正式训练和测试的代码，跑完整个流程
  - utils/: 与核心逻辑无关的工具类
- src_old/: 过时源码，无需查看。
- vagen/: vagen项目(另一个LLM based world model)中与环境interaction有关的代码【非必要不修改】

如果确实要修改上述非必要不修改的部分，征询人类开发者意见。