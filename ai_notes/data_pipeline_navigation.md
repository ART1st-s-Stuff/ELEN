# Data Pipeline (Navigation)

## 转换脚本

- 文件：`src/env/navigation/dataset.py`
- 输入：EB-Nav JSON + 图像目录
- 输出：HDF5（支持 train/test 划分）

## HDF5 字段

- `ep_len`, `ep_offset`, `pixels`, `action`, `instruction`, `is_end`

## 关键逻辑

- 动作映射：优先 `action[0]` 动作 ID（0..7），兼容字符串别名。
- 标签构造：
  - `instruction`：优先 episode 级字段，缺失时回退 trajectory 首条。
  - `is_end`：仅成功 episode 的最后一步为 `1.0`，其余 `0.0`。
- 性能：episode 级并行加载，主线程单写 HDF5。

## 常用参数

- `--download DIR`
- `--max-episodes N`
- `--split-test-ratio P --split-seed --output-test PATH`
- `--resize W H`

## 常见问题

- `Killed`：通常 OOM，优先降分辨率/减少 episode/降低 workers。
- `No valid episodes after conversion`：检查图像目录与 JSON 路径匹配。
