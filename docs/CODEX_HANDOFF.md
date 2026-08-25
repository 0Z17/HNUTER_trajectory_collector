# 新会话交接：free-flight pilot 准备状态

## 已完成

- 项目已独立为 `HNUTER_trajectory_collector`，free-flight 功能可用；inspection/surface 只有扩展契约，尚未实现任务 validity。
- 后台采集入口：`python -m expert_trajectory_collector.batch`。
- 环境级多进程、condition 级原子落盘、进程退出后 `--resume`、`PAUSED` 标记安全暂停/继续均已实现。
- 终端每 2 秒报告进度；轻量监控页提供状态以及暂停/继续按钮，无 3D/URDF 绘制。
- 同一环境跨 condition 复用 COAL/FCL 完整 URDF 碰撞检查器。
- 每个 condition 独立采样 start/goal 位置与非水平姿态，同时更新所有路线证书端点并重新验证。
- 保存原始 OMPL pose7、256 点 B-spline pose7、128 点 pose9 diffusion 标签和完整专家元数据。
- 已用真实 OMPL/COAL 执行 1 环境/1 路径 smoke：启动前暂停后 worker 保持 `paused`、继续后接受 1 条路径并写入全部文件、再次 `--resume` 不重复采集。

## 明确没有做

- 没有启动 100 环境/约 4,000 路径正式 pilot。
- 没有启用可视化、URDF 残影、TOPP-RA 或 MuJoCo/MPPI rollout。
- 没有开始 5,000 环境/200,000 路径正式采集。
- 当前逐 condition 存储尚未合并为训练 shard；这应在 pilot 验收后作为离线步骤完成。

## 推荐下一步

1. 在仓库根目录运行完整测试。
2. 读取 `docs/PILOT_COLLECTION_RUNBOOK.md`，确认输出目录、seed 与 12 workers。
3. 启动 100 环境 pilot，先观察前 5–10 个环境的内存、接受率和 family 耗时。
4. 必要时暂停；不要删除输出目录，修复后使用 `--resume`。
5. 完成后生成分布/质量报告，再决定 200k 配额。

## 关键实现位置

- `expert_trajectory_collector/batch.py`：CLI。
- `expert_trajectory_collector/campaign/runner.py`：进程池和 manifest。
- `expert_trajectory_collector/campaign/free_flight.py`：条件采样与环境 worker。
- `expert_trajectory_collector/campaign/state.py`：状态聚合和暂停标记。
- `expert_trajectory_collector/campaign/monitor.py`：轻量 Web 监控。
- `expert_trajectory_collector/campaign/encoding.py`：弧长重采样及 pose9。
- `expert_trajectory_collector/campaign/io.py`：原子状态与数组写入。

## 已知注意事项

- 监控默认端口为 8785；8765 可能被旧交互 UI 占用。
- 安全暂停在 condition 边界检查，不会中断正在执行的 OMPL/B-spline 调用。
- `capacity_exhausted` 表示达到最大 condition 数仍未凑满 40 条，应该保留并分析，不能盲目复制路径补齐。
- 采集总速率/ETA 的 elapsed 包括暂停时间，因此长时间暂停后 ETA 偏保守。
- `campaign_config.json` 是恢复的权威配置；恢复时只改变 worker/监控运行参数。
- 规划器默认使用 RRTConnect 的区域偏置 state sampler；普通引导不是硬 waypoint，orientation-sensitive passage 仍保留必要的硬姿态约束。
