# 新会话交接：free-flight v002 扩展修复状态

## v001 pilot 结果

- `free_flight_pilot_100env_v001` 已处理完 100 个环境，最终接受 3,544/4,000 条：86 个环境完成、14 个环境 capacity exhausted、0 个环境 failed。
- 主要失败为 583 个 condition 无法在 160 次尝试内让新端点同时保留全部路线证书，以及 106 个 condition 的 B-spline 姿态超限。
- 最慢的 mixed-industrial 环境有 47/48 个 condition 采样失败，每个失败约 57–63 秒，形成约 45 分钟长尾。
- 全部 condition 数据均已原子落盘；主进程结束时还暴露了监控线程与主线程共用状态临时文件名的竞争，因此 v001 没有生成最终 manifest。

## v002 已实现

- condition 只保留对当前 start/goal 有效的路线子集；普通/optional 场景至少一条，guaranteed-multi 至少两条，multi-homotopy 仍要求垂直+水平覆盖，frame-doorway 仍要求 through。
- condition 采样默认最多 64 次/4 秒；连续 8 个零接受 condition 后安全熔断为 capacity exhausted。
- 新环境正式计数前先做 4 个 condition probe，至少 2 个成功，否则换 seed；探测单次最多 24 次/1 秒。
- 端点与 orientation gate 为全局 B-spline 保留 5° 姿态余量，最终硬限制仍为 roll ±40°、pitch ±70°。
- `status.json` 新增 finished、shortfall、reachable upper bound 和 target reachable，目标不可达时不再显示伪 ETA。
- JSON/gzip 原子临时文件使用每次写入唯一名称，修复主线程与监控线程并发写状态的竞争。
- v001 配置恢复保持旧采样行为；上述采样行为只进入新的 config schema v002 数据集。
- 批处理仍为 free-flight only；inspection/surface 只有扩展契约，尚未实现各自 validity。

## v002 pilot 与单通道优化

- `free_flight_pilot_100env_v002` 已于 2026-08-25 完成：100/100 环境、4,000/4,000 路径、0 capacity exhausted、0 failed，总耗时 551.6 秒。
- 全部 1,477 个 condition 数组文件可读取，接受路径合计 4,000，未发现临时文件残留；后台服务正常退出。
- pilot 显示 `orientation_sensitive_passage` 与 `staggered_corridor` 都只有一个实际 route mode，每 condition 只产生一条非重复路径。前者为严格 aperture 固定路线；后者额外产生 4,290 次 `redundant_duplicate` 拒绝。
- 新 batch 配置通过 `experts_per_condition_overrides` 将这两个 family 固定为每 condition 1 条；其他 family 继续使用全局 `experts_per_condition=5`。Web UI 不读取该 batch 覆盖，交互行为不变。
- 已用真实 OMPL/COAL 验证两个 family 共 2 环境/4 路径：4 次 planner attempt 即完成，均为每 condition 请求并接受 1 条。

## 明确没有做

- 没有启用可视化、URDF 残影、TOPP-RA 或 MuJoCo/MPPI rollout。
- 没有开始 5,000 环境/200,000 路径正式采集。
- 当前逐 condition 存储尚未合并为训练 shard；应在 v002 pilot 验收后离线完成。

## 当前验证

- 新增 campaign 回归测试：11 passed。
- 完整测试：79 passed, 1 skipped（可选 `se3_diffusion` 对照未安装）。
- 10 个 family 的环境预检均为 4/4 probe 通过。
- 真实 OMPL/COAL smoke：`wall_protrusion_bracket` 与 `mixed_industrial` 共 2 环境、10/10 路径、0 exhausted、0 failed，17.1 秒完成并生成 manifest。
- 对 v001 最慢的 mixed-industrial 环境复放 5 个 condition：v002 为 5/5 成功、约 1.0 秒/condition；v001 同环境失败约 57–63 秒/condition。

## 推荐下一步

1. 为单通道覆盖后的新 pilot 使用新的 dataset ID 与输出目录；不要向已完成的 v002 目录写入不同配置。
2. 重点比较 `staggered_corridor` 的 planner attempt、condition 数和总耗时，并确认其他八个 family 的同 condition 多样性不变。
3. 完成 v001/v002/单通道优化版的分布与质量报告，再决定 200k 配额和训练 shard 方案。

## 关键实现位置

- `expert_trajectory_collector/batch.py`：CLI 和 v002 配置入口。
- `expert_trajectory_collector/campaign/runner.py`：进程池和 manifest。
- `expert_trajectory_collector/campaign/free_flight.py`：路线剪枝、预检、condition 采样和熔断。
- `expert_trajectory_collector/campaign/state.py`：状态、短缺和目标可达性。
- `expert_trajectory_collector/campaign/io.py`：并发安全原子落盘。
- `expert_trajectory_collector/campaign/monitor.py`：轻量 Web 监控。

## 已知注意事项

- 监控默认端口为 8785；8765 可能被旧交互 UI 占用。
- 安全暂停在 condition 边界检查，不会中断正在执行的 OMPL/B-spline 调用。
- `capacity_exhausted` 表示达到最大 condition 数或连续失败熔断，应该保留并分析，不能盲目复制路径补齐。
- `campaign_config.json` 是恢复的权威配置；恢复时只改变 worker/监控运行参数。
- 普通引导不是硬 waypoint；orientation-sensitive passage 仍保留必要的硬姿态约束。
