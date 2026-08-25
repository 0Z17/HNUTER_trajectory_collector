# Free-flight v002 pilot 后台采集运行手册

## 当前边界

本批处理器只采集**无时间参数的几何专家路径**。它不会启动三维可视化、URDF 残影、TOPP-RA 或 MuJoCo/MPPI rollout。v001 pilot 最终得到 3,544/4,000 条并暴露端点重采样过严、姿态平滑边界和状态并发写入问题；v002 pilot 已完成 100/100 环境和 4,000/4,000 路径。后续单通道 expert-count 优化版必须使用新的 dataset ID 与输出目录，不能混写已完成的 v002 数据集。

批处理不是浏览器循环请求。主进程按环境分派多进程 worker；同一环境只创建一次碰撞检查器，并连续采样多个 start/goal condition。每个 condition 完成后原子写入，因此异常退出后不会丢失已完成 condition。

`--experts-per-condition` 是普通 family 的默认值。已知只有单一有效路线容量的 `orientation_sensitive_passage` 和 `staggered_corridor` 在 batch 中固定覆盖为每 condition 1 条，避免为不存在的同端点非重复变体反复规划；Web UI 的交互式 expert 数量不受影响。该覆盖会写入 `campaign_config.json` 的 `experts_per_condition_overrides`，恢复时属于不可改变的数据集定义。

v002 不再要求每个随机端点同时保留环境生成时的全部路线。它会逐条重验并剪除失效模式：普通/optional 场景至少保留一条，guaranteed-multi 至少两条，`multi_homotopy` 仍保留垂直+水平覆盖，`frame_doorway` 仍保留 `through`。环境正式进入数据集前执行 4 次轻量 condition probe；不可稳定重采样的 seed 会被替换。condition 最多采样 64 次/4 秒，连续 8 个零接受 condition 后熔断，防止坏 seed 长时间占用 worker。

## 解释器与原生扩展

推荐使用已验证环境：

```bash
cd /home/z017/research/HNUTER_trajectory_collector
/home/z017/research/curobo_env/bin/python native/ompl_region_sampler/build_extension.py
```

确认 `ompl`、`coal` 与本地 `_ompl_region_sampler*.so` 都能被同一解释器导入。不要混用 Python 3.10 编译的扩展与 Python 3.12 虚拟环境。

## v002 pilot 参考命令（已完成；新运行必须更换输出目录和 dataset ID）

建议把输出放到仓库外，避免大型数据进入 Git：

```bash
cd /home/z017/research/HNUTER_trajectory_collector
mkdir -p /home/z017/research/HNUTER_datasets
nohup /home/z017/research/curobo_env/bin/python \
  -m expert_trajectory_collector.batch run \
  --output /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002 \
  --dataset-id free_flight_pilot_100env_v002 \
  --seed 20260825 \
  --environment-count 100 \
  --paths-per-environment 40 \
  --nominal-conditions-per-environment 8 \
  --experts-per-condition 5 \
  --maximum-conditions-per-environment 48 \
  --condition-sampling-max-attempts 64 \
  --condition-sampling-timeout 4.0 \
  --environment-precheck-condition-count 4 \
  --environment-precheck-minimum-successes 2 \
  --environment-precheck-max-attempts 24 \
  --environment-precheck-timeout 1.0 \
  --maximum-consecutive-condition-failures 8 \
  --terminal-attitude-margin-deg 5.0 \
  --workers 12 \
  --solve-time 0.20 \
  --maximum-planner-attempts 12 \
  --obstacle-count-min 6 \
  --obstacle-count-max 18 \
  --monitor-host 127.0.0.1 \
  --monitor-port 8785 \
  > /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002.log 2>&1 &
```

本机为 16 核/32 线程、约 30 GiB RAM，首轮推荐 12 个 worker。若内存稳定且 CPU 未饱和可增加到 14；不要直接使用 32。

监控页：`http://127.0.0.1:8785`。默认不用 8765，因为交互式场景服务可能占用该端口。

## 状态、暂停、继续和恢复

终端查询：

```bash
/home/z017/research/curobo_env/bin/python -m expert_trajectory_collector.batch status \
  --output /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002
```

安全暂停只在 condition 边界生效；正在规划的单个 condition 会先完成并落盘：

```bash
/home/z017/research/curobo_env/bin/python -m expert_trajectory_collector.batch pause \
  --output /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002

/home/z017/research/curobo_env/bin/python -m expert_trajectory_collector.batch resume \
  --output /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002
```

`resume` 控制命令只是移除暂停标记；如果进程已经退出，还需要重新启动采集器。恢复时不会重新提交已经 `complete` 或 `capacity_exhausted` 的环境：

```bash
nohup /home/z017/research/curobo_env/bin/python \
  -m expert_trajectory_collector.batch run \
  --output /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002 \
  --resume --workers 12 --monitor-port 8785 \
  >> /home/z017/research/HNUTER_datasets/free_flight_pilot_100env_v002.log 2>&1 &
```

恢复时数据集定义从已有 `campaign_config.json` 读取，只允许改变 worker 数和监控参数，避免无意中把两套分布写进同一个目录。

`status.json` 的 `target_reachable` 会考虑已经 capacity exhausted 的环境。若为 `false`，`eta_s` 为 `null`，并通过 `path_shortfall` 与 `reachable_path_upper_bound` 报告真实缺口；全部环境结束但路径不足时状态为 `finished_with_shortfall`，不再伪装成仍能达到目标。

## 落盘内容

```text
output_root/
  campaign_config.json
  provenance.json
  runtime.json
  status.json
  dataset_manifest.json
  PAUSED                         # 仅暂停时存在
  RUNNING                        # 仅主进程运行时存在
  environments/
    env_000000/
      environment.json
      progress.json
      conditions/
        condition_000/
          condition.json.gz
          expert_metadata.json
          paths.npz
        condition_001/...
```

`paths.npz` 保存：

- 变长原始 OMPL pose7：扁平 `ompl_path_pose7` + `ompl_path_offsets`；
- 256 点 B-spline pose7：`geometry_path_pose7`；
- 128 点 diffusion 标签：`training_path_pose9`；
- `normalized_arc_progress`；
- 完整 URDF/COAL 逐点净空：`clearance_m`。

元数据保留规划 seed、引导/实际拓扑、采样统计、拒绝原因、平滑和净空修复指标。环境与 condition 不会为每条路径重复保存。当前 pilot 是恢复友好的逐 condition 文件；正式 200k 冻结前再离线打包为 512–2,048 条/片的 shard，不在采集热路径中增加复杂度。

## pilot 完成后的验收

不要仅检查 `4000/4000`。至少汇总：

- 十个 family 的环境数、成功路径数、耗时和接受率；
- 每环境 condition 数以及每 condition 接受路径数；
- start/goal XYZ、高度差、姿态差分布；
- topology/mode 数量与熵、上下/左右覆盖；
- fixed fallback、region-biased、证书外可行拓扑比例；
- RRT exact、raw valid、B-spline valid、最终 accepted 的阶段率；
- minimum clearance、曲率、backtracking、长度和姿态分布；
- capacity exhausted/failed 环境及其日志。

正式 200k 采集前应根据 pilot 调整 family 配额、单 condition 专家数、worker 数和超时，随后冻结配置与 split 规则。
