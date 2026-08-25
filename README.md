# HNUTER Expert Trajectory Collector

面向无人机任务的独立专家轨迹采集器。项目将任务条件生成、场景验证、OMPL 专家规划、B-spline 平滑、URDF 精确碰撞检测、可视化检查和统一数据导出组织在稳定的任务插件边界之后。

当前完整启用 `free_flight`。`inspection` 与 `surface` 已注册条件/轨迹契约和扩展入口，尚未接入各自的精确 validity checker，因此 UI 会明确显示为“规划中”，不会把自由飞行逻辑冒充为其他任务。

## 快速启动

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[planner,test]'
.venv/bin/python native/ompl_region_sampler/build_extension.py
.venv/bin/python -m expert_trajectory_collector
```

打开 `http://127.0.0.1:8765`。如果已有包含 `numpy`、`ompl` 与 `coal` 的 Python 环境，也可以直接使用该解释器，不要求虚拟环境目录具有固定名称：

```bash
/path/to/python -m expert_trajectory_collector
```

查看任务目录或在命令行生成一个条件：

```bash
python3 -m expert_trajectory_collector --describe-tasks
python3 -m expert_trajectory_collector --generate \
  --family multi_homotopy --seed 42 --obstacle-count 8
```

历史入口仍兼容：

```bash
python3 obstacle_scene_builder.py
```

## 为什么需要原生扩展

区域偏置不是把锚点塞进最终路径，而是在 RRTConnect 查询内部注册自定义 `StateSamplerAllocator`，以 70% 区域采样和 30% 全局采样提高有效通道的探索效率。因此扩展必须针对当前 Python 与 OMPL ABI 构建，生成的 `.so` 只保留在本机，不提交到 Git。

构建脚本需要 `git`、CMake、C++ 编译器和可导入的 OMPL Python 包。首次构建会下载与 Python wheel 同版本的 OMPL 源码及 nanobind 子模块。

## 架构

```text
Web UI / CLI / batch client
            |
            v
ExpertCollectorService                 与任务无关的用例层
            |
            v
TaskRegistry -> TaskPlugin             稳定扩展边界
                    |
        +-----------+------------------+
        |                              |
 FreeFlightTaskPlugin        Inspection / Surface plugins
        |                              |
 scene + RRTConnect + spline    专属条件与 validity checker
```

- `expert_trajectory_collector/`：任务契约、注册表、服务、Web API 与 UI。
- `obstacle_scene_builder.py`：自由飞行场景族、路线证书、token 和验证。
- `obstacle_scene_experts.py`：专家生成与验收策略。
- `ompl_se3_planner.py`：SE(3) RRTConnect 与区域偏置采样接口。
- `multi_waypoint_planner.py`：多段规划、B-spline 平滑和净空修复。
- `coal_collision.py`：基于完整 URDF primitive 的 COAL 最近点/距离查询。
- `native/ompl_region_sampler/`：自定义 OMPL sampler 的 C++ 扩展源码。
- `expert_trajectory_collector/assets/`：随 Python 包分发的无人机碰撞 URDF。

统一采集记录使用 `expert_trajectory_collection_record_v001`，稳定保存 `task_type`、`task_contract`、`condition`、`conditioning`、`expert_set` 与 `collection_metadata`。任务专属 schema 保留在插件内部，便于后续 inspection 保存 FOV/LOS 与目标几何，surface 保存曲面参数域、接触约束、intrinsic states 和 lifted pose。

更多说明见 [采集器架构](docs/EXPERT_TRAJECTORY_COLLECTOR.md) 与 [自由飞行场景/专家策略](docs/OBSTACLE_SCENE_BUILDER.md)。

## 测试

```bash
python3 -m pytest
node --check expert_trajectory_collector/web_assets/app.js
```

项目使用 GPL-3.0 许可证。
