# HNUTER Expert Trajectory Collector

该工具把任务条件生成、专家规划、验证、conditioning 导出和专家记录封装为一个可扩展采集器。当前完整启用 `free_flight`；`inspection` 与 `surface` 已注册稳定扩展契约，但在各自的条件生成器和精确 validity checker 接入前不会伪装成可用任务。

## 启动

推荐入口：

```bash
python3 -m expert_trajectory_collector
```

浏览器访问 `http://127.0.0.1:8765`。历史命令仍兼容：

```bash
python3 obstacle_scene_builder.py
```

查看已注册任务及能力：

```bash
python3 -m expert_trajectory_collector --describe-tasks
```

## 架构

```text
Web UI / CLI / batch client
            |
            v
ExpertCollectorService                  task-agnostic use cases
            |
            v
TaskRegistry -> TaskPlugin              stable extension boundary
                    |
        +-----------+-----------+
        |                       |
 FreeFlightTaskPlugin     future Inspection/Surface plugins
        |                       |
 scene domain + OMPL       exact task validity + native collector
```

- `expert_trajectory_collector/contracts.py`：任务描述、能力和插件抽象。
- `expert_trajectory_collector/registry.py`：任务发现和注册，不依赖几何实现。
- `expert_trajectory_collector/service.py`：生成、验证、专家采集和统一记录导出。
- `expert_trajectory_collector/web.py`：仅负责 JSON/static HTTP 传输。
- `expert_trajectory_collector/tasks/free_flight.py`：现有障碍场景与专家规划的兼容适配层。
- `obstacle_scene_builder.py`：只保留 free-flight 几何、路线证书、token 和验证领域逻辑。
- `obstacle_scene_experts.py`：只保留 free-flight OMPL/B-spline 专家生成。

## 通用任务契约

采集器使用 `condition` 而不是把所有任务都称为 `scene`。不同任务可以拥有不同条件：

- Free-flight：环境 OBB、start/goal 区域、路线证书和规划引导；
- Inspection：环境、目标几何、传感器外参、视距/FOV/LOS 约束和 start/goal；
- Surface：环境、曲面几何与参数域、工具外参、接触约束和 start/goal。

每个插件分别实现：

1. `generate_condition`；
2. `validate_condition`；
3. `conditioning_payload`；
4. `collect_experts`；
5. 可选的 `generate_batch`。

Inspection 应复用 `inspection_v2_collector` 的碰撞与可见性联合 validity；Surface 应复用 `surface_v2_collector` 的 intrinsic OMPL，并同时保留 intrinsic states 与 lifted pose9。两者不应强行套用 free-flight 的 10-D OBB-only conditioning 或 pose7 轨迹格式。

## API

- `GET /api/tasks`：任务目录、状态、schema 和能力；
- `GET /api/health`：服务健康检查；
- `POST /api/generate`：生成任务条件；
- `POST /api/validate`：验证编辑后的任务条件；
- `POST /api/experts`：收集并验证专家轨迹；
- `POST /api/batch`：生成任务条件批次；
- `POST /api/collection`：导出统一专家采集记录。

未提供 `task_type` 时默认使用 `free_flight`，原有 `/api/generate`、`/api/validate`、`/api/experts` 客户端仍可工作。

## 统一采集记录

UI 的“导出专家采集记录”生成 `expert_trajectory_collection_record_v001`：

```text
task_type
task_contract
condition
conditioning
expert_set
collection_metadata
```

这个外层记录是跨任务稳定的；`condition`、`conditioning` 和 `expert_set` 的内部 schema 由任务插件声明。这样后续 inspection/surface 可以保存任务专属状态，而不会破坏 free-flight 数据格式。
