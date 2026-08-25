# HNUTER 统一专家路径数据规范 v001

状态：建议作为 5,000 环境 / 200,000 条专家路径采集的冻结前规范。

## 1. 数据集目标与边界

当前训练目标是让 diffusion 生成**无时间参数的几何 SE(3) 路径**，而不是直接生成控制轨迹。推荐的主监督为固定长度 `pose9` 路径：

```text
[x, y, z, rotation_6d]，shape = [128, 9]
```

路径按归一化弧长 `s∈[0,1]` 重采样，不包含速度、加速度、控制量或真实时间。free-flight 的姿态会影响完整 URDF 碰撞，因此不能退化成纯 XYZ 路径；inspection 和 surface 也继续保留各自的姿态、传感器或工具约束。

推荐部署链为：

```text
diffusion 几何路径
  -> 端点/旋转合法化
  -> 几何平滑与 COAL 连续碰撞检查
  -> TOPP-RA 或动力学感知的时间参数化
  -> MPPI / 实机控制器跟踪
  -> 必要时局部修复或重新生成
```

因此数据应把“几何专家”“时间参数化结果”“闭环 rollout”保存为三个相互关联但不混合的阶段。后两者可以缺省，不能反过来覆盖原始几何专家。

## 2. 从现有仓库继承的信息

### 2.1 当前 free-flight 采集器

当前 `scene_expert_trajectories_v001` 已经记录：

- 数据集级：scene、start/goal、采集 seed、规划模式、碰撞后端、可用拓扑类别、采样/回退阶段统计、接受流水线及失败原因；
- 单路径级：trajectory ID、planner seed、尝试序号、RRT range、solve time、目标引导类别和实际分类、生成阶段、waypoint/证书角色；
- 路径数组：变长 `ompl_path` 与 256 点 `bspline_path`，均为 `[x,y,z,qw,qx,qy,qz]`；
- 几何指标：长度、总旋转量、最大 roll/pitch、最小净空、曲率、纵向折返、局部弦长效率；
- 多样性指标：同类最近位置/姿态距离、是否放宽重复阈值；
- 采样诊断：区域/全局样本数、被拒区域样本数、sampler 分配次数；
- B-spline 净空修复：是否启用/尝试/成功、迭代次数、修复前后净空和最大控制点位移。

这些字段应完整保留，不应只导出最终的 128 点训练 tensor。

### 2.2 旧 free-flight 完整管线

旧数据管线还曾保存：

- `condition_start_goal`、归一化进度、reference/actual time；
- TOPP-RA reference state、线/角加速度和规划净空；
- MuJoCo actual state、tracking reference、MPPI action 和实际净空；
- planned/actual topology、reference duration、跟踪 RMSE、终点误差；
- 环境/轨迹文件路径与 SHA-256、运行命令、Git revision、Python/Torch/平台信息。

这些内容适合作为可选 `retiming` 与 `rollout` 审计，不作为当前 path diffusion 的主标签。

### 2.3 Inspection 与 Surface

现有 inspection 专家包含：

- `states_wxyz`、128 点 `trajectory_pose9`；
- minimum clearance、minimum visible fraction；
- planner seed/time、最近专家距离和 mode metadata；
- environment、sensor config、URDF 的路径与 SHA-256。

现有 surface 专家包含：

- 曲面参数域中的 `intrinsic_states = [u,v,(roll)]`；
- lifted 128 点 `trajectory_pose9`；
- minimum clearance、planner seed/time、最近专家距离和 mode metadata；
- surface task version、URDF 与环境哈希。

统一数据结构必须允许这些任务专属数组存在，不能把它们强制编码为 free-flight 的 OBB-only condition。

## 3. 数据组织

建议目录结构：

```text
dataset_root/
  dataset_manifest.json
  splits.json
  environments/
    environments_00000.jsonl
  conditions/
    conditions_00000.jsonl
  experts/
    expert_index_00000.jsonl
  arrays/
    geometry_00000.npz
    geometry_00001.npz
  attempts/
    attempts_00000.jsonl
  audits/
    retiming_00000.jsonl
    rollout_00000.jsonl
    audit_arrays_00000.npz
  reports/
    quality_summary.json
    distribution_summary.json
    normalization_stats.json
```

不要生成 200,000 个零散小文件。固定长度训练数组建议每 512–2,048 条组成一个 shard；JSONL 索引通过 `shard_path + row_index` 定位数组。变长 OMPL 路径可使用扁平数组加 offsets，或单独按较大的 shard 保存。

环境与 condition 只存一次，expert 使用 ID 引用，避免为每条路径重复保存相同的 32×10 obstacle tokens。

## 4. 共享外层记录

采集 API 继续保留现有的 `expert_trajectory_collection_record_v001` 外层：

```text
task_type
task_contract
condition
conditioning
expert_set
collection_metadata
```

这是单个 condition 的无损来源记录。构建大数据集时，再把它规范化拆成 environment、condition、expert 和 array shard，以避免重复存储；规范化记录必须保留 source record 的 ID/hash，能够反向追溯。

每个成功专家使用以下公共身份字段：

| 字段 | 要求 | 含义 |
|---|---:|---|
| `schema_version` | 必须 | `expert_path_record_v001` |
| `dataset_id` | 必须 | 数据集冻结版本 |
| `task_type` | 必须 | `free_flight` / `inspection` / `surface` |
| `task_version` | 必须 | 任务条件与 validity 的版本 |
| `environment_id` | 必须 | 物理环境实例 |
| `condition_id` | 必须 | 同一任务条件；包含同一 start/goal |
| `group_id` | 必须 | split 防泄漏分组 |
| `expert_id` | 必须 | 全数据集唯一 |
| `split` | 必须 | train/interpolation/task_ood/map_ood/compound_ood |
| `status` | 必须 | accepted/rejected/quarantined |
| `array_ref` | 必须 | shard、row 和数组 schema |
| `created_at_utc` | 必须 | UTC 时间 |

`group_id` 至少要把同一基础几何及其旋转、平移、轻微尺寸变体放在同一 split。不能按 trajectory 随机切分，否则同一地图和同一 start/goal 会同时出现在训练与测试中。

## 5. Dataset manifest

`dataset_manifest.json` 至少记录：

- schema/dataset version、状态 `collecting|complete|frozen`；
- 请求与实际的 environment、condition、expert 数量；
- task/family/split 数量及比例；
- collector、scene generator、planner、smoother、collision checker 的版本；
- Git commit、是否 dirty、采集配置路径和 SHA-256；
- Python、NumPy、OMPL、COAL 版本及原生扩展 ABI；
- URDF 路径、base link、collision primitive 列表和 SHA-256；
- 坐标系、长度/角度单位、quaternion 顺序、pose6D 约定；
- 轨迹长度、dtype、压缩与 shard 约定；
- seed 派生规则、恢复粒度和失败日志位置；
- 所有 shard 的路径、大小、样本数和 SHA-256；
- 质量阈值、汇总统计和冻结时间。

正式冻结后不得原地修改文件；修复应产生新 dataset version。

## 6. Environment 记录

### 6.1 公共字段

- `environment_id`、`group_id`、family/type、generator seed；
- 完整生成参数及其采样范围，不能只保存最终值；
- bounds、sampling bounds、坐标系与单位；
- 原始环境 JSON/XML 路径和 SHA-256；
- URDF reference/hash、安全边距和碰撞后端；
- 结构障碍和随机障碍列表、功能角色、物理支撑信息；
- 模型实际使用的 conditioning payload、schema、tokens 和 mask；
- generation/validation 状态与全部校验问题；
- 路线证书、可用 route modes、expert planning guides；
- 障碍功能统计：route selector、clearance shaper、distractor；
- 难度特征：障碍数、占据率、通道宽度、预计最小净空、可行模式数。

### 6.2 任务专属环境字段

- Inspection：目标几何、ROI、sensor extrinsics、FOV、range、LOS/遮挡设置；
- Surface：surface geometry/chart、参数域、tool extrinsics、接触/法向/高度约束。

## 7. Condition 记录

一个 condition 是模型的一次条件输入，而不仅是一张地图：

- `condition_id`、`environment_id`、`group_id`、task type/version；
- start/goal 的 canonical pose9，以及无损 pose7；
- start/goal 采样 seed、区域、采样尝试次数；
- start-goal 距离、高度差、姿态差和 direct-path validity；
- condition-specific task features；
- 可用拓扑/模式标签，但明确标记其是否只是有限证书集合；
- condition validation 结果；
- 当前 condition 保留/剪除的路线模式、最小模式要求、端点采样尝试数与耗时；环境级证书库不应被误当成每个随机端点都必须同时满足的全称约束；
- 该 condition 请求、尝试、接受、重复和失败的专家数。

Inspection 额外保存目标/可见性条件；Surface 额外保存 start/goal intrinsic state 和 phase/contact 条件。

## 8. Expert 路径记录

### 8.1 规划与来源

- planner name/version、seed、attempt、range、solve-time budget 和实际规划时间；
- planning mode、sampler 类型、区域/全局采样概率和采样统计；
- 目标 guide、实际 topology/mode、分类置信或模板距离；
- generation stage、fallback/recovery、是否使用固定姿态点；
- waypoint/guidance 只作 proposal 还是硬约束；
- OMPL exact/approximate 状态、termination reason；
- 接受/拒绝原因和完整 attempt diagnostics。

即使最终只保留成功路径，也必须单独保存所有失败 attempt 的 seed、阶段、耗时与原因，用于分析成功样本偏置和容量瓶颈。

### 8.2 几何数组

必须保存：

| 数组 | shape | dtype | 说明 |
|---|---:|---:|---|
| `ompl_path_pose7` | `[Nraw,7]` | float32/64 | 原始、变长、wxyz |
| `geometry_path_pose7` | `[Nsmooth,7]` | float32 | 最终碰撞自由 B-spline |
| `training_path_pose9` | `[128,9]` | float32 | diffusion 主监督 |
| `normalized_arc_progress` | `[128]` | float32 | 0 到 1 |
| `clearance_m` | `[128]` | float32 | 完整 URDF/COAL 净空 |

Quaternion 必须归一化、使用 `wxyz`，并进行相邻符号连续化；pose9 的 rotation-6D 约定必须在 manifest 中固定。`training_path_pose9` 必须由最终几何路径按弧长生成，不能从带控制误差的 MuJoCo rollout 反推。

任务专属可选数组：

- Inspection：逐点 visible fraction、LOS/FOV/range validity；
- Surface：`intrinsic_states`、逐点 surface/contact/normal error、phase ID；
- Free-flight：逐点 roll/pitch、route signature descriptor。

### 8.3 平滑与净空修复

- spline method/degree、knot/control point count、平滑权重；
- hard waypoint indices；
- clearance-gradient backend 与参数；
- 修复是否尝试/成功、迭代数、前后最小净空、最大控制点位移；
- 平滑前后路径长度、旋转量和碰撞状态。

### 8.4 几何质量指标

- path length、start-goal chord、detour ratio；
- orientation arc length；
- maximum absolute roll/pitch；
- minimum/percentile clearance；
- maximum curvature、纵向 backtracking、minimum local chord efficiency；
- waypoint/guide position 与 attitude RMS；
- 同 condition、同 topology 下的最近位置/姿态距离；
- duplicate、diversity relaxed、quality gate 结果；
- exact continuous collision audit 的采样间隔和最大角步长。

## 9. 可选 TOPP-RA 记录

TOPP-RA 不作为 diffusion 标签的一部分，使用 `retiming_status = not_run|passed|failed` 关联到 expert ID。

建议至少保存轻量 summary：

- retimer/config version、速度/角速度/线加速度/角加速度上限；
- safety scale、grid/validation 点数、solver、refinement 次数；
- duration、monotonic、start/end path speed；
- maximum linear/angular speed；
- maximum per-axis linear/angular acceleration；
- 失败原因。

需要重放时再保存可选 dense arrays：time、path position/speed/acceleration、pose、world linear velocity/acceleration、body angular velocity/acceleration。

TOPP-RA 在这里证明的是给定限制下的**几何路径运动学时间参数化**。它不能单独证明多旋翼推力、姿态与平动耦合后的完整动力学可行性。

## 10. 可选 MuJoCo/MPPI rollout 记录

使用 `rollout_status = not_run|passed|failed`，并记录：

- simulator/model/controller/config version 与 SHA-256；
- 控制周期、仿真周期、随机化参数和 rollout seed；
- reference 来源及其 retiming ID；
- `time`、`reference_state13`、`actual_state13`、`action`；
- reference/actual physical clearance、collision/contact events；
- position/attitude RMSE、最大误差、终点误差、完成时间；
- action magnitude/continuity/saturation、控制失败原因；
- planned topology 与 actual topology 是否一致。

Rollout 是部署链审计或后续可行性模型的标签，不应替换干净的几何专家路径。

## 11. 5,000 环境 / 200,000 路径如何分配

平均为每个环境 40 条路径。推荐把它拆成：

```text
每环境 8–10 个不同 condition（不同 start/goal 和任务参数）
每 condition 4–5 条有效且非重复的专家路径
```

另一种可接受配置是 5 个 condition × 8 个专家。不要为同一 start/goal 强行凑 40 条；很多场景没有 40 个有意义的几何模式，这会产生高度相关样本、固定 waypoint 回退和错误的“多样性”。

同一 condition 内的路径必须共享完全相同的 start/goal，才能让 diffusion 学习条件多模态；不同 condition 用于覆盖起终点分布。路径数量应按实际可行 mode 容量自适应：简单场景少留，拓扑丰富场景多留，最终通过全局配额平衡。

如果 5,000 个环境当前全部用于 10 个 free-flight family，可以先以每族约 500 个环境作为基线，再把配额从简单 clutter 向 narrow、orientation-sensitive、mixed industrial 等困难族倾斜。不要让“容易生成且接受率高”的 family 自然占据数据集多数。

一个可执行的 map-level 分配示例：

| split | 环境数 | 约路径数 | 用途 |
|---|---:|---:|---|
| train | 3,500 | 140,000 | 主训练 |
| interpolation | 500 | 20,000 | 同分布新 seed/参数 |
| task_ood | 400 | 16,000 | 起终点、姿态或任务条件外推 |
| map_ood | 400 | 16,000 | 留出结构组合/几何范围 |
| compound_ood | 200 | 8,000 | 地图与任务同时外推 |

OOD 不能只是随机换 seed；必须在生成前冻结被保留的结构组合、尺寸区间、通道类型或任务参数区间。

仅 `200000×128×9×float32` 的训练路径约占 0.86 GiB；再保留 256 点 pose7 几何路径约增加 1.34 GiB。条件去重和分 shard 后，完整几何库仍是易管理的量级；全量保存 dense TOPP-RA 与 MuJoCo 时序则会显著增加文件数、容量和采集时间，这也是将其设计成可选审计层的原因。

## 12. 数据量是否足够

结论：**对只生成 128 点 SE(3) 几何路径的首个正式 diffusion 模型，5,000 个有效环境和 200,000 条非重复路径通常已经是充足且偏大的起点；但数量本身不能保证足够。**

真正决定有效样本量的是：

1. 独立 environment/group 数，而不是同一条件的 RRT seed 数；
2. 独立 condition 数及 start/goal/姿态覆盖；
3. family、障碍功能、拓扑类别、上下/左右绕行和姿态敏感路径的平衡；
4. guide/fallback 产生的模式是否真实多样；
5. train 与 map/task OOD 是否严格无泄漏；
6. 低净空、高曲率和困难任务是否有足够覆盖。

建议目标约为 40,000–50,000 个 condition，而不是只有 5,000 个 condition 各重复 40 次。先生成嵌套子集 `D25k/D50k/D100k/D200k`，使用完全相同的验证/OOD 集绘制成功率、碰撞率、mode coverage 和 best-of-K 曲线。如果 100k 后已经稳定平台，则无需因为预设数字继续堆同质路径；若 map-OOD 仍明显提升，则优先增加环境和 condition，而不是增加同 condition 的 seed。

## 13. TOPP-RA 与真实 rollout 的建议

### TOPP-RA

- 不需要作为 path diffusion 的训练标签；
- 建议对 100% 最终几何路径运行轻量 retiming feasibility，并只强制保存 summary；
- dense retimed arrays 可按需生成或只对审计子集保存；
- 失败路径不必立即删除，应标记失败原因，判断是路径曲率/姿态问题还是参数上限过紧。

TOPP-RA 相比 OMPL 与 MuJoCo rollout 通常较便宜，覆盖全量可以提前验证未来后处理链是否闭合，但它仍只是运动学筛查。

### MuJoCo + MPPI

- 不建议对 200,000 条全部 rollout；这会把采集成本主要耗在控制仿真上，也会把特定控制器偏差混入几何专家定义；
- 建议先做 5,000–10,000 条分层 rollout（约 2.5%–5%）；
- 至少覆盖每个环境一条，再过采样 narrow/orientation-sensitive、低净空、高曲率、证书外拓扑、固定回退以及 TOPP-RA 边界样本；
- 将 rollout 成败作为审计标签。后续若发现几何指标无法预测闭环失败，再扩充 rollout 或训练独立的 feasibility/ranker。

由于多旋翼是欠驱动系统，最终实机可信度应由动力学感知后处理和代表性闭环 rollout 建立，而不是要求 diffusion 自身学习时间和控制量。

## 14. 采集前必须冻结的质量报告

正式跑 200k 前，先用小批次验证并冻结以下报告：

- 每 family/task/split 的 environment、condition、expert 数；
- start/goal XYZ、姿态差、高度差和直线距离分布；
- obstacle count/role/height、gap、净空和 route mode 分布；
- topology/mode 计数、Shannon entropy 和每 condition mode coverage；
- region-biased、fixed fallback、clearance repair 的比例；
- RRT exact、raw valid、B-spline valid、final accepted 各阶段接受率；
- 路径长度、detour、旋转、roll/pitch、曲率、backtracking 分布；
- duplicate rate、每 condition 有效专家数和容量耗尽率；
- TOPP-RA 通过率及失败分层；
- rollout 抽样的 collision、tracking 与终点成功率；
- 所有 split 的 group 泄漏检查与文件 hash 检查。

建议先进行约 100 环境 / 4,000 路径的 pilot。只有当 schema、恢复机制、失败日志和分布报告均稳定后，再启动完整采集，避免在 200k 结束后才发现字段缺失或 split 泄漏。
