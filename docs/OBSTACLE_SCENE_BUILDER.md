# Expert Trajectory Collector：Free-flight 任务插件

该插件在固定的 `8 × 8 × 4 m` 工作空间中生成参数化 OBB 条件，并通过通用专家轨迹采集器提供预览、编辑、校验、专家规划和导出。

## 启动

推荐在仓库根目录运行：

```bash
python3 -m expert_trajectory_collector
```

历史入口继续兼容：

```bash
python3 obstacle_scene_builder.py
```

若要启用完整 URDF 的 COAL 最近点梯度修复，必须使用已安装
`coal` 的 Python 环境启动。环境不提供 `bin/activate` 时，可以直接调用
对应解释器：

```bash
/path/to/python -m expert_trajectory_collector
```

区域偏置 RRT-Connect 首次使用前，需要用同一个解释器编译一次小型
`StateSamplerAllocator` 扩展：

```bash
/path/to/python \
  native/ompl_region_sampler/build_extension.py
```

构建脚本读取当前 Python wheel 的 OMPL 版本和 libstdc++ ABI，并从 OMPL
官方仓库取得完全匹配的头文件；生成的 `.so` 与依赖缓存均被 Git 忽略。
更换 Python 或 OMPL wheel 后应重新运行该命令。

状态栏会显示 `COAL梯度修复`；若解释器没有绑定，则明确显示
`COAL梯度不可用` 并回退到原 OBB 检查。

然后访问 <http://127.0.0.1:8765>。服务只监听本机；前端使用原生 Canvas，不需要安装 npm 包或访问 CDN。可用 `--host` 和 `--port` 修改监听地址。

也可以在命令行生成一个 environment JSON：

```bash
python3 obstacle_scene_builder.py --generate \
  --family staggered_corridor --seed 42 --obstacle-count 8 \
  > environment_staggered_42.json
```

## 支持的场景族

- `sparse_obb_clutter`
- `central_block`
- `multi_homotopy`
- `narrow_passage`
- `orientation_sensitive_passage`
- `wall_protrusion_bracket`
- `frame_doorway`
- `pillar_wall`
- `staggered_corridor`
- `mixed_industrial`

每个族支持确定性 seed，以及障碍物数量、尺寸、gap、整体 yaw 和 XY 平移的随机范围。勾选“每次随机”后，每次点击生成都会使用新的随机 seed；关闭后，相同 seed 与范围会严格复现同一个场景。模板核心的尺寸、偏置、镜像和结构相位也会随 seed 改变，而不仅仅是外围障碍物变化。默认 `1.00 m` 上限主要约束普通松散障碍的 XY 占地；高度单独采样，并对最终成功放置的障碍尝试达到 65% 的有效高度配额。参考旧 XML 中约 `0.28–0.30 m` 宽的高柱，生成器会使用 `0.24–0.40 m` 细柱承载 `2.45–3.60 m` 高度，或使用 `1.55–2.40 m` 的中型阻挡。只有在多条完整 URDF 扫掠通道已占满剩余高障碍位置时，才自适应回退到少量低矮可上飞障碍。

生成器从 `expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf` 的 `base_link` 读取 7 个 active collision primitive，推导保守碰撞 AABB 和安全边距。当前模型得到的包络约为 `1.219 × 1.295 × 0.547 m`，安全边距约为 `0.129 m`。

额外障碍物使用 OBB separating-axis test 进行增量采样，并按任务作用分成三类。`route_selector` 可以阻断部分旧路线证书，但必须满足场景族的最少路线生存约束；`clearance_shaper` 不阻断现有证书，但必须进入路线的 `0.42 m` 膨胀 URDF 净空壳层；`distractor` 不影响证书路线，目标比例约为 15%，硬上限为填充障碍的 35%。因此障碍数量增加时，不再把所有旧路线设为绝对禁入区，也不会为了凑满 cross-attention token 无限制地堆放低矮干扰物。若达到当前 seed 的有效障碍容量，生成会明确失败并要求降低数量或更换 seed。

填充障碍全部落地；需要离地的横梁只作为完整门框、管架等结构的一部分生成，并带有可见细柱支撑，贴近天花板的构件则显式标记为吊装结构。校验器会报告既不落地、也没有结构支撑声明的悬浮物。UI 建议将 8–18 个障碍作为常规训练范围，20 个以上用于少量压力测试；32 是 token 容量上限，不是每张地图的生成目标。

`sparse_obb_clutter`、`central_block`、`multi_homotopy`、`frame_doorway` 和 `pillar_wall` 是“保证多通道”族：生成时先构造共享同一 start/goal 的初始路线库，功能障碍加入后至少保留两条碰撞自由路线证书。门框的 `through` 证书不可被随机填充淘汰；`multi_homotopy` 的初始母版仍认证 `above / below / left / right` 四种路线，但最终场景允许淘汰部分证书，同时强制至少保留一个 `above/below` 垂直族和一个 `left/right` 水平族。`staggered_corridor`、`mixed_industrial` 和 `wall_protrusion_bracket` 在初始路线库存在多个证书时同样至少保留两个，否则保留一条。姿态敏感通道的四个 aperture 姿态点始终作为严格约束保护。这里的路线模式是场景生成与数据平衡的实用代理，不表示已穷举或严格证明了全部同伦类；后续 OMPL 仍可发现证书库之外的有效路线。

## 与 cross-attention 的契约

Environment 导出遵循仓库现有 box schema。所有满足 `collision=true`、`type=box` 且 `role != floor` 的障碍物按以下顺序编码：

```text
[x, y, z, size_x, size_y, size_z, qw, qx, qy, qz]
```

输出补齐至 `[32, 10]`，并附带 `[32]` boolean mask。实现与 `se3_diffusion._obstacle_tokens` 逐值测试一致。界面中的“导出 Environment”用于现有数据采集流程；“导出 Tokens”主要用于检查和调试。

## Web UI 操作

- 在左侧选择模板和随机化参数，然后生成场景。
- 在视口中左键拖拽环绕、右键/中键拖拽或 `Shift + 左键` 平移、滚轮缩放、点击 OBB 进行选择；也可切换顶视/前视并复位相机。环绕采用相机仰角定义并限制在地板上方 `5–88°`，不会再翻转到地板下方；右上角会显示当前 `ELEV`。
- 在右侧编辑选中 box 的位置、尺寸和 yaw，或添加/删除 box。
- 校验器检查 32-token 上限、box 字段、四元数归一化以及旋转 OBB 的固定边界。
- 下载前可直接查看 active token 数和校验状态。
- “导出随机化场景批次”会为全部 10 个族分别采样指定数量的变体，并随机化数量、gap、整体旋转/平移和 seed；批次生成会自动重采样越界实例。

视口中的两个半透明线框盒表示 start/goal 采样区域，圆点是本次经碰撞拒绝采样得到的示例端点。端点不再固定。start 和 goal 分别使用中心约为 `1.82–2.18 m`、高度跨度约为 `1.85–2.18 m` 的独立宽 Z 区域，两者始终具有大范围重叠。生成器不再预设“一端高、一端低”：两次独立采样既可能得到明显高度差，也可能落在近似同一水平面。UI 参数卡会直接显示两个区域实际的 Z 上下限。

生成器内部通过栅格搜索或受保护 portal 路线建立基于 URDF primitive OBB 的可行性证书；“保护通道（调试）”开关会用不同颜色显示当前保留的全部路线模式。普通显示默认隐藏这些生成约束，它们不是已经规划完成的专家轨迹。

start/goal 不再使用水平单位四元数：优先在 `4–40°` 的 roll 幅值、`3–70°` 的 pitch 幅值和小幅 yaw 内随机采样，并对完整姿态插值重新执行 URDF primitive 碰撞及姿态边界检查。特别狭窄的实例会退让到约 `3–6° roll`、`2–4° pitch`，但不会退回完全水平。非门框的内部参考姿态也会采样轻微 bank/pitch；姿态门框仍以其 `25–40°` 必要 roll 为主。

`orientation_sensitive_passage` 使用整体倾斜的矩形 frame，而不是要求无人机在水平窄缝中接近侧立。frame 的目标 roll 在 `25–40°` 内随机采样；正姿态无法匹配倾斜门框，而无人机以相近 roll 可以通过。该角度同时写入障碍物的 `required_roll_deg` 和路线证书。

场景与专家规划共享同一飞行姿态边界：`roll ∈ [-40°, 40°]`、`pitch ∈ [-70°, 70°]`。start/goal 在碰撞自由的前提下从该范围随机采样非水平姿态；OMPL 有效性检查和最终 B-spline 稠密复检都会拒绝越界姿态，因此平滑插值不能通过过冲绕开限制。返回 JSON 会在 `robot_reference.flight_attitude_limits_deg` 及专家指标中记录限制和实际最大绝对 roll/pitch。

## 交互式专家轨迹

生成场景后，可在左侧选择同一 start/goal 的专家数量并点击“生成专家轨迹”。后端复用项目的专家策略：

```text
one global start-goal OMPL query
  → C++ StateSampler draws 70% states inside scene guide regions
  → the same sampler draws 30% uniformly across the full workspace
  → OMPL RRT-Connect grows one bidirectional tree using both sources
  → fixed pose waypoints only for strict orientation-coupled passages
  → same-pair diversity filtering
  → low-control-point global degree-5 constrained SE(3) B-spline
  → COAL full-URDF closest-point repair and dense collision validation
```

UI 的正式专家流程使用场景随附的 `expert_planning_guides`。采样区域现在由模板结构决定：separator 类场景使用宽的 above/below 或 left/right 绕行带；doorway、narrow passage 和 bracket 使用门洞两侧相互关联的可行截面；staggered corridor 与 mixed industrial 按障碍物纵向站位生成 2–4 个局部净空单元。后两类共享整体绕行趋势，同时保留每个净空单元内的局部变化。

普通场景不再采样 1–2 个具体锚点，也不再拆成多个必须依次抵达的 RRTConnect 段。每次专家尝试只有一个 start→goal OMPL 查询：一个小型 C++ 扩展在现有 Python `SE3StateSpace` 上注册 `StateSamplerAllocator`，使 RRT-Connect 的随机状态初始以 `0.70` 概率来自当前候选的全部引导区域，以 `0.30` 概率来自完整工作空间。引导区域只改变树“更常向哪里扩展”，任何单个区域样本都不是 waypoint 或路径约束；返回轨迹的普通 `waypoints` 因而始终只有 start/goal。若 RRT-Connect 找到的有效轨迹更接近另一个证书路线，系统不再以 `route_mode_mismatch` 拒绝，而是按实际最近的路线代理类别接收并计数；原目标未命中只会提高后续该目标的区域偏置。证书因此只证明已知路线存在，不被当成全部可能拓扑的封闭集合。普通候选失败四次后不会退回固定证书，而是把区域采样概率逐步提高到最多 `0.90`，同时把该候选的求解预算最多提高到初始值的 `1.5` 倍。固定 waypoint 仅保留给 orientation-sensitive 等姿态与孔洞强耦合的严格通道。调试视图中的“引导区域 / 保护通道”开关会显示采样盒和路线证书。

`wall_protrusion_bracket` 的支架尖端现在位于墙洞进近中心线上，并与墙之间保留完整 URDF 包络能够完成横移的距离；地图会分别认证尖端内侧和外侧 dogleg 候选。`mixed_industrial` 的机器与控制柜分列任务走廊两侧，横向支撑梁覆盖中部；除结构化 winding 候选外，还会保留共享同一 start/goal、由自由空间 A* 发现且与主路线显著不同的候选。核心装配带有 `route_relevant` 标记，随机补充物仍只承担次要场景变化。

对于 `above/below/left/right`，采样区域不是固定小立方体，而是根据分隔障碍物的 OBB、start→goal 局部坐标轴、URDF 包络和工作空间边界计算的定向长方体走廊。上/下绕行区域横跨障碍物的大部分横向尺寸，左右绕行区域从最小安全侧向位置扩展到场地边界，并允许较宽的有效高度变化。RRT-Connect 会在所有区域与全局自由空间中反复采样，姿态在每个区域的参考姿态附近做有界 roll/pitch/yaw 随机化；区域之间如何连接完全由同一个双向树决定。最终平滑只硬约束 start/goal，因此不再因相邻分段从相反方向接入锚点而产生人为发卡弯。

orientation-sensitive passage 使用 `fixed_waypoints_required` 和 `orientation_coupled_aperture`：只固定孔前和孔后的必要倾斜姿态，不把位置与 roll 当成可以独立扰动的变量，也不在 start/goal 开放空间制造 via 扰动。严格固定模式或已确认无法采样的区域只有一条非重复路线时，可以返回少于 UI 请求数的专家，响应中的 `generation_exhausted_reason` 会说明原因。正式流程按路线模式覆盖、路径长度质量和“位置 + 姿态 + 长度”的近似重复判定验收，不再用单一全局位置 RMS 阈值强迫轨迹分叉。无引导纯 RRTConnect 仅保留为后端诊断兼容模式，不在 Web UI 中作为专家生成选项。

没有多路线模板的普通场景也使用自身的结构化净空链或 portal 截面，不再通过已接受路径附近的额外 via 扰动制造差异。start/goal 始终保持场景生成时采样的同一对端点，因此轨迹变化来自有效自由空间，而不是端点附近的纺锤形绕路。

所有“保证多通道”族会在地图生成阶段保护各路线模式，并为每个模式导出可采样区域；固定模板只承担可行性认证、轨迹分类和失败回退，不直接成为正常专家的完整几何曲线。返回 JSON 的每条专家包含 `topology_class`（兼容原接口的字段名），状态栏会显示本场景可用的类别。

平滑阶段不再每隔约 3 个 OMPL 状态设置一个插值节点。当前使用五次全局 waypoint-constrained B-spline，初始约每 14 个 raw 状态设置一个控制自由度，并同时惩罚位置/姿态的二、三阶导数；仅在碰撞复检失败时增加自由度。典型测试从 `88–174` 个 raw 状态降至约 `7–13` 个控制点。

普通区域偏置轨迹还会执行自然度验收：最大曲率不得超过 `8 m⁻¹`，沿 start→goal 主轴累计反向运动不得超过 `0.25 m`，在约 16 个等弧长采样间隔的局部窗口内，端点弦长与实际弧长之比不得低于 `0.65`。这三个条件分别拦截极小转弯半径、明显倒飞折返和局部发卡环；失败只会触发新的全局区域偏置查询，不会添加修复 waypoint。

当 COAL 可用时，普通的区域偏置候选在 B-spline 密集检查前增加局部净空修复：对 128 个曲线样本查询完整 URDF primitive 与场景 OBB 的有符号距离、最近点和法向，把净空损失通过 B-spline 基函数反传到位置控制点。更新被投影到硬 waypoint 约束的零空间，因此 start/goal 精确不动；姿态控制点完全不改。修复目标为额外 `0.018 m` 净空，单个控制点相对原拟合最多移动 `0.30 m`，并用回贴 OMPL guide 的正则项抑制跨通道漂移。严格的 orientation-sensitive 姿态点和固定回退不参与局部修复。无论优化是否声称成功，最终仍以整条曲线的密集 COAL 复检为准。

界面可分别开关虚线 OMPL guide 和实线 B-spline 结果。启用“URDF 姿态残影”后，会沿当前显示轨迹稀疏绘制 URDF 的 7 个 active collision primitive 和机体系 XYZ 轴；滑条控制每条轨迹的残影数量。这样可以直接检查 roll/pitch/yaw 是否连续、门洞姿态是否正确，以及平滑是否产生碰撞或姿态捷径。

专家接口为 `POST /api/experts`，输出 `scene_expert_trajectories_v001`，同时保留每条轨迹的 `ompl_path`、`bspline_path`、planner seed/range、多样性和几何指标。

交互后端优先使用 COAL 对完整 URDF 的 7 个 active collision primitive 与场景 OBB 做窄相查询；响应的 `collision_backend` 为 `coal_full_urdf_nearest_point`。当前 Python 解释器未提供 COAL 时才使用 `urdf_primitive_obb_conservative` 回退后端；回退后端没有伪造距离梯度，也不会执行控制点净空修复。

当前编辑器只生成 yaw-only OBB，但导出结构保留完整 `quaternion_wxyz`，与现有 SE(3) 环境读取和 cross-attention 接口兼容。
