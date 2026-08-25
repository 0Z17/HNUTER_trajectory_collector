"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const familyLabels = {
  sparse_obb_clutter: "随机稀疏 OBB clutter",
  central_block: "中央阻挡 / 必须绕行",
  multi_homotopy: "四通道母版（至少保留垂直+水平双族）",
  narrow_passage: "narrow passage",
  orientation_sensitive_passage: "orientation-sensitive passage",
  wall_protrusion_bracket: "wall + protrusion / bracket",
  frame_doorway: "frame / doorway",
  pillar_wall: "pillar + wall",
  staggered_corridor: "多障碍 staggered corridor",
  mixed_industrial: "mixed industrial structure",
};
const familyMinimums = {
  sparse_obb_clutter: 1, central_block: 1, multi_homotopy: 3,
  narrow_passage: 2, orientation_sensitive_passage: 6,
  wall_protrusion_bracket: 5, frame_doorway: 3, pillar_wall: 2,
  staggered_corridor: 2, mixed_industrial: 6,
};

const state = {
  scene: null, conditioning: null, tokens: [], mask: [], issues: [], selected: null,
  experts: null, expertRequestSerial: 0,
  taskType: "free_flight", taskCatalog: null,
  camera: { yaw: -0.72, pitch: 0.62, zoom: 1, panX: 0, panY: 0 },
  drag: null, hitFaces: [],
};
const CAMERA_ELEVATION_MIN = 5 * Math.PI / 180;
const CAMERA_ELEVATION_MAX = 88 * Math.PI / 180;

for (const [value, label] of Object.entries(familyLabels)) {
  const option = document.createElement("option");
  option.value = value; option.textContent = label;
  if (value === "staggered_corridor") option.selected = true;
  $("#family").append(option);
}

function parameters() {
  return {
    task_type: state.taskType,
    family: $("#family").value,
    seed: Number($("#seed").value),
    sample_ranges: true,
    obstacle_count_min: Number($("#countMin").value),
    obstacle_count_max: Number($("#countMax").value),
    size_min: Number($("#sizeMin").value),
    size_max: Number($("#sizeMax").value),
    gap_width_min: Number($("#gapMin").value),
    gap_width_max: Number($("#gapMax").value),
    global_yaw_min: Number($("#yawMin").value),
    global_yaw_max: Number($("#yawMax").value),
    translation_max: Number($("#translationMax").value),
  };
}

function randomSeed() {
  const values = new Uint32Array(1); crypto.getRandomValues(values);
  return values[0] & 0x7fffffff;
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

async function getJson(path) {
  const response = await fetch(path, {headers: {"Accept": "application/json"}});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

async function loadTaskCatalog() {
  const catalog = await getJson("/api/tasks");
  state.taskCatalog = catalog;
  const select = $("#taskType"); select.innerHTML = "";
  catalog.tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = task.task_type;
    option.textContent = `${task.label}${task.available ? "" : "（待接入）"}`;
    option.disabled = !task.available;
    option.selected = task.task_type === catalog.default_task;
    select.append(option);
  });
  state.taskType = select.value || catalog.default_task;
  renderTaskDescription();
}

function renderTaskDescription() {
  const task = state.taskCatalog?.tasks?.find((item) => item.task_type === state.taskType);
  if (!task) return;
  $("#taskDescription").textContent = `${task.summary} · ${task.condition_schema} → ${task.trajectory_schema}`;
}

async function generate() {
  setBusy(true);
  try {
    if ($("#autoSeed").checked) $("#seed").value = randomSeed();
    applyPayload(await post("/api/generate", parameters()));
  } catch (error) {
    showError(error.message);
  } finally { setBusy(false); }
}

async function generateExperts() {
  if (!state.scene) return;
  const requestSerial = ++state.expertRequestSerial;
  const expertSeed = randomSeed();
  const button = $("#generateExperts");
  button.disabled = true; button.textContent = "OMPL 规划中…";
  $("#downloadCollection").disabled = true;
  state.experts = null;
  draw();
  $("#expertStatus").className = "expert-status";
  $("#expertStatus").textContent = `正在重新规划 · expert seed ${expertSeed}…`;
  try {
    const experts = await post("/api/experts", {
      task_type: state.taskType,
      scene: state.scene, count: Number($("#expertCount").value),
      seed: expertSeed,
      solve_time: Number($("#expertSolveTime").value),
      planning_mode: $("#planningMode").value,
    });
    if (requestSerial !== state.expertRequestSerial) return;
    state.experts = experts;
    $("#downloadCollection").disabled = !(experts.experts?.length > 0);
    const elapsed = Number(state.experts.total_wall_time_s).toFixed(2);
    const recovery = Number(state.experts.recovery_expert_count || 0);
    const strictFixed = Number(state.experts.strict_fixed_expert_count || 0);
    const topology = (state.experts.available_topology_classes || []).join("/");
    const stages = state.experts.generation_stage_counts || {};
    const attempts = state.experts.generation_stage_attempt_counts || {};
    const pureMode = state.experts.planning_mode === "pure_rrtconnect";
    const pipeline = state.experts.acceptance_pipeline?.counts || {};
    const rejectionCounts = state.experts.acceptance_pipeline?.rejection_reason_counts || {};
    const dominantRejection = Object.entries(rejectionCounts).sort((a, b) => b[1] - a[1])[0];
    const rejectionLabels = {
      rrt_no_exact_solution: "RRT无精确解",
      raw_path_collision_check: "原始路径复检碰撞",
      bspline_collision: "B样条碰撞",
      bspline_workspace_bounds: "B样条越界",
      bspline_validation_other: "B样条校验",
      attitude_limit: "姿态超限",
      position_diversity: "轨迹过于相似",
      topology_class_repeat: "拓扑类别重复",
      redundant_duplicate: "重复专家",
      excessive_detour: "无效绕行",
      unnatural_geometry: "急弯或局部折返",
      invalid_endpoint: "端点无效",
      planner_or_pipeline_error: "规划异常",
    };
    const stageSummary = pureMode
      ? `RRT解 ${Number(pipeline.rrt_exact_solution || 0)}/${Number(pipeline.attempted || attempts.pure_rrtconnect || 0)} · 原始有效 ${Number(pipeline.raw_path_valid || 0)} · B样条有效 ${Number(pipeline.bspline_valid || 0)} · 最终 ${Number(pipeline.accepted || stages.pure_rrtconnect || 0)}`
      : `区域偏置RRT-Connect ${Number(stages.region_biased_global || 0)} · 固定回退 ${Number(stages.fixed_waypoint_fallback || 0)} · 状态采样 区域70%/全局30%`;
    const lastFailure = (state.experts.recent_failures || []).at(-1);
    const attitudeLimits = state.experts.flight_attitude_limits_deg;
    const attitude = attitudeLimits
      ? ` · |R|≤${attitudeLimits.roll[1]}° |P|≤${attitudeLimits.pitch[1]}°`
      : "";
    $("#expertStatus").className = "expert-status ready";
    const rejection = dominantRejection
      ? ` · 主要拒绝 ${rejectionLabels[dominantRejection[0]] || dominantRejection[0]}×${dominantRejection[1]}` : "";
    const fixedSummary = `${strictFixed ? ` · ${strictFixed} 条严格姿态点` : ""}${recovery ? ` · ${recovery} 条软通道失败回退` : ""}`;
    const discoveredOutsideGuide = Number(
      state.experts.accepted_outside_target_guide_count || 0
    );
    const discoverySummary = discoveredOutsideGuide
      ? ` · ${discoveredOutsideGuide} 条由OMPL发现其他通道`
      : "";
    const exhausted = state.experts.generation_exhausted_reason
      ? " · 已达到非重复路线容量" : "";
    const repair = state.experts.clearance_repair || {};
    const repairSummary = repair.backend_available
      ? ` · COAL梯度修复 ${Number(repair.accepted_succeeded_count || 0)}/${Number(repair.accepted_attempted_count || 0)}`
      : " · COAL梯度不可用";
    $("#expertStatus").textContent = `${state.experts.accepted_count}/${state.experts.requested_count} 条已接受 · seed ${expertSeed} · ${elapsed}s · ${stageSummary}${rejection}${repairSummary}${topology ? ` · ${topology}` : ""}${attitude}${fixedSummary}${discoverySummary}${exhausted}${!state.experts.accepted_count && lastFailure ? ` · ${lastFailure}` : ""}`;
    draw();
  } catch (error) {
    if (requestSerial !== state.expertRequestSerial) return;
    state.experts = null;
    $("#expertStatus").className = "expert-status bad";
    $("#expertStatus").textContent = error.message;
    showError(error.message);
  } finally { button.disabled = false; button.textContent = state.experts ? "重新生成专家轨迹" : "生成专家轨迹"; }
}

function clearExperts() {
  state.expertRequestSerial += 1;
  state.experts = null;
  $("#downloadCollection").disabled = true;
  $("#generateExperts").textContent = "生成专家轨迹";
  $("#expertStatus").className = "expert-status";
  $("#expertStatus").textContent = "任务条件已变化，请重新生成专家轨迹";
}

async function validate() {
  if (!state.scene) return;
  try { applyPayload(await post("/api/validate", {
    task_type: state.taskType, condition: state.scene,
  }), false); }
  catch (error) { showError(error.message); }
}

async function downloadBatch() {
  const button = $("#downloadBatch");
  button.disabled = true; button.textContent = "正在构建批次…";
  try {
    const p = parameters();
    const bank = await post("/api/batch", {
      task_type: state.taskType,
      variants_per_family: Number($("#variants").value), base_seed: p.seed,
      obstacle_count_min: p.obstacle_count_min, obstacle_count_max: p.obstacle_count_max,
      size_min: p.size_min, size_max: p.size_max,
      global_yaw_min: p.global_yaw_min, global_yaw_max: p.global_yaw_max,
      translation_max: p.translation_max,
    });
    download(`free_flight_condition_bank_seed_${p.seed}.json`, bank);
  } catch (error) { showError(error.message); }
  finally { button.disabled = false; button.textContent = "导出随机化条件批次"; }
}

async function downloadCollection() {
  if (!state.scene || !state.experts) return;
  const button = $("#downloadCollection");
  button.disabled = true; button.textContent = "正在封装采集记录…";
  try {
    const record = await post("/api/collection", {
      task_type: state.taskType,
      condition: state.scene,
      expert_set: state.experts,
    });
    download(`${state.scene.environment_id}_expert_collection.json`, record);
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false; button.textContent = "导出专家采集记录";
  }
}

function applyPayload(payload, resetSelection = true) {
  state.taskType = payload.task_type || state.taskType;
  state.scene = payload.scene;
  state.conditioning = payload.conditioning || null;
  state.tokens = payload.tokens;
  state.mask = payload.mask;
  state.issues = payload.validation;
  if (resetSelection) { state.selected = null; clearExperts(); }
  renderAll();
}

function activeObstacles() {
  return state.scene ? state.scene.obstacles.filter((item) => item.role !== "floor") : [];
}

function renderAll() {
  const obstacles = activeObstacles();
  const active = state.mask.filter(Boolean).length;
  $("#sceneId").textContent = state.scene?.environment_id?.toUpperCase() || "—";
  $("#obstacleSummary").textContent = `${obstacles.length} 个有效障碍物`;
  $("#tokenUsage").textContent = `${active} / 32`;
  $("#tokenFill").style.width = `${active / 32 * 100}%`;
  const sampled = state.scene?.generation_parameters;
  const robotSize = state.scene?.robot_reference?.collision_aabb_local?.size_xyz;
  const routeModes = Number(state.scene?.verified_route_mode_count || 0);
  const routePolicy = state.scene?.route_mode_policy || "unknown";
  const guides = state.scene?.expert_planning_guides || [];
  const modeNames = guides.map((item)=>item.id).join("/");
  const guidePolicy = [...new Set(guides.map((item)=>item.policy))].join(" + ");
  const startRegion = state.scene?.task_sampling?.start_region;
  const goalRegion = state.scene?.task_sampling?.goal_region;
  const generatedProps = obstacles.filter((item)=>item.role === "secondary_obstacle");
  const relevantProps = generatedProps.filter((item)=>Number(item.size_xyz?.[2]) >= 1.45);
  const functionSummary = state.scene?.obstacle_function_summary;
  const heightMix = generatedProps.length
    ? `${relevantProps.length}/${generatedProps.length} 个填充物高度≥1.45 m`
    : "无随机填充物";
  const functionMix = functionSummary && generatedProps.length
    ? `有效填充 ${(100 * Number(functionSummary.effective_prop_ratio || 0)).toFixed(0)}% · 路径选择 ${Number(functionSummary.route_selector_count || 0)} · 净空塑形 ${Number(functionSummary.clearance_shaper_count || 0)} · 干扰 ${Number(functionSummary.distractor_count || 0)}`
    : "";
  const zBand = (region)=>region
    ? `${(Number(region.center[2])-Number(region.size_xyz[2])/2).toFixed(2)}–${(Number(region.center[2])+Number(region.size_xyz[2])/2).toFixed(2)}`
    : "—";
  $("#sampledParameters").innerHTML = sampled
    ? `本次采样 · N=${sampled.obstacle_count}<br>gap=${Number(sampled.gap_width).toFixed(3)} m · yaw=${Number(sampled.global_yaw_deg).toFixed(1)}°<br>translate=(${Number(sampled.translate_x).toFixed(2)}, ${Number(sampled.translate_y).toFixed(2)}) m · seed=${sampled.seed}<br>route modes=${routeModes} · ${routePolicy}${modeNames ? `<br>${modeNames}` : ""}${guidePolicy ? `<br>guidance=${guidePolicy}` : ""}<br>start Z=${zBand(startRegion)} m · goal Z=${zBand(goalRegion)} m<br>${heightMix}${functionMix ? `<br>${functionMix}` : ""}${robotSize ? `<br>URDF AABB=${robotSize.map((value)=>Number(value).toFixed(3)).join("×")} m` : ""}`
    : "等待生成参数";
  renderValidation(); renderList(); renderEditor(); draw();
}

function renderValidation() {
  const bad = state.issues.length > 0;
  $("#statusPill").className = `status-pill ${bad ? "bad" : "ok"}`;
  $("#statusPill").innerHTML = `<i></i>${bad ? "CHECK CONDITION" : "CONDITION VALID"}`;
  $("#issueCount").textContent = `${state.issues.length} ISSUES`;
  $("#validationBox").className = `validation-box ${bad ? "bad" : "ok"}`;
  $("#validationBox").innerHTML = bad
    ? state.issues.map((issue) => `• ${escapeHtml(issue)}`).join("<br>")
    : "任务条件、规划证书与 conditioning contract 相容";
}

function renderList() {
  const list = $("#obstacleList"); list.innerHTML = "";
  activeObstacles().forEach((obstacle, index) => {
    const row = document.createElement("div");
    row.className = `obstacle-row ${index === state.selected ? "selected" : ""}`;
    const p = obstacle.pose.position, s = obstacle.size_xyz;
    const roleLabels = {route_selector: "路径选择", clearance_shaper: "净空塑形", distractor: "干扰"};
    const roleLabel = roleLabels[obstacle.functional_role];
    row.innerHTML = `<i></i><div><b>${escapeHtml(obstacle.id)}</b><small>${roleLabel ? `${roleLabel} · ` : ""}${p.map(format).join(" · ")}</small></div><em>${s.map(format).join("×")}</em>`;
    row.addEventListener("click", () => { state.selected = index; renderList(); renderEditor(); draw(); });
    list.append(row);
  });
}

function renderEditor() {
  const obstacle = activeObstacles()[state.selected];
  $("#editor").hidden = !obstacle;
  if (!obstacle) return;
  $("#editorTitle").textContent = obstacle.id;
  $$('[data-edit^="position"]').forEach((input, i) => input.value = obstacle.pose.position[i]);
  $$('[data-edit^="size"]').forEach((input, i) => input.value = obstacle.size_xyz[i]);
  $('[data-edit="yaw"]').value = Math.round(yawFromQuaternion(obstacle.pose.quaternion_wxyz) * 180 / Math.PI * 100) / 100;
}

function editObstacle(input) {
  const obstacle = activeObstacles()[state.selected];
  if (!obstacle || !Number.isFinite(Number(input.value))) return;
  const [field, index] = input.dataset.edit.split(".");
  if (field === "position") obstacle.pose.position[Number(index)] = Number(input.value);
  else if (field === "size") obstacle.size_xyz[Number(index)] = Number(input.value);
  else if (field === "yaw") obstacle.pose.quaternion_wxyz = quaternionFromYaw(Number(input.value) * Math.PI / 180);
  clearExperts(); validate();
}

function addObstacle() {
  if (!state.scene || activeObstacles().length >= 32) return;
  const index = activeObstacles().length;
  state.scene.obstacles.push({
    id: `manual_box_${String(index).padStart(2, "0")}`, type: "box",
    pose: { position: [0, 0, 1], quaternion_wxyz: [1, 0, 0, 0] },
    size_xyz: [0.6, 0.6, 2], collision: true, visual: true, static: true,
    role: "obstacle", family: "manual",
  });
  state.selected = index; clearExperts(); validate();
}

function deleteObstacle() {
  if (state.selected === null) return;
  const actual = state.scene.obstacles.findIndex((item, index) => item.role !== "floor" && state.scene.obstacles.filter((v, j) => j < index && v.role !== "floor").length === state.selected);
  if (actual >= 0) state.scene.obstacles.splice(actual, 1);
  state.selected = null; clearExperts(); validate();
}

function setBusy(busy) {
  $("#generate").disabled = busy;
  $("#generate").textContent = busy ? "正在生成…" : "生成任务条件";
}

function showError(message) {
  state.issues = [message]; renderValidation();
}

function download(name, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// Dependency-free 3-D OBB renderer. `pitch` is the camera elevation above the
// XY floor (not an object rotation), so it is kept in the upper hemisphere.
// Faces are depth-sorted and retained for click selection.
const canvas = $("#sceneCanvas"), context = canvas.getContext("2d");
const cubeFaces = [[0,1,3,2], [4,6,7,5], [0,4,5,1], [2,3,7,6], [0,2,6,4], [1,5,7,3]];

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  context.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}

function cameraPoint(point) {
  const [x, y, z] = point, cy = Math.cos(state.camera.yaw), sy = Math.sin(state.camera.yaw);
  const cp = Math.cos(state.camera.pitch), sp = Math.sin(state.camera.pitch);
  const rx = cy * x - sy * y, ry = sy * x + cy * y;
  // Conventional orbit-camera basis. Larger depth is nearer the camera;
  // positive elevation always places that camera above the ground plane.
  return [rx, -sp * ry - cp * (z - 1.7), cp * ry + sp * (z - 1.7)];
}

function project(point) {
  const [x, y, depth] = cameraPoint(point), rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width, rect.height) * 0.09 * state.camera.zoom;
  return [
    rect.width / 2 + state.camera.panX + x * scale,
    rect.height / 2 + state.camera.panY + y * scale,
    depth,
  ];
}

function boxVertices(obstacle) {
  const [cx, cy, cz] = obstacle.pose.position, [sx, sy, sz] = obstacle.size_xyz;
  const rotation = quaternionMatrix(obstacle.pose.quaternion_wxyz);
  const vertices = [];
  for (const z of [-sz/2, sz/2]) for (const y of [-sy/2, sy/2]) for (const x of [-sx/2, sx/2]) {
    const offset = matrixVector(rotation, [x,y,z]);
    vertices.push([cx + offset[0], cy + offset[1], cz + offset[2]]);
  }
  return vertices;
}

function quaternionMatrix(q) {
  let [w,x,y,z]=q.map(Number); const norm=Math.hypot(w,x,y,z)||1; w/=norm;x/=norm;y/=norm;z/=norm;
  return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
function matrixVector(matrix, vector) { return matrix.map((row)=>row[0]*vector[0]+row[1]*vector[1]+row[2]*vector[2]); }
function matrixMultiply(first, second) { return first.map((row)=>[0,1,2].map((column)=>row[0]*second[0][column]+row[1]*second[1][column]+row[2]*second[2][column])); }

function draw() {
  const rect = canvas.getBoundingClientRect(); context.clearRect(0, 0, rect.width, rect.height);
  drawGrid(); drawBounds(); state.hitFaces = [];
  if (!state.scene) return;
  const faces = [];
  activeObstacles().forEach((obstacle, obstacleIndex) => {
    const points = boxVertices(obstacle).map(project);
    cubeFaces.forEach((indices, faceIndex) => {
      const polygon = indices.map((i) => points[i]);
      faces.push({ polygon, depth: polygon.reduce((sum, p) => sum + p[2], 0) / 4, obstacleIndex, faceIndex });
    });
  });
  faces.sort((a, b) => a.depth - b.depth);
  faces.forEach((face) => drawFace(face));
  state.hitFaces = faces;
  drawCertifiedRoute();
  drawExpertPaths();
  drawRobotGhosts();
  drawEndpoints();
}

function drawFace(face) {
  const selected = face.obstacleIndex === state.selected;
  const light = [0.52, 0.76, 0.62, 0.88, 0.68, 0.72][face.faceIndex];
  context.beginPath(); face.polygon.forEach((p, i) => i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])); context.closePath();
  context.fillStyle = selected ? `rgba(75, 215, 232, ${light})` : `rgba(255, 176, 0, ${light})`;
  context.strokeStyle = selected ? "rgba(157,242,250,.95)" : "rgba(255,215,125,.72)";
  context.lineWidth = selected ? 1.35 : 0.65; context.fill(); context.stroke();
}

function line3(a, b, color, width = 1, dash = []) {
  const p = project(a), q = project(b); context.beginPath(); context.moveTo(p[0], p[1]); context.lineTo(q[0], q[1]);
  context.strokeStyle = color; context.lineWidth = width; context.setLineDash(dash); context.stroke(); context.setLineDash([]);
}

function drawGrid() {
  for (let v = -4; v <= 4; v += 0.5) {
    const major = Number.isInteger(v), color = major ? "rgba(96,119,135,.2)" : "rgba(96,119,135,.08)";
    line3([v,-4,0], [v,4,0], color, major ? 0.8 : 0.5); line3([-4,v,0], [4,v,0], color, major ? 0.8 : 0.5);
  }
  line3([-4,0,0], [4,0,0], "rgba(255,105,95,.45)", 1.2); line3([0,-4,0], [0,4,0], "rgba(92,210,130,.45)", 1.2);
}

function drawBounds() {
  const corners = [[-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0],[-4,-4,4],[4,-4,4],[4,4,4],[-4,4,4]];
  [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) => line3(corners[a], corners[b], "rgba(99,132,151,.48)", .8, [4,4]));
}

function drawEndpoints() {
  const pair = state.scene?.precheck_pairs?.[0]; if (!pair) return;
  const sampling = state.scene?.task_sampling;
  if (sampling) {
    drawRegion(sampling.start_region, "rgba(75,215,232,.12)", "rgba(75,215,232,.7)");
    drawRegion(sampling.goal_region, "rgba(243,239,207,.09)", "rgba(243,239,207,.58)");
  }
  [[pair.start_pose, "S", "#4bd7e8"], [pair.goal_pose, "G", "#f3efcf"]].forEach(([pose, label, color]) => {
    const p = project(pose.slice(0,3)); context.beginPath(); context.arc(p[0], p[1], 6, 0, Math.PI*2); context.fillStyle = color; context.fill();
    context.fillStyle = "#071015"; context.font = "800 8px ui-monospace"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(label, p[0], p[1]+.5);
  });
}

function drawRegion(region, fill, stroke) {
  const obstacle = {pose:{position:region.center,quaternion_wxyz:region.quaternion_wxyz},size_xyz:region.size_xyz};
  const points = boxVertices(obstacle).map(project);
  cubeFaces.forEach((indices) => {
    context.beginPath(); indices.forEach((index,i)=>i?context.lineTo(points[index][0],points[index][1]):context.moveTo(points[index][0],points[index][1]));
    context.closePath(); context.fillStyle=fill; context.strokeStyle=stroke; context.lineWidth=.7; context.fill(); context.stroke();
  });
}

const expertColors = ["#4bd7e8", "#ffca55", "#bb82ee", "#55dd9f", "#ff7f70", "#7fa8ff", "#d5e06d", "#e58fb5"];

function drawTrajectory(states, color, width, dash) {
  if (!states || states.length < 2) return;
  context.beginPath();
  states.forEach((pose,index)=>{const point=project(pose.slice(0,3));index?context.lineTo(point[0],point[1]):context.moveTo(point[0],point[1]);});
  context.strokeStyle=color; context.globalAlpha=dash.length?.62:.92; context.lineWidth=width; context.setLineDash(dash); context.stroke();
  context.setLineDash([]); context.globalAlpha=1;
}

function drawExpertPaths() {
  if (!state.experts) return;
  state.experts.experts.forEach((expert,index)=>{
    const color=expertColors[index%expertColors.length];
    if ($("#showOmpl").checked) drawTrajectory(expert.ompl_path,color,1,[5,4]);
    if ($("#showBspline").checked) drawTrajectory(expert.bspline_path,color,2.1,[]);
  });
}

function drawWireBox(center, rotation, size, color) {
  const [sx,sy,sz]=size, vertices=[];
  for(const z of[-sz/2,sz/2])for(const y of[-sy/2,sy/2])for(const x of[-sx/2,sx/2]){const d=matrixVector(rotation,[x,y,z]);vertices.push(project([center[0]+d[0],center[1]+d[1],center[2]+d[2]]));}
  [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b])=>{context.beginPath();context.moveTo(vertices[a][0],vertices[a][1]);context.lineTo(vertices[b][0],vertices[b][1]);context.strokeStyle=color;context.lineWidth=.55;context.stroke();});
}

function drawWireLoop(localPoints, center, rotation, color) {
  const points=localPoints.map((local)=>{const d=matrixVector(rotation,local);return project([center[0]+d[0],center[1]+d[1],center[2]+d[2]]);});
  context.beginPath();points.forEach((point,index)=>index?context.lineTo(point[0],point[1]):context.moveTo(point[0],point[1]));context.closePath();context.strokeStyle=color;context.lineWidth=.55;context.stroke();
}

function drawRobotPrimitive(primitive, center, rotation, color) {
  const half=primitive.half_extents.map(Number), steps=18;
  if(primitive.type==="sphere"){
    const radius=half[0];
    for(const plane of["xy","xz","yz"]){
      const ring=Array.from({length:steps},(_,index)=>{const angle=2*Math.PI*index/steps,c=radius*Math.cos(angle),s=radius*Math.sin(angle);return plane==="xy"?[c,s,0]:plane==="xz"?[c,0,s]:[0,c,s];});
      drawWireLoop(ring,center,rotation,color);
    }
    return;
  }
  if(primitive.type==="cylinder"){
    const radius=half[0], z=half[2];
    for(const level of[-z,z])drawWireLoop(Array.from({length:steps},(_,index)=>{const angle=2*Math.PI*index/steps;return[radius*Math.cos(angle),radius*Math.sin(angle),level];}),center,rotation,color);
    for(let index=0;index<6;index++){const angle=2*Math.PI*index/6,c=radius*Math.cos(angle),s=radius*Math.sin(angle),a=matrixVector(rotation,[c,s,-z]),b=matrixVector(rotation,[c,s,z]);line3([center[0]+a[0],center[1]+a[1],center[2]+a[2]],[center[0]+b[0],center[1]+b[1],center[2]+b[2]],color,.55);}
    return;
  }
  drawWireBox(center,rotation,half.map((value)=>2*value),color);
}

function drawRobotGhost(pose, color) {
  const primitives=state.scene?.robot_reference?.collision_primitives||[];
  const vehicleRotation=quaternionMatrix(pose.slice(3,7));
  primitives.forEach((primitive)=>{
    const local=primitive.local_pose, localPosition=matrixVector(vehicleRotation,local.position);
    const center=[pose[0]+localPosition[0],pose[1]+localPosition[1],pose[2]+localPosition[2]];
    const rotation=matrixMultiply(vehicleRotation,quaternionMatrix(local.quaternion_wxyz));
    drawRobotPrimitive(primitive,center,rotation,color);
  });
  const origin=pose.slice(0,3), axisColors=["rgba(255,100,92,.48)","rgba(86,220,130,.48)","rgba(91,145,255,.52)"];
  for(let axis=0;axis<3;axis++){const vector=[vehicleRotation[0][axis],vehicleRotation[1][axis],vehicleRotation[2][axis]].map((v)=>v*.42);line3(origin,[origin[0]+vector[0],origin[1]+vector[1],origin[2]+vector[2]],axisColors[axis],.8);}
}

function drawRobotGhosts() {
  if (!state.experts || !$("#showGhosts").checked) return;
  const count=Number($("#ghostDensity").value);
  state.experts.experts.forEach((expert,expertIndex)=>{
    const path=$("#showBspline").checked?expert.bspline_path:expert.ompl_path;
    const color=expertColors[expertIndex%expertColors.length]+"58";
    const indices=new Set(Array.from({length:count},(_,i)=>Math.round(i*(path.length-1)/Math.max(1,count-1))));
    indices.forEach((index)=>drawRobotGhost(path[index],color));
  });
}

function drawCertifiedRoute() {
  if (!$("#showCertificate").checked) return;
  const templates = state.scene?.expert_route_templates || [];
  const routes = templates.length
    ? templates.map((template,index)=>({route:template.route_poses,color:expertColors[index%expertColors.length]}))
    : [{route:state.scene?.feasibility_certificate?.route_poses,color:"rgba(96,123,145,.72)"}];
  routes.forEach(({route,color})=>{
    if (!route || route.length < 2) return;
    context.beginPath();
    route.forEach((pose,index)=>{const p=project(pose.slice(0,3));index?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]);});
    context.strokeStyle=color; context.globalAlpha=.58; context.lineWidth=1; context.setLineDash([3,5]); context.stroke(); context.setLineDash([]);context.globalAlpha=1;
    route.slice(1,-1).forEach((pose)=>{const p=project(pose.slice(0,3));context.beginPath();context.arc(p[0],p[1],1.8,0,Math.PI*2);context.fillStyle=color;context.globalAlpha=.72;context.fill();context.globalAlpha=1;});
  });
  (state.scene?.expert_planning_guides||[]).forEach((guide,index)=>{
    const color=expertColors[index%expertColors.length];
    (guide.sampled_waypoint_regions||[]).forEach((region)=>{
      const rotation=quaternionMatrix(region.quaternion_wxyz||[1,0,0,0]);
      drawWireBox(region.center_pose.slice(0,3),rotation,region.size_xyz,`${color}88`);
    });
  });
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i=0, j=polygon.length-1; i<polygon.length; j=i++) {
    const [xi,yi]=polygon[i], [xj,yj]=polygon[j];
    if ((yi>point[1]) !== (yj>point[1]) && point[0] < (xj-xi)*(point[1]-yi)/(yj-yi)+xi) inside=!inside;
  }
  return inside;
}

canvas.addEventListener("pointerdown", (event) => {
  const mode = event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "orbit";
  state.drag = {
    x: event.clientX, y: event.clientY, yaw: state.camera.yaw,
    pitch: state.camera.pitch, panX: state.camera.panX,
    panY: state.camera.panY, mode, moved: false,
  };
  canvas.setPointerCapture(event.pointerId); canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return; const dx=event.clientX-state.drag.x, dy=event.clientY-state.drag.y;
  if (Math.abs(dx)+Math.abs(dy)>3) state.drag.moved=true;
  if (state.drag.mode === "pan") {
    state.camera.panX = state.drag.panX + dx;
    state.camera.panY = state.drag.panY + dy;
  } else {
    state.camera.yaw=state.drag.yaw+dx*.008;
    state.camera.pitch=Math.max(CAMERA_ELEVATION_MIN,Math.min(CAMERA_ELEVATION_MAX,state.drag.pitch-dy*.008));
  }
  draw(); cameraReadout();
});
canvas.addEventListener("pointerup", (event) => {
  if (state.drag && !state.drag.moved && state.drag.mode === "orbit") {
    const rect=canvas.getBoundingClientRect(), point=[event.clientX-rect.left,event.clientY-rect.top];
    const hit=[...state.hitFaces].reverse().find((face)=>pointInPolygon(point,face.polygon));
    state.selected=hit ? hit.obstacleIndex : null; renderList(); renderEditor(); draw();
  }
  state.drag=null; canvas.classList.remove("dragging");
});
canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.camera.zoom=Math.max(.45,Math.min(2.4,state.camera.zoom*Math.exp(-event.deltaY*.001))); draw(); cameraReadout(); }, {passive:false});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

function setView(name) {
  if (name === "top") Object.assign(state.camera, {yaw: 0, pitch: CAMERA_ELEVATION_MAX, panX: 0, panY: 0});
  else if (name === "front") Object.assign(state.camera, {yaw: 0, pitch: CAMERA_ELEVATION_MIN, panX: 0, panY: 0});
  else Object.assign(state.camera, {yaw: -0.72, pitch: 0.62, panX: 0, panY: 0});
  $$(".view-tabs button").forEach((button)=>button.classList.toggle("active",button.dataset.view===name)); draw(); cameraReadout(name);
}
function resetCamera() {
  Object.assign(state.camera, {yaw: -0.72, pitch: 0.62, zoom: 1, panX: 0, panY: 0});
  $$(".view-tabs button").forEach((button)=>button.classList.toggle("active",button.dataset.view==="iso"));
  draw(); cameraReadout("ISO");
}
function cameraReadout(name="CUSTOM") { const elevation=Math.round(state.camera.pitch*180/Math.PI);$("#cameraReadout").textContent=`CAM · ${name.toUpperCase()} · ELEV ${elevation}° · ${Math.round(state.camera.zoom*100)}%`; }
function yawFromQuaternion(q) { return 2*Math.atan2(Number(q[3]),Number(q[0])); }
function quaternionFromYaw(yaw) { return [Math.cos(yaw/2),0,0,Math.sin(yaw/2)].map((v)=>Math.round(v*1e9)/1e9); }
function format(value) { return Number(value).toFixed(2); }
function escapeHtml(value) { const node=document.createElement("span"); node.textContent=String(value); return node.innerHTML; }

$("#generate").addEventListener("click", generate);
$("#taskType").addEventListener("change", () => {
  state.taskType = $("#taskType").value;
  renderTaskDescription();
  state.scene = null; state.experts = null; state.selected = null;
  $("#downloadCollection").disabled = true;
  draw();
});
$("#family").addEventListener("change", () => {
  const minimum = familyMinimums[$("#family").value];
  $("#countMin").min = minimum; $("#countMax").min = minimum;
  if (Number($("#countMin").value) < minimum) $("#countMin").value = minimum;
  if (Number($("#countMax").value) < minimum) $("#countMax").value = minimum;
  const attitude = $("#family").value === "orientation_sensitive_passage";
  $("#gapMin").value = attitude ? "1.64" : "1.53";
  $("#gapMax").value = attitude ? "1.72" : "1.90";
});
$("#randomize").addEventListener("click", () => { $("#seed").value = randomSeed(); generate(); });
$("#addObstacle").addEventListener("click", addObstacle);
$("#deleteObstacle").addEventListener("click", deleteObstacle);
$("#downloadScene").addEventListener("click", () => state.scene && download(`${state.scene.environment_id}_condition.json`, state.scene));
$("#downloadTokens").addEventListener("click", () => state.scene && download(`${state.scene.environment_id}_conditioning.json`, state.conditioning || {schema_version:"box_geometry_v001",feature_order:["x","y","z","size_x","size_y","size_z","qw","qx","qy","qz"],tokens:state.tokens,mask:state.mask}));
$("#downloadBatch").addEventListener("click", downloadBatch);
$("#generateExperts").addEventListener("click", generateExperts);
$("#downloadCollection").addEventListener("click", downloadCollection);
[$("#showOmpl"),$("#showBspline"),$("#showGhosts"),$("#showCertificate")].forEach((input)=>input.addEventListener("change",draw));
$("#ghostDensity").addEventListener("input",()=>{$("#ghostOut").textContent=$("#ghostDensity").value;draw();});
$$('[data-edit]').forEach((input) => input.addEventListener("change", () => editObstacle(input)));
$$(".view-tabs button[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("#resetCamera").addEventListener("click", resetCamera);
$("#variants").addEventListener("input", () => { const n=Number($("#variants").value); $("#variantsOut").textContent=n; $("#variantsMetric").textContent=n; $("#totalMetric").textContent=n*Object.keys(familyLabels).length; });
window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe($("#viewport"));
cameraReadout("ISO");
loadTaskCatalog().then(generate).catch((error) => showError(error.message));
