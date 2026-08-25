# Codex workspace context

This is the standalone HNUTER expert trajectory collector. Read these files before changing or launching data collection:

1. `docs/CODEX_HANDOFF.md` — current implementation decisions and next-session checklist.
2. `docs/PILOT_COLLECTION_RUNBOOK.md` — exact background, pause/resume, and recovery commands.
3. `docs/EXPERT_DATASET_SPECIFICATION.md` — unified free-flight/inspection/surface data contract.
4. `docs/EXPERT_TRAJECTORY_COLLECTOR.md` and `docs/OBSTACLE_SCENE_BUILDER.md` — architecture and planner behavior.

Do not start the 100-environment pilot automatically. The user will explicitly start it from a fresh session after reviewing the prepared workspace.

Use `/home/z017/research/curobo_env/bin/python` for the verified OMPL/COAL Python 3.10 ABI. Build the native region sampler with that same interpreter. Keep generated datasets outside the Git repository. The batch path intentionally excludes visualization, TOPP-RA, and MuJoCo/MPPI rollout.

Future `inspection` and `surface` work must use task plugins and task-specific validity checkers; do not reuse free-flight validity rules as substitutes.
