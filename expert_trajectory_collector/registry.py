"""Task plugin registry and built-in task declarations."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import TaskPlugin, UnknownTaskError


class TaskRegistry:
    def __init__(self, plugins: Iterable[TaskPlugin] = ()) -> None:
        self._plugins: dict[str, TaskPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: TaskPlugin) -> None:
        task_type = plugin.descriptor.task_type
        if not task_type or task_type in self._plugins:
            raise ValueError(f"duplicate or empty task type {task_type!r}")
        self._plugins[task_type] = plugin

    def get(self, task_type: str) -> TaskPlugin:
        try:
            return self._plugins[task_type]
        except KeyError as error:
            available = ", ".join(self._plugins) or "none"
            raise UnknownTaskError(
                f"unknown task {task_type!r}; registered tasks: {available}"
            ) from error

    def descriptors(self) -> list[dict[str, object]]:
        return [plugin.descriptor.to_dict() for plugin in self._plugins.values()]

    def __contains__(self, task_type: object) -> bool:
        return task_type in self._plugins


def create_default_registry() -> TaskRegistry:
    # Imports stay local so the collector core has no dependency on task
    # geometry modules and can be tested with lightweight synthetic plugins.
    from .tasks.free_flight import FreeFlightTaskPlugin
    from .tasks.planned import planned_task_plugins

    return TaskRegistry([FreeFlightTaskPlugin(), *planned_task_plugins()])
