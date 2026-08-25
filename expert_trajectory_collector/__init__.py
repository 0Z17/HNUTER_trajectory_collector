"""Task-pluggable expert trajectory collection services.

The package owns orchestration and transport.  Task-specific geometry,
validity checks, and planners live behind :class:`TaskPlugin`, allowing the
free-flight collector to remain stable while inspection and surface tasks are
added independently.
"""

from .contracts import (
    CollectorError,
    TaskCapabilities,
    TaskDescriptor,
    TaskPlugin,
    TaskUnavailableError,
    UnknownTaskError,
)
from .registry import TaskRegistry, create_default_registry
from .service import ExpertCollectorService

__all__ = [
    "CollectorError",
    "ExpertCollectorService",
    "TaskCapabilities",
    "TaskDescriptor",
    "TaskPlugin",
    "TaskRegistry",
    "TaskUnavailableError",
    "UnknownTaskError",
    "create_default_registry",
]
