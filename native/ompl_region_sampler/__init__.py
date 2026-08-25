"""C++ regional state sampler bridge for OMPL's Python bindings."""

try:
    from ._ompl_region_sampler import (  # type: ignore[import-not-found]
        SamplerStatistics,
        install_region_state_sampler,
    )
except ImportError as error:  # pragma: no cover - exercised without a build
    raise ImportError(
        "the OMPL regional sampler extension is not built; run "
        "the current Python interpreter with "
        "'native/ompl_region_sampler/build_extension.py'"
    ) from error

__all__ = ["SamplerStatistics", "install_region_state_sampler"]
