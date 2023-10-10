from enum import Enum

__all__ = [
    "ProcessingMode",
]


class ProcessingMode(str, Enum):
    """Workflow processing modes for SDS operation."""

    FORWARD = "forward"
    """New data: only output one incremental result."""

    HISTORICAL = "historical"
    """Past stack of data: output multiple results."""
