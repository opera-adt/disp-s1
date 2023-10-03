from enum import Enum

__all__ = [
    "ProcessingMode",
]


class ProcessingMode(str, Enum):
    """Method for finding SHPs during phase linking."""

    FORWARD = "forward"
    """New data: only output one incremental result."""

    HISTORICAL = "historical"
    """Past stack of data: output multiple results."""
