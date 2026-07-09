import logging
from logging import Formatter
from pathlib import Path

from dolphin._types import Filename


class InputValidationError(ValueError):
    """Raised when the input CSLC stack fails a precheck.

    Subclasses ``ValueError`` so existing handlers still catch it, while carrying
    a numeric ``error_code`` for the PGE to map to a specific failure.
    """

    def __init__(self, message: str, error_code: int):
        """Build the error, prefixing the message with its numeric code."""
        super().__init__(f"[error {error_code}] {message}")
        self.error_code = error_code


def setup_file_logging(filename: Filename) -> None:
    """Redirect all logging to a file."""
    logger = logging.getLogger()
    # In addition to stderr, log to a file if requested
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
