"""
Custom exceptions for MAGE.

Provides a hierarchy of typed exceptions for different failure modes:
LLM errors, validation errors, and memory errors.
"""

from __future__ import annotations


class MAGEError(Exception):
    """Base exception for all MAGE errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class MAGEValidationError(MAGEError):
    """Raised when generated CARL JSON fails validation.

    Parameters
    ----------
    message : str
        Description of the validation failure.
    raw_output : str | None
        The raw LLM output that failed validation.
    """

    def __init__(self, message: str, raw_output: str | None = None) -> None:
        self.raw_output = raw_output
        super().__init__(message)


class MAGELLMError(MAGEError):
    """Raised when an LLM call fails.

    Parameters
    ----------
    message : str
        Description of the LLM failure.
    stage : str | None
        The pipeline stage where the error occurred.
    """

    def __init__(self, message: str, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(message)


class MAGEMemoryError(MAGEError):
    """Raised when gigaevo-memory interaction fails.

    Parameters
    ----------
    message : str
        Description of the memory failure.
    """


class MAGECancelledError(MAGEError):
    """Raised when generation is cancelled via the caller's ``asyncio.Event``.

    Parameters
    ----------
    message : str
        Description (typically the stage at which cancellation was observed).
    stage : str | None
        Pipeline stage where the cancel was observed.
    """

    def __init__(self, message: str = "MAGE generation cancelled", stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(message)
