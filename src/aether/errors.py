"""Transport-independent failure categories understood by the runtime."""

from __future__ import annotations


class ModelError(RuntimeError):
    """Adapters supply a safe description, never a gateway body or credential."""

    def __init__(self, message: str, *, kind: str = "fatal", retry_after: float = 0,
                 usage: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after
        self.usage = usage
