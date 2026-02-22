from __future__ import annotations

from typing import Any, Callable


def traceable_safe(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Best-effort wrapper around langsmith.traceable.
    If LangSmith is unavailable, returns a no-op decorator.
    """
    try:
        from langsmith import traceable  # type: ignore

        return traceable(*args, **kwargs)
    except Exception:
        def _noop(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _noop

