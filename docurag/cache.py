"""Caching and retry resilience mechanisms for external API and expensive calls."""

import functools
import hashlib
import inspect
import json
import logging
import os
import time
import types
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from pydantic import BaseModel

from docurag.config import get_settings

logger = logging.getLogger(__name__)


def _compute_cache_key(args: tuple, kwargs: dict) -> str:
    """Compute deterministic SHA-256 hash for positional and keyword arguments."""
    hasher = hashlib.sha256()

    for arg in args:
        if isinstance(arg, bytes):
            hasher.update(b"__bytes__:")
            hasher.update(arg)
        elif isinstance(arg, (str, int, float, bool)):
            hasher.update(f"{type(arg).__name__}:{arg}".encode("utf-8"))
        elif hasattr(arg, "model_dump"):
            hasher.update(
                json.dumps(arg.model_dump(), sort_keys=True, default=str).encode(
                    "utf-8"
                )
            )
        else:
            try:
                hasher.update(
                    json.dumps(arg, sort_keys=True, default=str).encode("utf-8")
                )
            except Exception:
                hasher.update(repr(arg).encode("utf-8"))

    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        hasher.update(f"kw:{k}=".encode("utf-8"))
        if isinstance(v, bytes):
            hasher.update(b"__bytes__:")
            hasher.update(v)
        elif isinstance(v, (str, int, float, bool)):
            hasher.update(f"{type(v).__name__}:{v}".encode("utf-8"))
        elif hasattr(v, "model_dump"):
            hasher.update(
                json.dumps(v.model_dump(), sort_keys=True, default=str).encode(
                    "utf-8"
                )
            )
        else:
            try:
                hasher.update(
                    json.dumps(v, sort_keys=True, default=str).encode("utf-8")
                )
            except Exception:
                hasher.update(repr(v).encode("utf-8"))

    return hasher.hexdigest()


def _restore_return_type(cached_data: Any, fn: Callable) -> Any:
    """Reconstruct Pydantic model(s) from cached data using function return type annotation."""
    try:
        return_type = None
        try:
            hints = get_type_hints(fn)
            return_type = hints.get("return")
        except Exception:
            pass

        if return_type is None or return_type is inspect.Signature.empty:
            try:
                sig = inspect.signature(fn)
                if sig.return_annotation is not inspect.Signature.empty:
                    return_type = sig.return_annotation
            except Exception:
                pass

        if isinstance(return_type, str):
            fn_globals = getattr(fn, "__globals__", {})
            if return_type in fn_globals:
                return_type = fn_globals[return_type]

        if return_type is None or return_type is inspect.Signature.empty:
            return cached_data

        # Case 1: Direct Pydantic model (e.g. -> AskResult)
        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            if isinstance(cached_data, dict):
                return return_type.model_validate(cached_data)

        # Case 2: Container list of models (e.g. -> list[Chunk], -> List[PageData])
        origin = get_origin(return_type)
        if origin in (list, List):
            args = get_args(return_type)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                if isinstance(cached_data, list):
                    model_cls = args[0]
                    return [
                        model_cls.model_validate(item) if isinstance(item, dict) else item
                        for item in cached_data
                    ]

        # Case 3: Union or Optional (e.g. -> Optional[AskResult], -> AskResult | None)
        is_union = origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType)
        if is_union:
            args = get_args(return_type)
            model_classes = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
            if model_classes and isinstance(cached_data, dict):
                return model_classes[0].model_validate(cached_data)

    except Exception as exc:
        logger.debug(
            "Could not restore return type for function %s: %s",
            getattr(fn, "__name__", str(fn)),
            exc,
        )

    return cached_data


def disk_cache(namespace: str, custom_cache_dir: Optional[Path] = None):
    """Decorator to cache function results to disk based on SHA-256 of arguments."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            settings = get_settings()
            cache_dir = (
                custom_cache_dir
                if custom_cache_dir is not None
                else settings.cache_dir / namespace
            )
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Include DOCURAG_FAKES mode in the cache key to prevent fake/live pollution
            mode_flag = "fake" if settings.DOCURAG_FAKES else "live"
            key = _compute_cache_key(args, kwargs)
            key = f"{mode_flag}_{key}"
            cache_file = cache_dir / f"{key}.json"

            # Check cache hit with corruption recovery
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    return _restore_return_type(cached_data, fn)
                except (json.JSONDecodeError, OSError, ValueError) as err:
                    logger.warning(
                        "Corrupted cache file %s encountered (%s). Evicting.",
                        cache_file,
                        err,
                    )
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

            # Cache miss: execute function
            result = fn(*args, **kwargs)

            # Atomic file write to avoid race condition corruption
            tmp_file = cache_dir / f"{key}_{os.getpid()}_{time.time_ns()}.tmp"
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    if hasattr(result, "model_dump"):
                        json.dump(result.model_dump(), f)
                    elif isinstance(result, list):
                        payload = [
                            item.model_dump() if hasattr(item, "model_dump") else item
                            for item in result
                        ]
                        json.dump(payload, f, default=str)
                    else:
                        json.dump(result, f, default=str)
                os.replace(tmp_file, cache_file)
            except Exception as e:
                logger.warning("Failed to write disk cache %s: %s", cache_file, e)
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass

            return result

        return wrapper

    return decorator


def with_retry(
    fn_or_none: Optional[Callable] = None,
    tries: int = 3,
    base: float = 1.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator to retry function execution with exponential backoff."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            max_tries = max(1, tries)
            for attempt in range(1, max_tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_tries:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            fn.__name__,
                            max_tries,
                            exc,
                        )
                        raise
                    sleep_time = base * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d for %s failed with %s; retrying in %.2fs",
                        attempt,
                        max_tries,
                        fn.__name__,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
            if last_exc:
                raise last_exc

        return wrapper

    if callable(fn_or_none):
        return decorator(fn_or_none)
    return decorator


__all__ = ["_compute_cache_key", "_restore_return_type", "disk_cache", "with_retry"]
