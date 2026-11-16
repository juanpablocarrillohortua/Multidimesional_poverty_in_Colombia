"""Disk cache for slow notebook cells.

``cached`` stores a function's return value in a ``.pkl`` file and loads it
back when the function is called again with the same inputs, so a cell that
fits a pipeline for two minutes only does it once per change.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def _content(obj: Any) -> Any:
    """Replace pandas objects by their values, so equal data hashes equal.

    ``joblib.hash`` pickles a DataFrame's internal block layout, which
    differs between a frame just computed and the same frame loaded from a
    ``.pkl``. Hashing the values, labels and dtypes instead keeps the key
    stable when an upstream result is served from the cache.
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
        return (
            type(obj).__name__,
            list(frame.columns),
            [str(t) for t in frame.dtypes],
            pd.util.hash_pandas_object(frame, index=True).to_numpy(),
        )
    if isinstance(obj, (list, tuple)):
        return type(obj)(_content(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _content(v) for k, v in obj.items()}
    return obj


def content_hash(*objs: Any) -> str:
    """``joblib.hash`` of ``objs``, with DataFrames and Series by content."""
    return joblib.hash(_content(objs))


def _source(func: Callable[..., Any]) -> str:
    """The function's source, or its qualified name when there is none."""
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return f"{func.__module__}.{func.__qualname__}"


def cached(
    *,
    cache_dir: str | Path,
    watch: Iterable[str | Path] = (),
    label: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache the return value of the decorated function on disk.

    The result goes to ``<cache_dir>/<label>_<key>.pkl``. The key is a
    ``joblib.hash`` of the function's source code, every positional and
    keyword argument, and the bytes of each file in ``watch``. Change any of
    them and the next call recomputes; change them back and the older file
    is found again.

    Pass ``force_rerun=True`` to the decorated function to recompute and
    overwrite. That is the way out for what the key cannot see: a change
    inside another notebook function that the cached one calls, such as a
    scorer, since those are hashed by name only.

    Parameters
    ----------
    cache_dir : str or Path
        Folder for the ``.pkl`` files; created if missing.
    watch : iterable of str or Path, default=()
        Source files whose edits must invalidate the cache, e.g. the
        modules that define the pipeline's custom steps.
    label : str or None, default=None
        File-name prefix. ``None`` uses the function's name.

    Examples
    --------
    >>> @cached(cache_dir="cv_cache", watch=["utils/preprocessing.py"])
    ... def fit_report(pipeline, X, y): ...
    >>> slow = cached(cache_dir="cv_cache")(some_imported_function)
    """
    cache_dir = Path(cache_dir)
    watched = [Path(p) for p in watch]

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        prefix = label or func.__name__
        source = _source(func)

        @functools.wraps(func)
        def wrapper(*args: Any, force_rerun: bool = False, **kwargs: Any):
            files = [p.read_bytes() for p in watched]
            key = content_hash(prefix, source, args, kwargs, files)
            path = cache_dir / f"{prefix}_{key}.pkl"

            if path.exists() and not force_rerun:
                print(f"Resultado cargado desde el caché: {path}")
                return joblib.load(path)

            print(f"Sin caché para '{prefix}' o hubo cambios; calculando...")
            result = func(*args, **kwargs)
            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, path)
            print(f"Resultado guardado en: {path}")
            return result

        return wrapper

    return decorator
