import time
import logging
import functools
import pickle
from pathlib import Path
from typing import Callable, Any

log = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Decorator to retry a function if it raises an exception."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        log.error(f"Failed {func.__name__} after {max_attempts} attempts: {e}")
                        raise
                    log.warning(f"Attempt {attempts} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

def local_cache(cache_dir: Path):
    """Simple disk cache decorator for API responses."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create a semi-unique cache key
            key = f"{func.__name__}_{args}_{kwargs}".replace("/", "_").replace(" ", "")
            cache_file = cache_dir / f"{key}.pkl"
            
            if cache_file.exists():
                log.info(f"Loading cached response from {cache_file.name}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            
            result = func(self, *args, **kwargs)
            
            if result is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                log.info(f"Cached output to {cache_file.name}")
            
            return result
        return wrapper
    return decorator
