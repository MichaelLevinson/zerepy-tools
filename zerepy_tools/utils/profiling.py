# zerepy_tools/utils/profiling.py
"""
Profiling utilities for performance measurement.
"""

import time
import logging
import functools
from typing import Callable, Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """
    Simple utility for timing code execution.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a timer.
        
        Args:
            name: Optional name for the timer
        """
        self.name = name or "Timer"
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop the timer and return the elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        """
        Get the elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        """Start the timer when entering a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting a context manager."""
        self.stop()
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.6f} seconds")


@contextmanager
def timed_section(name: str, level: int = logging.DEBUG):
    """
    Context manager for timing a section of code.
    
    Args:
        name: Name of the section
        level: Logging level
    
    Yields:
        Timer instance
    """
    timer = Timer(name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        logger.log(level, f"{name}: {timer.elapsed:.6f} seconds")


def timed(func: Callable) -> Callable:
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__) as timer:
            result = func(*args, **kwargs)
        
        logger.debug(f"{func.__name__}: {timer.elapsed:.6f} seconds")
        return result
    
    return wrapper


class Profiler:
    """
    Simple profiler for tracking execution times of different code sections.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.timings: Dict[str, List[float]] = {}
    
    def start_section(self, name: str) -> Timer:
        """
        Start timing a section.
        
        Args:
            name: Section name
            
        Returns:
            Timer instance
        """
        timer = Timer()
        timer.start()
        return timer
    
    def end_section(self, name: str, timer: Timer) -> None:
        """
        End timing a section and record the result.
        
        Args:
            name: Section name
            timer: Timer instance
        """
        elapsed = timer.stop()
        
        if name not in self.timings:
            self.timings[name] = []
        
        self.timings[name].append(elapsed)
    
    @contextmanager
    def section(self, name: str):
        """
        Context manager for timing a section.
        
        Args:
            name: Section name
        
        Yields:
            Timer instance
        """
        timer = self.start_section(name)
        try:
            yield timer
        finally:
            self.end_section(name, timer)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics.
        
        Returns:
            Dictionary of timing statistics
        """
        stats = {}
        
        for name, times in self.timings.items():
            if not times:
                continue
                
            count = len(times)
            total = sum(times)
            avg = total / count
            
            stats[name] = {
                'count': count,
                'total': total,
                'avg': avg,
                'min': min(times),
                'max': max(times)
            }
        
        return stats
    
    def print_stats(self) -> None:
        """Print timing statistics."""
        stats = self.get_stats()
        
        if not stats:
            logger.info("No profiling data available")
            return
        
        logger.info("Profiling results:")
        for name, section_stats in stats.items():
            logger.info(
                f"  {name}: {section_stats['count']} calls, "
                f"avg: {section_stats['avg']:.6f}s, "
                f"total: {section_stats['total']:.6f}s, "
                f"min: {section_stats['min']:.6f}s, "
                f"max: {section_stats['max']:.6f}s"
            )
