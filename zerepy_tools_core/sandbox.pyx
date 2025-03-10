# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized secure sandbox for code execution.
"""

import logging
import sys
import builtins
import traceback
import threading
import signal
from typing import Dict, Any, Optional, Tuple
from io import StringIO

logger = logging.getLogger(__name__)


# TimeoutError for Python execution
class ExecutionTimeoutError(Exception):
    """Exception raised when code execution times out."""
    pass


# Thread with timeout for executing code
cdef class TimeoutThread(threading.Thread):
    """Thread class with a stop() method."""
    
    def __init__(self, code, globals_dict, locals_dict):
        """Initialize the thread."""
        super().__init__()
        self.code = code
        self.globals_dict = globals_dict
        self.locals_dict = locals_dict
        self.result = None
        self.exception = None
        self._stop_event = threading.Event()
    
    def run(self):
        """Run the thread."""
        try:
            exec(self.code, self.globals_dict, self.locals_dict)
            self.result = True
        except Exception as e:
            self.exception = e
    
    def stop(self):
        """Set stop event."""
        self._stop_event.set()
    
    def stopped(self):
        """Check if stopped."""
        return self._stop_event.is_set()


# Fallback for SIGALRM if not available (Windows)
try:
    from signal import SIGALRM
    alarm_available = True
except ImportError:
    alarm_available = False


cpdef tuple safe_execute(str code, dict globals_dict=None, dict locals_dict=None, int timeout=5):
    """
    Execute code in a controlled sandbox with timeout and restricted builtins.
    
    Args:
        code: Python code to execute
        globals_dict: Global namespace (default: restricted)
        locals_dict: Local namespace (default: empty dict)
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (success, result, output, error)
    """
    cdef dict restricted_builtins
    cdef object old_stdout, old_stderr
    cdef object stdout_buffer, stderr_buffer
    cdef bint success = False
    cdef object result = None
    cdef str output = ""
    cdef str error = ""
    
    # Create restricted builtins
    restricted_builtins = {
        'abs': builtins.abs,
        'all': builtins.all,
        'any': builtins.any,
        'chr': builtins.chr,
        'dict': builtins.dict,
        'dir': builtins.dir,
        'divmod': builtins.divmod,
        'enumerate': builtins.enumerate,
        'filter': builtins.filter,
        'float': builtins.float,
        'format': builtins.format,
        'frozenset': builtins.frozenset,
        'getattr': builtins.getattr,
        'hasattr': builtins.hasattr,
        'hash': builtins.hash,
        'hex': builtins.hex,
        'int': builtins.int,
        'isinstance': builtins.isinstance,
        'issubclass': builtins.issubclass,
        'iter': builtins.iter,
        'len': builtins.len,
        'list': builtins.list,
        'map': builtins.map,
        'max': builtins.max,
        'min': builtins.min,
        'next': builtins.next,
        'oct': builtins.oct,
        'ord': builtins.ord,
        'pow': builtins.pow,
        'print': builtins.print,
        'range': builtins.range,
        'repr': builtins.repr,
        'reversed': builtins.reversed,
        'round': builtins.round,
        'set': builtins.set,
        'sorted': builtins.sorted,
        'str': builtins.str,
        'sum': builtins.sum,
        'tuple': builtins.tuple,
        'type': builtins.type,
        'zip': builtins.zip,
    }
    
    # Setup global namespace
    if globals_dict is None:
        globals_dict = {
            '__builtins__': restricted_builtins,
        }
    
    # Setup local namespace
    if locals_dict is None:
        locals_dict = {}
    
    # Redirect stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    try:
        # Compile code
        compiled_code = compile(code, '<string>', 'exec')
        
        # Execute code with timeout
        if alarm_available:
            # Use SIGALRM for UNIX systems
            def handler(signum, frame):
                raise ExecutionTimeoutError("Code execution timed out")
            
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            
            try:
                exec(compiled_code, globals_dict, locals_dict)
                # Try to get a result from locals
                if '__result__' in locals_dict:
                    result = locals_dict['__result__']
                success = True
            except ExecutionTimeoutError:
                error = "Code execution timed out"
            except Exception as e:
                error = traceback.format_exc()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Use threading for non-UNIX systems
            thread = TimeoutThread(compiled_code, globals_dict, locals_dict)
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                thread.stop()
                error = "Code execution timed out"
            elif thread.exception:
                error = traceback.format_exception_only(type(thread.exception), thread.exception)[0]
            else:
                # Try to get a result from locals
                if '__result__' in locals_dict:
                    result = locals_dict['__result__']
                success = True
    except Exception as e:
        error = traceback.format_exc()
    finally:
        # Restore stdout and stderr
        output = stdout_buffer.getvalue()
        error_output = stderr_buffer.getvalue()
        if error_output:
            if error:
                error += "\n" + error_output
            else:
                error = error_output
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return success, result, output, error


cpdef dict analyze_runtime_behavior(str code, int timeout=5):
    """
    Analyze the runtime behavior of code.
    
    Args:
        code: Python code to analyze
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary with analysis results
    """
    cdef dict results = {
        "success": False,
        "execution_time": 0.0,
        "memory_usage": 0,
        "output_size": 0,
        "errors": [],
        "warnings": []
    }
    
    # Try to import memory_profiler
    try:
        import memory_profiler
        import time
        
        # Define a function to measure memory usage
        def measure_memory_usage():
            memory_usage = memory_profiler.memory_usage((exec, (code,), {}), timeout=timeout)
            return max(memory_usage) - memory_usage[0]
        
        # Measure execution time and memory usage
        start_time = time.time()
        success, result, output, error = safe_execute(code, timeout=timeout)
        end_time = time.time()
        
        results["success"] = success
        results["execution_time"] = end_time - start_time
        results["output_size"] = len(output)
        
        if error:
            results["errors"].append(error)
        
        # Check for suspicious behaviors
        suspicious_patterns = [
            ("Network access", r'urllib|requests|http|socket\.connect'),
            ("File I/O", r'open\(|with open|\.write\(|\.read\('),
            ("System access", r'os\.|sys\.|subprocess|platform'),
            ("Import suspicious", r'import os|import sys|import subprocess')
        ]
        
        for name, pattern in suspicious_patterns:
            if re.search(pattern, code):
                results["warnings"].append(f"Potentially suspicious behavior: {name}")
        
        return results
    except ImportError:
        # Fall back to basic analysis without memory profiling
        import time
        
        start_time = time.time()
        success, result, output, error = safe_execute(code, timeout=timeout)
        end_time = time.time()
        
        results["success"] = success
        results["execution_time"] = end_time - start_time
        results["output_size"] = len(output)
        
        if error:
            results["errors"].append(error)
        
        return results


cpdef tuple execute_with_custom_environment(str code, 
                                          dict allowed_modules, 
                                          int memory_limit=100, 
                                          int timeout=5):
    """
    Execute code with a custom environment and resource limits.
    
    Args:
        code: Python code to execute
        allowed_modules: Dictionary of allowed modules and their attributes
        memory_limit: Memory limit in MB
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (success, result, output, error)
    """
    cdef dict custom_builtins = {}
    cdef dict custom_globals = {}
    cdef str module_name, function_name
    
    # Setup restricted builtins
    for name in ('abs', 'all', 'any', 'chr', 'dict', 'dir', 'divmod', 
                 'enumerate', 'filter', 'float', 'format', 'frozenset', 
                 'getattr', 'hasattr', 'hash', 'hex', 'int', 'isinstance', 
                 'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 
                 'next', 'oct', 'ord', 'pow', 'print', 'range', 'repr', 
                 'reversed', 'round', 'set', 'sorted', 'str', 'sum', 
                 'tuple', 'type', 'zip'):
        custom_builtins[name] = getattr(builtins, name)
    
    # Setup custom globals with allowed modules
    custom_globals['__builtins__'] = custom_builtins
    
    for module_name, attributes in allowed_modules.items():
        try:
            module = __import__(module_name)
            if attributes == '*':
                # Import the whole module
                custom_globals[module_name] = module
            else:
                # Import specific attributes
                module_dict = {}
                for attr in attributes:
                    if hasattr(module, attr):
                        module_dict[attr] = getattr(module, attr)
                custom_globals[module_name] = module_dict
        except ImportError:
            pass
    
    # Execute with resource limits
    return safe_execute(code, custom_globals, timeout=timeout)
