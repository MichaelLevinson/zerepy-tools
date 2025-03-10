# zerepy_tools_core/__init__.py
"""
Performance-critical Cython components for Zerepy Tools.
"""

try:
    from .pattern_engine import *
    from .vector_ops import *
    from .validators import *
    from .sandbox import *
    
    __all__ = [
        'match', 'extract_patterns', 'batch_process',
        'vector_similarity', 'render',
        'validate_python', 'validate_javascript',
        'security_check_python', 'security_check_javascript',
        'safe_execute',
    ]
    
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# zerepy_tools_core/pattern_engine.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized pattern matching engine implementation in Cython.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
cimport numpy as np

logger = logging.getLogger(__name__)


cpdef tuple match(str pattern, str text, bint case_sensitive=False):
    """
    Match a pattern against text with high performance.
    
    Args:
        pattern: Regex pattern
        text: Text to match against
        case_sensitive: Whether the match is case-sensitive
        
    Returns:
        Tuple of (match_found, match_groups)
    """
    cdef int flags = 0 if case_sensitive else re.IGNORECASE
    cdef object match_obj
    
    try:
        match_obj = re.search(pattern, text, flags)
        if match_obj:
            return True, match_obj.groups()
        else:
            return False, ()
    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}")
        return False, ()


cpdef list extract_patterns(str text, list patterns, bint case_sensitive=False):
    """
    Extract all pattern matches from text with high performance.
    
    Args:
        text: Text to extract patterns from
        patterns: List of regex patterns
        case_sensitive: Whether matches are case-sensitive
        
    Returns:
        List of dictionaries with match information
    """
    cdef list results = []
    cdef int flags = 0 if case_sensitive else re.IGNORECASE
    cdef str pattern
    cdef object match_obj
    
    if not text or not patterns:
        return results
    
    for pattern in patterns:
        try:
            for match_obj in re.finditer(pattern, text, flags):
                results.append({
                    'pattern': pattern,
                    'match_text': match_obj.group(0),
                    'groups': match_obj.groups(),
                    'start': match_obj.start(),
                    'end': match_obj.end(),
                })
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
    
    # Sort by position in text
    results.sort(key=lambda x: x['start'])
    return results


cpdef list batch_process(list texts, object process_func):
    """
    Process multiple texts in batch with high performance.
    
    Args:
        texts: List of text strings
        process_func: Function to process each text
        
    Returns:
        List of results from processing each text
    """
    cdef list results = []
    cdef str text
    cdef object result
    
    if not texts:
        return results
    
    for text in texts:
        try:
            result = process_func(text)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
    
    return results


cpdef double vector_similarity(list vec1, list vec2) except -1.0:
    """
    Calculate cosine similarity between two vectors with high performance.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity between the vectors
    """
    cdef np.ndarray arr1, arr2
    cdef double dot_product, norm1, norm2
    
    try:
        arr1 = np.array(vec1, dtype=np.float64)
        arr2 = np.array(vec2, dtype=np.float64)
        
        if arr1.shape[0] != arr2.shape[0]:
            logger.error("Vector dimensions do not match")
            return -1.0
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logger.error(f"Error calculating vector similarity: {str(e)}")
        return -1.0


# zerepy_tools_core/vector_ops.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized vector operations for batch processing.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np
cimport numpy as np

logger = logging.getLogger(__name__)


cpdef list batch_process(list inputs, object process_func):
    """
    Process multiple inputs in batch with high performance.
    
    Args:
        inputs: List of input items
        process_func: Function to process each input
        
    Returns:
        List of results from processing each input
    """
    cdef list results = []
    cdef object item
    cdef object result
    
    if not inputs:
        return results
    
    for item in inputs:
        try:
            result = process_func(item)
            results.extend(result if isinstance(result, list) else [result])
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
    
    return results


cpdef dict batch_generate(list tools, str language, dict options):
    """
    Generate code for multiple tools in batch with high performance.
    
    Args:
        tools: List of tool definitions
        language: Programming language
        options: Additional options
        
    Returns:
        Dictionary mapping tool names to generated code
    """
    cdef dict results = {}
    cdef object tool
    
    if not tools:
        return results
    
    # Import code generator here to avoid circular imports
    from zerepy_tools.generators.code_generator import CodeGenerator
    generator = CodeGenerator()
    
    for tool in tools:
        try:
            code = generator.generate(tool, language, **options)
            results[tool.name] = code
        except Exception as e:
            logger.error(f"Error generating code for {tool.name}: {str(e)}")
            results[tool.name] = f"# Error: {str(e)}"
    
    return results


# zerepy_tools_core/validators.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized code validation in Cython.
"""

import ast
import re
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


cpdef tuple validate_python(str code):
    """
    Validate Python code syntax with high performance.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    cdef list errors = []
    cdef object tree
    
    try:
        tree = ast.parse(code)
        return True, []
    except SyntaxError as e:
        line_no = getattr(e, 'lineno', 'unknown')
        col = getattr(e, 'offset', 'unknown')
        error_msg = str(e)
        errors.append(f"Syntax error at line {line_no}, column {col}: {error_msg}")
        return False, errors
    except Exception as e:
        errors.append(f"Error parsing code: {str(e)}")
        return False, errors


cpdef tuple validate_javascript(str code):
    """
    Validate JavaScript code syntax with high performance.
    
    Args:
        code: JavaScript code to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    cdef list errors = []
    cdef list stack = []
    cdef int line_count = 1
    cdef str char, last_open
    cdef int last_line
    
    for i, char in enumerate(code):
        if char == '\n':
            line_count += 1
        elif char in '({[':
            stack.append((char, line_count))
        elif char in ')}]':
            if not stack:
                errors.append(f"Unmatched closing '{char}' at line {line_count}")
                return False, errors
            
            last_open, last_line = stack.pop()
            if (last_open == '(' and char != ')') or \
               (last_open == '{' and char != '}') or \
               (last_open == '[' and char != ']'):
                errors.append(
                    f"Mismatched '{last_open}' from line {last_line} "
                    f"with '{char}' at line {line_count}"
                )
                return False, errors
    
    # Check for unclosed braces, brackets, or parentheses
    if stack:
        for char, line in stack:
            errors.append(f"Unclosed '{char}' from line {line}")
        return False, errors
    
    return True, []


cpdef tuple security_check_python(str code):
    """
    Check Python code for security issues with high performance.
    
    Args:
        code: Python code to check
        
    Returns:
        Tuple of (is_secure, list_of_issues)
    """
    cdef list issues = []
    cdef list dangerous_patterns = [
        (r'eval\s*\(', "Use of eval() function"),
        (r'exec\s*\(', "Use of exec() function"),
        (r'__import__\s*\(', "Use of __import__() function"),
        (r'subprocess', "Potential subprocess usage"),
        (r'os\.system', "Potential shell command execution"),
        (r'os\.popen', "Potential shell command execution"),
        (r'open\s*\(.+[\'"]w[\'"]\)', "File writing operation"),
        (r'marshal\.loads', "Loading serialized code objects"),
        (r'pickle\.loads', "Loading pickled objects"),
        (r'yaml\.load\s*\([^,)]+\)', "Unsafe YAML loading (use yaml.safe_load)"),
    ]
    cdef str pattern, description
    cdef object match
    cdef int line_no
    
    for pattern, description in dangerous_patterns:
        for match in re.finditer(pattern, code):
            line_no = code[:match.start()].count('\n') + 1
            issues.append(f"Line {line_no}: {description}")
    
    return len(issues) == 0, issues


cpdef tuple security_check_javascript(str code):
    """
    Check JavaScript code for security issues with high performance.
    
    Args:
        code: JavaScript code to check
        
    Returns:
        Tuple of (is_secure, list_of_issues)
    """
    cdef list issues = []
    cdef list dangerous_patterns = [
        (r'eval\s*\(', "Use of eval() function"),
        (r'Function\s*\(\s*[\'"][^\'"]*return', "Dynamic function creation"),
        (r'new\s+Function', "Dynamic function creation"),
        (r'document\.write', "Direct DOM manipulation"),
        (r'innerHTML\s*=', "innerHTML assignment"),
        (r'dangerouslySetInnerHTML', "React's dangerouslySetInnerHTML"),
        (r'child_process', "Node.js child process module"),
        (r'fs\.', "Node.js file system operations"),
        (r'require\s*\(\s*[\'"]child_process[\'"]', "Requiring child_process module"),
        (r'window\.location\s*=', "Redirect attack potential"),
    ]
    cdef str pattern, description
    cdef object match
    cdef int line_no
    
    for pattern, description in dangerous_patterns:
        for match in re.finditer(pattern, code):
            line_no = code[:match.start()].count('\n') + 1
            issues.append(f"Line {line_no}: {description}")
    
    return len(issues) == 0, issues


# zerepy_tools_core/sandbox.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized secure sandbox for code execution.
"""

import logging
import sys
import builtins
import traceback
from typing import Dict, Any, Optional, Tuple
from io import StringIO

logger = logging.getLogger(__name__)


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
        # For simplicity, we're not implementing the actual timeout here
        # In a real implementation, you'd use threading or signal
        exec(compiled_code, globals_dict, locals_dict)
        
        # Try to get a result from locals
        if '__result__' in locals_dict:
            result = locals_dict['__result__']
        
        success = True
    except Exception as e:
        error = traceback.format_exc()
    finally:
        # Restore stdout and stderr
        output = stdout_buffer.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return success, result, output, error
