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
    cdef int string_mode = 0  # 0 = not in string, 1 = single quote, 2 = double quote
    cdef bint escape_next = False
    
    for i, char in enumerate(code):
        # Handle string literals properly
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if string_mode == 0:  # Not in a string
            if char == '"':
                string_mode = 2
            elif char == "'":
                string_mode = 1
            elif char == '\n':
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
        else:  # In a string
            if (char == "'" and string_mode == 1) or (char == '"' and string_mode == 2):
                string_mode = 0  # Exit string mode
            elif char == '\n':
                # Unclosed string literal
                errors.append(f"Unclosed string literal at line {line_count}")
                string_mode = 0
                line_count += 1
    
    # Check for unclosed strings
    if string_mode != 0:
        errors.append(f"Unclosed string literal at end of file")
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


cpdef list analyze_python_code(str code):
    """
    Perform advanced analysis of Python code.
    
    Args:
        code: Python code to analyze
        
    Returns:
        List of analysis results
    """
    cdef list results = []
    cdef object tree
    cdef object node
    
    # First validate syntax
    is_valid, errors = validate_python(code)
    if not is_valid:
        return [{"type": "error", "message": error} for error in errors]
    
    try:
        tree = ast.parse(code)
        
        # Collect imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        results.append({"type": "imports", "items": imports})
        
        # Find function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                }
                functions.append(func_info)
        
        results.append({"type": "functions", "items": functions})
        
        # Find class definitions
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                }
                classes.append(class_info)
        
        results.append({"type": "classes", "items": classes})
        
        # Check for potential issues
        potential_issues = []
        
        # Check for bare excepts
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                potential_issues.append({
                    "type": "warning", 
                    "message": f"Line {node.lineno}: Bare except clause"
                })
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        potential_issues.append({
                            "type": "warning",
                            "message": f"Line {node.lineno}: Mutable default argument in function {node.name}"
                        })
        
        if potential_issues:
            results.append({"type": "issues", "items": potential_issues})
            
        return results
    except Exception as e:
        return [{"type": "error", "message": f"Error analyzing code: {str(e)}"}]
