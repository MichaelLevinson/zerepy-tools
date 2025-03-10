# zerepy_tools/validators/syntax_validator.py
"""
Syntax validator for generated code.
"""

import ast
import logging
from typing import Tuple, List, Optional

from zerepy_tools.core.config import config

logger = logging.getLogger(__name__)


class SyntaxValidator:
    """
    Validates syntax of generated code.
    
    This class checks for syntax errors in different programming languages
    to ensure that generated code is syntactically valid.
    """
    
    def __init__(self):
        """Initialize the syntax validator."""
        # Try to import Cython implementation if available and enabled
        self._cython_impl = None
        if config.use_cython:
            try:
                from zerepy_tools_core import validators
                self._cython_impl = validators
            except ImportError:
                logger.debug("Cython validators not available")
    
    def validate_python(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python code syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            return self._cython_impl.validate_python(code)
        
        # Fall back to Python implementation
        errors = []
        
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            line_no = getattr(e, 'lineno', 'unknown')
            col = getattr(e, 'offset', 'unknown')
            error_msg = str(e)
            errors.append(f"Syntax error at line {line_no}, column {col}: {error_msg}")
            return False, errors
    
    def validate_javascript(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate JavaScript/TypeScript code syntax.
        
        Args:
            code: JavaScript/TypeScript code to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            return self._cython_impl.validate_javascript(code)
        
        # This is a minimal implementation since JS validation typically
        # requires external tools like esprima or TypeScript compiler
        
        # Check for basic syntax issues
        errors = []
        stack = []
        line_count = 1
        
        # Check for balanced braces, brackets, and parentheses
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
        
        # This is a very basic check, doesn't catch semantic errors
        return True, []


# zerepy_tools/validators/security_validator.py
"""
Security validator for generated code.
"""

import re
import logging
from typing import Tuple, List, Optional

from zerepy_tools.core.config import config

logger = logging.getLogger(__name__)


class SecurityValidator:
    """
    Validates security of generated code.
    
    This class checks for potential security issues in generated code,
    such as dangerous imports, eval usage, or shell command execution.
    """
    
    def __init__(self):
        """Initialize the security validator."""
        # Try to import Cython implementation if available and enabled
        self._cython_impl = None
        if config.use_cython:
            try:
                from zerepy_tools_core import validators
                self._cython_impl = validators
            except ImportError:
                logger.debug("Cython validators not available")
        
        # Patterns for dangerous Python constructs
        self.python_dangerous_patterns = [
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
        
        # Patterns for dangerous JavaScript constructs
        self.javascript_dangerous_patterns = [
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
    
    def validate_python(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_secure, list_of_issues)
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            return self._cython_impl.security_check_python(code)
        
        # Fall back to Python implementation
        issues = []
        
        # Check for dangerous patterns
        for pattern, description in self.python_dangerous_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                issues.append(f"Line {line_no}: {description}")
        
        return len(issues) == 0, issues
    
    def validate_javascript(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate JavaScript/TypeScript code for security issues.
        
        Args:
            code: JavaScript/TypeScript code to validate
            
        Returns:
            Tuple of (is_secure, list_of_issues)
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            return self._cython_impl.security_check_javascript(code)
        
        # Fall back to Python implementation
        issues = []
        
        # Check for dangerous patterns
        for pattern, description in self.javascript_dangerous_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                issues.append(f"Line {line_no}: {description}")
        
        return len(issues) == 0, issues
