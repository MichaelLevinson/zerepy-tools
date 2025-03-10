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
            try:
                return self._cython_impl.security_check_python(code)
            except Exception as e:
                logger.error(f"Error in Cython security check: {str(e)}")
                # Fall back to Python implementation
        
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
            try:
                return self._cython_impl.security_check_javascript(code)
            except Exception as e:
                logger.error(f"Error in Cython security check: {str(e)}")
                # Fall back to Python implementation
        
        # Fall back to Python implementation
        issues = []
        
        # Check for dangerous patterns
        for pattern, description in self.javascript_dangerous_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                issues.append(f"Line {line_no}: {description}")
        
        return len(issues) == 0, issues
    
    def check_allowed_imports(self, code: str, allowed_imports: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if the code only uses allowed imports.
        
        Args:
            code: Code to check
            allowed_imports: List of allowed import names
            
        Returns:
            Tuple of (is_secure, list_of_issues)
        """
        issues = []
        
        # Simple regex for Python imports
        python_import_pattern = r'^\s*(?:import|from)\s+([a-zA-Z0-9_.]+)'
        
        # Check each line for imports
        for i, line in enumerate(code.splitlines()):
            match = re.match(python_import_pattern, line)
            if match:
                import_name = match.group(1).split('.')[0]  # Get the base module name
                if import_name not in allowed_imports:
                    issues.append(f"Line {i+1}: Unauthorized import: {import_name}")
        
        return len(issues) == 0, issues
    
    def check_file_operations(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check for potentially unsafe file operations.
        
        Args:
            code: Code to check
            
        Returns:
            Tuple of (is_secure, list_of_issues)
        """
        issues = []
        
        # Patterns for file operations
        file_patterns = [
            (r'open\s*\(', "File open operation"),
            (r'\.write\s*\(', "File write operation"),
            (r'with\s+open', "File context manager"),
            (r'os\.remove', "File deletion"),
            (r'os\.unlink', "File deletion"),
            (r'shutil\.rmtree', "Directory deletion"),
        ]
        
        # Check for file operation patterns
        for pattern, description in file_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                issues.append(f"Line {line_no}: {description}")
        
        return len(issues) == 0, issues
