# zerepy_tools_core/__init__.py
"""
Performance-critical Cython components for Zerepy Tools.
"""

try:
    from .pattern_engine import match, extract_patterns, batch_process
    from .vector_ops import vector_similarity, batch_process as vector_batch_process
    from .validators import (
        validate_python, validate_javascript, 
        security_check_python, security_check_javascript
    )
    from .sandbox import safe_execute
    
    __all__ = [
        # Pattern engine
        'match',
        'extract_patterns',
        'batch_process',
        
        # Vector operations
        'vector_similarity',
        'vector_batch_process',
        
        # Validators
        'validate_python',
        'validate_javascript',
        'security_check_python',
        'security_check_javascript',
        
        # Sandbox
        'safe_execute',
    ]
    
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    # Print a message when running in verbose mode
    import logging
    logging.getLogger(__name__).debug(f"Cython modules not available: {e}")
