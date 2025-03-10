# zerepy_tools/generators/template_engine.py
"""
High-performance template engine for code generation.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Union

from zerepy_tools.core.config import config

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Template engine for generating code from templates.
    
    This class provides high-performance string templating
    with support for conditional logic and loops.
    """
    
    def __init__(self):
        """Initialize the template engine."""
        # Try to import Cython implementation if available and enabled
        self._cython_impl = None
        if config.use_cython:
            try:
                from zerepy_tools_core import template_engine
                self._cython_impl = template_engine
            except ImportError:
                logger.debug("Cython template engine not available")
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """
        Render a template with a context.
        
        Args:
            template: Template string with placeholders
            context: Dictionary of values to substitute
            
        Returns:
            Rendered string
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            try:
                return self._cython_impl.render(template, context)
            except Exception as e:
                logger.error(f"Error using Cython template engine: {str(e)}")
                # Fall back to Python implementation
        
        # Fall back to Python implementation
        return self._render_python(template, context)
    
    def _render_python(self, template: str, context: Dict[str, Any]) -> str:
        """
        Python implementation of template rendering.
        
        Args:
            template: Template string with placeholders
            context: Dictionary of values to substitute
            
        Returns:
            Rendered string
        """
        # Process conditional blocks
        template = self._process_conditionals(template, context)
        
        # Process loops
        template = self._process_loops(template, context)
        
        # Process variable substitutions
        template = self._process_variables(template, context)
        
        return template
    
    def _process_conditionals(self, template: str, context: Dict[str, Any]) -> str:
        """
        Process conditional blocks in a template.
        
        Args:
            template: Template string with conditionals
            context: Context dictionary
            
        Returns:
            Processed template string
        """
        # Match {% if condition %}...{% else %}...{% endif %} blocks
        pattern = r'{%\s*if\s+([^%]+?)\s*%}(.*?)(?:{%\s*else\s*%}(.*?))?{%\s*endif\s*%}'
        
        def replace(match):
            condition = match.group(1).strip()
            if_block = match.group(2)
            else_block = match.group(3) or ''
            
            # Evaluate the condition
            try:
                # Create a safe evaluation environment with context
                safe_dict = {k: v for k, v in context.items()}
                
                # Add some safe built-ins
                safe_dict.update({
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "isinstance": isinstance,
                    "list": list,
                    "dict": dict,
                    "True": True,
                    "False": False,
                    "None": None,
                })
                
                result = eval(condition, {"__builtins__": {}}, safe_dict)
                return if_block if result else else_block
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {str(e)}")
                return ''
        
        # Process all conditionals (may be nested)
        prev_template = ""
        current_template = template
        
        # Continue processing until no more changes are made
        # (this handles nested conditionals)
        while prev_template != current_template:
            prev_template = current_template
            current_template = re.sub(pattern, replace, current_template, flags=re.DOTALL)
        
        return current_template
    
    def _process_loops(self, template: str, context: Dict[str, Any]) -> str:
        """
        Process loop blocks in a template.
        
        Args:
            template: Template string with loops
            context: Context dictionary
            
        Returns:
            Processed template string
        """
        # Match {% for item in items %}...{% endfor %} blocks
        pattern = r'{%\s*for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_\.]+)\s*%}(.*?){%\s*endfor\s*%}'
        
        def replace(match):
            var_name = match.group(1)
            iterable_name = match.group(2)
            loop_content = match.group(3)
            
            # Get the iterable from context, supporting dot notation
            iterable = context
            for part in iterable_name.split("."):
                if isinstance(iterable, dict) and part in iterable:
                    iterable = iterable[part]
                elif hasattr(iterable, part):
                    iterable = getattr(iterable, part)
                else:
                    logger.error(f"Cannot find '{part}' in '{iterable_name}'")
                    return ""
            
            if not isinstance(iterable, (list, tuple, dict)):
                logger.error(f"'{iterable_name}' is not iterable")
                return ''
            
            # Generate the loop content
            result = []
            for i, item in enumerate(iterable):
                # Create a new context with the loop variable and loop metadata
                loop_context = context.copy()
                loop_context[var_name] = item
                loop_context["loop"] = {
                    "index": i + 1,
                    "index0": i,
                    "first": i == 0,
                    "last": i == len(iterable) - 1,
                }
                
                # Render the loop content with the new context
                # First process any nested loops or conditionals
                processed_content = self._process_conditionals(loop_content, loop_context)
                processed_content = self._process_loops(processed_content, loop_context)
                # Then process variables
                processed_content = self._process_variables(processed_content, loop_context)
                result.append(processed_content)
            
            return ''.join(result)
        
        # Process all loops (may be nested)
        prev_template = ""
        current_template = template
        
        # Continue processing until no more changes are made
        # (this handles nested loops)
        while prev_template != current_template:
            prev_template = current_template
            current_template = re.sub(pattern, replace, current_template, flags=re.DOTALL)
        
        return current_template
    
    def _process_variables(self, template: str, context: Dict[str, Any]) -> str:
        """
        Process variable substitutions in a template.
        
        Args:
            template: Template string with variables
            context: Context dictionary
            
        Returns:
            Processed template string
        """
        # Match {{ variable }} patterns
        pattern = r'{{(.*?)}}'
        
        def replace(match):
            var_name = match.group(1).strip()
            
            # Handle dotted notation (e.g., item.name)
            parts = var_name.split('.')
            value = context
            
            try:
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, '')
                    elif hasattr(value, part):
                        value = getattr(value, part)
                    else:
                        value = ''
                        break
                
                # Convert to string
                return str(value)
            except Exception as e:
                logger.error(f"Error accessing variable '{var_name}': {str(e)}")
                return ''
        
        return re.sub(pattern, replace, template)
