# zerepy_tools/generators/base_generator.py
"""
Base generator interface for Zerepy Tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from zerepy_tools.core.models import ToolDefinition


class BaseGenerator(ABC):
    """
    Abstract base class for code generators.
    
    Generators convert tool definitions into executable code
    implementing the tool's functionality.
    """
    
    @abstractmethod
    def generate(
        self, 
        tool: ToolDefinition,
        language: str = "python",
        **options
    ) -> str:
        """
        Generate code for a tool definition.
        
        Args:
            tool: Tool definition to generate code for
            language: Programming language to generate code in
            **options: Additional language-specific options
            
        Returns:
            Generated code as a string
        """
        pass
    
    def validate(self, code: str, language: str) -> bool:
        """
        Validate the generated code for syntax and security issues.
        
        Args:
            code: Generated code to validate
            language: Programming language of the code
            
        Returns:
            True if code is valid, False otherwise
        """
        return True  # Default implementation assumes valid
    
    @abstractmethod
    def get_template(self, language: str) -> str:
        """
        Get the code template for a specific language.
        
        Args:
            language: Programming language
            
        Returns:
            Template string for the language
        """
        pass


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
            return self._cython_impl.render(template, context)
        
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
                result = eval(condition, {"__builtins__": {}}, context)
                return if_block if result else else_block
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {str(e)}")
                return ''
        
        return re.sub(pattern, replace, template, flags=re.DOTALL)
    
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
        pattern = r'{%\s*for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_]+)\s*%}(.*?){%\s*endfor\s*%}'
        
        def replace(match):
            var_name = match.group(1)
            iterable_name = match.group(2)
            loop_content = match.group(3)
            
            # Get the iterable from context
            iterable = context.get(iterable_name, [])
            if not isinstance(iterable, (list, tuple, dict)):
                logger.error(f"'{iterable_name}' is not iterable")
                return ''
            
            # Generate the loop content
            result = []
            for item in iterable:
                # Create a new context with the loop variable
                loop_context = context.copy()
                loop_context[var_name] = item
                
                # Render the loop content with the new context
                rendered = self._process_variables(loop_content, loop_context)
                result.append(rendered)
            
            return ''.join(result)
        
        return re.sub(pattern, replace, template, flags=re.DOTALL)
    
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
                    else:
                        value = getattr(value, part, '')
                
                # Convert to string
                return str(value)
            except Exception as e:
                logger.error(f"Error accessing variable '{var_name}': {str(e)}")
                return ''
        
        return re.sub(pattern, replace, template)


# zerepy_tools/generators/code_generator.py
"""
Code generator for creating executable tool implementations.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union

from zerepy_tools.core.models import ToolDefinition, Parameter
from zerepy_tools.core.config import config
from zerepy_tools.generators.base_generator import BaseGenerator
from zerepy_tools.generators.template_engine import TemplateEngine
from zerepy_tools.validators.syntax_validator import SyntaxValidator

logger = logging.getLogger(__name__)


class CodeGenerator(BaseGenerator):
    """
    Generates executable code from tool definitions.
    
    This generator uses templates and language-specific logic
    to create code that implements a tool's functionality.
    """
    
    def __init__(self):
        """Initialize the code generator."""
        self.template_engine = TemplateEngine()
        self.syntax_validator = SyntaxValidator()
        
        # Load templates
        self.templates = {
            "python": self._load_template("python"),
            "javascript": self._load_template("javascript"),
            "typescript": self._load_template("typescript"),
        }
    
    def generate(
        self, 
        tool: ToolDefinition,
        language: str = "python",
        **options
    ) -> str:
        """
        Generate code for a tool definition.
        
        Args:
            tool: Tool definition to generate code for
            language: Programming language to generate code in
            **options: Additional language-specific options
            
        Returns:
            Generated code as a string
        """
        # Check if language is supported
        if language not in self.templates:
            supported = ", ".join(self.templates.keys())
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {supported}"
            )
        
        # Prepare context for template rendering
        context = {
            "tool": tool,
            "options": options,
            "parameters": tool.parameters,
            "required_params": [p for p in tool.parameters if p.required],
            "optional_params": [p for p in tool.parameters if not p.required],
            "language": language,
        }
        
        # Add language-specific context
        if language == "python":
            context.update(self._prepare_python_context(tool))
        elif language == "javascript":
            context.update(self._prepare_javascript_context(tool))
        elif language == "typescript":
            context.update(self._prepare_typescript_context(tool))
        
        # Render the template
        template = self.get_template(language)
        code = self.template_engine.render(template, context)
        
        # Validate syntax
        if language == "python":
            valid, errors = self.syntax_validator.validate_python(code)
            if not valid:
                logger.warning(f"Generated Python code has syntax errors: {errors}")
        elif language in ("javascript", "typescript"):
            valid, errors = self.syntax_validator.validate_javascript(code)
            if not valid:
                logger.warning(f"Generated JS/TS code has syntax errors: {errors}")
        
        return code
    
    def _prepare_python_context(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Prepare Python-specific context for template rendering.
        
        Args:
            tool: Tool definition
            
        Returns:
            Python-specific context dictionary
        """
        # Convert parameter types to Python types
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
            "date": "datetime.datetime",
            "file": "BinaryIO",
        }
        
        # Generate type annotations
        type_annotations = {}
        imports = set()
        
        for param in tool.parameters:
            py_type = type_mapping.get(param.type, "Any")
            
            # Add necessary imports
            if py_type == "datetime.datetime":
                imports.add("import datetime")
            elif py_type == "BinaryIO":
                imports.add("from typing import BinaryIO")
            
            # Make optional parameters Optional[Type]
            if not param.required:
                type_annotations[param.name] = f"Optional[{py_type}]"
                imports.add("from typing import Optional")
            else:
                type_annotations[param.name] = py_type
        
        # Add Any import if needed
        if "Any" in str(type_annotations):
            imports.add("from typing import Any")
        
        # Generate docstring parameter section
        docstring_params = []
        for param in tool.parameters:
            desc = param.description or f"The {param.name} parameter"
            docstring_params.append(f"    {param.name}: {desc}")
        
        return {
            "function_name": self._to_snake_case(tool.name),
            "class_name": self._to_pascal_case(tool.name),
            "imports": sorted(list(imports)),
            "type_annotations": type_annotations,
            "docstring_params": docstring_params,
        }
    
    def _prepare_javascript_context(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Prepare JavaScript-specific context for template rendering.
        
        Args:
            tool: Tool definition
            
        Returns:
            JavaScript-specific context dictionary
        """
        # Generate JSDoc parameter documentation
        jsdoc_params = []
        for param in tool.parameters:
            js_type = {
                "string": "string",
                "integer": "number",
                "number": "number",
                "boolean": "boolean",
                "array": "Array",
                "object": "Object",
                "date": "Date",
                "file": "Blob",
            }.get(param.type, "any")
            
            desc = param.description or f"The {param.name} parameter"
            jsdoc_params.append(f" * @param {{{js_type}}} {param.name} {desc}")
        
        return {
            "function_name": self._to_camel_case(tool.name),
            "class_name": self._to_pascal_case(tool.name),
            "jsdoc_params": jsdoc_params,
        }
    
    def _prepare_typescript_context(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Prepare TypeScript-specific context for template rendering.
        
        Args:
            tool: Tool definition
            
        Returns:
            TypeScript-specific context dictionary
        """
        # Convert parameter types to TypeScript types
        type_mapping = {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "any[]",
            "object": "Record<string, any>",
            "date": "Date",
            "file": "Blob",
        }
        
        # Generate type annotations
        type_annotations = {}
        
        for param in tool.parameters:
            ts_type = type_mapping.get(param.type, "any")
            
            # Make optional parameters optional in TypeScript
            if not param.required:
                type_annotations[param.name] = f"{ts_type} | undefined"
            else:
                type_annotations[param.name] = ts_type
        
        # Generate interface for parameters
        interface_props = []
        for param in tool.parameters:
            ts_type = type_mapping.get(param.type, "any")
            required = "" if param.required else "?"
            interface_props.append(
                f"  {param.name}{required}: {ts_type};"
            )
        
        return {
            "function_name": self._to_camel_case(tool.name),
            "class_name": self._to_pascal_case(tool.name),
            "interface_name": f"{self._to_pascal_case(tool.name)}Params",
            "type_annotations": type_annotations,
            "interface_props": interface_props,
            **self._prepare_javascript_context(tool),  # Include JSDoc comments
        }
    
    def get_template(self, language: str) -> str:
        """
        Get the code template for a specific language.
        
        Args:
            language: Programming language
            
        Returns:
            Template string for the language
        """
        if language not in self.templates:
            supported = ", ".join(self.templates.keys())
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {supported}"
            )
        
        return self.templates[language]
    
    def _load_template(self, language: str) -> str:
        """
        Load a template for a specific language.
        
        Args:
            language: Programming language
            
        Returns:
            Template string
        """
        # In a real implementation, these would be loaded from template files
        if language == "python":
            return """
import logging
{% for import_stmt in imports %}
{{ import_stmt }}
{% endfor %}

logger = logging.getLogger(__name__)


def {{ function_name }}(
    {% for param in parameters %}
    {{ param.name }}{% if not param.required %} = None{% endif %}{% if not loop.last %},{% endif %}
    {% endfor %}
) -> Any:
    """
    {{ tool.description }}
    
    Args:
{% for param_doc in docstring_params %}
{{ param_doc }}
{% endfor %}
        
    Returns:
        Implementation result
    """
    logger.info(f"Executing {{ tool.name }}")
    
    # Validate required parameters
    {% for param in required_params %}
    if {{ param.name }} is None:
        raise ValueError("{{ param.name }} is required")
    {% endfor %}
    
    # TODO: Implement the tool functionality
    # This is a placeholder implementation
    result = {
        "success": True,
        "message": "{{ tool.name }} executed successfully",
        "parameters": {
        {% for param in parameters %}
            "{{ param.name }}": {{ param.name }},
        {% endfor %}
        }
    }
    
    return result


class {{ class_name }}:
    """
    Class-based implementation of {{ tool.name }}.
    
    {{ tool.description }}
    """
    
    def __init__(self):
        """Initialize the {{ class_name }} tool."""
        self.logger = logging.getLogger(__name__)
    
    def execute(self,
        {% for param in parameters %}
        {{ param.name }}{% if not param.required %} = None{% endif %}{% if not loop.last %},{% endif %}
        {% endfor %}
    ) -> Any:
        """
        Execute the tool.
        
        Args:
{% for param_doc in docstring_params %}
{{ param_doc }}
{% endfor %}
            
        Returns:
            Implementation result
        """
        return {{ function_name }}(
            {% for param in parameters %}
            {{ param.name }}={{ param.name }},
            {% endfor %}
        )
"""
        elif language == "javascript":
            return """
/**
 * {{ tool.description }}
 *
{% for param_doc in jsdoc_params %}
{{ param_doc }}
{% endfor %}
 * @returns {Object} Implementation result
 */
function {{ function_name }}(
    {% for param in parameters %}
    {{ param.name }}{% if not param.required %} = null{% endif %}{% if not loop.last %},{% endif %}
    {% endfor %}
) {
    console.log(`Executing {{ tool.name }}`);
    
    // Validate required parameters
    {% for param in required_params %}
    if ({{ param.name }} === null || {{ param.name }} === undefined) {
        throw new Error("{{ param.name }} is required");
    }
    {% endfor %}
    
    // TODO: Implement the tool functionality
    // This is a placeholder implementation
    const result = {
        success: true,
        message: "{{ tool.name }} executed successfully",
        parameters: {
        {% for param in parameters %}
            {{ param.name }},
        {% endfor %}
        }
    };
    
    return result;
}

/**
 * Class-based implementation of {{ tool.name }}.
 * {{ tool.description }}
 */
class {{ class_name }} {
    /**
     * Initialize the {{ class_name }} tool.
     */
    constructor() {
        // Initialization code here
    }
    
    /**
     * Execute the tool.
     *
{% for param_doc in jsdoc_params %}
{{ param_doc }}
{% endfor %}
     * @returns {Object} Implementation result
     */
    execute(
        {% for param in parameters %}
        {{ param.name }}{% if not param.required %} = null{% endif %}{% if not loop.last %},{% endif %}
        {% endfor %}
    ) {
        return {{ function_name }}(
            {% for param in parameters %}
            {{ param.name }},
            {% endfor %}
        );
    }
}

// Export both function and class
module.exports = {
    {{ function_name }},
    {{ class_name }}
};
"""
        elif language == "typescript":
            return """
/**
 * Interface for {{ tool.name }} parameters
 */
interface {{ interface_name }} {
{% for prop in interface_props %}
{{ prop }}
{% endfor %}
}

/**
 * {{ tool.description }}
 *
{% for param_doc in jsdoc_params %}
{{ param_doc }}
{% endfor %}
 * @returns {Object} Implementation result
 */
function {{ function_name }}(
    {% for param in parameters %}
    {{ param.name }}{% if not param.required %} = undefined{% endif %}: {{ type_annotations[param.name] }}{% if not loop.last %},{% endif %}
    {% endfor %}
): Record<string, any> {
    console.log(`Executing {{ tool.name }}`);
    
    // Validate required parameters
    {% for param in required_params %}
    if ({{ param.name }} === undefined) {
        throw new Error("{{ param.name }} is required");
    }
    {% endfor %}
    
    // TODO: Implement the tool functionality
    // This is a placeholder implementation
    const result = {
        success: true,
        message: "{{ tool.name }} executed successfully",
        parameters: {
        {% for param in parameters %}
            {{ param.name }},
        {% endfor %}
        }
    };
    
    return result;
}

/**
 * Class-based implementation of {{ tool.name }}.
 * {{ tool.description }}
 */
class {{ class_name }} {
    /**
     * Initialize the {{ class_name }} tool.
     */
    constructor() {
        // Initialization code here
    }
    
    /**
     * Execute the tool.
     *
{% for param_doc in jsdoc_params %}
{{ param_doc }}
{% endfor %}
     * @returns {Object} Implementation result
     */
    execute(params: {{ interface_name }}): Record<string, any> {
        return {{ function_name }}(
            {% for param in parameters %}
            params.{{ param.name }},
            {% endfor %}
        );
    }
}

// Export both function and class
export {
    {{ function_name }},
    {{ class_name }},
    {{ interface_name }}
};
"""
        else:
            return ""
    
    def _to_snake_case(self, text: str) -> str:
        """
        Convert text to snake_case.
        
        Args:
            text: Input text
            
        Returns:
            snake_case version of the text
        """
        # Replace non-alphanumeric with spaces
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lowercase and replace spaces with underscores
        return re.sub(r'\s+', '_', text.lower())
    
    def _to_camel_case(self, text: str) -> str:
        """
        Convert text to camelCase.
        
        Args:
            text: Input text
            
        Returns:
            camelCase version of the text
        """
        # First convert to snake_case
        snake = self._to_snake_case(text)
        # Then convert to camelCase
        components = snake.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def _to_pascal_case(self, text: str) -> str:
        """
        Convert text to PascalCase.
        
        Args:
            text: Input text
            
        Returns:
            PascalCase version of the text
        """
        # First convert to snake_case
        snake = self._to_snake_case(text)
        # Then convert to PascalCase
        return ''.join(x.title() for x in snake.split('_'))


# Convenience function for the public API
def generate_code(
    tool: Union[ToolDefinition, str],
    language: str = "python",
    **options
) -> str:
    """
    Generate code for a tool definition.
    
    Args:
        tool: Tool definition or natural language prompt
        language: Programming language to generate code in
        **options: Additional language-specific options
        
    Returns:
        Generated code as a string
    """
    generator = CodeGenerator()
    
    # If tool is a string, extract it first
    if isinstance(tool, str):
        from zerepy_tools.extractors.composite_extractor import generate_tool_from_prompt
        tool_def = generate_tool_from_prompt(tool)
        if not tool_def:
            raise ValueError("Failed to extract tool definition from prompt")
    else:
        tool_def = tool
    
    return generator.generate(tool_def, language, **options)


def batch_generate_code(
    tools: List[ToolDefinition],
    language: str = "python",
    **options
) -> Dict[str, str]:
    """
    Generate code for multiple tool definitions in batch.
    
    Args:
        tools: List of tool definitions
        language: Programming language to generate code in
        **options: Additional language-specific options
        
    Returns:
        Dictionary mapping tool names to generated code
    """
    generator = CodeGenerator()
    results = {}
    
    # Use vectorization if enabled
    if config.enable_vectorization and config.use_cython:
        try:
            from zerepy_tools_core import vector_ops
            return vector_ops.batch_generate(tools, language, options)
        except (ImportError, AttributeError):
            logger.debug("Vectorized batch processing not available")
    
    # Fall back to sequential processing
    for tool in tools:
        try:
            code = generator.generate(tool, language, **options)
            results[tool.name] = code
        except Exception as e:
            logger.error(f"Error generating code for {tool.name}: {str(e)}")
            results[tool.name] = f"# Error: {str(e)}"
    
    return results
