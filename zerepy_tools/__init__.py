# zerepy_tools/__init__.py
"""
Zerepy Tools - High-performance tools add-on for Zerepy.

This package provides utilities for extracting tool definitions
from natural language prompts and generating executable code.
"""

from zerepy_tools.core.constants import __version__

# Import core modules
from zerepy_tools.core.models import ToolDefinition, Parameter, ParameterType
from zerepy_tools.core.config import config

# Import extractors
from zerepy_tools.extractors.composite_extractor import (
    ToolExtractor, 
    generate_tool_from_prompt,
    batch_extract_tools
)

# Import generators
from zerepy_tools.generators.code_generator import (
    CodeGenerator,
    generate_code,
    batch_generate_code
)

# Import prompt assistant
from zerepy_tools.prompt_assistant import PromptAssistant

# Convenience exports
__all__ = [
    # Version
    '__version__',
    
    # Core models
    'ToolDefinition',
    'Parameter',
    'ParameterType',
    
    # Configuration
    'config',
    
    # Main components
    'ToolExtractor',
    'CodeGenerator',
    'PromptAssistant',
    
    # High-level API functions
    'create_tool',
    'create_implementation',
    'generate_tool_from_prompt',
    'generate_code',
    'batch_extract_tools',
    'batch_generate_code',
]


# Main package entry points

def create_tool(prompt: str) -> ToolDefinition:
    """
    Create a single tool definition from a natural language prompt.
    
    Args:
        prompt: Natural language description of a tool
        
    Returns:
        Extracted tool definition or None if extraction failed
    """
    return generate_tool_from_prompt(prompt)


def create_implementation(tool, language="python", **options):
    """
    Create code implementation for a tool definition.
    
    Args:
        tool: Tool definition or natural language prompt
        language: Programming language to generate code in
        **options: Additional language-specific options
        
    Returns:
        Generated code as a string
    """
    return generate_code(tool, language, **options)
