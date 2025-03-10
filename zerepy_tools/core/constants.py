# zerepy_tools/core/constants.py
"""
Constants used throughout the Zerepy Tools package.
"""

from enum import Enum
from typing import Dict, Any

# Version constants
__version__ = "0.1.0"
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0

# Supported languages for code generation
SUPPORTED_LANGUAGES = ["python", "javascript", "typescript"]

# Parameter type mappings for different languages
PARAMETER_TYPE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "python": {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
        "date": "datetime.datetime",
        "file": "BinaryIO",
    },
    "javascript": {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "array": "Array",
        "object": "Object",
        "date": "Date",
        "file": "Blob",
    },
    "typescript": {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "array": "any[]",
        "object": "Record<string, any>",
        "date": "Date",
        "file": "Blob",
    },
}

# Default LLM settings
DEFAULT_LLM_SETTINGS = {
    "openai": {
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 1000,
    },
    "anthropic": {
        "model": "claude-3-opus-20240229",
        "temperature": 0.1,
        "max_tokens": 1000,
    },
}

# Common words to exclude from parameter extraction
COMMON_WORDS = {
    "a", "an", "the", "and", "or", "is", "are", "to", "from", "with",
    "that", "which", "this", "these", "those", "it", "of", "in", "on",
    "for", "by", "as", "at", "be", "was", "were", "has", "have", "had",
    "do", "does", "did", "but", "if", "not", "what", "when", "where",
    "who", "how", "why", "can", "will", "should", "would", "could",
    "might", "must", "may", "also", "tool", "function", "parameter",
    "create", "implement", "build", "use", "using", "accept", "take",
    "return", "get", "set", "value", "input", "output", "add", "remove"
}

# Parameter extraction patterns
PARAMETER_PATTERNS = [
    r"parameter\s+name:\s*([a-zA-Z0-9_]+)",
    r"(?:parameter|argument|input):\s*([a-zA-Z0-9_]+)",
    r"(?:accepts?|takes?|requires?)\s+(?:a|an)?\s*([a-zA-Z0-9_]+)\s+parameter",
    r"(?:with|using)\s+(?:a|an)?\s*([a-zA-Z0-9_]+)\s+(?:parameter|argument|input)",
    r"parameter\s+([a-zA-Z0-9_]+)\s+(?:is|represents|specifies)",
]

# Keywords that indicate parameters
PARAMETER_KEYWORDS = [
    "parameter", "argument", "input", "param", "arg", "option", "field",
    "variable", "data", "value"
]

# Structured format patterns
STRUCTURED_PATTERNS = [
    r"name:\s*([^\n]+)",
    r"description:\s*([^\n]+)",
    r"parameters?:\s*\[?([^\]]+)\]?",
]

class ToolCategory(Enum):
    """Enum representing different categories of tools."""
    DATA_PROCESSING = "data_processing"
    WEB = "web"
    MEDIA = "media"
    COMMUNICATION = "communication"
    UTILITY = "utility"
    SEARCH = "search"
    AI = "ai"
    UNKNOWN = "unknown"
