# zerepy_tools/__init__.py
"""
Zerepy Tools - High-performance tools add-on for Zerepy.

This package provides utilities for extracting tool definitions
from natural language prompts and generating executable code.
"""

__version__ = "0.1.0"

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

# Convenience exports
__all__ = [
    # Core models
    'ToolDefinition',
    'Parameter',
    'ParameterType',
    
    # Configuration
    'config',
    
    # Main components
    'ToolExtractor',
    'CodeGenerator',
    
    # High-level API functions
    'generate_tool_from_prompt',
    'generate_code',
    'batch_extract_tools',
    'batch_generate_code',
]


# Main package entry points

def extract_tools(prompt):
    """
    Extract tool definitions from a natural language prompt.
    
    Args:
        prompt: Natural language description of tools
        
    Returns:
        List of extracted tool definitions
    """
    extractor = ToolExtractor()
    return extractor.extract_from_prompt(prompt)


def create_tool(prompt):
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


# zerepy_tools/prompt_assistant.py
"""
Prompt assistant for improving tool descriptions.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class PromptAssistant:
    """
    Provides assistance for writing better tool prompts.
    
    This class offers utilities for improving, completing,
    and analyzing tool descriptions.
    """
    
    def __init__(self, use_llm: bool = True, provider: Optional[str] = None):
        """
        Initialize the prompt assistant.
        
        Args:
            use_llm: Whether to use LLM for assistance
            provider: LLM provider (default: config.default_llm_provider)
        """
        self.use_llm = use_llm
        self.provider = provider
        
        # Initialize LLM client if needed
        self.llm_client = None
        if use_llm:
            from zerepy_tools.core.config import config
            self.provider = provider or config.default_llm_provider
            
            if self.provider == "openai":
                try:
                    import openai
                    self.llm_client = openai.OpenAI()
                except ImportError:
                    logger.warning("OpenAI package not installed")
            elif self.provider == "anthropic":
                try:
                    import anthropic
                    self.llm_client = anthropic.Anthropic()
                except ImportError:
                    logger.warning("Anthropic package not installed")
    
    def improve_prompt(self, prompt: str) -> str:
        """
        Improve a tool prompt by making it more precise and complete.
        
        Args:
            prompt: Initial tool description
            
        Returns:
            Improved prompt
        """
        if not self.use_llm or not self.llm_client:
            return self._basic_improve_prompt(prompt)
        
        try:
            return self._llm_improve_prompt(prompt)
        except Exception as e:
            logger.error(f"Error improving prompt with LLM: {str(e)}")
            return self._basic_improve_prompt(prompt)
    
    def _basic_improve_prompt(self, prompt: str) -> str:
        """
        Basic prompt improvement without LLM.
        
        Args:
            prompt: Initial tool description
            
        Returns:
            Improved prompt
        """
        lines = []
        
        # Add a standardized header if missing
        if not re.search(r'\b(?:create|implement|build)\s+(?:a|an)\s+', prompt.lower()):
            lines.append(f"Create a tool that {prompt.strip()}")
        else:
            lines.append(prompt.strip())
        
        # Add parameter section if missing
        if "parameter" not in prompt.lower() and "input" not in prompt.lower():
            lines.append("\nParameters:")
            
            # Try to guess parameters from context
            words = set(re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', prompt))
            potential_params = [
                word for word in words 
                if word.lower() not in [
                    'a', 'an', 'the', 'and', 'or', 'is', 'are', 'to', 'from',
                    'that', 'which', 'this', 'these', 'those', 'it', 'tool',
                    'function', 'create', 'implement', 'build', 'should', 'can',
                    'will', 'would', 'may', 'might', 'must', 'for', 'with', 'by'
                ]
            ]
            
            # Add potential parameters
            for param in potential_params[:3]:  # Limit to top 3
                lines.append(f"- {param}: Parameter description")
        
        # Add description section if needed
        if "description" not in prompt.lower():
            lines.append("\nDescription: Tool for performing the specified functionality.")
        
        return "\n".join(lines)
    
    def _llm_improve_prompt(self, prompt: str) -> str:
        """
        Improve prompt using LLM.
        
        Args:
            prompt: Initial tool description
            
        Returns:
            Improved prompt
        """
        system_prompt = """You are a tool prompt improvement assistant. Your job is to:
1. Analyze the user's initial tool description
2. Identify any missing details or ambiguities
3. Suggest a more complete and precise tool description
4. Make sure the improved description includes:
   - Clear tool name
   - Detailed description of functionality
   - Well-defined parameters with types and descriptions
   - Any necessary constraints or requirements

Your improved prompt should be clear, specific, and comprehensive.
Return ONLY the improved prompt without any other explanations or formatting.
"""
        
        if self.provider == "openai":
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please improve this tool description:\n\n{prompt}"}
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.llm_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\nPlease improve this tool description:\n\n{prompt}"}
                ]
            )
            return response.content[0].text
        
        # Fallback
        return self._basic_improve_prompt(prompt)
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a tool prompt and provide feedback.
        
        Args:
            prompt: Tool description
            
        Returns:
            Analysis results
        """
        if not self.use_llm or not self.llm_client:
            return self._basic_analyze_prompt(prompt)
        
        try:
            return self._llm_analyze_prompt(prompt)
        except Exception as e:
            logger.error(f"Error analyzing prompt with LLM: {str(e)}")
            return self._basic_analyze_prompt(prompt)
    
    def _basic_analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Basic prompt analysis without LLM.
        
        Args:
            prompt: Tool description
            
        Returns:
            Analysis results
        """
        analysis = {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        # Check length
        if len(prompt) < 50:
            analysis["weaknesses"].append("Tool description is too short")
            analysis["suggestions"].append("Provide a more detailed description")
        else:
            analysis["strengths"].append("Sufficient description length")
        
        # Check for tool name
        if re.search(r'\b(?:create|implement|build)\s+(?:a|an)\s+([a-zA-Z0-9_\s]+?)\s+tool', prompt):
            analysis["strengths"].append("Tool name is specified")
        else:
            analysis["weaknesses"].append("Tool name is not clearly specified")
            analysis["suggestions"].append("Clearly specify the tool name")
        
        # Check for parameters
        if "parameter" in prompt.lower() or "input" in prompt.lower():
            analysis["strengths"].append("Parameters are mentioned")
        else:
            analysis["weaknesses"].append("Parameters are not defined")
            analysis["suggestions"].append("Define the required parameters")
        
        # Check for parameter types
        param_types = ["string", "integer", "number", "boolean", "array", "object", "date"]
        if any(p_type in prompt.lower() for p_type in param_types):
            analysis["strengths"].append("Parameter types are specified")
        else:
            analysis["weaknesses"].append("Parameter types are not specified")
            analysis["suggestions"].append("Specify the types for all parameters")
        
        # Calculate score
        score = 5  # Base score
        score += len(analysis["strengths"]) * 1.5
        score -= len(analysis["weaknesses"]) * 1.0
        analysis["score"] = max(1, min(10, round(score)))
        
        return analysis
    
    def _llm_analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt using LLM.
        
        Args:
            prompt: Tool description
            
        Returns:
            Analysis results
        """
        system_prompt = """You are a tool prompt analyzer. Your job is to:
1. Assess the quality and completeness of a tool description
2. Identify strengths and weaknesses
3. Provide a structured analysis

Your analysis should cover:
- Clarity: Is the purpose of the tool clear?
- Completeness: Are all necessary details included?
- Specificity: Are inputs and outputs clearly defined?
- Constraints: Are limitations or requirements specified?
- Error handling: Is error handling mentioned?

Return a JSON object with the following fields:
- "score": Overall quality score from 1-10
- "strengths": Array of strong points
- "weaknesses": Array of weak points or missing elements
- "suggestions": Array of specific improvement suggestions
"""
        
        try:
            if self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please analyze this tool description:\n\n{prompt}"}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\nPlease analyze this tool description:\n\n{prompt}"}
                    ]
                )
                # Extract JSON from response
                content = response.content[0].text
                match = re.search(r'```json\n(.*?)\n```|(\{.*\})', content, re.DOTALL)
                if match:
                    json_str = match.group(1) or match.group(2)
                    return json.loads(json_str)
                else:
                    return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            
        # Fallback
        return self._basic_analyze_prompt(prompt)
    
    def suggest_parameters(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Suggest parameters for a tool based on its description.
        
        Args:
            prompt: Tool description
            
        Returns:
            List of parameter dictionaries
        """
        if not self.use_llm or not self.llm_client:
            return self._basic_suggest_parameters(prompt)
        
        try:
            return self._llm_suggest_parameters(prompt)
        except Exception as e:
            logger.error(f"Error suggesting parameters with LLM: {str(e)}")
            return self._basic_suggest_parameters(prompt)
    
    def _basic_suggest_parameters(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Basic parameter suggestion without LLM.
        
        Args:
            prompt: Tool description
            
        Returns:
            List of parameter dictionaries
        """
        suggested_params = []
        
        # Try to extract mentioned parameters
        param_mentions = re.findall(
            r'\b([a-zA-Z][a-zA-Z0-9_]*)\s+(?:parameter|argument|input|field)\b', 
            prompt,
            re.IGNORECASE
        )
        
        # Also try other patterns
        param_mentions.extend(re.findall(
            r'(?:parameter|argument|input|field)s?\s+(?:called|named)\s+([a-zA-Z][a-zA-Z0-9_]*)',
            prompt,
            re.IGNORECASE
        ))
        
        # If no explicit mentions, try to guess from context
        if not param_mentions:
            words = set(re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', prompt))
            param_mentions = [
                word for word in words 
                if word.lower() not in [
                    'a', 'an', 'the', 'and', 'or', 'is', 'are', 'to', 'from',
                    'that', 'which', 'this', 'these', 'those', 'it', 'tool',
                    'function', 'create', 'implement', 'build', 'should', 'can',
                    'will', 'would', 'may', 'might', 'must', 'for', 'with', 'by'
                ]
            ]
        
        # Generate parameter dictionaries
        for param in param_mentions:
            # Try to infer type
            param_type = "string"  # Default type
            
            # Basic type inference
            if param.lower() in ["count", "number", "amount", "quantity", "limit"]:
                param_type = "integer"
            elif param.lower() in ["enabled", "active", "flag", "visible"]:
                param_type = "boolean"
            elif param.lower() in ["items", "list", "array", "values"]:
                param_type = "array"
            elif param.lower() in ["date", "time", "timestamp", "datetime"]:
                param_type = "date"
            
            suggested_params.append({
                "name": param,
                "type": param_type,
                "description": f"The {param} parameter",
                "required": True
            })
        
        return suggested_params
    
    def _llm_suggest_parameters(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Suggest parameters using LLM.
        
        Args:
            prompt: Tool description
            
        Returns:
            List of parameter dictionaries
        """
        system_prompt = """You are a tool parameter suggestion assistant. Your job is to:
1. Analyze a tool description
2. Identify what parameters would be needed for the tool
3. Suggest appropriate parameter names, types, and descriptions

For each parameter, provide:
- name: A camelCase or snake_case identifier
- type: One of string, integer, number, boolean, array, object, or date
- description: A clear explanation of what the parameter is for
- required: Whether the parameter should be required (true or false)

Return a JSON array of parameter objects, e.g.:
[
  {
    "name": "query",
    "type": "string",
    "description": "The search query to execute",
    "required": true
  },
  {
    "name": "maxResults",
    "type": "integer",
    "description": "Maximum number of results to return",
    "required": false
  }
]
"""
        
        try:
            if self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please suggest parameters for this tool:\n\n{prompt}"}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\nPlease suggest parameters for this tool:\n\n{prompt}"}
                    ]
                )
                # Extract JSON from response
                content = response.content[0].text
                match = re.search(r'```json\n(.*?)\n```|(\[.*\])', content, re.DOTALL)
                if match:
                    json_str = match.group(1) or match.group(2)
                    return json.loads(json_str)
                else:
                    return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            
        # Fallback
        return self._basic_suggest_parameters(prompt)


# Add to main exports
__all__.extend(['PromptAssistant'])
