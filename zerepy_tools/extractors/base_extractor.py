# zerepy_tools/extractors/base_extractor.py
"""
Base extractor interface for Zerepy Tools.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from zerepy_tools.core.models import ToolDefinition


class BaseExtractor(ABC):
    """
    Abstract base class for tool extractors.
    
    Extractors convert natural language descriptions into structured
    tool definitions by analyzing text and identifying parameters.
    """
    
    @abstractmethod
    def extract(self, text: str) -> List[ToolDefinition]:
        """
        Extract tool definitions from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            List of extracted tool definitions
        """
        pass
    
    @abstractmethod
    def extract_single(self, text: str) -> Optional[ToolDefinition]:
        """
        Extract a single tool definition from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Extracted tool definition or None if not found
        """
        pass
    
    def get_confidence_score(self, text: str) -> float:
        """
        Calculate a confidence score for extraction from this text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        return 0.5  # Default implementation
    
    def identify_required_parameters(
        self,
        tool: ToolDefinition,
        text: str
    ) -> List[str]:
        """
        Identify which parameters are required based on the text.
        
        Args:
            tool: Tool definition with parameters
            text: Original text description
            
        Returns:
            List of required parameter names
        """
        # Default implementation: all parameters are required
        return [param.name for param in tool.parameters]


# zerepy_tools/extractors/pattern_extractor.py
"""
Pattern-based tool extraction using regex patterns and heuristics.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from zerepy_tools.core.models import ToolDefinition, Parameter, ParameterType
from zerepy_tools.core.config import config
from zerepy_tools.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PatternExtractor(BaseExtractor):
    """
    Extracts tool definitions using regex patterns and heuristics.
    
    This extractor uses a combination of pattern matching and text analysis
    to identify tool names, descriptions, and parameters.
    """
    
    def __init__(self):
        """Initialize the pattern extractor."""
        # Try to import Cython implementation if available and enabled
        self._cython_impl = None
        if config.use_cython:
            try:
                from zerepy_tools_core import pattern_engine
                self._cython_impl = pattern_engine
            except ImportError:
                logger.debug("Cython pattern engine not available")
        
        # Common patterns for tool extraction
        self.tool_name_patterns = [
            r"(?:create|implement|build)\s+(?:a|an)\s+([a-zA-Z0-9_\s]+?)\s+tool",
            r"tool\s+name:\s*([a-zA-Z0-9_\s]+)",
            r"name:\s*([a-zA-Z0-9_\s]+)",
        ]
        
        self.tool_desc_patterns = [
            r"description:\s*(.*?)(?:\n|$)",
            r"(?:tool|function)\s+description:\s*(.*?)(?:\n|$)",
            r"(?:that|which)\s+(.*?)(?:\.|$)",
        ]
        
        self.param_patterns = [
            r"parameter\s+name:\s*([a-zA-Z0-9_]+)",
            r"(?:parameter|argument|input):\s*([a-zA-Z0-9_]+)",
            r"(?:accept|take|require)s?\s+(?:a|an)?\s*([a-zA-Z0-9_]+)\s+parameter",
        ]
        
        self.param_type_patterns = {
            ParameterType.STRING: [
                r"string", r"text", r"str", r"character", r"word"
            ],
            ParameterType.INTEGER: [
                r"integer", r"int", r"whole\s+number", r"count"
            ],
            ParameterType.NUMBER: [
                r"number", r"float", r"decimal", r"double", r"numeric"
            ],
            ParameterType.BOOLEAN: [
                r"boolean", r"bool", r"flag", r"true/false", r"true or false"
            ],
            ParameterType.ARRAY: [
                r"array", r"list", r"\[\]", r"sequence", r"collection"
            ],
            ParameterType.OBJECT: [
                r"object", r"dict", r"dictionary", r"map", r"json", r"struct"
            ],
            ParameterType.DATE: [
                r"date", r"time", r"datetime", r"timestamp"
            ],
            ParameterType.FILE: [
                r"file", r"attachment", r"document", r"upload"
            ],
        }
    
    def extract(self, text: str) -> List[ToolDefinition]:
        """
        Extract tool definitions from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            List of extracted tool definitions
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            return self._extract_with_cython(text)
        
        # Extract a single tool by default
        tool = self.extract_single(text)
        return [tool] if tool else []
    
    def extract_single(self, text: str) -> Optional[ToolDefinition]:
        """
        Extract a single tool definition from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Extracted tool definition or None if not found
        """
        # Use Cython implementation if available
        if self._cython_impl is not None:
            tools = self._extract_with_cython(text)
            return tools[0] if tools else None
        
        # Extract tool name
        name = self._extract_tool_name(text)
        if not name:
            logger.debug("No tool name found")
            return None
        
        # Extract tool description
        description = self._extract_tool_description(text)
        if not description:
            # Use a default description based on the name
            description = f"A tool for {name.lower().strip()}"
        
        # Create tool definition
        tool = ToolDefinition(name=name, description=description)
        
        # Extract parameters
        parameters = self._extract_parameters(text)
        for param in parameters:
            tool.add_parameter(param)
        
        return tool
    
    def get_confidence_score(self, text: str) -> float:
        """
        Calculate a confidence score for extraction from this text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        
        # Check for tool naming patterns
        for pattern in self.tool_name_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
                break
        
        # Check for parameter patterns
        for pattern in self.param_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += min(len(matches) * 0.1, 0.3)
        
        # Check for structured format indicators
        if re.search(r"parameters?:", text, re.IGNORECASE):
            score += 0.2
        
        # Check for description patterns
        if re.search(r"description:", text, re.IGNORECASE):
            score += 0.2
        
        return min(score, 1.0)
    
    def _extract_with_cython(self, text: str) -> List[ToolDefinition]:
        """
        Extract tools using the Cython implementation.
        
        Args:
            text: Input text to extract from
            
        Returns:
            List of extracted tool definitions
        """
        # This is a placeholder for the actual Cython implementation
        # In a real implementation, we would call the Cython methods directly
        logger.debug("Using Cython pattern engine")
        
        # For now, fall back to Python implementation
        tool = self.extract_single(text)
        return [tool] if tool else []
    
    def _extract_tool_name(self, text: str) -> str:
        """
        Extract tool name from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extracted tool name or empty string if not found
        """
        for pattern in self.tool_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r"\s+", " ", name)
                # Convert to CamelCase
                return "".join(word.capitalize() for word in name.split())
        
        return ""
    
    def _extract_tool_description(self, text: str) -> str:
        """
        Extract tool description from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extracted description or empty string if not found
        """
        for pattern in self.tool_desc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                # Clean up the description
                description = re.sub(r"\s+", " ", description)
                return description
        
        return ""
    
    def _extract_parameters(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract parameters from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of parameter dictionaries
        """
        parameters = []
        
        # Try to extract parameter blocks
        param_blocks = self._extract_parameter_blocks(text)
        if param_blocks:
            for block in param_blocks:
                param = self._parse_parameter_block(block)
                if param:
                    parameters.append(param)
        
        # If no parameter blocks, try to extract from patterns
        if not parameters:
            for pattern in self.param_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.strip()
                    if name and not any(p["name"] == name for p in parameters):
                        param_type = self._infer_parameter_type(name, text)
                        description = self._extract_parameter_description(name, text)
                        parameters.append({
                            "name": name,
                            "type": param_type,
                            "description": description,
                            "required": True
                        })
        
        return parameters
    
    def _extract_parameter_blocks(self, text: str) -> List[str]:
        """
        Extract parameter description blocks from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of parameter block strings
        """
        # Look for parameter sections
        param_section_match = re.search(
            r"parameters?:(.*?)(?:\n\s*\n|$)", 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if param_section_match:
            param_section = param_section_match.group(1)
            # Split by parameter entries
            return re.split(r"\n\s*[-*]\s+", param_section)
        
        return []
    
    def _parse_parameter_block(self, block: str) -> Optional[Dict[str, Any]]:
        """
        Parse a parameter block into a parameter dictionary.
        
        Args:
            block: Parameter block text
            
        Returns:
            Parameter dictionary or None if parsing failed
        """
        # Extract name
        name_match = re.search(r"([a-zA-Z0-9_]+)(?:\s*[:-]|$)", block)
        if not name_match:
            return None
        
        name = name_match.group(1).strip()
        
        # Extract type
        param_type = self._infer_parameter_type(name, block)
        
        # Extract description
        description_match = re.search(r"[:|-]\s*(.*?)(?:\n|$)", block)
        description = (
            description_match.group(1).strip() if description_match else f"{name} parameter"
        )
        
        # Determine if required
        required = "optional" not in block.lower()
        
        return {
            "name": name,
            "type": param_type,
            "description": description,
            "required": required
        }
    
    def _infer_parameter_type(self, name: str, context: str) -> ParameterType:
        """
        Infer parameter type from name and context.
        
        Args:
            name: Parameter name
            context: Context text
            
        Returns:
            Inferred parameter type
        """
        context_lower = context.lower()
        
        # Check for explicit type mentions in context
        for param_type, patterns in self.param_type_patterns.items():
            for pattern in patterns:
                if re.search(
                    rf"{name}\s+(?:is|as|type|parameter)?\s*(?:a|an)?\s*{pattern}",
                    context_lower
                ):
                    return param_type
        
        # Infer from name
        name_lower = name.lower()
        
        # Boolean indicators
        if (name_lower.startswith(("is_", "has_", "should_", "can_", "enable")) or
                name_lower in ("enabled", "active", "visible", "approved", "flag")):
            return ParameterType.BOOLEAN
        
        # Array indicators
        if (name_lower.endswith(("s", "list", "array", "items", "collection")) or
                name_lower in ("tags", "options", "ids", "values")):
            return ParameterType.ARRAY
        
        # Date indicators
        if any(date_term in name_lower for date_term in 
               ("date", "time", "day", "month", "year", "schedule", "birthday")):
            return ParameterType.DATE
        
        # Number indicators
        if any(num_term in name_lower for num_term in 
               ("count", "number", "amount", "quantity", "price", "cost", 
                "rate", "percent", "id", "score", "age", "size")):
            return ParameterType.INTEGER
        
        # Default to string
        return ParameterType.STRING
    
    def _extract_parameter_description(self, name: str, text: str) -> str:
        """
        Extract a description for a parameter.
        
        Args:
            name: Parameter name
            text: Text to extract from
            
        Returns:
            Parameter description
        """
        # Look for explicit description patterns
        desc_match = re.search(
            rf"{name}\s+(?:is|parameter|argument)?\s*(?:-|:|–)?\s*(.*?)(?:\.|$)", 
            text, 
            re.IGNORECASE
        )
        
        if desc_match:
            return desc_match.group(1).strip()
        
        # Generate a generic description
        return f"The {name} parameter"


# zerepy_tools/extractors/llm_extractor.py
"""
LLM-based tool extraction using language models.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union

from zerepy_tools.core.models import ToolDefinition
from zerepy_tools.core.config import config
from zerepy_tools.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """
    Extracts tool definitions using a Language Model.
    
    This extractor uses an LLM to parse natural language descriptions
    into structured tool definitions.
    """
    
    # Prompt template for extraction
    EXTRACTION_PROMPT = """
    Extract tool definitions from the following natural language description.
    Return a JSON object with the following structure:
    
    {
      "tools": [
        {
          "name": "ToolName",
          "description": "Tool description",
          "parameters": {
            "properties": {
              "param1": {
                "type": "string",
                "description": "Parameter description"
              }
            },
            "required": ["param1"]
          }
        }
      ]
    }
    
    The supported parameter types are: string, integer, number, boolean, array, object, date.
    
    Description:
    {text}
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM extractor.
        
        Args:
            provider: LLM provider to use (default: config.default_llm_provider)
        """
        self.provider = provider or config.default_llm_provider
        self.llm_config = config.get_llm_config(self.provider)
        
        # Import the appropriate LLM client
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
            except ImportError:
                logger.error("OpenAI package not installed")
                self.client = None
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                logger.error("Anthropic package not installed")
                self.client = None
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            self.client = None
    
    def extract(self, text: str) -> List[ToolDefinition]:
        """
        Extract tool definitions from text using an LLM.
        
        Args:
            text: Input text to extract from
            
        Returns:
            List of extracted tool definitions
        """
        if not self.client:
            logger.error("LLM client not initialized")
            return []
        
        # Generate prompt
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        
        try:
            # Call the appropriate LLM based on provider
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.llm_config.get("model", "gpt-4"),
                    messages=[
                        {"role": "system", "content": "You extract tool definitions from text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.get("temperature", 0.1),
                    max_tokens=self.llm_config.get("max_tokens", 1000),
                )
                response_text = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.llm_config.get("model", "claude-3-opus-20240229"),
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.get("temperature", 0.1),
                    max_tokens=self.llm_config.get("max_tokens", 1000),
                )
                response_text = response.content[0].text
            else:
                logger.error(f"Unsupported LLM provider: {self.provider}")
                return []
            
            # Parse the JSON response
            try:
                data = json.loads(response_text)
                tool_defs = []
                
                for tool in data.get("tools", []):
                    tool_def = ToolDefinition.from_dict(tool)
                    tool_defs.append(tool_def)
                
                return tool_defs
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return []
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return []
    
    def extract_single(self, text: str) -> Optional[ToolDefinition]:
        """
        Extract a single tool definition from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Extracted tool definition or None if not found
        """
        tools = self.extract(text)
        return tools[0] if tools else None
    
    def get_confidence_score(self, text: str) -> float:
        """
        Calculate a confidence score for extraction from this text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        # LLM extractor has high confidence in natural language descriptions
        # but lower confidence in structured formats
        
        # Check for structured format indicators
        structured_indicators = [
            "```json", 
            '"name":', 
            '"description":', 
            '"parameters":'
        ]
        
        score = 0.8  # Base score for LLM extractor
        
        # Reduce score for structured formats
        for indicator in structured_indicators:
            if indicator in text:
                score -= 0.1
        
        return max(0.1, score)
