# zerepy_tools/extractors/llm_extractor.py
"""
LLM-based tool extraction using language models.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union

from zerepy_tools.core.models import ToolDefinition
from zerepy_tools.core.config import config
from zerepy_tools.core.constants import DEFAULT_LLM_SETTINGS
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
        
        # Initialize LLM client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        self.client = None
        
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
            except ImportError:
                logger.error("OpenAI package not installed")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                logger.error("Anthropic package not installed")
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
    
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
            response_text = self._call_llm(prompt)
            
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
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM provider with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_config.get("model", DEFAULT_LLM_SETTINGS["openai"]["model"]),
                messages=[
                    {"role": "system", "content": "You extract tool definitions from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.get(
                    "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"]
                ),
                max_tokens=self.llm_config.get(
                    "max_tokens", DEFAULT_LLM_SETTINGS["openai"]["max_tokens"]
                ),
            )
            return response.choices[0].message.content
            
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_config.get(
                    "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                ),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.get(
                    "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"]
                ),
                max_tokens=self.llm_config.get(
                    "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                ),
            )
            return response.content[0].text
        
        raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
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
