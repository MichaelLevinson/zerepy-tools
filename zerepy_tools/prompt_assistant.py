# zerepy_tools/prompt_assistant.py
"""
Prompt assistant for improving tool descriptions.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union

from zerepy_tools.core.config import config
from zerepy_tools.core.constants import DEFAULT_LLM_SETTINGS

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
            self.provider = provider or config.default_llm_provider
            self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the LLM client."""
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
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}")
    
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
                model=config.get_llm_config(self.provider).get(
                    "model", DEFAULT_LLM_SETTINGS["openai"]["model"]
                ),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please improve this tool description:\n\n{prompt}"}
                ],
                temperature=config.get_llm_config(self.provider).get(
                    "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"]
                ),
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.llm_client.messages.create(
                model=config.get_llm_config(self.provider).get(
                    "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                ),
                max_tokens=config.get_llm_config(self.provider).get(
                    "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                ),
                temperature=config.get_llm_config(self.provider).get(
                    "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"]
                ),
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\nPlease improve this tool description:\n\n{prompt}"}
                ]
            )
            return response.content[0].text
        
        # Fallback
        return self._basic_improve_prompt(prompt)
    
    def complete_prompt(self, partial_prompt: str) -> str:
        """
        Complete a partial tool description.
        
        Args:
            partial_prompt: Beginning of a tool description
            
        Returns:
            Completed tool description
        """
        if not self.use_llm or not self.llm_client:
            return self._basic_complete_prompt(partial_prompt)
        
        try:
            return self._llm_complete_prompt(partial_prompt)
        except Exception as e:
            logger.error(f"Error completing prompt with LLM: {str(e)}")
            return self._basic_complete_prompt(partial_prompt)
    
    def _basic_complete_prompt(self, partial_prompt: str) -> str:
        """
        Basic prompt completion without LLM.
        
        Args:
            partial_prompt: Beginning of a tool description
            
        Returns:
            Completed tool description
        """
        # First improve the partial prompt
        improved = self._basic_improve_prompt(partial_prompt)
        
        # Add standard sections if missing
        lines = improved.splitlines()
        
        has_parameters = any("parameter" in line.lower() for line in lines)
        has_description = any("description" in line.lower() for line in lines)
        
        if not has_parameters:
            lines.append("\nParameters:")
            lines.append("- input: The primary input for the tool (string)")
            lines.append("- options: Optional configuration settings (object)")
        
        if not has_description:
            tool_name = "this tool"
            for line in lines:
                match = re.search(r'(?:create|implement|build)\s+(?:a|an)\s+([a-zA-Z0-9_\s]+?)\s+tool', line, re.IGNORECASE)
                if match:
                    tool_name = match.group(1).strip()
                    break
            
            lines.append(f"\nDescription: A tool for {tool_name} functionality.")
        
        # Add usage example
        if not any("example" in line.lower() for line in lines):
            lines.append("\nExample usage:")
            lines.append("```")
            lines.append("// Example code showing how to use this tool")
            lines.append("```")
        
        return "\n".join(lines)
    
    def _llm_complete_prompt(self, partial_prompt: str) -> str:
        """
        Complete a prompt using LLM.
        
        Args:
            partial_prompt: Beginning of a tool description
            
        Returns:
            Completed tool description
        """
        system_prompt = """You are a tool prompt completion assistant. Your job is to:
1. Analyze the user's partial tool description
2. Complete it with all necessary details
3. Ensure the final description is comprehensive and precise

A good tool description includes:
- What the tool does (its purpose and functionality)
- What inputs/parameters it needs
- What output it produces
- Any relevant constraints or requirements
- Error handling considerations

Maintain the user's original intent while adding clarity and completeness.
Provide the FULL completed description, not just the additions.
"""
        
        if self.provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=config.get_llm_config(self.provider).get(
                    "model", DEFAULT_LLM_SETTINGS["openai"]["model"]
                ),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please complete this partial tool description:\n\n{partial_prompt}"}
                ],
                temperature=config.get_llm_config(self.provider).get(
                    "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"]
                ),
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.llm_client.messages.create(
                model=config.get_llm_config(self.provider).get(
                    "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                ),
                max_tokens=config.get_llm_config(self.provider).get(
                    "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                ),
                temperature=config.get_llm_config(self.provider).get(
                    "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"]
                ),
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\nPlease complete this partial tool description:\n\n{partial_prompt}"}
                ]
            )
            return response.content[0].text
        
        # Fallback
        return self._basic_complete_prompt(partial_prompt)
    
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
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["openai"]["model"]
                    ),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please analyze this tool description:\n\n{prompt}"}
                    ],
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"]
                    ),
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                    ),
                    max_tokens=config.get_llm_config(self.provider).get(
                        "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                    ),
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"]
                    ),
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
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["openai"]["model"]
                    ),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please suggest parameters for this tool:\n\n{prompt}"}
                    ],
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"]
                    ),
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                    ),
                    max_tokens=config.get_llm_config(self.provider).get(
                        "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                    ),
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"]
                    ),
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
    
    def generate_examples(self, tool_type: str, count: int = 3) -> List[str]:
        """
        Generate example prompts for a particular type of tool.
        
        Args:
            tool_type: Category of tool (e.g., 'data processing', 'web search')
            count: Number of examples to generate
            
        Returns:
            List of example prompts
        """
        if not self.use_llm or not self.llm_client:
            return self._basic_generate_examples(tool_type, count)
        
        try:
            return self._llm_generate_examples(tool_type, count)
        except Exception as e:
            logger.error(f"Error generating examples with LLM: {str(e)}")
            return self._basic_generate_examples(tool_type, count)
    
    def _basic_generate_examples(self, tool_type: str, count: int = 3) -> List[str]:
        """
        Generate basic example prompts without using LLM.
        
        Args:
            tool_type: Category of tool
            count: Number of examples to generate
            
        Returns:
            List of example prompts
        """
        examples = []
        
        # Example templates based on tool type
        if tool_type.lower() == "data processing":
            examples = [
                "Create a data processing tool that accepts a CSV file and filters rows based on a column value",
                "Create a tool that aggregates numerical data by categories and returns summary statistics",
                "Create a tool that converts between different data formats (JSON, CSV, XML)"
            ]
        elif tool_type.lower() == "web search":
            examples = [
                "Create a web search tool that accepts a query parameter and returns relevant results",
                "Create a tool that searches for news articles based on keywords and date range",
                "Create a tool that performs image search based on descriptive terms"
            ]
        elif tool_type.lower() == "communication":
            examples = [
                "Create a messaging tool that sends notifications to users via email",
                "Create a tool that posts updates to social media platforms",
                "Create a tool that translates text between languages"
            ]
        else:
            # Generic examples
            examples = [
                f"Create a {tool_type} tool that processes input data and returns results",
                f"Build a {tool_type} tool with configuration options for customizing behavior",
                f"Implement a {tool_type} tool that integrates with external services"
            ]
        
        # Return requested number of examples
        return examples[:count]
    
    def _llm_generate_examples(self, tool_type: str, count: int = 3) -> List[str]:
        """
        Generate example prompts using LLM.
        
        Args:
            tool_type: Category of tool
            count: Number of examples to generate
            
        Returns:
            List of example prompts
        """
        system_prompt = f"""You are a tool prompt example generator. Your job is to:
1. Generate {count} distinct example prompts for {tool_type} tools
2. Make each example clear, specific, and realistic
3. Ensure each example covers all necessary details

Each example should specify:
- What the tool does
- What inputs/parameters it needs
- What output it produces
- Any relevant constraints or requirements

Format your response as follows:
EXAMPLE 1: [Your first example prompt]
EXAMPLE 2: [Your second example prompt]
...and so on.

Use only this format with the exact markers "EXAMPLE N:" at the start of each example.
"""
        
        try:
            if self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["openai"]["model"]
                    ),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please generate {count} example prompts for {tool_type} tools."}
                    ],
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["openai"]["temperature"] + 0.2
                    ),  # Slightly higher temperature for creativity
                )
                response_text = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=config.get_llm_config(self.provider).get(
                        "model", DEFAULT_LLM_SETTINGS["anthropic"]["model"]
                    ),
                    max_tokens=config.get_llm_config(self.provider).get(
                        "max_tokens", DEFAULT_LLM_SETTINGS["anthropic"]["max_tokens"]
                    ),
                    temperature=config.get_llm_config(self.provider).get(
                        "temperature", DEFAULT_LLM_SETTINGS["anthropic"]["temperature"] + 0.2
                    ),  # Slightly higher temperature for creativity
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\nPlease generate {count} example prompts for {tool_type} tools."}
                    ]
                )
                response_text = response.content[0].text
            else:
                return self._basic_generate_examples(tool_type, count)
            
            # Parse the response to extract examples
            examples = []
            
            # First try with the explicit EXAMPLE N: format we requested
            pattern = r"EXAMPLE\s*(\d+)\s*:\s*(.*?)(?=EXAMPLE\s*\d+\s*:|$)"
            matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                example = match.group(2).strip()
                if example:
                    examples.append(example)
            
            # If that didn't work, try other common formats
            if not examples:
                # Try numbered list with periods (1. Example...)
                pattern = r"(?:^|\n)\s*(\d+)[\.\)]\s*(.*?)(?=(?:^|\n)\s*\d+[\.\)]|$)"
                matches = re.finditer(pattern, response_text, re.DOTALL)
                
                for match in matches:
                    example = match.group(2).strip()
                    if example:
                        examples.append(example)
            
            # If still no examples, try looking for "Example N:" format
            if not examples:
                pattern = r"Example\s*(\d+)\s*:?\s*(.*?)(?=Example\s*\d+|$)"
                matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)
                
                for match in matches:
                    example = match.group(2).strip()
                    if example:
                        examples.append(example)
            
            # If all else fails, split by double newlines as a last resort
            if not examples and response_text.strip():
                examples = [ex.strip() for ex in response_text.split("\n\n") if ex.strip()]
                
            # Ensure we return at most the requested number of examples
            return examples[:count] if examples else self._basic_generate_examples(tool_type, count)
            
        except Exception as e:
            logger.error(f"Error generating examples with LLM: {str(e)}")
            return self._basic_generate_examples(tool_type, count)
