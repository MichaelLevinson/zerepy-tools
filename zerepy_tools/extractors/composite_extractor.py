# zerepy_tools/extractors/composite_extractor.py
"""
Composite extractor that combines multiple extraction strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from zerepy_tools.core.models import ToolDefinition, Parameter
from zerepy_tools.core.config import config
from zerepy_tools.extractors.base_extractor import BaseExtractor
from zerepy_tools.extractors.pattern_extractor import PatternExtractor
from zerepy_tools.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


class CompositeExtractor(BaseExtractor):
    """
    Combines multiple extraction strategies for better results.
    
    This extractor uses a combination of pattern matching and LLM-based
    extraction, selecting the most appropriate method based on confidence
    scores and merging results when beneficial.
    """
    
    def __init__(self, extractors: Optional[List[BaseExtractor]] = None):
        """
        Initialize the composite extractor.
        
        Args:
            extractors: List of extractors to use (default: PatternExtractor and LLMExtractor)
        """
        if extractors is None:
            self.extractors = [
                PatternExtractor(),
                LLMExtractor()
            ]
        else:
            self.extractors = extractors
        
        # Confidence threshold for considering a result valid
        self.confidence_threshold = 0.3
    
    def extract(self, text: str) -> List[ToolDefinition]:
        """
        Extract tool definitions using multiple strategies.
        
        Args:
            text: Input text to extract from
            
        Returns:
            List of extracted tool definitions
        """
        extractor_results = []
        confidence_scores = {}
        
        # Get results from each extractor
        for extractor in self.extractors:
            confidence = extractor.get_confidence_score(text)
            
            # Skip extractors with low confidence
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"Skipping {extractor.__class__.__name__}: confidence {confidence:.2f}"
                )
                continue
                
            logger.debug(
                f"Using {extractor.__class__.__name__} with confidence {confidence:.2f}"
            )
            
            try:
                tools = extractor.extract(text)
                extractor_results.append((extractor, tools))
                confidence_scores[extractor] = confidence
            except Exception as e:
                logger.error(
                    f"Error with {extractor.__class__.__name__}: {str(e)}"
                )
        
        # If no results, return empty list
        if not extractor_results:
            logger.warning("No extractors produced results")
            return []
        
        # If only one extractor produced results, return those
        if len(extractor_results) == 1:
            return extractor_results[0][1]
        
        # Merge results from different extractors
        merged_tools = self._merge_results(extractor_results, confidence_scores)
        return merged_tools
    
    def extract_single(self, text: str) -> Optional[ToolDefinition]:
        """
        Extract a single tool definition using multiple strategies.
        
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
        # Return the maximum confidence score from all extractors
        if not self.extractors:
            return 0.0
            
        return max(
            extractor.get_confidence_score(text) for extractor in self.extractors
        )
    
    def _merge_results(
        self, 
        extractor_results: List[tuple], 
        confidence_scores: Dict[BaseExtractor, float]
    ) -> List[ToolDefinition]:
        """
        Merge results from different extractors.
        
        Args:
            extractor_results: List of (extractor, tools) tuples
            confidence_scores: Dictionary mapping extractors to confidence scores
            
        Returns:
            List of merged tool definitions
        """
        # Group tools by name
        tools_by_name = {}
        
        for extractor, tools in extractor_results:
            confidence = confidence_scores[extractor]
            
            for tool in tools:
                # Normalize tool name for comparison
                normalized_name = self._normalize_name(tool.name)
                
                if normalized_name in tools_by_name:
                    # Merge with existing tool
                    existing_tool, existing_confidence = tools_by_name[normalized_name]
                    
                    if confidence > existing_confidence:
                        # Use the tool with higher confidence for base properties
                        merged_tool = self._merge_tools(tool, existing_tool)
                        tools_by_name[normalized_name] = (merged_tool, confidence)
                    else:
                        # Keep existing tool as base and merge in new properties
                        merged_tool = self._merge_tools(existing_tool, tool)
                        tools_by_name[normalized_name] = (merged_tool, existing_confidence)
                else:
                    # Add new tool
                    tools_by_name[normalized_name] = (tool, confidence)
        
        # Return just the tools without confidence scores
        return [tool for tool, _ in tools_by_name.values()]
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a tool name for comparison.
        
        Args:
            name: Tool name
            
        Returns:
            Normalized name
        """
        # Remove spaces, convert to lowercase
        return name.lower().replace(" ", "")
    
    def _merge_tools(
        self, 
        primary: ToolDefinition, 
        secondary: ToolDefinition
    ) -> ToolDefinition:
        """
        Merge two tool definitions, with primary taking precedence.
        
        Args:
            primary: Primary tool definition
            secondary: Secondary tool definition
            
        Returns:
            Merged tool definition
        """
        # Create new tool with primary properties
        merged = ToolDefinition(
            name=primary.name,
            description=primary.description
        )
        
        # Track parameters by name to avoid duplicates
        params_by_name = {}
        
        # Add primary parameters
        for param in primary.parameters:
            params_by_name[param.name] = param
        
        # Add/merge secondary parameters
        for param in secondary.parameters:
            if param.name in params_by_name:
                # Parameter exists in primary, merge descriptions if needed
                primary_param = params_by_name[param.name]
                
                # Use longer description
                if len(param.description) > len(primary_param.description):
                    primary_param.description = param.description
                    
                # If primary is string and secondary is more specific, use secondary type
                if primary_param.type == "string" and param.type != "string":
                    primary_param.type = param.type
            else:
                # Add new parameter from secondary
                params_by_name[param.name] = param
        
        # Add all merged parameters to the tool
        for param in params_by_name.values():
            merged.add_parameter(param)
        
        return merged


class ToolExtractor:
    """
    High-level tool extraction API.
    
    This class provides a simplified interface for tool extraction,
    using the most appropriate extraction strategy automatically.
    """
    
    def __init__(self):
        """Initialize the tool extractor."""
        self.extractor = CompositeExtractor()
    
    def extract_from_prompt(self, prompt: str) -> List[ToolDefinition]:
        """
        Extract tool definitions from a natural language prompt.
        
        Args:
            prompt: Natural language prompt
            
        Returns:
            List of extracted tool definitions
        """
        return self.extractor.extract(prompt)
    
    def extract_single_tool(self, prompt: str) -> Optional[ToolDefinition]:
        """
        Extract a single tool definition from a prompt.
        
        Args:
            prompt: Natural language prompt
            
        Returns:
            Extracted tool definition or None if not found
        """
        return self.extractor.extract_single(prompt)
    
    def identify_required_parameters(
        self,
        tool: ToolDefinition,
        prompt: str
    ) -> List[str]:
        """
        Identify which parameters are required based on the prompt.
        
        Args:
            tool: Tool definition with parameters
            prompt: Original prompt
            
        Returns:
            List of required parameter names
        """
        return self.extractor.identify_required_parameters(tool, prompt)


# Convenience function for the public API
def generate_tool_from_prompt(prompt: str) -> Optional[ToolDefinition]:
    """
    Generate a tool definition from a natural language prompt.
    
    Args:
        prompt: Natural language prompt
        
    Returns:
        Generated tool definition or None if extraction failed
    """
    extractor = ToolExtractor()
    return extractor.extract_single_tool(prompt)


def batch_extract_tools(prompts: List[str]) -> List[ToolDefinition]:
    """
    Extract tools from multiple prompts in batch.
    
    Args:
        prompts: List of natural language prompts
        
    Returns:
        List of extracted tool definitions
    """
    extractor = ToolExtractor()
    results = []
    
    # Use vectorization if enabled
    if config.enable_vectorization and config.use_cython:
        try:
            from zerepy_tools_core import vector_ops
            return vector_ops.batch_process(prompts, extractor.extract_from_prompt)
        except (ImportError, AttributeError):
            logger.debug("Vectorized batch processing not available")
    
    # Fall back to sequential processing
    for prompt in prompts:
        tools = extractor.extract_from_prompt(prompt)
        results.extend(tools)
    
    return results
