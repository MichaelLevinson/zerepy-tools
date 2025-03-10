# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized pattern matching engine implementation in Cython.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
cimport numpy as np

logger = logging.getLogger(__name__)


cpdef tuple match(str pattern, str text, bint case_sensitive=False):
    """
    Match a pattern against text with high performance.
    
    Args:
        pattern: Regex pattern
        text: Text to match against
        case_sensitive: Whether the match is case-sensitive
        
    Returns:
        Tuple of (match_found, match_groups)
    """
    cdef int flags = 0 if case_sensitive else re.IGNORECASE
    cdef object match_obj
    
    try:
        match_obj = re.search(pattern, text, flags)
        if match_obj:
            return True, match_obj.groups()
        else:
            return False, ()
    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}")
        return False, ()


cpdef list extract_patterns(str text, list patterns, bint case_sensitive=False):
    """
    Extract all pattern matches from text with high performance.
    
    Args:
        text: Text to extract patterns from
        patterns: List of regex patterns
        case_sensitive: Whether matches are case-sensitive
        
    Returns:
        List of dictionaries with match information
    """
    cdef list results = []
    cdef int flags = 0 if case_sensitive else re.IGNORECASE
    cdef str pattern
    cdef object match_obj
    
    if not text or not patterns:
        return results
    
    for pattern in patterns:
        try:
            for match_obj in re.finditer(pattern, text, flags):
                results.append({
                    'pattern': pattern,
                    'match_text': match_obj.group(0),
                    'groups': match_obj.groups(),
                    'start': match_obj.start(),
                    'end': match_obj.end(),
                })
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
    
    # Sort by position in text
    results.sort(key=lambda x: x['start'])
    return results


cpdef list batch_process(list texts, object process_func):
    """
    Process multiple texts in batch with high performance.
    
    Args:
        texts: List of text strings
        process_func: Function to process each text
        
    Returns:
        List of results from processing each text
    """
    cdef list results = []
    cdef str text
    cdef object result
    
    if not texts:
        return results
    
    for text in texts:
        try:
            result = process_func(text)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
    
    return results


cpdef double vector_similarity(list vec1, list vec2) except -1.0:
    """
    Calculate cosine similarity between two vectors with high performance.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity between the vectors
    """
    cdef np.ndarray arr1, arr2
    cdef double dot_product, norm1, norm2
    
    try:
        arr1 = np.array(vec1, dtype=np.float64)
        arr2 = np.array(vec2, dtype=np.float64)
        
        if arr1.shape[0] != arr2.shape[0]:
            logger.error("Vector dimensions do not match")
            return -1.0
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logger.error(f"Error calculating vector similarity: {str(e)}")
        return -1.0


cpdef list find_best_matches(list patterns, str text, int max_results=5):
    """
    Find the best matching patterns in text.
    
    Args:
        patterns: List of regex patterns
        text: Text to search in
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries with match information, sorted by quality
    """
    cdef list matches = extract_patterns(text, patterns)
    cdef list scored_matches = []
    
    # Score matches based on position, length, etc.
    for match in matches:
        score = 1.0
        
        # Earlier matches are better
        score -= match['start'] / len(text) * 0.5
        
        # Longer matches are better
        match_length = match['end'] - match['start']
        score += min(match_length / 50.0, 0.5)  # Cap at 0.5
        
        # Add to scored matches
        scored_matches.append((score, match))
    
    # Sort by score (descending) and return top matches
    scored_matches.sort(reverse=True, key=lambda x: x[0])
    return [match for score, match in scored_matches[:max_results]]


cpdef dict extract_tool_components(str text):
    """
    Extract tool components (name, description, parameters) from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        Dictionary with extracted components
    """
    cdef dict result = {
        'name': '',
        'description': '',
        'parameters': []
    }
    
    # Define patterns for tool components
    cdef list name_patterns = [
        r"(?:create|implement|build)\s+(?:a|an)\s+([a-zA-Z0-9_\s]+?)\s+tool",
        r"tool\s+name:\s*([a-zA-Z0-9_\s]+)",
        r"name:\s*([a-zA-Z0-9_\s]+)",
    ]
    
    cdef list desc_patterns = [
        r"description:\s*(.*?)(?:\n|$)",
        r"(?:tool|function)\s+description:\s*(.*?)(?:\n|$)",
        r"(?:that|which)\s+(.*?)(?:\.|$)",
    ]
    
    cdef list param_patterns = [
        r"parameter\s+name:\s*([a-zA-Z0-9_]+)",
        r"(?:parameter|argument|input):\s*([a-zA-Z0-9_]+)",
        r"(?:accepts?|takes?|requires?)\s+(?:a|an)?\s*([a-zA-Z0-9_]+)\s+parameter",
    ]
    
    # Extract name
    cdef list name_matches = []
    for pattern in name_patterns:
        found, groups = match(pattern, text)
        if found and groups:
            name_matches.append(groups[0])
    
    if name_matches:
        # Clean up the name
        name = name_matches[0].strip()
        name = re.sub(r"\s+", " ", name)
        # Convert to CamelCase
        result['name'] = "".join(word.capitalize() for word in name.split())
    
    # Extract description
    cdef list desc_matches = []
    for pattern in desc_patterns:
        found, groups = match(pattern, text)
        if found and groups:
            desc_matches.append(groups[0])
    
    if desc_matches:
        result['description'] = desc_matches[0].strip()
    
    # Extract parameters
    cdef list param_matches = []
    for pattern in param_patterns:
        for match_obj in re.finditer(pattern, text, re.IGNORECASE):
            if match_obj.groups():
                param_matches.append(match_obj.group(1))
    
    # Add unique parameters
    cdef set unique_params = set()
    for param in param_matches:
        if param not in unique_params:
            unique_params.add(param)
            result['parameters'].append({'name': param})
    
    return result
