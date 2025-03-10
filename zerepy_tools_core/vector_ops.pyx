# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
Optimized vector operations for batch processing.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
import multiprocessing

import numpy as np
cimport numpy as np

logger = logging.getLogger(__name__)


cpdef list batch_process(list inputs, object process_func, int num_workers=0):
    """
    Process multiple inputs in batch with high performance.
    
    Args:
        inputs: List of input items
        process_func: Function to process each input
        num_workers: Number of worker processes (0 = sequential)
        
    Returns:
        List of results from processing each input
    """
    cdef list results = []
    
    if not inputs:
        return results
    
    # Use multiprocessing if requested and available
    if num_workers > 1:
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(process_func, inputs)
        except Exception as e:
            logger.error(f"Error in parallel batch processing: {str(e)}")
            # Fall back to sequential processing
            for item in inputs:
                try:
                    result = process_func(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
    else:
        # Sequential processing
        for item in inputs:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
    
    return results


cpdef list transform_batch(np.ndarray[double, ndim=2] data, object transform_func):
    """
    Apply a transformation function to each row of a data matrix.
    
    Args:
        data: 2D numpy array where each row is an input vector
        transform_func: Function to transform each row
        
    Returns:
        List of transformed results
    """
    cdef int i, n_rows
    cdef list results = []
    cdef np.ndarray[double, ndim=1] row
    
    if data is None or data.size == 0:
        return results
    
    n_rows = data.shape[0]
    for i in range(n_rows):
        row = data[i, :]
        try:
            result = transform_func(row)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in row transformation: {str(e)}")
            results.append(None)
    
    return results


cpdef dict batch_generate(list tools, str language, dict options):
    """
    Generate code for multiple tools in batch with high performance.
    
    Args:
        tools: List of tool definitions
        language: Programming language
        options: Additional options
        
    Returns:
        Dictionary mapping tool names to generated code
    """
    cdef dict results = {}
    cdef object tool
    
    if not tools:
        return results
    
    # Import code generator here to avoid circular imports
    from zerepy_tools.generators.code_generator import CodeGenerator
    generator = CodeGenerator()
    
    for tool in tools:
        try:
            code = generator.generate(tool, language, **options)
            results[tool.name] = code
        except Exception as e:
            logger.error(f"Error generating code for {tool.name}: {str(e)}")
            results[tool.name] = f"# Error: {str(e)}"
    
    return results


cpdef np.ndarray[double, ndim=2] compute_pairwise_similarity(np.ndarray[double, ndim=2] vectors):
    """
    Compute pairwise cosine similarity between vectors.
    
    Args:
        vectors: 2D numpy array where each row is a vector
        
    Returns:
        2D numpy array of pairwise similarities
    """
    cdef int n_vectors
    cdef np.ndarray[double, ndim=2] normalized
    cdef np.ndarray[double, ndim=2] similarity_matrix
    
    if vectors is None or vectors.size == 0:
        return np.array([], dtype=np.float64)
    
    try:
        n_vectors = vectors.shape[0]
        
        # Normalize the vectors (for cosine similarity)
        norms = np.linalg.norm(vectors, axis=1)
        norms = norms.reshape(-1, 1)  # Column vector for broadcasting
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        normalized = vectors / norms
        
        # Compute similarity matrix using dot product
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    except Exception as e:
        logger.error(f"Error computing similarity matrix: {str(e)}")
        return np.array([], dtype=np.float64)


cpdef list find_nearest_neighbors(np.ndarray[double, ndim=2] vectors, 
                                 np.ndarray[double, ndim=1] query_vector, 
                                 int k=5):
    """
    Find k nearest neighbors to a query vector.
    
    Args:
        vectors: 2D numpy array where each row is a vector
        query_vector: Query vector to find neighbors for
        k: Number of neighbors to return
        
    Returns:
        List of (index, similarity) tuples for top k neighbors
    """
    cdef int n_vectors
    cdef np.ndarray[double, ndim=1] similarities
    cdef np.ndarray[long, ndim=1] indices
    cdef list results = []
    cdef int i
    
    if vectors is None or vectors.size == 0 or query_vector is None:
        return []
    
    try:
        # Normalize the query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Normalize the data vectors
        n_vectors = vectors.shape[0]
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized = vectors / norms.reshape(-1, 1)
        
        # Compute cosine similarities
        similarities = np.dot(normalized, query_vector)
        
        # Get top k indices
        k = min(k, n_vectors)
        indices = np.argsort(-similarities)[:k]
        
        # Return (index, similarity) pairs
        for i in range(k):
            index = indices[i]
            similarity = similarities[index]
            results.append((int(index), float(similarity)))
        
        return results
    except Exception as e:
        logger.error(f"Error finding nearest neighbors: {str(e)}")
        return []
