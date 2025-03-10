# zerepy_tools/core/config.py
"""
Configuration management for Zerepy Tools.
"""

import os
import platform
import multiprocessing
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class for Zerepy Tools.
    
    This class manages configuration options for optimizations,
    performance settings, and feature toggles.
    """
    
    def __init__(self):
        """Initialize with default configuration."""
        # Detect if Cython modules are available
        self._cython_available = self._check_cython_available()
        self._rust_available = self._check_rust_available()
        
        # Core settings
        self._use_cython = self._cython_available
        self._use_rust = self._rust_available
        self._enable_vectorization = True
        self._worker_threads = min(multiprocessing.cpu_count(), 4)
        
        # LLM settings
        self._default_llm_provider = "openai"
        self._llm_config: Dict[str, Any] = {
            "openai": {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 1000,
            }
        }
        
        # Load from environment variables
        self._load_from_env()
    
    def _check_cython_available(self) -> bool:
        """Check if Cython optimized modules are available."""
        try:
            from zerepy_tools_core import pattern_engine  # noqa
            return True
        except ImportError:
            return False
    
    def _check_rust_available(self) -> bool:
        """Check if Rust extensions are available."""
        try:
            import zerepy_tools_rust  # noqa
            return True
        except ImportError:
            return False
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Performance settings
        if "ZEREPY_USE_CYTHON" in os.environ:
            self._use_cython = os.environ["ZEREPY_USE_CYTHON"].lower() == "true"
        
        if "ZEREPY_USE_RUST" in os.environ:
            self._use_rust = os.environ["ZEREPY_USE_RUST"].lower() == "true"
        
        if "ZEREPY_ENABLE_VECTORIZATION" in os.environ:
            self._enable_vectorization = (
                os.environ["ZEREPY_ENABLE_VECTORIZATION"].lower() == "true"
            )
        
        if "ZEREPY_WORKER_THREADS" in os.environ:
            try:
                self._worker_threads = int(os.environ["ZEREPY_WORKER_THREADS"])
            except ValueError:
                pass
        
        # LLM settings
        if "ZEREPY_DEFAULT_LLM" in os.environ:
            self._default_llm_provider = os.environ["ZEREPY_DEFAULT_LLM"]
    
    @property
    def use_cython(self) -> bool:
        """Whether to use Cython implementations."""
        return self._use_cython and self._cython_available
    
    @use_cython.setter
    def use_cython(self, value: bool) -> None:
        """Set whether to use Cython implementations."""
        self._use_cython = value
    
    @property
    def use_rust(self) -> bool:
        """Whether to use Rust implementations."""
        return self._use_rust and self._rust_available
    
    @use_rust.setter
    def use_rust(self, value: bool) -> None:
        """Set whether to use Rust implementations."""
        self._use_rust = value
    
    @property
    def enable_vectorization(self) -> bool:
        """Whether to enable vectorization for batch processing."""
        return self._enable_vectorization
    
    @enable_vectorization.setter
    def enable_vectorization(self, value: bool) -> None:
        """Set whether to enable vectorization."""
        self._enable_vectorization = value
    
    @property
    def worker_threads(self) -> int:
        """Number of worker threads for parallelization."""
        return self._worker_threads
    
    @worker_threads.setter
    def worker_threads(self, value: int) -> None:
        """Set number of worker threads."""
        self._worker_threads = max(1, value)
    
    @property
    def default_llm_provider(self) -> str:
        """The default LLM provider to use."""
        return self._default_llm_provider
    
    @default_llm_provider.setter
    def default_llm_provider(self, value: str) -> None:
        """Set the default LLM provider."""
        self._default_llm_provider = value
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific LLM provider.
        
        Args:
            provider: Provider name (default: default_llm_provider)
            
        Returns:
            Configuration dictionary for the provider
        """
        if provider is None:
            provider = self._default_llm_provider
        
        return self._llm_config.get(provider, {})
    
    def set_llm_config(self, provider: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a specific LLM provider.
        
        Args:
            provider: Provider name
            config: Configuration dictionary
        """
        self._llm_config[provider] = config


# Singleton instance
config = Config()
