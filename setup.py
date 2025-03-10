# setup.py
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Check if Cython is available
try:
    from Cython.Distutils import build_ext
    have_cython = True
except ImportError:
    have_cython = False

# Define Cython extensions
ext_modules = []
if have_cython:
    extensions = [
        Extension(
            "zerepy_tools_core.pattern_engine",
            ["zerepy_tools_core/pattern_engine.pyx"],
            include_dirs=[np.get_include()]
        ),
        Extension(
            "zerepy_tools_core.vector_ops",
            ["zerepy_tools_core/vector_ops.pyx"],
            include_dirs=[np.get_include()]
        ),
        Extension(
            "zerepy_tools_core.validators",
            ["zerepy_tools_core/validators.pyx"],
        ),
        Extension(
            "zerepy_tools_core.sandbox",
            ["zerepy_tools_core/sandbox.pyx"],
        )
    ]
    ext_modules = cythonize(extensions)

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="zerepy-tools",
    version="0.1.0",
    description="High-performance tools add-on for Zerepy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Levinson",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    cmdclass={'build_ext': build_ext} if have_cython else {},
)

# pyproject.toml
"""
[build-system]
requires = ["setuptools>=42", "wheel", "Cython>=0.29.21", "numpy>=1.19.0"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py37', 'py38', 'py39', 'py310']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.7"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.cython]
compiler_directives = {language_level = "3", boundscheck = false, wraparound = false, initializedcheck = false, cdivision = true}
"""

# requirements.txt
"""
# Core dependencies
pydantic>=1.8.0
numpy>=1.19.0
zerepy>=0.1.0

# Build dependencies
cython>=0.29.21
setuptools>=42.0.0
wheel>=0.35.0

# Optional dependencies
openai>=0.27.0
transformers>=4.5.0
"""

# Directory Structure
"""
zerepy-tools/
├── README.md
├── setup.py
├── pyproject.toml               
├── requirements.txt
├── zerepy_tools/                
│   ├── __init__.py              
│   ├── core/                    
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── config.py
│   │   └── constants.py
│   ├── extractors/              
│   │   ├── __init__.py
│   │   ├── base_extractor.py    
│   │   ├── pattern_extractor.py 
│   │   ├── llm_extractor.py     
│   │   └── composite_extractor.py
│   ├── generators/             
│   │   ├── __init__.py
│   │   ├── base_generator.py
│   │   ├── code_generator.py    
│   │   └── template_engine.py  
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── syntax_validator.py
│   │   └── security_validator.py
│   └── utils/
│       ├── __init__.py
│       ├── string_utils.py
│       └── profiling.py
└── zerepy_tools_core/          
    ├── __init__.py
    ├── pattern_engine.pyx       
    ├── vector_ops.pyx           
    ├── validators.pyx           
    └── sandbox.pyx
"""
