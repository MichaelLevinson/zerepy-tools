# Zerepy Tools

High-performance tools add-on for Zerepy that leverages multi-language programming techniques to optimize speed and efficiency. It transforms natural language descriptions into functional tools through a combination of optimized Python, Cython for performance-critical paths, and appropriate use of vectorization.

## Features

- **High-Performance Extraction**: Extract tool definitions from natural language prompts with optimized pattern matching
- **Code Generation**: Generate executable code implementations in Python, JavaScript, and TypeScript
- **Vectorized Processing**: Process multiple prompts/tools simultaneously for improved throughput
- **Parameter Inference**: Automatically infer parameter types, descriptions, and requirements
- **Security Validation**: Validate generated code for syntax errors and security vulnerabilities
- **Prompt Assistant**: Get help writing better tool prompts with suggestions and improvements

## Installation

```bash
# Basic installation
pip install zerepy-tools

# With Cython optimizations (recommended)
pip install zerepy-tools[cython]

# With LLM integration for enhanced capabilities
pip install zerepy-tools[llm]

# Full installation with all features
pip install zerepy-tools[all]
```

## Quick Start

```python
from zerepy_tools import create_tool, create_implementation

# Extract a tool definition from a natural language prompt
prompt = """
Create a weather tool that accepts a location parameter for the city
and an optional units parameter that defaults to metric.
"""

# Extract the tool definition
tool = create_tool(prompt)
print(f"Extracted tool: {tool.name}")
print(f"Parameters: {[p.name for p in tool.parameters]}")

# Generate a Python implementation
python_code = create_implementation(tool, language="python")
print(python_code)

# Generate a JavaScript implementation
js_code = create_implementation(tool, language="javascript")
print(js_code)
```

## High-Level API

```python
# Generate a tool definition
from zerepy_tools import generate_tool_from_prompt

tool = generate_tool_from_prompt("Create a calculator tool that adds two numbers")

# Generate code in different languages
from zerepy_tools import generate_code

python_code = generate_code(tool, language="python")
js_code = generate_code(tool, language="javascript")
ts_code = generate_code(tool, language="typescript")

# Process multiple prompts in batch
from zerepy_tools import batch_extract_tools

prompts = [
    "Create a weather tool that takes a location parameter",
    "Create a translation tool that translates text from one language to another",
    "Create a calculator tool that performs basic arithmetic operations"
]

tools = batch_extract_tools(prompts)

# Generate code for multiple tools in batch
from zerepy_tools import batch_generate_code

code_map = batch_generate_code(tools, language="python")
```

## Configuration

```python
from zerepy_tools import config

# Use Cython implementations (default: True if available)
config.use_cython = True

# Enable vectorization for batch processing
config.enable_vectorization = True

# Set number of worker threads for parallelization
config.worker_threads = 4

# Set default LLM provider
config.default_llm_provider = "openai"

# Configure LLM settings
config.set_llm_config("openai", {
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 1000,
})
```

## Prompt Assistant

```python
from zerepy_tools import PromptAssistant

assistant = PromptAssistant()

# Improve a tool prompt
improved_prompt = assistant.improve_prompt(
    "Create a tool that searches for news articles"
)

# Analyze a prompt for quality
analysis = assistant.analyze_prompt(
    "Create a weather tool that takes a location"
)
print(f"Prompt score: {analysis['score']}/10")
print(f"Strengths: {analysis['strengths']}")
print(f"Suggestions: {analysis['suggestions']}")

# Suggest parameters for a tool
parameters = assistant.suggest_parameters(
    "Create a tool that searches for images based on keywords"
)
for param in parameters:
    print(f"- {param['name']} ({param['type']}): {param['description']}")
```

## Advanced Usage

### Creating Custom Extractors

```python
from zerepy_tools.extractors.base_extractor import BaseExtractor
from zerepy_tools.core.models import ToolDefinition

class MyCustomExtractor(BaseExtractor):
    def extract(self, text: str) -> List[ToolDefinition]:
        # Custom extraction logic
        # ...
        return tools
    
    def extract_single(self, text: str) -> Optional[ToolDefinition]:
        # Custom extraction logic for a single tool
        # ...
        return tool
        
    def get_confidence_score(self, text: str) -> float:
        # Custom confidence scoring
        # ...
        return score
```

### Creating Custom Code Generators

```python
from zerepy_tools.generators.base_generator import BaseGenerator
from zerepy_tools.core.models import ToolDefinition

class MyCustomGenerator(BaseGenerator):
    def generate(self, tool: ToolDefinition, language: str = "python", **options) -> str:
        # Custom code generation logic
        # ...
        return code
        
    def get_template(self, language: str) -> str:
        # Custom template for the language
        # ...
        return template
```

## Performance Benchmarks

| Operation | Python Implementation | Cython Implementation | Speedup |
|-----------|----------------------|----------------------|---------|
| Pattern Matching | 1.00x | 10.50x | 10.50x |
| Parameter Extraction | 1.00x | 8.75x | 8.75x |
| Template Rendering | 1.00x | 6.20x | 6.20x |
| Code Generation | 1.00x | 4.30x | 4.30x |
| Batch Processing (100 items) | 1.00x | 25.00x | 25.00x |

## License

MIT License
