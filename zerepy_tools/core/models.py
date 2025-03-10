# zerepy_tools/core/models.py
"""
Core data models for Zerepy Tools.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class ParameterType(str, Enum):
    """Enum representing parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    FILE = "file"


class Parameter(BaseModel):
    """Model representing a tool parameter."""
    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class ToolDefinition(BaseModel):
    """Model representing a tool definition."""
    name: str
    description: str
    parameters: List[Parameter] = Field(default_factory=list)
    
    def add_parameter(self, parameter: Union[Parameter, Dict[str, Any]]) -> None:
        """
        Add a parameter to the tool definition.
        
        Args:
            parameter: Parameter object or dictionary
        """
        if isinstance(parameter, dict):
            parameter = Parameter(**parameter)
        self.parameters.append(parameter)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool definition to a dictionary format.
        
        Returns:
            Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **({"default": param.default} if param.default is not None else {})
                    } for param in self.parameters
                },
                "required": [param.name for param in self.parameters if param.required]
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        """
        Create a ToolDefinition from a dictionary.
        
        Args:
            data: Dictionary with tool definition
            
        Returns:
            ToolDefinition object
        """
        name = data["name"]
        description = data["description"]
        
        # Create ToolDefinition
        tool = cls(name=name, description=description)
        
        # Extract parameters
        if "parameters" in data and "properties" in data["parameters"]:
            properties = data["parameters"]["properties"]
            required = data["parameters"].get("required", [])
            
            for param_name, param_data in properties.items():
                tool.add_parameter({
                    "name": param_name,
                    "type": param_data.get("type", ParameterType.STRING),
                    "description": param_data.get("description", ""),
                    "required": param_name in required,
                    "default": param_data.get("default", None)
                })
        
        return tool
