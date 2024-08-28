from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel, Field

from .const import DEFAULT_TEMPERATURE

class BaseLLM(BaseModel, ABC):
    temperature: int = Field(default=DEFAULT_TEMPERATURE,
                             description="Temperature for LLM generation.")
    
    @abstractmethod
    def complete(self):
        pass