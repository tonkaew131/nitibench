from openai import OpenAI
from typing import Dict, Any, Optional


class OpenaiModel:
    def __init__(self,
                 model: str,
                 system_prompt: str,
                 temperature: float,
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.seed = seed
        
        self._create_client()
    
    def _create_client(self) -> None:
        self.client = OpenAI()
        
    def complete(self,
                 query: str) -> Dict[str, Any]:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
                ],
            temperature=self.temperature,
            seed=self.seed
            )