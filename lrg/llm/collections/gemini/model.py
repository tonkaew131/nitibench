import google.generativeai as genai
from typing import Dict, Any
from .config import GeminiConfig

class GoogleGeminiModel:
    def __init__(self,
                 model: str,
                 system_prompt: str,
                 temperature: float,
                 max_output_tokens: int,
                 candidate_count: int) -> None:
        
        self.tools = [] # only text generation for now
        self.model = model
        self.system_prompt = system_prompt
        self.generation_config = GeminiConfig(temperature=temperature,
                                              max_output_tokens=max_output_tokens,
                                              candidate_count=candidate_count).model_dump()
        
        # hard code safety level
        self.safety_level: Dict[str, str] = {
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"
            }
        
        self._create_client()
    
    def _create_client(self) -> None:
        """Creates the Google Gemini client."""
        self._client = genai.GenerativeModel(
                            model_name=self.model,
                            safety_settings=self.safety_level,
                            generation_config=self.generation_config,
                            system_instruction=self.system_prompt,
                            tools=self.tools
                        )
        
    def complete(self,
                 query: str) -> Dict[str, Any]:
        
        return self._client.generate_content(query)