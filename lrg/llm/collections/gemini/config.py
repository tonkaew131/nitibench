from pydantic import BaseModel, Field

class GeminiConfig(BaseModel):
    temperature: float = Field(default=float,
                       description="LLM temperature")
    
    max_output_tokens: int = Field(default=int,
                                     description="Max output tokens from LLM")
    
    candidate_count: int = Field(default=int,
                                 description="Candidate output from Gemini, recommended as 1")
    
    