from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel
import os

from openai import AsyncOpenAI

import time

from .config import OpenAIConfig
import re

bias = '''```json
{
  '''
    
def convert_from_str(content):

    pattern = r'\{(.*)\}'
    if not isinstance(content, str):
        return content

    try:
        x = re.search(pattern, content, re.DOTALL).group().strip()
        return eval(x) 
    
    except SyntaxError:
        try:
            content = content.replace(bias, "").replace("```", "").replace("json", "").strip()
            if content[-1] != "}":
                content += "}"
                
            x = re.search(pattern, content, re.DOTALL).group().strip()
            return eval(x) 
        except:
            
            return content


load_dotenv("/app/setting.env")

class OpenAIModel(object):
    
    def __init__(self, config: OpenAIConfig = OpenAIConfig()) -> None:
        """
        Just initialize everything including set attributes and start client
        """
        self.config = config.model_dump()  
        
        self.model_name = self.config["model"]
        
        self.base_url = self.config.pop("base_url")
        
        self.api_key = self.config.pop("api_key")
        
        self._create_client()
        
    def _create_client(self) -> None:
        """
        Create the client
        """
        client = AsyncOpenAI(base_url = self.base_url, api_key=self.api_key)
        self.client = client
        
    async def complete(self, messages: List[Dict], structure: BaseModel, system: str = "") -> Dict:
        """
        Complete chat message. For openai api, system prompt are formatted in messages and structure is in the form of pydantic
        """
        start_time = time.time()
        
        if "typhoon" in self.model_name:
            messages += [{"role": "assistant", "content": bias}]
            response = await self.client.beta.chat.completions.parse(
                messages = messages,
                **self.config
            )
            
            response.choices[0].message.content = bias + response.choices[0].message.content
            
        elif structure is not None:
            response = await self.client.beta.chat.completions.parse(
                messages=messages,
                response_format = structure,
                **self.config
                )
        else:
            response = await self.client.beta.chat.completions.parse(
                messages=messages,
                **self.config
                )
        
        time_taken = time.time()-start_time
        
        if response.choices[0].message.parsed is not None:
            content = response.choices[0].message.parsed
        else:
            content = convert_from_str(response.choices[0].message.content)
            
        
        if isinstance(content, str):
            content = content
        elif not isinstance(content, dict):
            content = content.model_dump()   
            
        return {"content": content, "usage": response.usage.model_dump(), "llm_time": time_taken}
    
    async def multi_complete(self, messages: List[List[Dict]], structures: List[BaseModel]) -> List[Dict]:
        """
        For calling async function
        """
        
        jobs = [self.complete(message, structure) for message, structure in zip(messages, structures)]
        
        return await asyncio.gather(*jobs)