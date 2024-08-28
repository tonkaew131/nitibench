from pydantic import BaseModel, Field
from typing import Callable, List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, meta

from .loader import prompt_loader
from .const import DEFAULT_SYSTEM_PROMPT_TEMPLATE_NAME

class PromptManager:
    def __init__(self,
                 system_prompt: Optional[str] = None) -> None:
        self.prompt_loader: Callable = prompt_loader
        self.system_prompt: str = system_prompt or self.get_system_prompt()
        self.available_templates: List[str] = self.list_available_templates()
    
    def get_system_prompt(self) -> str:
        """Return system prompt as string."""
        return self.prompt_loader.get_template(DEFAULT_SYSTEM_PROMPT_TEMPLATE_NAME).render()
    
    def list_available_templates(self) -> List[str]:
        """Get list of avaliable templates from package.""" 
        return self.prompt_loader.list_templates()
    
    def render(self,
               template_name: str,
               render_config: Dict[str, Any]) -> str:
        """
        template_name: tempalte name with .md
        render_config: dict which contains information for prompt rendering
        """
        if template_name not in self.available_templates:
            raise ValueError(f"Template name: {template_name} is not avaiable. Please check available prompt templates using list_available_templates()")
        
        return prompt_loader.get_template(template_name).render(render_config)
        
    def _check_render_config(self,
                             template: str,
                             render_config: dict) -> None:
        """Check whether the render_config contains all required variables for prompt rendering or not."""
        raise NotImplementedError("Waiting for best practice.")
    
        # with open(template.filename, "r") as reader:
        #     template_str = reader.read()
            
        # env = Environment()
        # parsed_content = env.parse(template_str)
        # variables = meta.find_undeclared_variables(parsed_content)
        
        # missing_variables = variables - set(render_config.keys())
        # if missing_variables:
        #     raise ValueError(f"Missing required variables: {', '.join(missing_variables)}")