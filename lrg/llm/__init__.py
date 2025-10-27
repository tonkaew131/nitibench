from typing import Dict
from .collections.claude import ClaudeConfig, ClaudeModel
from .collections.gemini import GeminiConfig, GeminiModel
from .collections.openai import OpenAIConfig, OpenAIModel

MAP_MODEL = {
    "gpt": (OpenAIConfig, OpenAIModel),
    "claude": (ClaudeConfig, ClaudeModel),
    "gemini": (GeminiConfig, GeminiModel),
    "o1": (OpenAIConfig, OpenAIModel),
    "aisingapore/gemma2": (OpenAIConfig, OpenAIModel),
    "typhoon": (OpenAIConfig, OpenAIModel),
}


def init_llm(config: Dict):

    model_type = config["type"]

    assert model_type in MAP_MODEL, "Unrecognize model model_type: {}".format(model_type)
    config_class, model_class = MAP_MODEL[model_type]

    model_config = config_class(**config, inference_type=model_type)

    _dump = model_config.model_dump()
    if isinstance(_dump, dict) and "api_key" in _dump:
        _val = _dump["api_key"]
        if isinstance(_val, str) and len(_val) > 12:
            _dump["api_key"] = f"{_val[:8]}...{_val[-4:]}"
        elif _val:
            _dump["api_key"] = "*****"
        else:
            _dump["api_key"] = _val
    print(_dump)

    model = model_class(config=model_config)

    return model
