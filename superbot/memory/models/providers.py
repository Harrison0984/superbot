"""内置 Provider 实现"""
from typing import List, Dict, Any, Optional
import numpy as np
from .protocols import LLMProvider, EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer Embedding Provider"""

    def __init__(self, model):
        """
        初始化

        参数:
            model: sentence-transformers 模型实例
        """
        self._model = model

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text)

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self._model.encode(texts)

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class LocalMLXProvider(LLMProvider):
    """本地 MLX LLM Provider (Qwen3)"""

    def __init__(self, model, tokenizer, default_max_tokens: int = 256, default_temperature: float = 0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    def generate(self, prompt: str, **kwargs) -> str:
        import mlx_lm
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        temperature = kwargs.get("temperature", self.default_temperature)
        return mlx_lm.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def extract_triples(self, text: str, max_tokens: int = 256) -> List[Dict[str, Any]]:
        prompt = f"""<|im_start|>system
你是一个关系抽取专家。从文本中提取三元组并评估置信度。

## 任务
提取 (主语, 关系, 宾语, 置信度) 格式的三元组

## 置信度规则
- 0.9-1.0: 明确事实
- 0.6-0.8: 推断关系
- 0.3-0.5: 模糊关系

## 输出格式 (JSON)
[
    {{"subject": "主语", "relation": "关系", "object": "宾语", "confidence": 0.95}}
]
<|im_end|>
<|im_start|>user
文本: "{text}"

## 三元组
<|im_end|>
<|im_start|>assistant
"""
        response = self.generate(prompt, max_tokens=max_tokens, temperature=0.1)
        return self._parse_json(response)

    def compress(self, text: str, max_tokens: int = 128) -> str:
        prompt = f"""<|im_start|>system
压缩成长度不超过100字的简洁摘要，保留关键信息。
<|im_end|>
<|im_start|>user
{text}

摘要:
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.3)

    def understand_context(
        self,
        query: str,
        memory: List[str],
        history: List[str],
        max_tokens: int = 128
    ) -> str:
        prompt = f"""<|im_start|>system
你是一个上下文理解专家。根据对话历史和上下文，推断用户当前消息的含义。

## 任务
1. 如果用户使用了代词(它/这个/那/也/同样等)，根据上下文推断指代内容
2. 结合相关记忆理解用户意图
3. 直接给出理解结果，一句话即可
<|im_end|>
<|im_start|>user
## 相关记忆
{memory[:3] if memory else '无'}

## 对话历史
{history[-3:] if history else '无'}

## 当前消息
"{query}"

## 理解
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.3)

    def _parse_json(self, response: str) -> List[Dict[str, Any]]:
        import json
        import re
        try:
            return json.loads(response)
        except:
            match = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', response)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return []
