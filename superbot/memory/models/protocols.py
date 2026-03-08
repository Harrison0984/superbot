"""模型层协议定义"""
from typing import Protocol, List, Dict, Any
import numpy as np


class LLMProvider(Protocol):
    """LLM 提供商抽象"""

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        ...

    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """提取三元组"""
        ...

    def compress(self, text: str) -> str:
        """压缩文本"""
        ...

    def understand_context(
        self,
        query: str,
        memory: List[str],
        history: List[str]
    ) -> str:
        """上下文理解"""
        ...


class EmbeddingProvider(Protocol):
    """Embedding 提供商抽象"""

    def encode(self, text: str) -> np.ndarray:
        """编码单个文本"""
        ...

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量编码"""
        ...

    def dimension(self) -> int:
        """返回向量维度"""
        ...
