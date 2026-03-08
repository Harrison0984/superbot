"""Embedding 缓存管理器 - 避免重复计算"""
from typing import Dict, Optional
import hashlib
import numpy as np


class EmbeddingCache:
    """Embedding 结果缓存"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}

    def _hash(self, text: str) -> str:
        """生成文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """获取缓存的向量"""
        key = self._hash(text)
        return self._cache.get(key)

    def set(self, text: str, vector: np.ndarray) -> None:
        """设置缓存"""
        # 简单的 LRU: 超过上限时清空
        if len(self._cache) >= self.max_size:
            # 删除最旧的 20%
            keys_to_remove = list(self._cache.keys())[:self.max_size // 5]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._hash(text)
        self._cache[key] = vector

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)
