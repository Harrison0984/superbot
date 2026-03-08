"""统一缓存模块：通用的数量/大小限制缓冲区"""
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional

import numpy as np


class CacheBuffer:
    """
    统一缓存：支持数量和大小两个维度触发

    通过传入不同的处理函数来实现不同功能：
    - 语义去重：传入相似度检测函数
    - 批量处理：传入 LLM 处理函数
    """

    def __init__(
        self,
        buffer_count: int = 3,
        buffer_size: int = 2048,
        # 可选：用于语义去重的相似度检测
        similarity_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        similarity_threshold: float = 0.95
    ):
        self.buffer_count = buffer_count
        self.buffer_size = buffer_size
        self.similarity_func = similarity_func
        self.similarity_threshold = similarity_threshold

        self.buffer: List[Dict[str, Any]] = []
        self._total_bytes: int = 0

    def push(self, text: str, vector: np.ndarray) -> bool:
        """
        添加内容到缓冲区

        返回:
            True 表示添加成功
            False 表示被去重过滤（当设置了 similarity_func 时）
        """
        # 如果设置了相似度检测，先检查是否重复
        if self.similarity_func is not None and self.buffer:
            for existing in self.buffer:
                similarity = self.similarity_func(vector, existing["vector"])
                if similarity >= self.similarity_threshold:
                    return False  # 语义重复，拒绝添加

        text_bytes = len(text.encode('utf-8'))
        self.buffer.append({
            "text": text,
            "vector": vector,
            "text_bytes": text_bytes,
            "timestamp": datetime.now()
        })
        self._total_bytes += text_bytes
        return True

    def should_process(self) -> bool:
        """
        判断是否应该处理

        任一条件满足即触发：
        - 数量 >= buffer_count
        - 总大小 >= buffer_size
        """
        if len(self.buffer) >= self.buffer_count:
            return True
        if self._total_bytes >= self.buffer_size:
            return True
        return False

    def get_batch(self) -> List[Dict[str, Any]]:
        """获取批次并清空缓冲区"""
        batch = self.buffer.copy()
        self.clear()
        return batch

    def get_batch_texts(self) -> List[str]:
        """获取批次的文本列表"""
        texts = [item["text"] for item in self.buffer]
        self.clear()
        return texts

    def get_batch_vectors(self) -> List[np.ndarray]:
        """获取批次的向量列表"""
        vectors = [item["vector"] for item in self.buffer]
        self.clear()
        return vectors

    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self._total_bytes = 0

    def size(self) -> int:
        """返回当前缓冲区条目数"""
        return len(self.buffer)

    def total_bytes(self) -> int:
        """返回当前缓冲区总字节数"""
        return self._total_bytes

    def status(self) -> Dict[str, Any]:
        """返回缓冲区状态"""
        return {
            "count": len(self.buffer),
            "count_limit": self.buffer_count,
            "bytes": self._total_bytes,
            "size_limit": self.buffer_size,
            "should_process": self.should_process()
        }
