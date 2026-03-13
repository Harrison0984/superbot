"""统一缓存模块：通用的数量/大小限制缓冲区"""
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger


class CacheBuffer:
    """
    统一缓存：支持数量和大小两个维度触发
    """

    def __init__(
        self,
        buffer_count: int = 10,
        buffer_size: int = 4096,
    ):
        self.buffer_count = buffer_count
        self.buffer_size = buffer_size

        self.buffer: List[Dict[str, Any]] = []
        self._total_bytes: int = 0

    def push(self, text: str, source: str = "USER") -> bool:
        """
        添加内容到缓冲区

        参数:
            text: 文本内容
            source: 来源标识，可选值为 "USER" 或 "ASSISTANT"，默认为 "USER"

        返回:
            True 表示添加成功
        """
        text_bytes = len(text.encode('utf-8'))

        # 先添加新内容
        self.buffer.append({
            "text": text,
            "text_bytes": text_bytes,
            "timestamp": datetime.now(),
            "source": source
        })
        self._total_bytes += text_bytes
        logger.debug("[CacheBuffer] pushed: buffer_size={}/{}, total_bytes={}", len(self.buffer), self.buffer_count, self._total_bytes)

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
        """获取所有 buffer 内容，不清空"""
        return self.buffer.copy()

    def get_batch_texts(self) -> List[str]:
        """获取批次的文本列表"""
        return [item["text"] for item in self.buffer]

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


class FIFOBuffer(CacheBuffer):
    """
    先进先出缓冲区：当超过 buffer_count 或 buffer_size 阈值时，移除最旧的内容
    """

    def push(self, text: str, source: str = "USER") -> bool:
        """
        添加内容到缓冲区（FIFO 淘汰）
        """
        text_bytes = len(text.encode('utf-8'))

        # 先添加新内容
        self.buffer.append({
            "text": text,
            "text_bytes": text_bytes,
            "timestamp": datetime.now(),
            "source": source
        })
        self._total_bytes += text_bytes
        logger.debug("[FIFOBuffer] pushed: buffer_size={}/{}, total_bytes={}", len(self.buffer), self.buffer_count, self._total_bytes)

        # 如果超过阈值，FIFO 淘汰最旧的内容
        while (len(self.buffer) > self.buffer_count or
               self._total_bytes > self.buffer_size) and self.buffer:
            removed = self.buffer.pop(0)  # 移除最旧的
            self._total_bytes -= removed["text_bytes"]
            logger.debug("[FIFOBuffer] evicted oldest item, buffer_size={}", len(self.buffer))

        return True
