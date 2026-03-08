"""物理过滤模块：基于压缩率计算信息增量密度"""
from typing import Optional

import zstandard as zstd


class EntropyGatekeeper:
    """物理过滤门卫：基于压缩率计算信息增量密度"""

    def __init__(
        self,
        threshold: float = 0.5,
        buffer_count: int = 100,
        buffer_size: int = 64 * 1024,  # 64KB
        min_text_length: int = 20
    ):
        self.threshold = threshold
        self.buffer_count = buffer_count
        self.buffer_size = buffer_size
        self.min_text_length = min_text_length

        # 使用列表而不是字节串，便于管理
        self.window_items: list[str] = []
        self._total_bytes: int = 0

        # 创建压缩器 (level 3 平衡速度和压缩率)
        self.cctx = zstd.ZstdCompressor(level=3)

    def _clean_old_entries(self):
        """清理超量的条目"""
        # 按数量清理
        while len(self.window_items) > self.buffer_count:
            removed = self.window_items.pop(0)
            self._total_bytes -= len(removed.encode('utf-8'))

        # 按大小清理
        while self._total_bytes > self.buffer_size and self.window_items:
            removed = self.window_items.pop(0)
            self._total_bytes -= len(removed.encode('utf-8'))

    def _calculate_word_repetition_ratio(self, text: str) -> float:
        """
        计算字符重复率（用于短文本）

        返回值范围 [0, 1]：
        - 值越高说明重复越多，信息量越低
        """
        chars = list(text.lower())
        if len(chars) <= 1:
            return 0.0

        unique_chars = len(set(chars))
        repetition = 1 - (unique_chars / len(chars))
        return repetition

    def _calculate_incremental_density(self, new_text: str) -> float:
        """
        计算增量密度（基于压缩率）

        返回值范围 [0, 1]：
        - 值越高说明信息重复度高（可压缩）
        - 值越低说明信息增量高（新鲜）
        """
        if len(new_text) < self.min_text_length:
            return 0.9  # 短文本默认高重复

        # 压缩新文本
        new_bytes = new_text.encode('utf-8')
        new_compressed = self.cctx.compress(new_bytes)
        new_ratio = len(new_compressed) / len(new_bytes)

        # 压缩新旧组合
        combined = b"".join(item.encode('utf-8') for item in self.window_items) + new_bytes
        combined_compressed = self.cctx.compress(combined)
        combined_ratio = len(combined_compressed) / len(combined)

        # 如果组合压缩率更低，说明有信息增益
        incremental_density = max(0, combined_ratio - new_ratio)

        return incremental_density

    def should_accept(self, new_text: str) -> bool:
        """
        判断新文本是否应该被接受

        拒绝条件：
        1. 文本太短
        2. 字符重复率太高
        3. 增量密度太高（与历史太重复）
        """
        if not new_text or len(new_text.strip()) < self.min_text_length:
            return False

        # 检查字符重复率
        repetition_ratio = self._calculate_word_repetition_ratio(new_text)
        if repetition_ratio > 0.8:  # 超过 80% 重复字符
            return False

        # 检查增量密度
        incremental_density = self._calculate_incremental_density(new_text)
        if incremental_density > self.threshold:
            return False

        # 接受，添加到窗口
        self.window_items.append(new_text)
        self._total_bytes += len(new_text.encode('utf-8'))
        self._clean_old_entries()

        return True

    def status(self):
        """返回状态"""
        return {
            "count": len(self.window_items),
            "bytes": self._total_bytes,
            "count_limit": self.buffer_count,
            "size_limit": self.buffer_size
        }
