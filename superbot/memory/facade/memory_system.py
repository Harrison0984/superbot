"""Memory System Facade - Unified Entry"""
import os
import sys
import re
import json
import threading
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
from loguru import logger

# Add src to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import using superbot.memory paths (memmory is an internal name)
from superbot.memory.config import Config, config
from superbot.memory.storage import VectorStore, EnhancedRelationStore
from superbot.memory.pipeline.ingestion.cache_buffer import CacheBuffer, FIFOBuffer
from superbot.memory.pipeline.retrieval import EnhancedRetriever

# Type hints
from superbot.memory.models import LLMProvider, EmbeddingProvider
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superbot.memory.models import LLMProvider, EmbeddingProvider


class MemorySystem:
    """Memory Management System - Facade Pattern"""

    def __init__(
        self,
        data_dir: str = "./data",
        config: Optional[Config] = None
    ):
        self.config = config if config is not None else Config()
        # Convert to absolute path to avoid issues with relative paths
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self._llm: Optional[LLMProvider] = None
        self._embedding: Optional[EmbeddingProvider] = None

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize storage layer (unified in data directory)
        self.vector_store = VectorStore(
            persist_directory=os.path.join(self.data_dir, "chroma")
        )
        self.relation_store = EnhancedRelationStore(
            db_path=os.path.join(self.data_dir, "memmory.db")
        )

        # 批量处理缓存
        self.process_buffer = CacheBuffer(
            buffer_count=self.config.process_buffer_count,
            buffer_size=self.config.process_buffer_size
        )

        # 对话历史
        self.history = FIFOBuffer(
            buffer_count=self.config.history_buffer_count,
            buffer_size=self.config.history_buffer_size
        )

        # 初始化检索器 (embedding_provider 由外部注入)
        self.retriever = EnhancedRetriever(
            self.vector_store,
            self.relation_store,
            embedding_provider=None,
            llm_provider=None  # 由外部通过 set_embedding 设置
        )

        # remember 生成的最新摘要
        self._latest_summary: Optional[str] = None

        # 线程锁
        self._lock = threading.Lock()

    # ==================== 模型注入 ====================

    def set_embedding(self, provider: EmbeddingProvider) -> "MemorySystem":
        """注入 Embedding provider"""
        self._embedding = provider
        self.retriever.embedding_provider = provider
        # 如果有 LLM，也设置到 retriever
        if self._llm:
            self.retriever.llm_provider = self._llm
        return self

    def set_llm(self, provider: LLMProvider) -> "MemorySystem":
        """注入 LLM provider"""
        self._llm = provider
        self.retriever.llm_provider = provider
        return self

    def _get_llm(self) -> Optional[LLMProvider]:
        """获取 LLM"""
        return self._llm

    # ==================== 核心接口 ====================

    def remember(self, raw_text: str, similarity_threshold: float = 0.97) -> None:
        """
        添加记忆 v2 - 仅提取摘要和三元组（去重后返回）

        参数:
            raw_text: 原始文本
            similarity_threshold: 相似度阈值，默认 0.97

        返回:
            无返回值
        """
        with self._lock:
            # 1. raw_text 缓存到 process_buffer（来源固定为 ASSISTANT）
            self.history.push(raw_text, source="ASSISTANT")
            self.process_buffer.push(raw_text, source="ASSISTANT")
            logger.debug("[Memory] remember() pushed to process_buffer, size: {}", self.process_buffer.size())

            # 2. 检查是否需要处理
            if not self.process_buffer.should_process():
                logger.debug("[Memory] remember() buffer not ready, waiting for more items")
                self._latest_summary = None
                return

            # 3. 获取缓冲区内容并处理
            self._process_buffer(similarity_threshold)

    def _process_buffer(self, similarity_threshold: float) -> None:
        """
        处理 process_buffer 中的内容，提取摘要和三元组

        参数:
            similarity_threshold: 相似度阈值
        """
        batch_items = self.process_buffer.get_batch()
        if not batch_items:
            logger.debug("[Memory] _process_buffer() skipped, no batch items")
            return

        logger.debug("[Memory] _process_buffer() processing {} items", len(batch_items))

        # 获取 LLM
        llm = self._get_llm()
        if llm is None:
            logger.warning("[Memory] _process_buffer() no LLM provider")
            self._latest_summary = None
            return

        # 整合所有文本为对话格式（摘要放在前面）
        combined_text = ""
        if self._latest_summary:
            combined_text += f"[SUMMARY]:{self._latest_summary}\n"

        for item in batch_items:
            source = item.get("source", "USER")
            combined_text += f"[{source}]:{item['text']}\n"

        logger.debug("[Memory] _process_buffer() combined_text: {}", combined_text[:200])

        # 记录 raw_logs（每个 item 都要记录）
        raw_log_ids = []
        for item in batch_items:
            raw_log_id = self.relation_store.add_raw_log(
                content=item["text"],
                source=item.get("source", "USER"),
                incremental_density=None
            )
            raw_log_ids.append(raw_log_id)

        # 一次性提交给 LLM 提取摘要和三元组
        summary, triples = self._extract_summary_and_triples(combined_text, llm)

        # 保存最新的摘要到 self
        self._latest_summary = summary
        logger.debug("[Memory] _process_buffer() summary: {}", summary[:100] if summary else None)

        # 对每个三元组进行去重
        if self._embedding and triples:
            embedding = self._embedding

            for triple in triples:
                # 构建 subject+relation 文本
                subject = triple.get("s", "") or triple.get("subject", "")
                relation = triple.get("r", "") or triple.get("relation", "")
                obj = triple.get("o", "") or triple.get("object", "")
                action_text = f"{subject} {relation}"

                if not action_text.strip():
                    continue

                # 向量化
                action_vector = embedding.encode(action_text).tolist()

                # 查询 user_actions 集合
                results = self.vector_store.search(
                    query_vector=action_vector,
                    n=1,
                    collection="user_actions"
                )

                # 检查相似度
                existing_id = None
                if results and results[0].get("similarity", 0) >= similarity_threshold:
                    existing_id = results[0]["id"]
                    logger.debug("[Memory] _process_buffer() found existing action: {} (similarity: {:.4f})",
                               existing_id, results[0]["similarity"])
                else:
                    # 写入新记录
                    new_id = str(uuid.uuid4())
                    self.vector_store.add(
                        ids=[new_id],
                        vectors=[action_vector],
                        documents=[action_text],
                        collection="user_actions"
                    )
                    existing_id = new_id
                    logger.debug("[Memory] _process_buffer() added new action: {}", new_id)

                # 写入 sqlite 表 action_objects（关联到所有 raw_log_id）
                if existing_id:
                    for raw_log_id in raw_log_ids:
                        self.relation_store.add_action_object(
                            raw_log_id=raw_log_id,
                            vector_id=existing_id,
                            subject=subject,
                            relation=relation,
                            object_=obj
                        )

        self.process_buffer.clear()
        logger.debug("[Memory] _process_buffer() completed, latest_summary: {}",
                    self._latest_summary[:100] if self._latest_summary else None)

    def _extract_summary_and_triples(self, text: str, llm) -> tuple:
        """同时提取摘要和三元组

        使用优化后的 prompt，一次 LLM 调用同时产出摘要和三元组

        返回:
            (summary, triples) 元组
        """
        # 从配置获取 prompt
        prompt_template = self.config.summary_triples_prompt
        prompt = prompt_template.format(text=text)
        max_tokens = self.config.summary_triples_max_tokens

        try:
            # 调用 LLM generate 方法
            # mlx_lm.generate 不支持 temperature 参数
            response = llm.generate(prompt, max_tokens=max_tokens)
            logger.debug("[Memory] _extract_summary_and_triples() LLM response: {}", response[:200])

            # 解析响应
            summary = ""
            triples = []

            # 提取摘要 - 兼容 V1 和 V2 格式
            # V1: 摘要：xxx 三元组：...
            # V2: 摘要：xxx\n三元组：[...]
            # 也兼容没有"摘要："前缀的情况
            summary_match = re.search(r'(?:摘要[：:]\s*)?(.+?)(?=\n三元组|三元组\[|$)', response, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
                # 清理摘要中的思考过程残留
                if '<think>' in summary:
                    summary = summary.split('</think>')[-1].strip()

            # 提取三元组
            triple_match = re.search(r'三元组[：:]?\s*(\[.+?\])', response, re.DOTALL)
            if triple_match:
                try:
                    triples = json.loads(triple_match.group(1))
                except:
                    # 备用解析
                    try:
                        objs = re.findall(r'\{[^{}]*\}', triple_match.group(1))
                        for o in objs:
                            try:
                                t = json.loads(o)
                                if 's' in t or 'subject' in t:
                                    triples.append(t)
                            except:
                                pass
                    except:
                        pass

            logger.debug("[Memory] _extract_summary_and_triples() extracted: {} triples, summary: {}",
                        len(triples), summary[:50] if summary else "")

            return summary, triples

        except Exception as e:
            logger.error("[Memory] _extract_summary_and_triples() error: {}", e)
            return "", []

    def recall(self, query_text: str, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        检索记忆 v2 - 基于摘要检索

        参数:
            query_text: 查询文本
            similarity_threshold: 相似度阈值，默认 0.9

        返回:
            包含 triples（三元组语句列表）的字典
        """
        with self._lock:
            # 使用 self 中的摘要
            summary = self._latest_summary or ""
            logger.debug("[Memory] recall() called with query: {}, summary: {}", query_text[:100], summary[:100])

            # 0. 将 query_text 保存到 process_buffer（来源固定为 USER）
            self.history.push(query_text, source="USER")
            self.process_buffer.push(query_text, source="USER")
            logger.debug("[Memory] recall() pushed query to process_buffer, size: {}", self.process_buffer.size())

            if not self._embedding:
                logger.warning("[Memory] recall() no embedding provider")
                return {"triples": [], "query": query_text}

            # 1. 向量化 summary 和 query_text
            summary_vector = self._embedding.encode(summary).tolist()
            query_vector = self._embedding.encode(query_text).tolist()

            # 2. 计算相似度
            from datetime import datetime
            import numpy as np

            similarity = np.dot(summary_vector, query_vector) / (np.linalg.norm(summary_vector) * np.linalg.norm(query_vector) + 1e-8)
            logger.debug("[Memory] recall() summary vs query_text similarity: {:.4f}", similarity)

            # 3. 如果相似度 < 0.8，从 query_summary 集合查询最接近的 summary
            used_summary = summary
            if similarity < 0.8:
                query_summary_results = self.vector_store.search(
                    query_vector=summary_vector,
                    n=1,
                    collection="query_summary"
                )

                if query_summary_results and query_summary_results[0].get("similarity", 0) > similarity:
                    used_summary = query_summary_results[0].get("document", "")
                    logger.debug("[Memory] recall() found better summary: {}", used_summary[:50])
                    # 使用找到的 summary 向量
                    summary_vector = self._embedding.encode(used_summary).tolist()

            # 4. 保存当前 summary 到 query_summary 集合
            save_id = str(uuid.uuid4())
            self.vector_store.add(
                ids=[save_id],
                vectors=[summary_vector],
                documents=[used_summary],
                metadatas=[{"timestamp": datetime.now().isoformat()}],
                collection="query_summary"
            )
            logger.debug("[Memory] recall() saved summary to query_summary: {}", save_id)

            # 5. 查询 user_actions 集合
            vector_results = self.vector_store.search(
                query_vector=summary_vector,
                n=10,
                collection="user_actions"
            )

            logger.debug("[Memory] recall() vector results: {}", len(vector_results))

            # 6. 过滤匹配度 > threshold 的结果，并查询 action_objects 组合成完整三元组
            triples = []

            for result in vector_results:
                if result.get("similarity", 0) <= similarity_threshold:
                    continue

                vector_id = result.get("id")
                action_obj = self.relation_store.get_action_object(vector_id)
                if action_obj:
                    subject = action_obj.get("subject", "")
                    relation = action_obj.get("relation", "")
                    obj = action_obj.get("object", "")

                    triples.append({
                        "subject": subject,
                        "relation": relation,
                        "object": obj,
                        "triple_text": f"{subject} {relation} {obj}" if obj else f"{subject} {relation}",
                        "similarity": result.get("similarity", 0),
                        "vector_id": vector_id
                    })

            logger.debug("[Memory] recall() returning {} triples", len(triples))

            # 获取 history 中的数据
            batch_items = self.history.get_batch()
            buffer_data = batch_items

            return {
                "summary": used_summary,
                "triples": triples,
                "history": buffer_data
            }

    def get_memory_context(self, query: str) -> str:
        """获取格式化后的记忆上下文，用于 system prompt。

        Args:
            query: 当前查询文本

        Returns:
            格式化的记忆上下文字符串，如果无结果则返回空字符串
        """
        logger.debug("[Memory] get_memory_context() called with query: {}", query[:100])
        results = self.recall(query)
        lines = []

        # 0. Add instruction to prioritize facts
        summary = results.get("summary", "")
        triples = results.get("triples", [])
        history = results.get("history", [])

        if summary or triples:
            lines.append("IMPORTANT: Prioritize the facts below when answering.")
            lines.append("")

        # 1. Format summary
        if summary:
            lines.append("## Summary")
            lines.append(summary)
            lines.append("")

        # 2. Format triples
        if triples:
            lines.append("## Knowledge Triples")
            for triple in triples[:10]:
                triple_text = triple.get("triple_text", "")
                similarity = triple.get("similarity", 0)
                if triple_text:
                    lines.append(f"- {triple_text} (similarity: {similarity:.2f})")
            lines.append("")

        # 3. Format history (recent conversations)
        if history:
            lines.append("## Recent Conversations")
            for item in history[-6:]:
                text = item.get("text", "")
                source = item.get("source", "USER")
                role = "User" if source == "USER" else "Assistant"
                if text:
                    lines.append(f"- [{role}]: {text[:100]}")
            lines.append("")

        result = "\n".join(lines) if lines else ""
        logger.debug("[Memory] get_memory_context() returning {} chars", len(result))
        return result


    def shutdown(self):
        """关闭系统"""
        pass
