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
from superbot.memory.pipeline.ingestion.entropy_gatekeeper import EntropyGatekeeper
from superbot.memory.pipeline.ingestion.cache_buffer import CacheBuffer
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

        # Initialize input processing pipeline components
        self.entropy_gatekeeper = EntropyGatekeeper(
            threshold=self.config.entropy_threshold,
            buffer_count=self.config.entropy_buffer_count,
            buffer_size=self.config.entropy_buffer_size
        )

        # 语义去重缓存
        self.semantic_cache = CacheBuffer(
            buffer_count=self.config.semantic_buffer_count,
            buffer_size=float('inf'),  # 语义去重不需要大小限制
            similarity_func=self._cosine_similarity,
            similarity_threshold=self.config.semantic_threshold
        )

        # 批量处理缓存
        self.process_buffer = CacheBuffer(
            buffer_count=self.config.process_buffer_count,
            buffer_size=self.config.process_buffer_size
        )

        # 初始化检索器 (embedding_provider 由外部注入)
        self.retriever = EnhancedRetriever(
            self.vector_store,
            self.relation_store,
            embedding_provider=None,
            llm_provider=None  # 由外部通过 set_embedding 设置
        )

        # 对话历史
        self.history: List[str] = []
        self.active_topic: Optional[str] = None

        # 线程锁
        self._lock = threading.Lock()

    # ==================== 模型注入 ====================

    def set_llm(self, provider: LLMProvider) -> "MemorySystem":
        """注入 LLM provider"""
        self._llm = provider
        return self

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

    def _get_embedding(self) -> Optional[EmbeddingProvider]:
        """获取 Embedding"""
        return self._embedding

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # ==================== 核心接口 ====================

    def remember(self, raw_text: str) -> bool:
        """
        添加记忆

        参数:
            raw_text: 原始文本

        返回:
            是否成功处理
        """
        with self._lock:
            logger.debug("[Memory] remember() called with text: {}", raw_text[:100])

            # 1. 物理过滤
            if not self.entropy_gatekeeper.should_accept(raw_text):
                logger.debug("[Memory] remember() rejected by entropy gatekeeper: {}", raw_text[:100])
                return False

            # 2. 获取向量
            if self._embedding is None:
                raise RuntimeError("Embedding provider not set. Please call set_embedding() first.")
            vector = self._embedding.encode(raw_text)
            logger.debug("[Memory] remember() encoded vector shape: {}", vector.shape)

            # 3. 语义去重
            if not self.semantic_cache.push(raw_text, vector):
                logger.debug("[Memory] remember() rejected by semantic cache (duplicate): {}", raw_text[:100])
                return False

            # 4. 加入缓冲区并处理（调用 LLM 提取三元组）
            self.process_buffer.push(raw_text, vector)
            logger.debug("[Memory] remember() pushed to process_buffer, size: {}", len(self.process_buffer.buffer))
            self._process_buffer()

            # 5. 更新历史和话题
            self.history.append(raw_text)
            self._track_topic(raw_text)

            logger.debug("[Memory] remember() success, history length: {}", len(self.history))
            return True

    def remember_2(self, raw_text: str, similarity_threshold: float = 0.97) -> dict:
        """
        添加记忆 v2 - 仅提取摘要和三元组（去重后返回）

        参数:
            raw_text: 原始文本
            similarity_threshold: 相似度阈值，默认 0.97

        返回:
            摘要字符串
        """
        # 1. 记录 raw_logs
        raw_log_id = self.relation_store.add_raw_log(
            content=raw_text,
            source=None,
            incremental_density=None
        )

        # 2. 调用 LLM 同时提取摘要和三元组
        llm = self._get_llm()
        if llm is None:
            logger.warning("[Memory] remember_2() no LLM provider")
            return raw_text

        # 使用优化后的 prompt 同时提取摘要和三元组
        summary, triples = self._extract_summary_and_triples(raw_text, llm)

        # 3. 对每个三元组进行去重
        if self._embedding and triples:
            embedding = self._embedding
            action_ids = []

            for triple in triples:
                # 构建 subject+relation 文本
                subject = triple.get("s", "") or triple.get("subject", "")
                relation = triple.get("r", "") or triple.get("relation", "")
                obj = triple.get("o", "") or triple.get("object", "")
                action_text = f"{subject} {relation}"

                if not action_text.strip():
                    action_ids.append(None)
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
                    logger.debug("[Memory] remember_2() found existing action: {} (similarity: {:.4f})",
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
                    logger.debug("[Memory] remember_2() added new action: {}", new_id)

                # 写入 sqlite 表 action_objects
                if existing_id:
                    self.relation_store.add_action_object(
                        raw_log_id=raw_log_id,
                        vector_id=existing_id,
                        subject=subject,
                        relation=relation,
                        object_=obj
                    )

                action_ids.append(existing_id)

            logger.debug("[Memory] remember_2() success, extracted {} triples, {} new actions",
                        len(triples), sum(1 for aid in action_ids if aid))
            return summary

        logger.debug("[Memory] remember_2() success, extracted {} triples", len(triples))
        return summary

    def _extract_summary_and_triples(self, text: str, llm) -> tuple:
        """同时提取摘要和三元组

        使用优化后的 prompt，一次 LLM 调用同时产出摘要和三元组

        返回:
            (summary, triples) 元组
        """
        # 优化后的 prompt（跳过思考过程）
        prompt = f'''<|im_start|>system
Output ONLY summary and triples, no thinking.
Format:
摘要：<summary>
三元组：[{{"s":"subject","r":"relation","o":"object"}}]
<|im_end|>
<|im_start|>user
1. 压缩成简短摘要，不超过200字
2. 提取三元组：{text}
<|im_end|>
<|im_start|>assistant
摘要：'''

        try:
            # 调用 LLM generate 方法
            # SuperbotLLMAdapter 有 generate 方法
            response = llm.generate(prompt, max_tokens=512, temperature=0.3)
            logger.debug("[Memory] _extract_summary_and_triples() LLM response: {}", response[:200])

            # 解析响应
            summary = ""
            triples = []

            # 提取摘要
            summary_match = re.search(r'摘要[：:]\s*(.+?)(?=三元组|$)', response, re.DOTALL)
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

    def _split_long_text(self, text: str) -> list[str]:
        """Split text into multiple parts if it contains multiple intents.

        Split by:
        - Chinese comma '，' followed by new intent
        - Period '。' or '！'
        - Multiple statements separated by ','
        """
        # Remove prefix for analysis
        clean_text = text
        prefix = ""
        if text.startswith("[USER]"):
            prefix = "[USER] "
            clean_text = text[6:].strip()
        elif text.startswith("[ASSISTANT]"):
            prefix = "[ASSISTANT] "
            clean_text = text[12:].strip()

        # Split by common delimiters
        import re
        # Split by: Chinese period, exclamation, question mark, or comma+space
        parts = re.split(r'[。！？\n]+|(?<=[，]),', clean_text)
        parts = [p.strip() for p in parts if p.strip()]

        # If only one part, return original
        if len(parts) <= 1:
            return [text]

        # Add prefix back
        return [f"{prefix}{p}" for p in parts]

    def _process_buffer(self):
        """处理缓冲区中的内容"""
        import uuid
        call_id = str(uuid.uuid4())[:8]

        if not self.process_buffer.should_process():
            logger.debug("[Memory] _process_buffer() skipped, buffer not ready")
            return

        batch_items = self.process_buffer.get_batch()
        if not batch_items:
            logger.debug("[Memory] _process_buffer() skipped, no batch items")
            return

        logger.debug("[Memory] _process_buffer() [{}] processing {} items", call_id, len(batch_items))
        batch_text = [item["text"] for item in batch_items]

        # 调用 LLM 提纯
        llm = self._get_llm()
        extracted = []
        logger.debug("[Memory] _process_buffer() [{}] batch_text: {}", call_id, batch_text)

        # Build conversation context from recent history (increased window)
        context = self._build_context()

        for text in batch_text:
            if llm is not None:
                try:
                    # Use action-metadata extraction with context
                    action_metadata = llm.extract_triples(text, context=context)
                    logger.debug("[Memory] _process_buffer() [{}] raw result type: {}, value: {}", call_id, type(action_metadata).__name__, action_metadata)
                    logger.debug("[Memory] _process_buffer() [{}] is list: {}, is truthy: {}", call_id, isinstance(action_metadata, list), bool(action_metadata))

                    # New format: single dict with "triple": [subject, relation, object]
                    if isinstance(action_metadata, dict) and "triple" in action_metadata:
                        triple_list = action_metadata.get("triple", [])
                        if len(triple_list) >= 3:
                            subject, relation, obj = triple_list[0], triple_list[1], triple_list[2]
                            triple_summary = action_metadata.get("summary", "")
                            action = triple_summary if triple_summary else f"{subject} {relation} {obj}"
                            triple_data = {"subject": subject, "relation": relation, "object": obj, "summary": triple_summary}
                            metadata = {"triples": [triple_data], "summary": triple_summary}
                            extracted.append({
                                "tag": "triple",
                                "action": action,
                                "metadata": metadata,
                            })
                            logger.debug("[Memory] _process_buffer() [{}] extracted 1 triple with summary: {}", call_id, triple_summary)
                    # New format: single dict with subject, relation/predicate, object, summary
                    elif isinstance(action_metadata, dict) and "subject" in action_metadata and "summary" in action_metadata:
                        triple = action_metadata
                        triple_summary = triple.get("summary", "")
                        subject = triple.get("subject", "")
                        relation = triple.get("relation", "") or triple.get("predicate", "")
                        obj = triple.get("object", "")
                        action = triple_summary if triple_summary else f"{subject} {relation} {obj}"
                        metadata = {"triples": [triple], "summary": triple_summary}
                        extracted.append({
                            "tag": "triple",
                            "action": action,
                            "metadata": metadata,
                        })
                        logger.debug("[Memory] _process_buffer() [{}] extracted 1 triple with summary: {}", call_id, triple_summary)
                    # New format: list of {"subject": ..., "relation": ..., "object": ..., "summary": ...}
                    elif isinstance(action_metadata, list) and action_metadata:
                        first = action_metadata[0]
                        if isinstance(first, dict) and "subject" in first and "summary" in first:
                            # New format: each triple has its own summary
                            # Create separate entry for each triple
                            for triple in action_metadata:
                                triple_summary = triple.get("summary", "")
                                subject = triple.get("subject", "")
                                relation = triple.get("relation", "") or triple.get("predicate", "")
                                obj = triple.get("object", "")
                                # Use triple's own summary as action
                                action = triple_summary if triple_summary else f"{subject} {relation} {obj}"
                                metadata = {"triples": [triple], "summary": triple_summary}
                                extracted.append({
                                    "tag": "triple",
                                    "action": action,
                                    "metadata": metadata,
                                })
                            logger.debug("[Memory] _process_buffer() [{}] extracted {} triples with individual summaries", call_id, len(action_metadata))
                        elif isinstance(first, dict) and "subject" in first and "relation" in first and "object" in first:
                            # Old format: list of triples without individual summaries
                            first_triple = action_metadata[0]
                            action = f"{first_triple.get('subject', '')} {first_triple.get('relation', '')} {first_triple.get('object', '')}"
                            metadata = {"triples": action_metadata, "summary": ""}
                            extracted.append({
                                "tag": "triple",
                                "action": action,
                                "metadata": metadata,
                            })
                            logger.debug("[Memory] _process_buffer() [{}] extracted {} triples", call_id, len(action_metadata))
                        else:
                            # Old action-metadata format
                            data = action_metadata[0]
                            action = data.get("action", text)
                            metadata = data.get("metadata", {})
                            extracted.append({
                                "tag": "action",
                                "action": action,
                                "metadata": metadata,
                            })
                    elif isinstance(action_metadata, dict) and action_metadata:
                        if "subject" in action_metadata:
                            # Single triple
                            triples_data = [action_metadata]
                            action = f"{action_metadata.get('subject', '')} {action_metadata.get('relation', '')} {action_metadata.get('object', '')}"
                            metadata = {"subject": action_metadata.get("subject", ""), "relation": action_metadata.get("relation", ""), "object": action_metadata.get("object", "")}
                            extracted.append({
                                "tag": "triple",
                                "action": action,
                                "metadata": metadata,
                            })
                        else:
                            # Old action-metadata format
                            action = action_metadata.get("action", text)
                            metadata = action_metadata.get("metadata", {})
                            extracted.append({
                                "tag": "action",
                                "action": action,
                                "metadata": metadata,
                            })
                    else:
                        logger.debug("[Memory] _process_buffer() [{}] empty result, using fallback", call_id)
                        extracted.append({"tag": "未分类", "action": text, "metadata": {}})
                except Exception as e:
                    logger.warning("[Memory] _process_buffer() [{}] LLM extraction failed: {}", call_id, e)
                    extracted.append({"tag": "未分类", "action": text, "metadata": {}})
            else:
                extracted.append({"tag": "未分类", "action": text, "metadata": {}})

        # 存储：每个 extracted 项单独存储 (一个输入可能产生多个 triple)
        import json

        # 先为每个输入创建 raw_log
        raw_ids = []
        for item in batch_items:
            raw_text = item["text"]
            raw_id = self.relation_store.add_raw_log(
                content=raw_text,
                source="memory",
                incremental_density=None
            )
            raw_ids.append(raw_id)
            logger.debug("[Memory] _process_buffer() added raw_log id: {}", raw_id)

        # 为每个 extracted 项创建 memory_node
        extracted_idx = 0
        for batch_idx, item in enumerate(batch_items):
            raw_text = item["text"]
            original_vector = item["vector"].tolist()
            raw_id = raw_ids[batch_idx]

            # 处理该 batch 对应的所有 extracted (可能有多个 triple)
            while extracted_idx < len(extracted):
                data = extracted[extracted_idx]
                extracted_idx += 1

                action = data.get("action", raw_text)
                metadata = data.get("metadata", {})
                tag = data.get("tag", "action")

                # 跳过未分类的
                if tag == "未分类":
                    vector_id = str(uuid.uuid4())
                    memory_node_id = self.relation_store.add_memory_node(
                        raw_id=raw_id,
                        tag="raw",
                        summary=raw_text[:50],
                        vector_id=vector_id,
                        entities=json.dumps({}),
                        facts=json.dumps([raw_text])
                    )
                    store_metadata = {"tag": "raw", "memory_node_id": str(memory_node_id)}
                    self.vector_store.add(
                        ids=[vector_id],
                        vectors=[original_vector],
                        documents=[raw_text],
                        metadatas=[store_metadata]
                    )
                    continue

                # 存储 triple
                vector_id = str(uuid.uuid4())
                memory_node_id = self.relation_store.add_memory_node(
                    raw_id=raw_id,
                    tag=tag,
                    summary=action,
                    vector_id=vector_id,
                    entities=json.dumps(metadata),
                    facts=json.dumps([action])
                )
                logger.debug("[Memory] _process_buffer() added memory_node id: {}, action: {}", memory_node_id, action)

                # 存储三元组到 relationships 表
                triples_from_meta = metadata.get("triples", [])
                if triples_from_meta:
                    for triple in triples_from_meta:
                        head = triple.get("subject", "").strip()
                        relation = triple.get("relation", "").strip()
                        tail = triple.get("object", "").strip()
                        if head and relation and tail:
                            self.relation_store.upsert_relation(
                                head=head,
                                relation=relation,
                                tail=tail,
                                ref_id=memory_node_id
                            )
                            print(f"[Memory] STORE Triple: ({head}, {relation}, {tail}), Summary: {action}")

                # 存入向量库 (ChromaDB metadata 只支持简单类型)
                store_metadata = {"tag": tag, "memory_node_id": str(memory_node_id)}
                # 把复杂 metadata 转成 JSON 字符串
                if metadata:
                    store_metadata["metadata_json"] = json.dumps(metadata)
                    # 同时提取关键字段用于检索
                    if triples_from_meta and len(triples_from_meta) > 0:
                        first_triple = triples_from_meta[0]
                        store_metadata["subject"] = first_triple.get("subject", "")
                        store_metadata["relation"] = first_triple.get("relation", "")
                        store_metadata["object"] = first_triple.get("object", "")
                # 存向量用 raw_text（用于匹配），但文档内容用 action（事实摘要）
                doc_content = action if action and action != raw_text else raw_text
                self.vector_store.add(
                    ids=[vector_id],
                    vectors=[original_vector],
                    documents=[doc_content],
                    metadatas=[store_metadata]
                )
                logger.debug("[Memory] _process_buffer() added vector_store id: {}, action: {}", vector_id, action)

        logger.debug("[Memory] _process_buffer() completed for {} items", len(batch_items))

        # Keep only buffer_count items after processing
        current_count = self.process_buffer.size()
        keep_count = self.config.process_buffer_count
        if current_count > keep_count:
            # Keep only the last keep_count items
            remaining = self.process_buffer.buffer[-keep_count:]
            self.process_buffer.buffer = remaining
            # Recalculate total_bytes
            self.process_buffer._total_bytes = sum(item["text_bytes"] for item in self.process_buffer.buffer)
            logger.debug("[Memory] _process_buffer() kept {} items, removed {}", keep_count, current_count - keep_count)

    def _build_context(self) -> str:
        """Build conversation context from recent history.

        Returns last 6 conversation turns (3 user + 3 assistant pairs) as context.
        """
        if not self.history or len(self.history) < 2:
            return ""

        # Get last 6 messages (3 conversation turns) for better context
        recent = self.history[-6:] if len(self.history) >= 6 else self.history

        # Format as conversation turns
        context_parts = []
        for i, msg in enumerate(recent):
            if msg.startswith("[USER]") or msg.startswith("[ASSISTANT]"):
                context_parts.append(msg)
            else:
                # Handle messages without prefix
                role = "User" if i % 2 == 0 else "Assistant"
                context_parts.append(f"[{role}] {msg}")

        return "\n".join(context_parts)

    def _track_topic(self, text: str) -> Optional[str]:
        """滑动窗口话题追踪"""
        if not self.history or len(self.history) < 2:
            self.active_topic = None
            return None

        prev_text = self.history[-2]
        prev_vec = self._embedding.encode(prev_text)
        curr_vec = self._embedding.encode(text)

        similarity = self._cosine_similarity(prev_vec, curr_vec)

        if similarity > 0.7:
            return self.active_topic

        self.active_topic = None
        return None

    def recall(
        self,
        query_text: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        检索记忆

        参数:
            query_text: 查询文本
            top_n: 返回结果数量，默认 5

        返回:
            包含 facts、raw_logs、relations 的字典
        """
        logger.debug("[Memory] recall() called with query: {}, top_n: {}", query_text[:100], top_n)

        results = self.retriever.retrieve(query_text, top_n=top_n)

        # Print recall results
        print(f"[Memory] RECALL Query: {query_text}")
        print(f"[Memory] RECALL Facts: {len(results.get('facts', []))}")
        for f in results.get('facts', []):
            meta = f.get('metadata', {})
            print(f"  - {f.get('content', '')[:50]}, subject: {meta.get('subject', '')}")
        print(f"[Memory] RECALL Relations: {len(results.get('relations', []))}")
        for r in results.get('relations', []):
            print(f"  - ({r.get('head', '')}, {r.get('relation', '')}, {r.get('tail', '')})")

        logger.debug("[Memory] recall() retrieved {} facts, {} relations, {} raw_logs",
                     len(results.get("facts", [])), len(results.get("relations", [])), len(results.get("raw_logs", [])))

        # raw_logs 已经在 retriever.retrieve() 中获取

        llm = self._get_llm()
        if llm is not None and self.history:
            memory = [f["content"] for f in results.get("facts", [])][:3]
            understanding = llm.understand_context(
                query=query_text,
                memory=memory,
                history=self.history[-3:]
            )
            results["understanding"] = understanding
            results["type"] = "llm_understanding"
            logger.debug("[Memory] recall() using LLM understanding: {}", understanding[:100] if understanding else "")
        else:
            results["type"] = "direct_retrieval"
            logger.debug("[Memory] recall() using direct retrieval (no LLM or no history)")

        return results

    def recall_2(self, query_text: str, summary: str, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        检索记忆 v2 - 基于摘要检索

        参数:
            query_text: 查询文本
            summary: 摘要（用于检索）
            similarity_threshold: 相似度阈值，默认 0.9

        返回:
            包含 triples（三元组语句列表）的字典
        """
        logger.debug("[Memory] recall_2() called with query: {}, summary: {}", query_text[:100], summary[:100])

        if not self._embedding:
            logger.warning("[Memory] recall_2() no embedding provider")
            return {"triples": [], "query": query_text}

        # 1. 向量化 summary 和 query_text
        summary_vector = self._embedding.encode(summary).tolist()
        query_vector = self._embedding.encode(query_text).tolist()

        # 2. 计算相似度
        from datetime import datetime
        import numpy as np

        similarity = np.dot(summary_vector, query_vector) / (np.linalg.norm(summary_vector) * np.linalg.norm(query_vector) + 1e-8)
        logger.debug("[Memory] recall_2() summary vs query_text similarity: {:.4f}", similarity)

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
                logger.debug("[Memory] recall_2() found better summary: {}", used_summary[:50])
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
        logger.debug("[Memory] recall_2() saved summary to query_summary: {}", save_id)

        # 5. 查询 user_actions 集合
        vector_results = self.vector_store.search(
            query_vector=summary_vector,
            n=10,
            collection="user_actions"
        )

        logger.debug("[Memory] recall_2() vector results: {}", len(vector_results))

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

        logger.debug("[Memory] recall_2() returning {} triples", len(triples))

        return {
            "triples": triples,
            "query": query_text,
            "summary": used_summary
        }

    def get_memory_context(self, query: str, top_n: int = 5) -> str:
        """获取格式化后的记忆上下文，用于 system prompt。

        Args:
            query: 当前查询文本
            top_n: 返回结果数量

        Returns:
            格式化的记忆上下文字符串，如果无结果则返回空字符串
        """
        logger.debug("[Memory] get_memory_context() called with query: {}", query[:100])
        results = self.recall(query, top_n=top_n)
        lines = []

        # 0. Add instruction to prioritize facts
        if results.get("facts") or results.get("relations"):
            lines.append("IMPORTANT: Prioritize the facts below when answering.")
            lines.append("")

        # 1. Format semantic facts
        facts = results.get("facts", [])
        if facts:
            lines.append("## Relevant Facts")
            tag_tracker = {}  # Track tags for "latest" vs "history" markers

            for fact in facts[:5]:
                content = fact.get("content", "")
                if not content:
                    continue

                # Get tag from metadata
                metadata = fact.get("metadata", {})
                tag = metadata.get("tag", "")

                # Status label logic
                if tag not in tag_tracker:
                    status_label = ""  # Latest entry has no special label
                    tag_tracker[tag] = 1
                else:
                    status_label = "(history)"  # Same tag, non-first entry marked as history

                # Extract date
                timestamp = fact.get("metadata", {}).get("timestamp", "")
                date_str = timestamp.split(" ")[0] if timestamp and " " in timestamp else (timestamp[:10] if timestamp else "")

                line = f"- [{date_str}] {tag}{status_label}: {content}"
                lines.append(line)

        lines.append("")  # Blank line separator

        # 2. Format action metadata (from action-metadata pairs)
        # Collect metadata from facts for display
        metadata_entries = []
        for fact in facts[:5]:
            metadata = fact.get("metadata", {})
            if metadata:
                # Format: action + metadata attributes
                action = fact.get("content", "")
                if action and metadata:
                    meta_str = ", ".join([f"{k}={v}" for k, v in metadata.items() if k not in ["tag", "memory_node_id", "timestamp"]])
                    if meta_str:
                        metadata_entries.append(f"- {action} ({meta_str})")
                    else:
                        metadata_entries.append(f"- {action}")
        if metadata_entries:
            lines.append("## Action Metadata")
            lines.extend(metadata_entries[:5])

        # 2. Format relations from relationships table
        relations = results.get("relations", [])
        if relations:
            lines.append("")
            lines.append("## Knowledge Graph Relations")
            # Group by subject
            subject_relations = {}
            for rel in relations:
                head = rel.get("head", "")
                relation = rel.get("relation", "")
                tail = rel.get("tail", "")
                if head:
                    if head not in subject_relations:
                        subject_relations[head] = []
                    subject_relations[head].append(f"{relation} {tail}")
            for subject, rels in subject_relations.items():
                lines.append(f"- {subject}: {', '.join(rels)}")

        lines.append("")  # Blank line separator

        # 3. Add raw conversations
        raw_logs = results.get("raw_logs", [])
        if raw_logs:
            lines.append("## Recent Conversations")
            for log in raw_logs[:3]:
                content = log.get("content", "")
                timestamp = log.get("timestamp", "")
                if content:
                    # Extract [USER] or [ASSISTANT] prefix
                    prefix = ""
                    if content.startswith("[USER]"):
                        prefix = "[User]"
                        content = content[7:]
                    elif content.startswith("[ASSISTANT]"):
                        prefix = "[Assistant]"
                        content = content[11:]

                    date_str = timestamp.split(" ")[0] if timestamp and " " in timestamp else (timestamp[:10] if timestamp else "")

                    if prefix and date_str:
                        lines.append(f"- {date_str} {prefix}: {content[:100]}")
                    elif prefix:
                        lines.append(f"- {prefix}: {content[:100]}")
                    else:
                        lines.append(f"- {content[:100]}")

        # 4. Add LLM understanding (if exists)
        understanding = results.get("understanding")
        if understanding:
            lines.append("")
            lines.append(f"## Context Summary\n{understanding}")

        result = "\n".join(lines) if lines else ""
        logger.debug("[Memory] get_memory_context() returning {} chars", len(result))
        return result

    def normalize_entities(self, similarity_threshold: float = 0.9) -> int:
        """Normalize entities using embedding similarity.

        Args:
            similarity_threshold: Threshold for considering entities as same (0-1)

        Returns:
            Number of entities normalized
        """
        if self._embedding is None:
            logger.warning("[Memory] normalize_entities() skipped: no embedding provider")
            return 0

        logger.info("[Memory] normalize_entities() started, threshold: {}", similarity_threshold)

        # Get all memory nodes with entities
        all_nodes = self.relation_store.get_all_memory_nodes()
        if not all_nodes:
            return 0

        # Build entity list with node info
        entities = []  # (node_id, entity_key, entity_value, vector)
        for node in all_nodes:
            try:
                import json
                node_id = node["id"]
                entities_data = json.loads(node.get("entities", "{}")) if node.get("entities") else {}
                # Extract entity from metadata
                for key, value in entities_data.items():
                    if isinstance(value, str) and value:
                        # Encode entity
                        vec = self._embedding.encode(value)
                        entities.append((node_id, key, value, vec))
            except Exception:
                pass

        if not entities:
            return 0

        # Find and merge similar entities
        normalized_count = 0
        processed_nodes = set()  # Track processed node IDs

        for i in range(len(entities)):
            node_id1, key1, value1, vec1 = entities[i]
            if node_id1 in processed_nodes:
                continue

            for j in range(i + 1, len(entities)):
                node_id2, key2, value2, vec2 = entities[j]
                if node_id2 in processed_nodes:
                    continue

                # Calculate similarity
                similarity = self._cosine_similarity(vec1, vec2)
                if similarity >= similarity_threshold:
                    logger.info("[Memory] normalize_entities() merging: {} -> {} (similarity: {:.2f})",
                              value2, value1, similarity)

                    # Keep the newer one (higher id), merge older into newer
                    older_id, older_key, older_value = node_id2, key2, value2
                    newer_id, newer_key, newer_value = node_id1, key1, value1
                    if node_id2 > node_id1:
                        older_id, older_key, older_value = node_id1, key1, value1
                        newer_id, newer_key, newer_value = node_id2, key2, value2

                    # Get current newer node data
                    newer_node = self.relation_store.get_memory(newer_id)
                    if newer_node:
                        # Update newer node: add merged_from info
                        import json
                        current_entities = json.loads(newer_node.get("entities", "{}")) if newer_node.get("entities") else {}
                        # Add the merged entity as an alias
                        if "merged_from" not in current_entities:
                            current_entities["merged_from"] = []
                        current_entities["merged_from"].append({
                            "key": older_key,
                            "value": older_value,
                            "similarity": float(similarity)
                        })
                        # Update newer node
                        self.relation_store.update_memory_node(
                            newer_id,
                            entities=json.dumps(current_entities),
                            tag="normalized"
                        )

                    # Mark older node as merged
                    self.relation_store.update_memory_node(
                        older_id,
                        tag="merged"
                    )

                    normalized_count += 1
                    processed_nodes.add(older_id)
                    processed_nodes.add(newer_id)

        logger.info("[Memory] normalize_entities() completed, normalized: {}", normalized_count)
        return normalized_count

    def shutdown(self):
        """关闭系统"""
        pass
