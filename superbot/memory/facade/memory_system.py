"""记忆系统门面 - 统一入口"""
import os
import sys
import threading
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np

# 添加 src 到路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import using superbot.memory paths (memmory is an internal name)
from superbot.memory.config import Config, config
from superbot.memory.storage import VectorStore, EnhancedRelationStore
from superbot.memory.pipeline.ingestion.entropy_gatekeeper import EntropyGatekeeper
from superbot.memory.pipeline.ingestion.cache_buffer import CacheBuffer
from superbot.memory.pipeline.retrieval import EnhancedRetriever

# 类型提示
from superbot.memory.models import LLMProvider, EmbeddingProvider
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superbot.memory.models import LLMProvider, EmbeddingProvider


class MemorySystem:
    """记忆管理系统 - 门面模式"""

    def __init__(
        self,
        data_dir: str = "./data",
        config: Optional[Config] = None
    ):
        self.config = config if config is not None else Config()
        self.data_dir = data_dir
        self._llm: Optional[LLMProvider] = None
        self._embedding: Optional[EmbeddingProvider] = None

        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 初始化存储层 (统一在 data 目录下)
        self.vector_store = VectorStore(
            persist_directory=os.path.join(data_dir, "chroma")
        )
        self.relation_store = EnhancedRelationStore(
            db_path=os.path.join(data_dir, "memmory.db")
        )

        # 初始化输入处理管道组件
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
            embedding_provider=None
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
            # 1. 物理过滤
            if not self.entropy_gatekeeper.should_accept(raw_text):
                return False

            # 2. 获取向量
            if self._embedding is None:
                raise RuntimeError("Embedding provider not set. Please call set_embedding() first.")
            vector = self._embedding.encode(raw_text)

            # 3. 语义去重
            if not self.semantic_cache.push(raw_text, vector):
                return False

            # 4. 加入批量处理缓冲区
            self.process_buffer.push(raw_text, vector)

            # 5. 处理缓冲区
            self._process_buffer()

            # 6. 更新历史和话题
            self.history.append(raw_text)
            self._track_topic(raw_text)

            return True

    def _process_buffer(self):
        """处理缓冲区中的内容"""
        if not self.process_buffer.should_process():
            return

        batch_items = self.process_buffer.get_batch()
        if not batch_items:
            return

        batch_text = [item["text"] for item in batch_items]

        # 调用 LLM 提纯（仅处理有意义的文本，忽略过短的消息）
        llm = self._get_llm()
        extracted = []
        for text in batch_text:
            # Skip triple extraction for very short messages
            if llm is not None and len(text) > 50:
                try:
                    triples = llm.extract_triples(text)
                    if isinstance(triples, list) and triples:
                        entities = [{"value": t.get("subject", ""), "type": "subject"}
                                   for t in triples if t.get("subject")]
                        entities += [{"value": t.get("object", ""), "type": "object"}
                                   for t in triples if t.get("object")]
                        extracted.append({
                            "tag": "三元组",
                            "summary": text,  # 使用原始文本作为 summary
                            "entities": entities,
                            "facts": [t.get("subject", "") + "-" + t.get("relation", "") + "-" + t.get("object", "")
                                     for t in triples if t.get("subject") and t.get("relation") and t.get("object")]
                        })
                    else:
                        extracted.append({"tag": "未分类", "summary": text, "entities": [], "facts": []})
                except Exception as e:
                    # Skip triple extraction on error
                    extracted.append({"tag": "未分类", "summary": text, "entities": [], "facts": []})
            else:
                extracted.append({"tag": "未分类", "summary": text, "entities": [], "facts": []})

        # 存储：raw_logs → memory_nodes → relationships
        for i, item in enumerate(batch_items):
            raw_text = item["text"]
            vector = item["vector"].tolist()

            if i < len(extracted):
                data = extracted[i]
                tag = data.get("tag", "未分类")
                summary = data.get("summary", raw_text)
                entities = data.get("entities", [])
                facts = data.get("facts", [])

                # 1. 存储原始日志
                raw_id = self.relation_store.add_raw_log(
                    content=raw_text,
                    source="memory",
                    incremental_density=None  # 可从 EntropyGatekeeper 获取
                )

                # 2. 生成 vector_id (使用 UUID)
                vector_id = str(uuid.uuid4())

                # 3. 存储记忆节点
                memory_node_id = self.relation_store.add_memory_node(
                    raw_id=raw_id,
                    tag=tag,
                    summary=summary,
                    vector_id=vector_id,
                    entities=entities,
                    facts=facts
                )

                # 4. 存入向量库（处理空值）
                metadata = {"tag": tag, "memory_node_id": memory_node_id}
                if entities:
                    metadata["entities"] = entities
                if facts:
                    metadata["facts"] = facts
                self.vector_store.add(
                    ids=[vector_id],
                    vectors=[vector],
                    documents=[summary],
                    metadatas=[metadata]
                )

                # 5. 存储实体关系
                for fact in facts:
                    if isinstance(fact, str) and "-" in fact:
                        parts = fact.split("-", 2)
                        if len(parts) >= 3:
                            subject = parts[0].strip()
                            relation = parts[1].strip()
                            obj = parts[2].strip()
                            if subject and relation and obj:
                                self.relation_store.upsert_relation(
                                    head=subject,
                                    relation=relation,
                                    tail=obj,
                                    ref_id=raw_id
                                )

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
        results = self.retriever.retrieve(query_text, top_n=top_n)

        # 获取原始数据
        raw_logs = []
        for fact in results.get("facts", []):
            memory_node_id = fact.get("id")
            if memory_node_id:
                memory_node = self.relation_store.get_memory(memory_node_id)
                if memory_node and memory_node.get("raw_id"):
                    raw_log = self.relation_store.get_raw_log(memory_node["raw_id"])
                    if raw_log:
                        raw_logs.append({
                            "memory_node_id": memory_node_id,
                            "raw_log_id": raw_log["id"],
                            "content": raw_log["content"],
                            "source": raw_log.get("source"),
                            "timestamp": raw_log.get("timestamp")
                        })

        results["raw_logs"] = raw_logs

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
        else:
            results["type"] = "direct_retrieval"

        return results

    def shutdown(self):
        """关闭系统"""
        pass
