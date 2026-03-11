"""增强版检索模块：使用 RRF 融合 + 时间衰减 + 自动实体提取"""
import logging
from typing import List, Dict, Any, Optional

from superbot.memory.storage import VectorStore, EnhancedRelationStore

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """增强版检索器：RRF 融合 + 时间衰减 + 自动实体提取"""

    def __init__(
        self,
        vector_store: VectorStore,
        relation_store: EnhancedRelationStore,
        embedding_provider,
        w_sim: float = 0.7,
        w_time: float = 0.3,
        alpha: float = 0.05,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.relation_store = relation_store
        self.embedding_provider = embedding_provider
        self.w_sim = w_sim
        self.w_time = w_time
        self.alpha = alpha
        self.rrf_k = rrf_k

    def _get_model(self):
        """获取 embedding 模型"""
        if self.embedding_provider is None:
            raise RuntimeError(
                "Embedding provider not set. "
                "Please call set_embedding() on MemorySystem first."
            )
        return self.embedding_provider

    def _extract_entities_from_metadata(self, vector_results: List[Dict]) -> List[str]:
        """从向量检索结果的 metadata 中自动提取实体"""
        import json
        entities = set()
        for result in vector_results:
            metadata = result.get("metadata", {})
            # 从 entities 字段提取
            if "entities" in metadata:
                entities_list = metadata["entities"]
                # Handle both JSON string and list formats (for backward compatibility)
                if isinstance(entities_list, str):
                    try:
                        entities_list = json.loads(entities_list)
                    except (json.JSONDecodeError, TypeError):
                        entities_list = []
                if isinstance(entities_list, list):
                    for entity in entities_list:
                        if isinstance(entity, dict):
                            value = entity.get("value", "")
                            if value:
                                entities.add(value)
                        elif isinstance(entity, str) and entity:
                            entities.add(entity)
        return list(entities)

    def retrieve(
        self,
        query_text: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        检索记忆

        参数:
            query_text: 查询文本
            top_n: 返回数量

        返回:
            包含 facts 和 relations 的字典
        """
        logger.debug("[Retriever] retrieve() query: {}, top_n: {}", query_text[:100], top_n)

        # 1. 向量检索
        model = self._get_model()
        query_vector = model.encode(query_text).tolist()

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            n=top_n * 2  # 多取一些用于后续筛选
        )
        logger.debug("[Retriever] vector_store.search() returned {} results", len(vector_results))

        # 2. 自动从检索结果中提取实体
        query_entities = self._extract_entities_from_metadata(vector_results)

        # 构建 vector_hits: {id: score}
        vector_hits = {}
        for result in vector_results:
            doc_id = result["id"]
            # ChromaDB 返回 distance，转换为相似度
            score = 1 - (result.get("distance") or 0)
            vector_hits[doc_id] = score

        # 3. SQL 增强路 (RRF 融合)
        sql_ranked_ids = self._get_sql_recall_ranked(vector_hits)

        # 4. Get max memory_node_id for recency boost
        max_node_id = 0
        # Build lookup dict for O(1) access instead of O(n) search
        node_id_map = {}
        for result in vector_results:
            doc_id = result["id"]
            metadata = result.get("metadata", {})
            node_id = metadata.get("memory_node_id", 0)
            node_id_map[doc_id] = node_id
            try:
                max_node_id = max(max_node_id, int(node_id))
            except (ValueError, TypeError):
                pass

        # 5. RRF 融合 + Recency Boost
        rrf_scores = {}
        recency_boost = 1.2  # Recent items get 20% boost

        # 向量路排名
        vector_ranked_ids = sorted(vector_hits.keys(), key=lambda x: vector_hits[x], reverse=True)
        for rank, v_id in enumerate(vector_ranked_ids, start=1):
            # Calculate recency boost using lookup dict
            recency_factor = 1.0
            if max_node_id > 0:
                node_id = node_id_map.get(v_id, 0)
                try:
                    node_id_int = int(node_id)
                    # Normalize: newest gets boost, oldest gets 1.0
                    recency_factor = 1.0 + (recency_boost - 1.0) * (node_id_int / max_node_id)
                except (ValueError, TypeError):
                    pass

            rrf_scores[v_id] = rrf_scores.get(v_id, 0) + (1 / (self.rrf_k + rank)) * recency_factor

        # SQL 路排名
        for rank, v_id in enumerate(sql_ranked_ids, start=1):
            rrf_scores[v_id] = rrf_scores.get(v_id, 0) + 1 / (self.rrf_k + rank)

        # 5. 排序取 Top N
        final_ids = [item[0] for item in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

        # 6. 获取完整数据
        facts = []
        if final_ids:
            for doc_id in final_ids:
                for result in vector_results:
                    if result["id"] == doc_id:
                        metadata = result.get("metadata", {})
                        # 从 metadata 获取 memory_node_id
                        memory_node_id = metadata.get("memory_node_id", doc_id)
                        facts.append({
                            "id": memory_node_id,
                            "vector_id": doc_id,
                            "content": result.get("document", ""),
                            "metadata": metadata,
                            "score": vector_hits.get(doc_id, 0)
                        })
                        break

        # 7. 关系路召回（基于自动提取的实体）
        relations = self._relation_search(query_entities)
        logger.debug("[Retriever] _relation_search() returned {} relations", len(relations))

        result = {
            "facts": facts,
            "relations": relations
        }
        logger.debug("[Retriever] retrieve() returning {} facts, {} relations", len(facts), len(relations))
        return result

    def _get_sql_recall_ranked(self, vector_hits: Dict[str, float]) -> List[str]:
        """SQL 增强召回"""
        if not vector_hits:
            return []

        conn = self.relation_store._get_conn()
        cursor = conn.cursor()

        ids = list(vector_hits.keys())

        # 构建 CASE 语句
        sim_cases = " ".join([
            f"WHEN id='{vid}' THEN {score}"
            for vid, score in vector_hits.items()
        ])

        sql = f"""
        WITH WeightedRecall AS (
            SELECT
                id,
                vector_id,
                tag,
                timestamp,
                ({self.w_sim} * (CASE {sim_cases} ELSE 0 END) +
                 {self.w_time} * (1.0 / (1.0 + {self.alpha} * (julianday('now') - julianday(timestamp))))
                ) AS unified_score
            FROM memory_nodes
            WHERE vector_id IN ({",".join(['?' for _ in ids])})
        ),
        RankedByTag AS (
            SELECT vector_id, unified_score,
                   ROW_NUMBER() OVER (PARTITION BY tag ORDER BY unified_score DESC) as tag_rank
            FROM WeightedRecall
        )
        SELECT vector_id FROM RankedByTag
        WHERE tag_rank <= 2
        ORDER BY unified_score DESC;
        """

        try:
            cursor.execute(sql, ids)
            result = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"SQL 增强召回失败: {e}")
            result = []
        finally:
            conn.close()

        return result

    def _relation_search(self, query_entities: List[str]) -> List[Dict[str, Any]]:
        """
        关系路召回（基于实体）

        使用实体关系表 (head - relation - tail) 进行查询
        使用 IN 查询一次获取所有实体的关系
        """
        if not query_entities:
            return []

        conn = self.relation_store._get_conn()
        cursor = conn.cursor()

        results = []
        try:
            # 使用 IN 查询一次获取所有实体的关系
            placeholders = ','.join(['?' for _ in query_entities])
            sql = f"""
                SELECT id, head, relation, tail, last_seen, ref_id
                FROM relationships
                WHERE head IN ({placeholders}) OR tail IN ({placeholders})
                ORDER BY id DESC
                LIMIT 20;
            """
            cursor.execute(sql, query_entities + query_entities)
            rows = cursor.fetchall()

            for row in rows:
                results.append({
                    "id": row[0],
                    "head": row[1],
                    "relation": row[2],
                    "tail": row[3],
                    "last_seen": row[4],
                    "ref_id": row[5]
                })

            logger.debug(f"关系查询: entities={query_entities}, found={len(results)}")

        except Exception as e:
            logger.warning(f"关系路召回失败: {e}")
        finally:
            conn.close()

        return results
