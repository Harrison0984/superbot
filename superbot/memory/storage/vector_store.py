"""向量存储模块：使用 ChromaDB 存储和检索向量"""
from typing import List, Dict, Any, Optional
import chromadb


class VectorStore:
    """向量存储：使用 ChromaDB"""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "memmory"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """添加向量"""
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

    def search(
        self,
        query_vector: List[float],
        n: int = 5,
        filter_: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where=filter_
        )

        # 格式化返回结果
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return formatted

    def delete(self, ids: List[str]):
        """删除向量"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """返回向量数量"""
        return self.collection.count()
