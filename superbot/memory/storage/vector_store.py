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
        # 缓存其他集合
        self._collections: Dict[str, chromadb.Collection] = {}

    def get_collection(self, name: str) -> chromadb.Collection:
        """获取指定名称的集合"""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection: Optional[str] = None
    ):
        """添加向量"""
        if collection:
            coll = self.get_collection(collection)
            coll.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas
            )
        else:
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
        filter_: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if collection:
            coll = self.get_collection(collection)
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=n,
                where=filter_
            )
        else:
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
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else None
                })

        return formatted

    def delete(self, ids: List[str], collection: Optional[str] = None):
        """删除向量"""
        if collection:
            coll = self.get_collection(collection)
            coll.delete(ids=ids)
        else:
            self.collection.delete(ids=ids)

    def count(self, collection: Optional[str] = None) -> int:
        """返回向量数量"""
        if collection:
            return self.get_collection(collection).count()
        return self.collection.count()
