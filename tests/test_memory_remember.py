"""Memory System 测试用例 - remember_2 和 recall_2"""
import os
import sys
import time
import json
import pytest

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superbot.memory.config import Config
from superbot.memory.facade.memory_system import MemorySystem
from superbot.memory.storage.vector_store import VectorStore
from superbot.memory.storage.relation_store import RelationStore


@pytest.fixture
def memory_system():
    """创建测试用的 MemorySystem"""
    # 初始化配置
    config = Config()
    config.chroma_persist_dir = "./data/test_chroma"
    config.sqlite_path = "./data/test_memory.db"

    # 创建 MemorySystem
    ms = MemorySystem(config=config, data_dir="./data/test")

    # 设置 embedding provider
    from superbot.agent.memory_providers import SuperbotEmbeddingAdapter
    embedding = SuperbotEmbeddingAdapter(
        model_path="/Users/heyunpeng/workstation/src/nomic-embed-text-v1.5",
        trust_remote_code=True
    )
    ms.set_embedding(embedding)

    # 设置 LLM provider (mock)
    ms._llm = MockLLMProvider()

    return ms


class MockLLMProvider:
    """Mock LLM Provider for testing"""

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        # 简单提取三元组
        if "三元组" in prompt:
            # 从 prompt 中提取关键信息
            lines = prompt.split("\n")
            text = ""
            for line in lines:
                if "提取三元组" in line or "1." in line:
                    text = line.replace("提取三元组：", "").replace("1. 压缩成简短摘要", "").strip()
                    break

            if not text:
                for line in lines:
                    if line.strip() and not line.startswith("<") and not line.startswith("1"):
                        text = line.strip()
                        break

            # 提取实体
            triples = []
            if "我叫" in text:
                triples.append({"s": "我", "r": "叫", "o": text.split("叫")[1].split("，")[0].split("。")[0].strip()})
            if "老婆" in text or "妻子" in text:
                parts = text.split("，")
                for p in parts:
                    if "叫" in p:
                        name = p.split("叫")[-1].strip()
                        triples.append({"s": "我", "r": "妻子是", "o": name})
            if "住在" in text or "住" in text:
                parts = text.split("，")
                for p in parts:
                    if "住" in p:
                        loc = p.replace("住在", "").replace("住", "").strip()
                        triples.append({"s": "我们", "r": "住在", "o": loc})

            summary = text[:50] if text else "摘要测试"
            return f"摘要：{summary}\n三元组：{json.dumps(triples)}"

        return "测试回复"


def test_remember_2(memory_system):
    """测试 remember_2 方法"""
    print("=" * 60)
    print("测试 remember_2")
    print("=" * 60)

    ms = memory_system

    # 测试用例
    test_texts = [
        "我叫张三，是一名工程师",
        "我老婆叫李梅，是一名医生",
        "我们住在上海",
    ]

    print("\n--- 写入数据 ---")
    for text in test_texts:
        result = ms.remember_2(text)
        print(f"输入: {text}")
        print(f"摘要: {result}")
        print()

    # 检查数据库
    print("\n--- raw_logs 表 ---")
    conn = ms.relation_store._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM raw_logs")
    rows = cursor.fetchall()
    for row in rows:
        print(f"id: {row[0]}, content: {row[1][:50]}...")
    conn.close()

    print("\n--- action_objects 表 ---")
    conn = ms.relation_store._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, subject, relation, object FROM action_objects")
    rows = cursor.fetchall()
    for row in rows:
        print(f"id: {row[0]}, ({row[1]}, {row[2]}, {row[3]})")
    conn.close()

    print("\n--- user_actions 集合 ---")
    count = ms.vector_store.count(collection="user_actions")
    print(f"向量数量: {count}")


def test_recall_2(memory_system: MemorySystem):
    """测试 recall_2 方法"""
    print("\n" + "=" * 60)
    print("测试 recall_2")
    print("=" * 60)

    # 测试用例
    test_cases = [
        ("我老婆是谁", "张三的妻子是李梅"),
        ("我叫什么", "我叫张三"),
        ("住在哪", "住在上海"),
    ]

    for query, summary in test_cases:
        print(f"\n查询: {query}")
        print(f"摘要: {summary}")

        result = memory_system.recall_2(query, summary, similarity_threshold=0.85)

        print(f"使用的摘要: {result.get('summary', '')}")
        print(f"三元组数量: {len(result.get('triples', []))}")

        for t in result.get("triples", []):
            print(f"  - {t.get('triple_text')} (相似度: {t.get('similarity', 0):.4f})")


def test_query_summary_collection(memory_system: MemorySystem):
    """测试 query_summary 集合"""
    print("\n" + "=" * 60)
    print("测试 query_summary 集合")
    print("=" * 60)

    # 写入一些 query_summary
    test_queries = [
        ("我老婆是谁", "张三的妻子是李梅是一名医生"),
        ("我叫什么", "我叫张三是一名工程师"),
    ]

    for query, summary in test_queries:
        print(f"\n写入: query={query}, summary={summary}")
        result = memory_system.recall_2(query, summary)
        print(f"使用的摘要: {result.get('summary', '')}")

    # 检查集合
    print("\n--- query_summary 集合 ---")
    count = memory_system.vector_store.count(collection="query_summary")
    print(f"向量数量: {count}")


def cleanup():
    """清理测试数据"""
    import shutil
    if os.path.exists("./data/test_chroma"):
        shutil.rmtree("./data/test_chroma")
    if os.path.exists("./data/test_memory.db"):
        os.remove("./data/test_memory.db")
    print("清理完成")


def main():
    print("Memory System 测试")
    print("=" * 60)

    # 清理旧数据
    cleanup()

    # 测试 remember_2
    ms = test_remember_2()

    # 测试 recall_2
    test_recall_2(ms)

    # 测试 query_summary 集合
    test_query_summary_collection(ms)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
