"""Memory System 测试用例 - remember_2 和 recall_2"""
import os
import sys
import time
import json
import pytest
import warnings

# 过滤第三方库警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superbot.memory.config import Config
from superbot.memory.facade.memory_system import MemorySystem
from superbot.memory.storage.vector_store import VectorStore
from superbot.memory.storage.relation_store import RelationStore

# 配置日志 - 启用 DEBUG 级别
os.environ["LOGURU_LEVEL"] = "DEBUG"


def get_real_llm():
    """获取真实的 LLM provider (MLX)"""
    from superbot.memory.models.providers import LocalMLXProvider
    from mlx_lm import load

    model_path = "/Users/heyunpeng/workstation/src/MLX-Qwen3.5-4B-Claude-4.6-Opus"
    model, tokenizer = load(model_path)

    return LocalMLXProvider(model, tokenizer)


@pytest.fixture
def memory_system():
    """创建测试用的 MemorySystem"""
    return create_memory_system(use_real_llm=False)


def create_memory_system(use_real_llm: bool = False):
    """创建 MemorySystem 实例

    参数:
        use_real_llm: 是否使用真实的 LLM
    """
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

    # 设置 LLM provider
    if use_real_llm:
        print("使用真实 LLM (MLX)")
        ms._llm = get_real_llm()
    else:
        ms._llm = MockLLMProvider()

    return ms


class MockLLMProvider:
    """Mock LLM Provider for testing"""

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        # 简单提取三元组
        if "三元组" in prompt:
            # 从 prompt 中提取用户输入的原始文本
            # prompt 格式: ...提取三元组：{text}...
            import re

            # 尝试提取 "提取三元组：" 后面的内容
            match = re.search(r'提取三元组[：:]\s*(.+?)(?:<|im_end|$)', prompt, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                # 提取最后一个自然段
                text = prompt.split("\n")[-1].strip()

            # 提取实体 - 更健壮的解析
            triples = []

            # 喜欢/爱吃/爱吃等关系
            like_match = re.search(r'喜欢(吃)?([^，,\s]+)', text)
            if like_match:
                obj = like_match.group(2) or ""
                if obj:
                    triples.append({"s": "我", "r": "喜欢", "o": obj})

            # 讨厌/不喜欢
            hate_match = re.search(r'讨肦|不喜欢|讨厌([^，,\s]+)', text)
            if hate_match:
                obj = hate_match.group(1) or ""
                if obj:
                    triples.append({"s": "我", "r": "讨厌", "o": obj})

            # 名字相关
            if "张三" in text or "李四" in text or "名字" in text or "我叫" in text:
                name_match = re.search(r'叫([^，,\s]+)', text)
                if name_match:
                    triples.append({"s": "我", "r": "叫", "o": name_match.group(1)})

            # 老婆/妻子
            if "老婆" in text or "妻子" in text:
                name_match = re.search(r'老婆[叫是]([^，,\s]+)', text)
                if not name_match:
                    name_match = re.search(r'妻子[叫是]([^，,\s]+)', text)
                if name_match:
                    triples.append({"s": "我", "r": "妻子是", "o": name_match.group(1)})

            # 职业
            if "医生" in text:
                triples.append({"s": "我老婆", "r": "是", "o": "医生"})

            if "工程师" in text:
                triples.append({"s": "我", "r": "是", "o": "工程师"})

            if "设计师" in text:
                triples.append({"s": "我", "r": "是", "o": "设计师"})

            # 地点
            if "上海" in text or "北京" in text or "深圳" in text:
                loc_match = re.search(r'住在([^，,\s]+)', text)
                if loc_match:
                    triples.append({"s": "我们", "r": "住在", "o": loc_match.group(1)})

            summary = f"用户信息：{text[:80]}"
            return f"摘要：{summary}\n三元组：{json.dumps(triples)}"

        return "测试回复"


def test_remember_2(memory_system, interactive: bool = False):
    """测试 remember_2 方法

    参数:
        memory_system: MemorySystem 实例
        interactive: 是否启用交互模式
    """
    print("=" * 60)
    print("测试 remember_2")
    print("=" * 60)

    ms = memory_system

    if interactive:
        # 交互模式
        print("\n--- 交互模式 (输入 q 退出) ---")
        while True:
            try:
                text = input("\n请输入 raw_text: ").strip()
                if not text:
                    continue
                if text.lower() in ['q', 'quit', 'exit', '退出']:
                    print("退出交互模式")
                    break

                result = ms.remember_2(text)
                print(f"摘要: {result}")

                # 显示统计
                conn = ms.relation_store._get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM action_objects")
                count = cursor.fetchone()[0]
                print(f"当前 action_objects 数量: {count}")
                conn.close()

            except KeyboardInterrupt:
                print("\n退出交互模式")
                break
            except Exception as e:
                print(f"错误: {e}")
        return ms

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

    return ms


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


def cleanup():
    """清理测试数据"""
    import shutil
    if os.path.exists("./data/test_chroma"):
        shutil.rmtree("./data/test_chroma")
    if os.path.exists("./data/test_memory.db"):
        os.remove("./data/test_memory.db")
    print("清理完成")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Memory System 测试")
    parser.add_argument("--interactive", "-i", action="store_true", help="启用交互模式")
    parser.add_argument("--cleanup", "-c", action="store_true", help="清理测试数据后退出")
    parser.add_argument("--real-llm", "-r", action="store_true", help="使用真实 LLM (MLX)")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    print("Memory System 测试")
    print("=" * 60)

    # 清理旧数据
    cleanup()

    # 初始化 MemorySystem
    ms = create_memory_system(use_real_llm=args.real_llm)

    if args.interactive:
        # 交互模式 - 测试 remember_2
        test_remember_2(ms, interactive=True)
    else:
        # 测试 remember_2
        test_remember_2(ms)

        # 测试 recall_2
        test_recall_2(ms)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
