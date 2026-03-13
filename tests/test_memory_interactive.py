"""Memory System 交互式测试"""
import os
import sys
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superbot.memory.facade.memory_system import MemorySystem
from superbot.memory.config import Config

# 使用 MiniMax 作为 LLM
from superbot.providers.minimax_provider import MiniMaxProvider

# 使用本地 embedding
from superbot.agent.memory_providers import SuperbotEmbeddingAdapter


def create_memory_system() -> MemorySystem:
    """创建 MemorySystem 实例"""
    import shutil
    data_dir = "./data/test_memory"
    # 测试开始前清空存储记录
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"[System]: 已清空数据目录: {data_dir}")

    config = Config()
    # 减少 buffer 阈值以便快速测试
    config.process_buffer_count = 30
    config.process_buffer_size = 100

    # 使用 MiniMax 作为 LLM provider
    llm = MiniMaxProvider(
        api_key="sk-cp-aqz5l61MF4GvW6_0r181IMrOAeziU1WcDtYJV4V0htJulfc00Ik4JdVPm8_IRN8eoAPC-YYm7k_6etJ1A7zM0W-fcvcY2Y7StAJR4214cRLb9slXN1pyCfM",
        default_model="MiniMax-M2.5",
    )

    # 使用本地 sentence-transformers 作为 embedding provider
    embedding = SuperbotEmbeddingAdapter(
        model_path="/Users/heyunpeng/workstation/src/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    # 创建 MemorySystem
    memory_system = MemorySystem(
        data_dir="./data/test_memory",
        config=config
    )

    # 注入 provider
    memory_system.set_llm(llm)
    memory_system.set_embedding(embedding)

    return memory_system


def build_prompt(user_input: str, memory_context: str) -> str:
    """构建 LLM 提示词"""
    prompt = f"""你是一个友好的AI助手。请根据以下记忆上下文来回答用户的问题。

记忆上下文:
{memory_context}

用户问题: {user_input}

请回答:"""
    return prompt


def chat_loop():
    """交互式聊天循环"""
    print("=" * 50)
    print("Memory System 交互式测试")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除历史")
    print("输入 'context' 查看记忆上下文")
    print("输入 'summary' 查看当前摘要")
    print("输入 'db' 查看数据库存储内容")
    print("=" * 50)

    memory = create_memory_system()
    llm = memory._get_llm()  # 获取 LLM 实例

    while True:
        try:
            user_input = input("\n[You]: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("再见!")
                break

            if user_input.lower() == "clear":
                memory.history.clear()
                print("历史已清除")
                continue

            if user_input.lower() == "context":
                context = memory.get_memory_context(user_input)
                print(f"\n[Memory Context]:\n{context}")
                continue

            if user_input.lower() == "summary":
                summary = memory._latest_summary
                print(f"\n[Current Summary]: {summary}")
                continue

            if user_input.lower() == "db":
                # 查看数据库存储内容
                print("\n" + "=" * 50)
                print("数据库存储内容:")
                print("=" * 50)

                # 检测向量维度
                try:
                    # 获取 embedding 维度
                    test_vec = memory._embedding.encode("test").tolist()
                    vec_dim = len(test_vec)
                except:
                    vec_dim = 768  # 默认

                # 查看 user_actions collection
                try:
                    actions_results = memory.vector_store.search(
                        query_vector=[0] * vec_dim,  # 任意向量，只获取列表
                        n=100,
                        collection="user_actions"
                    )
                    print(f"\n[user_actions] 共 {len(actions_results)} 条记录:")
                    for i, r in enumerate(actions_results[:10]):
                        action_obj = memory.relation_store.get_action_object(r.get("id", ""))
                        if action_obj:
                            print(f"  {i+1}. ID: {r.get('id', '')[:20]}...")
                            print(f"     三元组: {action_obj.get('subject', '')} | {action_obj.get('relation', '')} | {action_obj.get('object', '')}")
                except Exception as e:
                    print(f"  读取 user_actions 失败: {e}")

                # 查看 query_summary collection
                try:
                    summary_results = memory.vector_store.search(
                        query_vector=[0] * vec_dim,
                        n=100,
                        collection="query_summary"
                    )
                    print(f"\n[query_summary] 共 {len(summary_results)} 条记录:")
                    for i, r in enumerate(summary_results[:5]):
                        doc = r.get("document", "")[:100]
                        print(f"  {i+1}. {doc}...")
                except Exception as e:
                    print(f"  读取 query_summary 失败: {e}")

                # 查看 process_buffer 内容
                try:
                    buffer_items = memory.process_buffer.get_batch()
                    print(f"\n[process_buffer] 共 {len(buffer_items)} 条记录:")
                    for i, item in enumerate(buffer_items):
                        src = item.get("source", "USER")
                        txt = item.get("text", "")[:60]
                        print(f"  {i+1}. [{src}]: {txt}...")
                except Exception as e:
                    print(f"  读取 process_buffer 失败: {e}")

                # 查看 history 内容
                try:
                    history_items = memory.history.get_batch()
                    print(f"\n[history] 共 {len(history_items)} 条记录:")
                    for i, item in enumerate(history_items):
                        src = item.get("source", "USER")
                        txt = item.get("text", "")[:60]
                        print(f"  {i+1}. [{src}]: {txt}...")
                except Exception as e:
                    print(f"  读取 history 失败: {e}")

                print("\n" + "=" * 50)
                continue

            if not user_input:
                continue

            # 2. 获取记忆上下文
            memory_context = memory.get_memory_context(user_input)

            # 3. 调用 LLM 生成回复
            print(f"[System]: Generating response...")
            prompt = build_prompt(user_input, memory_context)
            # MiniMaxProvider.chat 是 async 方法
            response = asyncio.run(llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            ))
            response = response.content

            print(f"\n[Assistant]: {response}")

            # 4. 调用 remember 记住助手回复
            print(f"[System]: Remembering assistant response...")
            memory.remember(response)

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    chat_loop()
