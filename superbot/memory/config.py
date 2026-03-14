"""记忆系统配置"""
from pydantic import BaseModel


class Config(BaseModel):
    """记忆系统配置"""

    # ==================== 对话历史缓冲区 ====================
    # FIFOBuffer 使用，用于存储对话历史

    history_buffer_count: int = 3
    """
    对话历史缓冲区数量

    含义: 保留的历史条目数
    - 值越大: 保留更多历史记录
    - 值越小: 只保留近期记录
    """

    history_buffer_size: int = 2 * 1024
    """
    对话历史缓冲区大小 (字节)

    含义: 保留的历史字节数
    - 值越大: 保留更多历史内容
    - 值越小: 只保留近期内容
    """

    # ==================== 批量处理缓冲区 ====================
    # CacheBuffer 使用，用于批量提交给 LLM 提纯

    process_buffer_count: int = 30
    """
    批量处理缓冲区数量

    含义: 达到此数量时触发 LLM 处理
    - 值越大: 批量处理更多内容，减少 LLM 调用
    - 值越小: 更及时处理，但增加 LLM 调用
    """

    process_buffer_size: int = 2048
    """
    批量处理缓冲区大小 (字节)

    含义: 达到此大小时触发 LLM 处理
    - 值越大: 积累更多内容，减少 LLM 调用
    - 值越小: 更及时处理，但增加 LLM 调用
    """

    # ==================== 存储配置 ====================

    data_dir: str = "./data"
    """
    数据根目录
    """

    chroma_persist_dir: str = "./data/chroma"
    """
    ChromaDB 持久化目录
    """

    sqlite_path: str = "./data/memmory.db"
    """
    SQLite 数据库路径
    """

    # ==================== 摘要+三元组提取配置 ====================

    summary_triples_prompt: str = '''<|im_start|>system
你是一个严格的事实提取助手。只提取用户输入中明确提到的信息，不要添加任何额外信息。
如果输入中没有提及的信息，绝对不要编造。

对话角色定义：
- USER（用户）= 我
- Assistant（AI）= 你
- 第三方身份保持原样（如人名、公司名等）

规则：
1. 只提取输入文本中明确存在的实体和关系
2. 不要解释、扩展或补充任何不存在的内容
3. 如果不确定或信息不足，返回空三元组
4. 三元组中的主语和宾语：用户相关用"我"，AI相关用"你"，第三方保持原样

输出格式（必须严格按格式）：
摘要：[不超过200字的摘要，只描述输入中明确存在的信息]
三元组：[{{"s":"主语","r":"关系","o":"宾语"}}]，如果无信息则输出[]
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
摘要：'''
    """
    摘要+三元组提取的提示词模板（V2 强约束版本，减少幻觉）

    占位符:
    - {text}: 待提取的文本
    """
    summary_triples_max_tokens: int = 2048
    """
    摘要+三元组提取的最大 token 数
    """

    # ==================== 反思配置 ====================

    reflection_timeout: int = 3600
    """
    反思触发空闲超时时间（秒）

    含义：系统空闲多长时间后触发反思
    - 默认 1 小时（3600秒）
    """

    reflection_prompt: str = '''<|im_start|>system
你是一个记忆分析助手。请对以下三元组进行逻辑评分（0-1分），评估其准确性和有效性：
- 0分：无意义、重复或错误的三元组
- 0.5分：基本有效但不够准确
- 1分：完全正确且有价值

请返回 JSON 格式的评分结果：
[{"id": "xxx", "s": "主语", "r": "关系", "o": "宾语", "score": 0.85}, ...]
<|im_end|>
<|im_start|>user
三元组列表：
{triples}
<|im_end|>
<|im_start|>assistant
'''
    """
    三元组评分反思提示词模板

    占位符:
    - {triples}: 三元组列表 JSON
    """

    reflection_summary_prompt: str = '''<|im_start|>system
你是一个记忆分析助手。请对以下摘要进行逻辑评分（0-1分），评估其准确性和有效性：
- 0分：无意义、重复或错误的摘要
- 0.5分：基本有效但不够准确
- 1分：完全正确且有价值

请返回 JSON 格式的评分结果：
[{"id": "xxx", "summary": "摘要内容", "score": 0.85}, ...]
<|im_end|>
<|im_start|>user
摘要列表：
{summaries}
<|im_end|>
<|im_start|>assistant
'''
    """
    摘要评分反思提示词模板

    占位符:
    - {summaries}: 摘要列表 JSON
    """

    reflection_insights_prompt: str = '''<|im_start|>system
你是一个记忆分析助手。请根据以下新数据和已有用户画像，更新用户特点描述：

1. 参考已有用户画像
2. 根据新数据更新用户特点
3. 如果没有新变化，可以保留原有内容

请返回 JSON 格式：
{
  "user_profile": "更新后的用户特点描述"
}
<|im_end|>
<|im_start|>user
已有用户画像：
{existing_profile}

---
新数据摘要：
{summaries}

---
新数据三元组：
{triples}
<|im_end|>
<|im_start|>assistant
'''
    """
    用户洞察反思提示词模板

    占位符:
    - {existing_profile}: 已有用户画像
    - {summaries}: 摘要列表
    - {triples}: 三元组列表
    """


# 全局默认配置
config = Config()
