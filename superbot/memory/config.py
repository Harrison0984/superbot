"""记忆系统配置"""
from pydantic import BaseModel


class Config(BaseModel):
    """记忆系统配置"""

    # ==================== 物理过滤缓冲区 ====================
    # EntropyGatekeeper 使用，用于过滤低信息量内容

    entropy_buffer_count: int = 1
    """
    物理过滤缓冲区数量

    含义: 用于计算增量密度的历史窗口大小（条数）
    - 值越大: 考虑更多历史内容，检测长期重复
    - 值越小: 只考虑近期内容
    """

    entropy_buffer_size: int = 64 * 1024
    """
    物理过滤缓冲区大小 (字节)

    含义: 用于计算增量密度的历史窗口大小（字节）
    - 值越大: 考虑更多历史内容
    - 值越小: 只考虑近期内容
    """

    entropy_threshold: float = 0.4
    """
    增量密度阈值

    含义: ID Score > threshold 时拒绝内容
    范围: [0, 1]
    - 值越小(接近0): 过滤越严格，只接受高信息密度内容
    - 值越大(接近1): 过滤越宽松，接受更多内容
    """

    # ==================== 语义去重缓冲区 ====================
    # CacheBuffer 使用，用于检测语义重复

    semantic_buffer_count: int = 1
    """
    语义去重缓冲区数量

    含义: 用于语义去重的缓冲区最大条目数
    - 值越大: 检测更多历史条目的语义重复
    - 值越小: 只检测近期的语义重复
    """

    semantic_threshold: float = 0.95
    """
    语义相似度阈值

    含义: 相似度 >= threshold 时视为重复
    范围: [0, 1]
    - 值越大(接近1): 相似度要求越高
    - 值越小: 允许更多语义相似内容
    """

    # ==================== 批量处理缓冲区 ====================
    # CacheBuffer 使用，用于批量提交给 LLM 提纯

    process_buffer_count: int = 1
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

    # ==================== 检索配置 ====================

    retrieval_w_sim: float = 0.7
    """
    语义相似度权重

    unified_score = w_sim * similarity + w_time * time_decay
    """

    retrieval_w_time: float = 0.3
    """
    时间衰减权重
    """

    retrieval_alpha: float = 0.05
    """
    时间衰减系数

    time_decay = 1 / (1 + alpha * days)
    """

    retrieval_rrf_k: int = 60
    """
    RRF 常数

    rrf_score = 1 / (k + rank)
    """

    # ==================== Action-Metadata Extraction Config ====================

    action_metadata_prompt: str = ""
    """
    动作-元数据提取的提示词模板

    占位符:
    - {text}: 待提取的文本
    - {context}: 对话上下文 (可选)
    - {example}: 示例
    """
    action_metadata_example: str = ""
    """
    动作-元数据提取的示例
    """

    # ==================== 摘要+三元组提取配置 ====================

    summary_triples_prompt: str = '''<|im_start|>system
你是一个严格的事实提取助手。只提取用户输入中明确提到的信息，不要添加任何额外信息。
如果输入中没有提及的信息，绝对不要编造。

规则：
1. 只提取输入文本中明确存在的实体和关系
2. 不要解释、扩展或补充任何不存在的内容
3. 如果不确定或信息不足，返回空三元组

输出格式（必须严格按格式）：
摘要：[不超过50字的摘要，只描述输入中明确存在的信息]
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


# 全局默认配置
config = Config()
