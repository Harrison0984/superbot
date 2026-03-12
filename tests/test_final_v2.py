"""三元组提取 + 摘要 - 优化最终版 v2 (幻觉优化测试)"""
import time
import sys
import json
import re
import logging
import mlx_lm
from mlx_lm import load

# ==================== 配置 ====================
MODEL_PATH = "/Users/heyunpeng/workstation/src/MLX-Qwen3.5-4B-Claude-4.6-Opus"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载模型一次
logger.info("加载模型...")
model, tokenizer = load(MODEL_PATH)
logger.info("模型加载完成\n")


# ==================== 优化后的 Prompt（减少幻觉） ====================

# Prompt V1: 基础版本（当前使用）
PROMPT_V1 = '''<|im_start|>system
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

# Prompt V2: 更强的约束，减少幻觉
PROMPT_V2 = '''<|im_start|>system
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

# Prompt V3: 更简洁的格式
PROMPT_V3 = '''<|im_start|>system
从用户输入中提取事实三元组。只输出明确提到的内容，不添加任何信息。
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
['''


def extract_with_prompt(text: str, prompt_template: str, max_tokens: int = 512) -> tuple:
    """使用指定 prompt 提取摘要和三元组"""
    prompt = prompt_template.format(text=text)

    start_time = time.time()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    elapsed = time.time() - start_time

    # 解析响应
    summary = ""
    triples = []

    # 清理思考过程
    if '<think>' in response:
        clean_response = response.split('<think>')[-1].strip()
    else:
        clean_response = response

    # 尝试匹配 "摘要：xxx 三元组"
    summary_match = re.search(r'摘要[：:]\s*(.+?)\s*三元组', clean_response, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()

    # 提取三元组
    triple_match = re.search(r'三元组[：:]?\s*(\[.*?\])', clean_response, re.DOTALL)
    if triple_match:
        try:
            triples = json.loads(triple_match.group(1))
        except:
            try:
                objs = re.findall(r'\{[^{}]*\}', triple_match.group(1))
                for o in objs:
                    try:
                        t = json.loads(o)
                        if 's' in t and 'r' in t:
                            triples.append(t)
                    except:
                        pass
            except:
                pass

    return summary, triples, elapsed, response


def test_prompt_versions():
    """测试不同 prompt 版本的效果"""
    test_texts = [
        "我叫包子",
        "我爱吃葡萄",
        "我老婆叫花卷，是一名医生",
        "今天天气真好",
        "我老婆叫李梅，是一名医生，我们住在上海",
    ]

    versions = [
        ("V1 (基础)", PROMPT_V1),
        ("V2 (强约束)", PROMPT_V2),
        ("V3 (简洁)", PROMPT_V3),
    ]

    for text in test_texts:
        logger.info("=" * 70)
        logger.info(f"输入: {text}")
        logger.info("=" * 70)

        for name, prompt in versions:
            logger.info(f"\n--- {name} ---")
            summary, triples, elapsed, raw = extract_with_prompt(text, prompt)
            logger.info(f"耗时: {elapsed:.2f}秒")
            logger.info(f"摘要: {summary[:50] if summary else '(无)'}")
            logger.info(f"三元组: {triples}")
            logger.info(f"原始输出: {raw[:200]}...")
        logger.info("")


def test_max_tokens():
    """测试不同 max_tokens 的效果"""
    test_texts = [
        "我爱吃葡萄",
        "我老婆叫花卷，是一名医生",
    ]

    token_values = [256, 512, 1024]

    for text in test_texts:
        logger.info("=" * 70)
        logger.info(f"输入: {text}")
        logger.info("=" * 70)

        for tokens in token_values:
            logger.info(f"\n--- max_tokens={tokens} ---")
            summary, triples, elapsed, raw = extract_with_prompt(text, PROMPT_V2, max_tokens=tokens)
            logger.info(f"耗时: {elapsed:.2f}秒")
            logger.info(f"摘要: {summary[:50] if summary else '(无)'}")
            logger.info(f"三元组: {triples}")


def test_system_prompt():
    """测试不同的 system prompt 效果"""
    prompts = [
        ("无约束", PROMPT_V1),
        ("强约束", PROMPT_V2),
    ]

    # 关键测试用例：容易产生幻觉的输入
    hallucination_tests = [
        "我爱吃葡萄",  # 可能被误解为其他
        "我叫包子",  # 可能被解释为食物
        "今天天气不错",  # 可能产生额外信息
    ]

    logger.info("=" * 70)
    logger.info("幻觉测试")
    logger.info("=" * 70)

    for text in hallucination_tests:
        for name, prompt in prompts:
            summary, triples, elapsed, raw = extract_with_prompt(text, prompt, max_tokens=512)
            logger.info(f"\n输入: {text}")
            logger.info(f"Prompt: {name}")
            logger.info(f"三元组: {triples}")
            if not triples:
                logger.warning(">>> 幻觉: 生成了不存在的三元组!" if triples else ">>> 正确: 无三元组")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prompt 幻觉优化测试")
    parser.add_argument("--test", choices=["prompt", "tokens", "hallucination", "all"], default="all")
    args = parser.parse_args()

    if args.test in ["prompt", "all"]:
        test_prompt_versions()

    if args.test in ["tokens", "all"]:
        test_max_tokens()

    if args.test in ["hallucination", "all"]:
        test_system_prompt()


def extract_triples_only(text: str) -> list:
    """仅提取三元组 - 使用 V2 强约束 prompt"""
    logger.info(f"[仅三元组] 输入文本 ({len(text)}字): {text[:30]}...")

    # 使用 V2 强约束 prompt
    prompt = f'''<|im_start|>system
你是一个严格的事实提取助手。只提取用户输入中明确提到的信息，不要添加任何额外信息。
如果输入中没有提及的信息，绝对不要编造。

规则：
1. 只提取输入文本中明确存在的实体和关系
2. 不要解释、扩展或补充任何不存在的内容
3. 如果不确定或信息不足，返回空三元组

输出格式（必须严格按格式）：
三元组：[{{"s":"主语","r":"关系","o":"宾语"}}]，如果无信息则输出[]
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
['''

    start_time = time.time()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False
    )
    elapsed = time.time() - start_time
    logger.info(f"耗时: {elapsed:.2f}秒")

    triples = []
    try:
        if '</think>' in response:
            response = response.split('</think>')[-1]
        if not response.startswith('['):
            response = '[' + response
        if not response.endswith(']'):
            response = response + ']'
        triples = json.loads(response)
    except:
        try:
            objs = re.findall(r'\{[^{}]*\}', response)
            for o in objs:
                try:
                    t = json.loads(o)
                    if 's' in t and 'r' in t:
                        triples.append(t)
                except:
                    pass
        except:
            pass

    if triples:
        for i, t in enumerate(triples[:3], 1):
            logger.info(f"  三元组 {i}: ({t.get('s', '')}, {t.get('r', '')}, {t.get('o', '')})")
    else:
        logger.warning("未提取到三元组")

    return triples, elapsed


def extract_summary_only(text: str) -> str:
    """仅提取摘要 - 使用 V2 强约束 prompt"""
    logger.info(f"[仅摘要] 输入文本 ({len(text)}字): {text[:30]}...")

    # 使用 V2 强约束 prompt
    prompt = f'''<|im_start|>system
你是一个严格的事实提取助手。只提取用户输入中明确提到的信息，不要添加任何额外信息。
如果输入中没有提及的信息，绝对不要编造。

规则：
1. 只提取输入文本中明确存在的信息
2. 不要解释、扩展或补充任何不存在的内容
3. 如果不确定或信息不足，返回简短说明

输出格式：
摘要：[不超过50字的摘要，只描述输入中明确存在的信息]
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
摘要：'''

    start_time = time.time()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False
    )
    elapsed = time.time() - start_time
    logger.info(f"耗时: {elapsed:.2f}秒")

    summary = response.strip()
    logger.info(f"摘要: {summary[:100]}...")

    return summary, elapsed


def extract_both(text: str) -> tuple:
    """同时提取摘要和三元组"""
    logger.info(f"[同时提取] 输入文本 ({len(text)}字): {text[:30]}...")

    # 优化: 强制跳过思考，直接输出
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

    start_time = time.time()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=512,
        verbose=False
    )
    elapsed = time.time() - start_time
    logger.info(f"耗时: {elapsed:.2f}秒")

    # 解析响应
    summary = ""
    triples = []

    # 提取摘要 - 支持多种格式
    # 格式1: 摘要：xxx 三元组：[...]
    # 格式2: xxx 三元组：[...]
    # 格式3: 思考过程 摘要内容
    summary = ""

    # 先清理思考过程
    clean_response = response
    if '<think>' in response:
        clean_response = response.split('</think>')[-1].strip()

    # 尝试匹配 "摘要：xxx 三元组"
    summary_match = re.search(r'摘要[：:]\s*(.+?)\s*三元组', clean_response, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        # 尝试匹配 "xxx 三元组"（没有"摘要"前缀）
        summary_match = re.search(r'^(.+?)\s*三元组', clean_response, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()

    # 提取三元组
    triple_match = re.search(r'三元组[：:]?\s*(\[.+?\])', response, re.DOTALL)
    if triple_match:
        try:
            triples = json.loads(triple_match.group(1))
        except:
            try:
                objs = re.findall(r'\{[^{}]*\}', triple_match.group(1))
                for o in objs:
                    try:
                        t = json.loads(o)
                        if 's' in t and 'r' in t:
                            triples.append(t)
                    except:
                        pass
            except:
                pass

    logger.info(f"摘要: {summary[:80]}...")
    if triples:
        for i, t in enumerate(triples[:3], 1):
            logger.info(f"  三元组 {i}: ({t.get('s', '')}, {t.get('r', '')}, {t.get('o', '')})")

    return summary, triples, elapsed


def run_test(text: str, mode: str = "both"):
    """运行单次测试"""
    logger.info("=" * 60)
    logger.info(f"模式: {mode}, 输入: {text[:50]}...")

    if mode == "both":
        summary, triples, elapsed = extract_both(text)
        return {"summary": summary, "triples": triples}, elapsed
    elif mode == "triples":
        triples, elapsed = extract_triples_only(text)
        return {"triples": triples}, elapsed
    elif mode == "summary":
        summary, elapsed = extract_summary_only(text)
        return {"summary": summary}, elapsed


def extract_both_with_max_tokens(text: str, max_tokens: int = 512) -> tuple:
    """同时提取摘要和三元组 - 可配置 max_tokens"""
    # 优化: 强制跳过思考，直接输出
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

    start_time = time.time()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    elapsed = time.time() - start_time

    # 解析响应
    summary = ""
    triples = []

    # 先清理思考过程
    clean_response = response
    if '<think>' in response:
        clean_response = response.split('</think>')[-1].strip()

    # 提取摘要
    summary_match = re.search(r'摘要[：:]\s*(.+?)\s*三元组', clean_response, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        summary_match = re.search(r'^(.+?)\s*三元组', clean_response, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()

    # 提取三元组
    triple_match = re.search(r'三元组[：:]?\s*(\[.+?\])', clean_response, re.DOTALL)
    if triple_match:
        try:
            triples = json.loads(triple_match.group(1))
        except:
            try:
                objs = re.findall(r'\{[^{}]*\}', triple_match.group(1))
                for o in objs:
                    try:
                        t = json.loads(o)
                        if 's' in t and 'r' in t:
                            triples.append(t)
                    except:
                        pass
            except:
                pass

    return summary, triples, elapsed


def parameter_test():
    """测试 max_tokens 对性能和结果的影响"""
    test_text = "我老婆叫李梅，她是一名医生。我们住在上海。我叫张三，在腾讯工作。"

    # 测试不同 max_tokens
    token_values = [64, 128, 256, 384, 512]

    logger.info("=" * 60)
    logger.info("参数测试: max_tokens 对性能和结果的影响")
    logger.info("=" * 60)
    logger.info(f"测试文本: {test_text} ({len(test_text)}字)")
    logger.info("=" * 60)

    results = []

    for max_tokens in token_values:
        logger.info(f"\n--- max_tokens={max_tokens} ---")

        # 运行3次取平均
        times = []
        for i in range(3):
            summary, triples, elapsed = extract_both_with_max_tokens(test_text, max_tokens)
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        # 打印结果
        logger.info(f"耗时: {avg_time:.2f}秒")
        logger.info(f"摘要: {summary[:60] if summary else '(无)'}...")
        logger.info(f"三元组数: {len(triples)}")
        for j, t in enumerate(triples[:5], 1):
            s = t.get('s', '') or t.get('subject', '')
            r = t.get('r', '') or t.get('relation', '')
            o = t.get('o', '') or t.get('object', '')
            logger.info(f"  {j}. ({s}, {r}, {o})")

        results.append({
            "max_tokens": max_tokens,
            "time": avg_time,
            "triples": len(triples),
            "summary": summary
        })

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("【汇总】")
    logger.info("=" * 60)
    logger.info(f"{'max_tokens':<15} {'耗时':<12} {'三元组数':<10} {'摘要长度'}")
    logger.info("-" * 60)
    for r in results:
        summary_len = len(r['summary']) if r['summary'] else 0
        logger.info(f"{r['max_tokens']:<15} {r['time']:.2f}秒      {r['triples']:<10} {summary_len}")

    # 分析
    logger.info("\n【分析】")
    fastest = min(results, key=lambda x: x['time'])
    most_triples = max(results, key=lambda x: x['triples'])
    logger.info(f"最快: max_tokens={fastest['max_tokens']} ({fastest['time']:.2f}秒)")
    logger.info(f"最多三元组: max_tokens={most_triples['max_tokens']} ({most_triples['triples']}个)")


def performance_test():
    """性能对比测试"""
    test_cases = [
        ("短句", "我叫张三"),
        ("中句", "我老婆叫李梅，她是一名医生。我们住在上海。"),
        ("100字", """李明2015年毕业于清华大学计算机系，之后加入北京一家互联网公司负责后端开发。2018年他跳槽到深圳腾讯公司担任高级工程师，2020年升为技术团队负责人。他妻子王芳是深圳第三人民医院医生，儿子李轩三岁。李明爱好跑步，参加过深圳马拉松。"""),
        ("200字", """李明是一名拥有十年经验的软件工程师，2015年毕业于清华大学计算机系。毕业后，他加入了一家位于北京的互联网公司，负责后端开发工作。在公司工作期间，李明主导了多个关键项目的开发，包括分布式缓存系统的设计和实现。2018年，李明跳槽到深圳的腾讯公司，担任高级工程师职务。在腾讯工作期间，他参与了微信支付后端架构的优化工作，将系统的并发处理能力提升了50%。2020年，李明被提升为技术团队负责人，管理着一个15人的开发团队。李明的妻子王芳是一名医生，在深圳第三人民医院工作，他们是2019年结婚，住在深圳南山区，儿子三岁。"""),
    ]

    logger.info("\n" + "=" * 60)
    logger.info("性能对比测试")
    logger.info("=" * 60)

    results = {
        "only_triples": {"time": 0, "count": 0},
        "only_summary": {"time": 0, "count": 0},
        "both": {"time": 0, "count": 0},
    }

    for name, text in test_cases:
        logger.info(f"\n--- 测试: {name} ({len(text)}字) ---")

        # 模式1: 仅三元组
        r1, t1 = run_test(text, "triples")
        results["only_triples"]["time"] += t1
        results["only_triples"]["count"] += len(r1.get("triples", []))

        # 模式2: 仅摘要
        r2, t2 = run_test(text, "summary")
        results["only_summary"]["time"] += t2

        # 模式3: 同时提取
        r3, t3 = run_test(text, "both")
        results["both"]["time"] += t3
        results["both"]["count"] += len(r3.get("triples", []))

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("【性能对比结果】")
    logger.info("=" * 60)

    n = len(test_cases)

    logger.info(f"\n1. 仅提取三元组:")
    logger.info(f"   总耗时: {results['only_triples']['time']:.2f}秒")
    logger.info(f"   平均耗时: {results['only_triples']['time']/n:.2f}秒/次")
    logger.info(f"   平均三元组数: {results['only_triples']['count']/n:.1f}个/次")

    logger.info(f"\n2. 仅提取摘要:")
    logger.info(f"   总耗时: {results['only_summary']['time']:.2f}秒")
    logger.info(f"   平均耗时: {results['only_summary']['time']/n:.2f}秒/次")

    logger.info(f"\n3. 同时提取摘要+三元组:")
    logger.info(f"   总耗时: {results['both']['time']:.2f}秒")
    logger.info(f"   平均耗时: {results['both']['time']/n:.2f}秒/次")
    logger.info(f"   平均三元组数: {results['both']['count']/n:.1f}个/次")

    # 对比
    logger.info("\n" + "=" * 60)
    logger.info("【分析】")
    logger.info("=" * 60)

    both_avg = results['both']['time'] / n
    separate_total = (results['only_triples']['time'] + results['only_summary']['time']) / n

    logger.info(f"\n分开调用平均耗时: {separate_total:.2f}秒")
    logger.info(f"合并调用平均耗时: {both_avg:.2f}秒")
    logger.info(f"合并节省时间: {separate_total - both_avg:.2f}秒 ({((separate_total - both_avg)/separate_total*100):.1f}%)")

    # 准确度对比
    logger.info("\n【三元组质量对比】")
    logger.info(f"仅三元组模式平均: {results['only_triples']['count']/n:.1f}个")
    logger.info(f"合并模式平均: {results['both']['count']/n:.1f}个")


def interactive_mode():
    """交互模式"""
    logger.info("=" * 60)
    logger.info("三元组+摘要提取 - 交互模式")
    logger.info("命令: 1=仅三元组, 2=仅摘要, 3=同时提取, q=退出")
    logger.info("=" * 60)

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if not text:
                continue

            if text.lower() in ['q', 'quit', 'exit', '退出']:
                logger.info("退出")
                break

            logger.info("\n--- 模式选择 ---")
            logger.info("1. 仅提取三元组")
            logger.info("2. 仅提取摘要")
            logger.info("3. 同时提取摘要+三元组")

            mode = input("选择模式 [3]: ").strip() or "3"

            if mode == "1":
                run_test(text, "triples")
            elif mode == "2":
                run_test(text, "summary")
            elif mode == "3":
                run_test(text, "both")
            else:
                logger.warning("无效模式")

        except KeyboardInterrupt:
            logger.info("\n退出")
            break
        except Exception as e:
            logger.error(f"错误: {e}")


import random


def generate_random_texts():
    """生成随机测试文本"""
    templates = [
        "我叫{name}，今年{age}岁，住在{city}，从事{job}。",
        "{name}是我的朋友，他在{company}工作，是一名{job}。",
        "今天天气{weather}，我和{name}一起去{place}玩。",
        "我老婆叫{name}，她是一名{job}。我们住在{city}。",
        "我的爱好是{hobby}，每周都会去{place}练习。",
        "{name}告诉我他喜欢{effect}，特别是{thing}。",
        "我在{company}工作了{year}年，负责{job}。",
        "我的家人包括{name}（{relation}）和{name2}（{relation2}）。",
    ]

    names = ["张三", "李梅", "王明", "赵丽", "刘强", "陈芳", "杨洋", "林涛"]
    cities = ["北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安"]
    jobs = ["医生", "工程师", "教师", "设计师", "律师", "会计", "销售", "经理"]
    companies = ["腾讯", "阿里", "百度", "字节", "美团", "京东", "华为", "小米"]
    hobbies = ["跑步", "游泳", "篮球", "足球", "钢琴", "绘画", "阅读", "旅行"]
    places = ["公园", "健身房", "图书馆", "电影院", "商场", "餐厅", "咖啡馆"]
    weather = ["很好", "不错", "有点热", "有点冷", "晴朗", "多云"]
    relations = ["爸爸", "妈妈", "哥哥", "姐姐", "弟弟", "妹妹", "老婆", "老公"]
    years = ["3", "5", "10", "15", "8", "12"]
    effects = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓"]
    things = ["电子产品", "时尚服饰", "运动装备", "书籍", "音乐", "电影"]

    def fill_template(tpl):
        text = tpl
        text = text.replace("{name}", random.choice(names))
        text = text.replace("{name2}", random.choice(names))
        text = text.replace("{city}", random.choice(cities))
        text = text.replace("{job}", random.choice(jobs))
        text = text.replace("{company}", random.choice(companies))
        text = text.replace("{hobby}", random.choice(hobbies))
        text = text.replace("{place}", random.choice(places))
        text = text.replace("{weather}", random.choice(weather))
        text = text.replace("{relation}", random.choice(relations))
        text = text.replace("{relation2}", random.choice(relations))
        text = text.replace("{year}", random.choice(years))
        text = text.replace("{effect}", random.choice(effects))
        text = text.replace("{thing}", random.choice(things))
        return text

    texts = []
    for _ in range(10):
        # 随机选择1-3个模板组合
        count = random.randint(1, 3)
        combined = "".join([fill_template(random.choice(templates)) for _ in range(count)])
        texts.append(combined)

    return texts


def batch_random_test():
    """批量随机测试 + 交互模式"""
    logger.info("=" * 60)
    logger.info("批量随机测试")
    logger.info("=" * 60)

    # 生成随机文本
    random_texts = generate_random_texts()

    total_time = 0
    total_triples = 0

    for i, text in enumerate(random_texts, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试 {i}/{len(random_texts)} ({len(text)}字)")
        logger.info(f"{'='*60}")
        logger.info(f"输入: {text[:80]}...")

        summary, triples, elapsed = extract_both(text)

        total_time += elapsed
        total_triples += len(triples)

        # 打印三元组
        logger.info(f"\n【三元组】({len(triples)}个)")
        for j, t in enumerate(triples, 1):
            s = t.get('s', '') or t.get('subject', '')
            r = t.get('r', '') or t.get('relation', '')
            o = t.get('o', '') or t.get('object', '')
            logger.info(f"  {j}. ({s}, {r}, {o})")

        # 打印摘要
        logger.info(f"\n【摘要】")
        logger.info(f"  {summary[:200] if summary else '(无)'}...")

        logger.info(f"\n耗时: {elapsed:.2f}秒")

    # 统计
    logger.info("\n" + "=" * 60)
    logger.info("【统计】")
    logger.info("=" * 60)
    logger.info(f"总测试数: {len(random_texts)}")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"平均耗时: {total_time/len(random_texts):.2f}秒/次")
    logger.info(f"平均三元组: {total_triples/len(random_texts):.1f}个/次")

    # 开启交互模式
    logger.info("\n" + "=" * 60)
    logger.info("进入交互模式（输入 q 退出）")
    logger.info("=" * 60)

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if not text:
                continue

            if text.lower() in ['q', 'quit', 'exit', '退出']:
                logger.info("退出")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"输入: {text}")
            logger.info(f"{'='*60}")

            summary, triples, elapsed = extract_both(text)

            # 打印三元组
            logger.info(f"\n【三元组】({len(triples)}个)")
            for j, t in enumerate(triples, 1):
                s = t.get('s', '') or t.get('subject', '')
                r = t.get('r', '') or t.get('relation', '')
                o = t.get('o', '') or t.get('object', '')
                logger.info(f"  {j}. ({s}, {r}, {o})")

            # 打印摘要
            logger.info(f"\n【摘要】")
            logger.info(f"  {summary[:200] if summary else '(无)'}...")

            logger.info(f"\n耗时: {elapsed:.2f}秒")

        except KeyboardInterrupt:
            logger.info("\n退出")
            break
        except Exception as e:
            logger.error(f"错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--perf':
            performance_test()
        elif sys.argv[1] == '--batch':
            batch_random_test()
        elif sys.argv[1] == '--param':
            parameter_test()
        else:
            interactive_mode()
    else:
        interactive_mode()
