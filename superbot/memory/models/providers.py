"""内置 Provider 实现"""
from typing import List, Dict, Any, Optional
import numpy as np
from .protocols import LLMProvider, EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer Embedding Provider"""

    def __init__(self, model):
        """
        初始化

        参数:
            model: sentence-transformers 模型实例
        """
        self._model = model

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text)

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self._model.encode(texts)

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class LocalMLXProvider(LLMProvider):
    """本地 MLX LLM Provider (Qwen3)"""

    def __init__(self, model, tokenizer, default_max_tokens: int = 256, default_temperature: float = 0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    def generate(self, prompt: str, **kwargs) -> str:
        import mlx_lm
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        # mlx_lm.generate doesn't accept temperature kwarg, use default
        return mlx_lm.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    def extract_triples(self, text: str, max_tokens: int = 4096) -> Dict[str, Any]:
        """提取事实列表
        返回: {"facts": ["事实1", "事实2", ...]}
        """
        messages = [
            {"role": "system", "content": "你是一个事实提取器。请将输入文本拆解为独立的、短小的\"事实断言\"。\n要求：\n1. 每条事实必须是独立的陈述句。\n2. 去除所有形容词和修饰语。\n3. 保持 JSON 格式：{\"facts\": [\"事实1\", \"事实2\", ...]}"},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = self.generate(prompt, max_tokens=max_tokens)

        # 解析响应
        import json
        import re

        result = {"facts": []}

        # 移除思考过程
        if "</think>" in response:
            response = response.split("</think>")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[-1]

        # 尝试解析JSON
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                if "facts" in data and isinstance(data["facts"], list):
                    result["facts"] = [f.strip() for f in data["facts"] if f.strip()]
                    return result
        except:
            pass

        return result

    def _extract_from_response(self, response: str) -> Dict[str, Any]:
        """从响应中提取压缩文本和三元组"""
        import json
        import re

        result = {"compressed": "", "triples": []}

        # 模板值
        template_values = ["...", "主语", "subject", "关系", "relation", "宾语", "object", ""]

        # 改进JSON解析：只在思考过程结束后查找JSON
        json_part = response

        # 如果有思考过程，尝试找到最终的JSON输出
        if "</think>" in response:
            # 分割并反向查找包含JSON的部分
            parts = response.split("</think>")
            for part in reversed(parts):
                # 跳过思考过程中的模板提示
                if "摘要" in part and "主体" in part:
                    continue
                if "compressed" in part.lower() or "subject" in part.lower():
                    json_part = part
                    break
            else:
                json_part = parts[-1] if parts else response

        # 移除markdown代码块标记
        json_part = re.sub(r'^```json\s*', '', json_part)
        json_part = re.sub(r'^```\s*', '', json_part)
        json_part = re.sub(r'\s*```$', '', json_part)
        # 方法1: 尝试直接解析JSON - 改进解析逻辑
        try:
            # 尝试多种 JSON 匹配模式
            json_patterns = [
                r'\{[\s\S]*\}',  # 原始模式
                r'\{[^{}]*"compressed"[^{}]*"triples"[^{}]*\}',  # 完整结构
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, json_part)
                for match_str in matches:
                    try:
                        data = json.loads(match_str)
                        compressed = data.get("compressed", "")
                        triples_list = data.get("triples", [])

                        if triples_list and isinstance(triples_list, list):
                            cleaned = self._clean_triples(triples_list)
                            if cleaned:
                                result["compressed"] = compressed.strip() if compressed else ""
                                result["triples"] = cleaned
                                return result
                        # 即使 triples 为空，也返回 compressed
                        elif compressed:
                            result["compressed"] = compressed.strip()
                            return result
                    except:
                        continue
        except:
            pass

        # 方法2: 从思考过程中提取
        thinking_part = response
        if "Thinking Process:" in response:
            thinking_part = response.split("Thinking Process:")[-1]
            if "</think>" in thinking_part:
                thinking_part = thinking_part.split("</think>")[0]

        try:
            # 尝试提取 Subject, Relation, Object
            pattern = r'Subject.*?[:：]\s*([^\n\(,]+)'
            rel_pat = r'Relation.*?[:：]\s*([^\n\(,]+)'
            obj_pat = r'Object.*?[:：]\s*([^\n\(,]+)'

            subjects = re.findall(pattern, thinking_part)
            relations = re.findall(rel_pat, thinking_part)
            objects = re.findall(obj_pat, thinking_part)

            if subjects and relations and objects:
                triples = []
                min_len = min(len(subjects), len(relations), len(objects))

                for i in range(min_len):
                    s = subjects[i].strip()
                    r = relations[i].strip()
                    o = objects[i].strip()

                    # 移除括号内容
                    s = re.sub(r'\s*\([^\)]*\)\s*', '', s).strip()
                    r = re.sub(r'\s*\([^\)]*\)\s*', '', r).strip()
                    o = re.sub(r'\s*\([^\)]*\)\s*', '', o).strip()

                    # 移除引号
                    s = s.strip('"\'「」')
                    r = r.strip('"\'「」')
                    o = o.strip('"\'「」')

                    # 过滤模板和无效内容
                    if (s and r and o and
                        len(s) <= 20 and len(r) <= 15 and len(o) <= 20 and
                        s not in template_values and
                        r not in template_values and
                        o not in template_values):
                        triples.append({"subject": s, "relation": r, "object": o})

                if triples:
                    result["triples"] = triples
                    return result
        except:
            pass

        # 方法3: 尝试从JSON模板中提取实际内容（模型可能在思考过程中输出了完整内容）
        try:
            # 查找 {"compressed": ..., "triples": ...} 格式
            all_matches = re.findall(r'\{[^{}]*"compressed"[^{}]*\}[^{}]*\{[^{}]*"triples"[^{}]*\}', response)
            for match_str in all_matches:
                try:
                    data = json.loads(match_str)
                    triples_list = data.get("triples", [])
                    if triples_list:
                        cleaned = self._clean_triples(triples_list)
                        if cleaned:
                            result["compressed"] = data.get("compressed", "")
                            result["triples"] = cleaned
                            return result
                except:
                    pass
        except:
            pass

        return result

    def _clean_triples(self, triples: List[Dict]) -> List[Dict]:
        """清理三元组数据"""
        cleaned = []
        # 模板内容和无效值
        template_values = ["...", "主语", "subject", "关系", "relation", "宾语", "object", ""]
        invalid_patterns = ["**", "*", "null", "none", "undefined", "->"]

        for triple in triples:
            if not isinstance(triple, dict):
                continue
            subject = triple.get("subject", "").strip().strip('"')
            relation = triple.get("relation", "").strip().strip('"')
            obj = triple.get("object", "").strip().strip('"')

            # 检查是否包含无效模式
            has_invalid = any(p in subject.lower() or p in relation.lower() or p in obj.lower()
                            for p in invalid_patterns)

            # 检查 relation 是否包含多个值（逗号分隔）
            has_multiple_values = ',' in relation or ',' in obj

            # 过滤模板内容和无效内容
            if (subject and subject not in template_values and
                relation and relation not in template_values and
                obj and obj not in template_values and
                not has_invalid and
                not has_multiple_values and
                len(subject) <= 20 and len(relation) <= 15 and len(obj) <= 25):
                cleaned.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj
                })
        return cleaned

    def compress(self, text: str, max_tokens: int = 128) -> str:
        prompt = f"""<|im_start|>system
压缩成长度不超过100字的简洁摘要，保留关键信息。
<|im_end|>
<|im_start|>user
{text}

摘要:
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=max_tokens)

    def understand_context(
        self,
        query: str,
        memory: List[str],
        history: List[str],
        max_tokens: int = 128
    ) -> str:
        prompt = f"""<|im_start|>system
你是一个上下文理解专家。根据对话历史和上下文，推断用户当前消息的含义。

## 任务
1. 如果用户使用了代词(它/这个/那/也/同样等)，根据上下文推断指代内容
2. 结合相关记忆理解用户意图
3. 直接给出理解结果，一句话即可
<|im_end|>
<|im_start|>user
## 相关记忆
{memory[:3] if memory else '无'}

## 对话历史
{history[-3:] if history else '无'}

## 当前消息
"{query}"

## 理解
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=max_tokens)

    def _parse_json(self, response: str) -> List[Dict[str, Any]]:
        import json
        import re
        try:
            return json.loads(response)
        except:
            match = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', response)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return []


class LlamaCppProvider(LLMProvider):
    """本地 llama.cpp LLM Provider (Qwen3-8B-Q4)"""

    _llm = None

    def __init__(self, model_path: str = "/Users/heyunpeng/workstation/src/Qwen3-8B-Q4_K_M/Qwen3-8B-Q4_K_M.gguf"):
        self.model_path = model_path
        self.default_max_tokens = 2048

    def _get_llm(self):
        """获取或初始化 llm 实例"""
        if LlamaCppProvider._llm is None:
            from llama_cpp import Llama
            LlamaCppProvider._llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_gpu_layers=99,  # 使用 Metal GPU
                verbose=False,
                logits_all=False,  # 只返回最后一个 token 的 logits
            )
        return LlamaCppProvider._llm

    def generate(self, prompt: str, **kwargs) -> str:
        llm = self._get_llm()
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

    def extract_triples(self, text: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """提取事实列表
        返回: {"facts": ["事实1", "事实2", ...]}
        """
        llm = self._get_llm()

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一个事实提取器。请将输入文本拆解为独立的，短小的事实断言。要求：1. 每条事实必须是独立的陈述句。2. 去除所有形容词和修饰语。3. 保持 JSON 格式：{'facts': ['事实1', '事实2']}"},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens
        )

        content = response['choices'][0]['message']['content']
        return self._parse_facts(content)

    def _parse_facts(self, content: str) -> Dict[str, Any]:
        """解析事实列表"""
        import json
        import re

        result = {"facts": []}

        # 移除思考过程
        if "<think>" in content:
            content = content.split("</think>")[-1]

        # 解析 JSON
        try:
            data = json.loads(content)
            if "facts" in data and isinstance(data["facts"], list):
                result["facts"] = [f.strip() for f in data["facts"] if f.strip()]
                return result
        except:
            pass

        # 尝试提取 JSON 部分
        try:
            match = re.search(r'\{[^{}]*"facts"[^{}]*\}', content)
            if match:
                data = json.loads(match.group())
                if "facts" in data:
                    result["facts"] = [f.strip() for f in data["facts"] if f.strip()]
                    return result
        except:
            pass

        return result

    def compress(self, text: str, max_tokens: int = 128) -> str:
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "压缩成简短摘要，不超过100字"},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

    def understand_context(self, query: str, memory: List[str], history: List[str], max_tokens: int = 128) -> str:
        llm = self._get_llm()
        context = f"相关记忆：{memory[:3]}\n对话历史：{history[-3:]}\n当前消息：{query}"
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "根据上下文推断用户意图"},
                {"role": "user", "content": context}
            ],
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

