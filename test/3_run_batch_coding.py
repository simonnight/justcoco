
# 3_run_batch_coding.py
# 权威最终版 (重新设计): 多线程 + 数据清洗增强 + 动态K + 去重 + 校正同步 + 提示词优化 + 新编码规则后处理修正

import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import google.api_core.exceptions
import os
import json
import time
import pickle
from tqdm import tqdm
import traceback
import re
import shutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
import concurrent.futures
from threading import local

# =============================================================================
# --- 配置区 (用户修改) ---
# =============================================================================

# 1. 精确复制/粘贴您当前要处理的问题文本
BATCH_QUESTION_TEXT = "评分的原因"
# 2. 指定对应的批次文件名 (确保文件在04_input_batches_to_code文件夹内)
BATCH_INPUT_FILE = "04_input_batches_to_code/batch_问题A.csv"
# 3. 设置并发线程数
NUM_THREADS = 10  # 根据你的API配额和系统资源调整

# =============================================================================
# --- 常量与配置 (通常无需修改) ---
# =============================================================================

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("错误：请先设置 GOOGLE_API_KEY 环境变量。")
    exit()

# --- API 配置 ---
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash-lite"
# --- 处理逻辑阈值 ---
SIMILARITY_COPY_THRESHOLD = 0.2
K_VALUE = 3 # 基础 K 值，动态 K 会基于此调整

# --- 文件路径 ---
CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"
KB_FOLDER = "03_knowledge_base"
ANSWER_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "answer.index")
QUESTION_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "question.index")
DATA_MAP_FILE = os.path.join(KB_FOLDER, "data_map.pkl")
UNIQUE_QUESTIONS_FILE = os.path.join(KB_FOLDER, "unique_questions.pkl")
QUESTION_TO_SUB_INDEX_MAP_FILE = os.path.join(KB_FOLDER, "question_to_sub_index_map.pkl")
RESULTS_FOLDER = "05_coded_results/old_questions"
REVIEWED_RESULTS_FOLDER = "06_reviewed_results" # 已审核结果文件夹

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- 无意义内容过滤器 ---
MEANINGLESS_WORDS = {
    "ok", "good", "nice", "fine", "test", "null", "none", "na", "n/a",
    "好", "好的", "好好好", "好的好的", "不错", "可以", "没问题",
    "测试", "test", "无", "没有", "无意见", "不详", "不知道", "不清楚",
    "1", "11", "111", "666"
}
# 无意义内容的默认编码 (情感已按要求修改为 "负面")
MEANINGLESS_CODE_JSON = [
    {
        "sentiment": "负面", "net": "其他", "subnet": "无效回答", "code": "999",
        "label": "无明确意义回答", "is_new_suggestion": False
    }
]

# =============================================================================
# --- 辅助函数区 ---
# =============================================================================

def is_meaningless(text: str) -> bool:
    """检查文本是否属于无意义内容"""
    if not text: return True
    cleaned_text = text.strip()
    if not cleaned_text: return True
    if cleaned_text.lower() in MEANINGLESS_WORDS: return True
    if cleaned_text.isdigit(): return True
    if re.fullmatch(r"[^\w\u4e00-\u9fa5]+", cleaned_text): return True
    return False

def find_best_matching_question_index(question_index, unique_questions, new_question: str) -> int:
    """在历史问题中找到语义最相近的问题的索引号"""
    response = genai.embed_content(model=EMBEDDING_MODEL, content=new_question, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([response['embedding']]).astype('float32')
    distances, indices = question_index.search(query_vector, 1)
    return indices[0][0]

def format_examples_rich(df_slice, codebook):
    """格式化案例，展示完整的层级信息"""
    lines = []
    for _, row in df_slice.iterrows():
        code = str(row['code'])
        if code in codebook.index:
            code_info = codebook.loc[code]
            lines.append(
                f"- 问题: \"{row['question']}\"\n"
                f"  回答: \"{row['text']}\"\n"
                f"  -> 编码结果: {{ sentiment: '{code_info.sentiment}', net: '{code_info.net}', subnet: '{code_info.subnet}', code: {code}, label: '{code_info.label}' }}"
            )
    return "\n".join(lines)

def clean_code_info(code_series, code: str, is_new: bool) -> dict:
    """将Pandas Series转换为纯净的字典"""
    if isinstance(code_series, pd.DataFrame):
        record = code_series.iloc[0]
    else:
        record = code_series
    sentiment = record.sentiment if pd.notna(record.sentiment) else ""
    net = record.net if pd.notna(record.net) else ""
    subnet = record.subnet if pd.notna(record.subnet) else ""
    label = record.label if pd.notna(record.label) else ""
    return {
        'sentiment': str(sentiment), 'net': str(net), 'subnet': str(subnet),
        'code': code, 'label': str(label), 'is_new_suggestion': is_new
    }

def clean_json_string(raw_text: str) -> str:
    """从AI返回的原始文本中，提取纯净的JSON字符串。"""
    start_brace = raw_text.find('{')
    start_bracket = raw_text.find('[')
    start_index = min(start_brace if start_brace != -1 else float('inf'),
                      start_bracket if start_bracket != -1 else float('inf'))
    end_brace = raw_text.rfind('}')
    end_bracket = raw_text.rfind(']')
    end_index = max(end_brace, end_bracket)
    
    if start_index < float('inf') and end_index != -1:
        return raw_text[start_index : end_index + 1]
    return raw_text

# --- 动态 K 值 (根据新要求修改) ---
def estimate_k_value(text: str) -> int:
    """
    根据回答文本的长度，估算一个合适的 K 值。
    短文本(<=5 chars) K=3, 中等(<=15 chars) K=4, 长(>15 chars) K=7
    """
    text_length = len(text)
    if text_length <= 5:
        return 3
    elif text_length <= 15:
        return 4
    else:
        return 7

# --- 线程局部存储 ---
thread_local_storage = local()

def get_thread_local_model():
    """为每个线程惰性初始化一个独立的 GenerativeModel 实例"""
    if not hasattr(thread_local_storage, "model"):
        system_instruction = (
            "作为质性研究编码专家，你的任务是：\n"
            "1. 识别反馈中的所有独立观点。\n"
            "2. 为每个观点精确匹配现有编码。\n"
            "3. 若现有编码过宽泛，则在相同层级下建议更具体的新编码。\n"
            "4. 必须严格返回JSON格式。"
        )
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        thread_local_storage.model = genai.GenerativeModel(
            GENERATION_MODEL, system_instruction=system_instruction, generation_config=generation_config
        )
    return thread_local_storage.model

# --- 增强的重试逻辑，包含服务器内部错误 ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=(
        retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) |
        retry_if_exception_type(google.api_core.exceptions.InternalServerError) |
        retry_if_exception_type(google.api_core.exceptions.DeadlineExceeded)
    )
)
def generate_content_with_retry(model, user_prompt):
    """带自动重试机制的API调用函数"""
    return model.generate_content(user_prompt)

# =============================================================================
# --- 核心处理函数区 ---
# =============================================================================

def rag_pipeline_final(
    resources, new_text: str, base_k: int, similarity_threshold: float,
    stats_ref, preloaded_sub_index, original_indices_for_sub_index
):
    """
    无缓存的RAG流水线，包含相似度匹配和AI深度分析。
    集成动态K值、Zero-Shot优化、新编码规则后处理修正。
    """
    try:
        df_map = resources["df_map"]
        codebook_df = resources["codebook_df"]
        model = get_thread_local_model() # 正确获取线程局部模型
        answer_index = resources["answer_index"]
        # system_instruction 已包含在线程模型中
        
        # --- 动态 K 值 ---
        dynamic_k = estimate_k_value(new_text)
        k = dynamic_k

        # --- 第二道防线：历史相似匹配 ---
        if preloaded_sub_index.ntotal > 0 and similarity_threshold is not None:
            response = genai.embed_content(model=EMBEDDING_MODEL, content=new_text, task_type="RETRIEVAL_QUERY")
            query_vector = np.array([response['embedding']]).astype('float32')
            distances, sub_indices = preloaded_sub_index.search(query_vector, 1)
            if distances[0][0] < similarity_threshold:
                with stats_ref['lock']: # 线程安全更新
                    stats_ref['similarity_hits'] += 1
                best_match_original_idx = original_indices_for_sub_index[sub_indices[0][0]]
                codes_to_copy = df_map[df_map.index == best_match_original_idx]['code'].tolist()
                unique_codes = sorted(list(set(codes_to_copy)))
                copied_codes = [clean_code_info(codebook_df.loc[str(c)], str(c), is_new=False) for c in unique_codes]
                return copied_codes, "SIMILARITY_MATCH_COPY"

        # --- 第三道防线：完整RAG+LLM调用 ---
        with stats_ref['lock']: # 线程安全更新
            stats_ref['api_calls'] += 1
            
        prompt_mode, relevant_examples = "zero_shot", pd.DataFrame()
        if 'query_vector' not in locals():
            response = genai.embed_content(model=EMBEDDING_MODEL, content=new_text, task_type="RETRIEVAL_QUERY")
            query_vector = np.array([response['embedding']]).astype('float32')
            
        if preloaded_sub_index.ntotal > 0:
            distances, sub_indices = preloaded_sub_index.search(query_vector, min(k, preloaded_sub_index.ntotal))
            if distances[0][0] < 0.9:
                prompt_mode, original_indices = "standard", [original_indices_for_sub_index[i] for i in sub_indices[0]]
                relevant_examples = df_map.loc[original_indices]
                
        if prompt_mode == "zero_shot":
            distances, indices = answer_index.search(query_vector, k)
            if distances[0][0] < 0.9:
                prompt_mode, relevant_examples = "cross_context_warning", df_map.iloc[indices[0]]

        relevant_definitions_str, contextual_codes_str = "", ""
        if not relevant_examples.empty:
            unique_codes_from_examples = relevant_examples['code'].unique().astype(str)
            relevant_code_info_df = codebook_df.loc[codebook_df.index.intersection(unique_codes_from_examples)]
            relevant_definitions_str = relevant_code_info_df.to_string()
            top_example_code = str(relevant_examples.iloc[0]['code'])
            if top_example_code in codebook_df.index:
                top_example_info = codebook_df.loc[top_example_code]
                if isinstance(top_example_info, pd.DataFrame):
                    top_example_info = top_example_info.iloc[0]
                main_net, main_subnet = top_example_info['net'], top_example_info['subnet']
                contextual_codes_df = codebook_df[(codebook_df['net'] == main_net) & (codebook_df['subnet'] == main_subnet)]
                if not contextual_codes_df.empty:
                    contextual_codes_str = f"相关上下文编码 (供您生成新编码时参考):\n{contextual_codes_df.to_string()}"

        prompt_intro = f"""
# 任务：多维度编码
为以下“待编码文本”进行专业的多层级编码。一个回答可能包含多个可独立编码的主题，找出所有适用的编码。
---
## 待编码文本
**问题**: "{BATCH_QUESTION_TEXT}"
**回答**: "{new_text}"
---
"""
        # --- Zero-Shot 模式优化 (如果 Net 索引可用) ---
        if prompt_mode == "zero_shot":
            full_codebook_str = codebook_df.to_string()
            # 尝试使用 Net 索引进行优化
            net_index = resources.get("net_index")
            net_names_list = resources.get("net_names", [])
            
            if net_index is not None and net_names_list:
                try:
                    query_response = genai.embed_content(model=EMBEDDING_MODEL, content=new_text, task_type="RETRIEVAL_QUERY")
                    query_vector = np.array([query_response['embedding']]).astype('float32')
                    faiss.normalize_L2(query_vector)
                    
                    num_nets_to_fetch = 3
                    scores, net_indices = net_index.search(query_vector, min(num_nets_to_fetch, len(net_names_list)))
                    relevant_net_names = [net_names_list[idx] for idx in net_indices[0]]
                    
                    filtered_codebook_df = codebook_df[codebook_df['net'].isin(relevant_net_names)]
                    if not filtered_codebook_df.empty:
                        # 可以选择只保留关键列以进一步减少 token
                        # filtered_codebook_for_prompt = filtered_codebook_df[['net', 'subnet', 'code', 'label']]
                        filtered_codebook_for_prompt = filtered_codebook_df
                        full_codebook_str = filtered_codebook_for_prompt.to_string()
                except Exception as e:
                    print(f"Warning: Failed to use Net index for zero-shot filtering: {e}. Falling back to full codebook.")
            
            prompt_references = f"## 相关码表参考 (基于文本预测的可能分类)\n{full_codebook_str}"
        else:
            warning_message = "# [重要警告]: 检索到的案例与当前问题不完全一致，请格外谨慎地判断是否直接沿用其编码。\n" if prompt_mode == "cross_context_warning" else ""
            prompt_references = f"## 相关历史案例\n{warning_message}{format_examples_rich(relevant_examples, codebook_df)}\n## 相关编码定义\n{relevant_definitions_str}\n{contextual_codes_str}"

        # --- 优化后的 Prompt ---
        user_prompt = f"""
{prompt_intro}
# 参考信息
{prompt_references}
---
# 分析与决策指令
请严格、一步一步地遵循以下思考过程和输出要求：

## 1. 核心原则
- **准确性优先**: 确保每个编码准确反映了回答中的观点。
- **完整性**: 不遗漏回答中任何独立且有意义的观点。
- **一致性**: 编码风格、层级、术语需与现有码表保持高度一致。

## 2. 决策流程
### 第一步：精确匹配现有编码
- **全面扫描**: 仔细检查“待编码文本”中的每个观点。
- **高优先级匹配**: 在“相关历史案例”和“相关编码定义”中寻找`label`与观点**高度一致**的编码。
- **直接复用**: 如果找到强匹配项，**必须**直接使用该编码的**完整信息**（sentiment, net, subnet, code, label），并设置 `is_new_suggestion: false`。

### 第二步：必要时创造新编码
仅当**所有**观点都无法精确匹配到现有编码时，才考虑生成新编码。对于每个需要新编码的观点：
1.  **确定分类**:
    -   明确该观点应归属于哪个现有的 `net` (主类别) 和 `subnet` (子类别)。
    -   参考“相关上下文编码”或“完整码表参考”来确定最合适的分类。
2.  **生成新标签 (`label`)**:
    -   创建一个**具体、清晰、无歧义**的标签，精准描述该观点。
    -   标签语言应与现有码表风格一致。
3.  **生成新编码 (`code`)**:
    -   **冲突检查**: **严格检查**你生成的新 `code` 是否与“完整码表参考”或“相关编码定义”中的**任何** `code` 相同。**绝对禁止**使用已存在的 `code`。
    -   **编号策略**:
        -   在确定了 `net` 和 `subnet` 后，查看该分类下现有的所有 `code`。
        -   新 `code` 应为该分类下当前**最大**的 `code` 数字 **加一**。
        -   例如，若 `netA/subnetB` 下现有编码为 ["101", "102", "105"]，则新编码应为 "106"。
        -   如果该 `net`/`subnet` 下无现有编码，则从项目约定的该分类起始编号开始（请根据现有码表推断，如 101, 201 等）。
    -   **格式要求**: `code` 必须是**字符串格式的数字**，不包含前导零 (例如: "1", "23", "123", "1234")。
4.  **确定情感 (`sentiment`)**:
    -   根据观点内容，从 "正面", "负面" 中选择最贴切的一个。
5.  **标记新建议**:
    -   对于所有新生成的编码，**必须**设置 `is_new_suggestion: true`。

## 3. 输出要求
- **格式**: **严格**以一个【JSON数组】的格式输出你的最终决策。
- **内容**: 数组中的每个元素对应一个独立编码观点。
- **完整性**: 如果没有任何编码适用，请返回一个空数组 `[]`。
- **示例格式**:
```json
[
  {{
    "sentiment": "从 '正面', '负面' 中选择",
    "net": "必须与现有码表中的主类别一致",
    "subnet": "必须与现有码表中的子类别一致",
    "code": "一个有效的、不与现有码表冲突的数字字符串 (例如: '1', '23', '123', '1234')",
    "label": "具体、清晰、无歧义的标签描述",
    "is_new_suggestion": false (对于匹配的编码) 或 true (对于新生成的编码)
  }}
]
```
**请务必确保最终输出的 JSON 是有效的、可解析的，并且严格遵守上述所有指令。**
"""

        chat_response = generate_content_with_retry(model, user_prompt)
        cleaned_text = clean_json_string(chat_response.text)
        if not cleaned_text:
            return [{"error": "API returned empty response after cleaning."}], "ERROR"

        # --- 【关键】新编码规则后处理校验修正 ---
        try:
            raw_results = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            return [{"error": f"AI返回的JSON无法解析: {str(e)}", "raw_response": cleaned_text}], "ERROR"

        validated_results = []
        for item in raw_results:
            if "error" in item:
                validated_results.append(item)
                continue

            is_new_suggestion = item.get("is_new_suggestion", False)
            suggested_code = item.get("code", "")
            original_net = item.get("net", "")
            original_subnet = item.get("subnet", "")
            item_label = item.get("label", "Unknown Label")

            if is_new_suggestion:
                # 1. 基本格式检查
                if not isinstance(suggested_code, str) or not suggested_code.isdigit() or not (1 <= len(suggested_code) <= 4):
                    print(f"警告：AI为 '{item_label}' 建议的新编码 '{suggested_code}' 格式无效。")
                    item["validation_warning"] = f"新编码 '{suggested_code}' 格式无效 (应为1到4位数字字符串)。"
                    validated_results.append(item)
                    continue

                # 2. 核心检查：code 是否与现有码表冲突
                if suggested_code in codebook_df.index:
                    print(f"警告：AI建议的新编码 '{suggested_code}' 与现有码表冲突。正在尝试修正...")
                    existing_in_context_df = codebook_df[(codebook_df['net'] == original_net) & (codebook_df['subnet'] == original_subnet)]
                    if not existing_in_context_df.empty:
                        try:
                            existing_codes_int = [int(code) for code in existing_in_context_df.index if code.isdigit()]
                            if existing_codes_int:
                                max_code_in_context_int = max(existing_codes_int)
                                new_available_code_int = max_code_in_context_int + 1
                                new_available_code_str = str(new_available_code_int)

                                attempt_count = 0
                                max_attempts = 1000
                                while new_available_code_str in codebook_df.index and attempt_count < max_attempts:
                                    new_available_code_int += 1
                                    new_available_code_str = str(new_available_code_int)
                                    attempt_count += 1

                                if attempt_count >= max_attempts:
                                    error_msg = f"无法为 net='{original_net}', subnet='{original_subnet}' 找到不冲突的新编码 (尝试了 {max_attempts} 次)。"
                                    print(f"错误: {error_msg}")
                                    item["validation_error"] = error_msg
                                    validated_results.append(item)
                                    continue

                                info_msg = f"信息：已将冲突编码 '{suggested_code}' 自动修正为 '{new_available_code_str}'。"
                                print(info_msg)
                                item["code"] = new_available_code_str
                                item["original_suggested_code"] = suggested_code
                                item["validation_note"] = f"原始建议编码 '{suggested_code}' 冲突，已自动修正为 '{new_available_code_str}'。"
                            else:
                                warning_msg = f"net='{original_net}', subnet='{original_subnet}' 下未找到有效的现有数字编码。"
                                print(f"警告: {warning_msg}")
                                item["validation_warning"] = f"{warning_msg} 无法自动修正新编码 '{suggested_code}'。"
                                validated_results.append(item)
                                continue
                        except ValueError as ve:
                            error_msg = f"处理 net='{original_net}', subnet='{original_subnet}' 的现有编码时出错: {ve}。"
                            print(f"错误: {error_msg}")
                            item["validation_error"] = f"{error_msg} 无法自动修正新编码 '{suggested_code}'。"
                            validated_results.append(item)
                            continue
                    else:
                        error_msg = f"新编码 '{suggested_code}' 冲突，且其 net='{original_net}', subnet='{original_subnet}' 下无现有编码。"
                        print(f"错误: {error_msg}")
                        item["validation_error"] = error_msg
                        validated_results.append(item)
                else:
                    # 没有冲突，检查并可能移除前导零
                    if suggested_code.startswith('0') and len(suggested_code) > 1:
                        try:
                            normalized_code = str(int(suggested_code))
                            if normalized_code not in codebook_df.index:
                                item["code"] = normalized_code
                                item["validation_note"] = f"原始建议编码 '{suggested_code}' 含有前导零，已规范化为 '{normalized_code}'。"
                        except ValueError:
                            pass
                    validated_results.append(item)
            else:
                # 不是新建议，直接添加
                if suggested_code not in codebook_df.index:
                    item["validation_warning"] = f"非新建议编码 '{suggested_code}' 未在现有码表中找到。"
                validated_results.append(item)

        return validated_results, "API_CALL"

    except google.api_core.exceptions.ResourceExhausted as e:
        return [{"error": str(e)}], "RATE_LIMIT_ERROR"
    except google.api_core.exceptions.InternalServerError as e:
        return [{"error": f"Google API 500 Internal Server Error: {str(e)}"}], "INTERNAL_SERVER_ERROR"
    except Exception as e:
        traceback.print_exc()
        return [{"error": f"rag_pipeline_final 内部错误: {str(e)}", "traceback": traceback.format_exc()}], "ERROR"


def process_single_row(args, shared_data):
    """处理单行数据的函数，供 ThreadPoolExecutor 调用"""
    index, row = args
    current_text = row['text']
    process_method = ""
    coding_results_list = None

    # 从 shared_data 中解包所需资源
    exact_match_lookup = shared_data['exact_match_lookup']
    reviewed_lookup = shared_data['reviewed_lookup']
    canonical_question_text = shared_data['canonical_question_text']
    preloaded_sub_index = shared_data['preloaded_sub_index']
    original_indices_for_sub_index = shared_data['original_indices_for_sub_index']
    resources = shared_data['resources']
    stats_ref = shared_data['stats_ref']

    try:
        # --- 最高优先级：检查已校正 lookup ---
        reviewed_key = (BATCH_QUESTION_TEXT, current_text)
        if reviewed_key in reviewed_lookup:
            coding_results_list = reviewed_lookup[reviewed_key]
            process_method = "REVIEWED_CORRECTED"
            with stats_ref['lock']:
                stats_ref['reviewed_hits'] += 1
        else:
            # --- 未在校正库中找到，执行原有逻辑 ---
            if is_meaningless(current_text):
                coding_results_list = MEANINGLESS_CODE_JSON
                process_method = "PREFILTERED_MEANINGLESS"
                with stats_ref['lock']:
                    stats_ref['prefiltered_hits'] += 1
            else:
                lookup_key = (canonical_question_text, current_text)
                if lookup_key in exact_match_lookup:
                    coding_results_list = exact_match_lookup[lookup_key]
                    process_method = "KB_EXACT_MATCH"
                    with stats_ref['lock']:
                        stats_ref['kb_exact_hits'] += 1
                else:
                    # --- 调用 RAG 流水线 ---
                    coding_results_list, call_type = rag_pipeline_final(
                        resources, current_text, K_VALUE, SIMILARITY_COPY_THRESHOLD,
                        stats_ref, # 传递 stats_ref 用于更新
                        preloaded_sub_index, original_indices_for_sub_index
                    )
                    process_method = call_type.upper()
                    # 错误计数已在 rag_pipeline_final 内部通过 stats_ref 更新

    except Exception as e:
        coding_results_list = [{"error": f"Worker内部错误: {str(e)}", "traceback": traceback.format_exc()}]
        process_method = "WORKER_ERROR"
        with stats_ref['lock']:
            stats_ref['errors'] += 1

    final_result = {
        "uuid": row.get('uuid', ''),
        "question": BATCH_QUESTION_TEXT,
        "text": current_text,
        "process_method": process_method,
        "coding_results": coding_results_list,
        "original_index": index
    }
    return final_result

# =============================================================================
# --- 主执行区 ---
# =============================================================================

def main():
    """主执行函数"""
    print("--- 步骤 1: 正在加载所有共享资源 ---")
    try:
        resources = {
            "df_map": pd.read_pickle(DATA_MAP_FILE),
            "codebook_df": pd.read_csv(CODEBOOK_FILE, dtype={'code': str}).set_index('code'),
            "answer_index": faiss.read_index(ANSWER_FAISS_INDEX_FILE),
        }
        uncoded_df = pd.read_csv(BATCH_INPUT_FILE, dtype=str).fillna('')
        print("共享资源加载成功！")
    except Exception as e:
        print(f"错误：启动时加载共享资源失败: {e}")
        traceback.print_exc()
        return

    print(f"--- 步骤 2: 为批处理问题“{BATCH_QUESTION_TEXT[:30]}...”准备上下文 ---")
    try:
        question_index = faiss.read_index(QUESTION_FAISS_INDEX_FILE)
        with open(UNIQUE_QUESTIONS_FILE, 'rb') as f:
            unique_questions = pickle.load(f)
        with open(QUESTION_TO_SUB_INDEX_MAP_FILE, 'rb') as f:
            question_to_sub_index_map = pickle.load(f)
            
        best_match_q_idx = find_best_matching_question_index(question_index, unique_questions, BATCH_QUESTION_TEXT)
        canonical_question_text = unique_questions[best_match_q_idx]
        print(f"智能匹配到最相似的历史问题: “{canonical_question_text}”")
        
        sub_index_info = question_to_sub_index_map[canonical_question_text]
        preloaded_sub_index = faiss.read_index(sub_index_info["index_file"])
        original_indices_for_sub_index = sub_index_info["original_indices"]
        print("专属迷你索引信息已加载。")
        
    except Exception as e:
        print(f"错误：加载上下文失败: {e}")
        traceback.print_exc()
        return

    print("--- 步骤 3: 正在构建知识库精准匹配查找表 ---")
    try:
        exact_match_lookup = {}
        grouped = resources["df_map"].groupby(['question', 'text'])['code'].apply(lambda codes: sorted(list(set(codes.astype(str)))))
        for (question, text), codes in grouped.items():
            key = (question, text)
            code_info_list = []
            for code in codes:
                try:
                    code_series = resources["codebook_df"].loc[code]
                    code_info = clean_code_info(code_series, code, is_new=False)
                    code_info_list.append(code_info)
                except KeyError:
                    continue
            exact_match_lookup[key] = code_info_list
        print(f"精准匹配查找表构建完成，包含 {len(exact_match_lookup)} 条独特问答对。")
    except Exception as e:
        print(f"错误：构建精准匹配查找表失败: {e}")
        traceback.print_exc()
        return

    # --- 步骤 3.5: 构建已校正编码查找表 ---
    print("--- 步骤 3.5: 正在构建已校正编码查找表 ---")
    reviewed_lookup = {}
    if os.path.exists(REVIEWED_RESULTS_FOLDER):
        reviewed_files = [f for f in os.listdir(REVIEWED_RESULTS_FOLDER) if f.endswith('.jsonl')]
        for filename in reviewed_files:
            filepath = os.path.join(REVIEWED_RESULTS_FOLDER, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                q = record.get("question", "")
                                t = record.get("text", "")
                                results = record.get("coding_results", [])
                                if q and t and results is not None:
                                    reviewed_lookup[(q, t)] = results
                            except json.JSONDecodeError:
                                print(f"警告：无法解析 {filepath} 中的一行，已跳过。")
            except Exception as e:
                print(f"警告：读取已审核文件 {filepath} 时出错: {e}")
                # 不因单个文件错误而停止
    print(f"已校正编码查找表构建完成，包含 {len(reviewed_lookup)} 条已校正问答对。")

    # --- 步骤 3.6: 为 zero-shot 模式构建 Net 向量索引 (可选优化) ---
    print("--- 步骤 3.6: 正在为 zero-shot 模式构建 Net 向量索引 ---")
    try:
        codebook_df = resources["codebook_df"]
        net_descriptions = codebook_df.groupby('net')['label'].apply(lambda labels: ' '.join(labels)).to_dict()
        net_names = list(net_descriptions.keys())
        net_texts = list(net_descriptions.values())
        
        if not net_texts:
            raise ValueError("没有找到任何 net 的描述文本，无法构建 Net 索引。")

        embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=net_texts, task_type="RETRIEVAL_DOCUMENT")
        net_embeddings = np.array(embedding_response['embeddings']).astype('float32')
        
        dimension = net_embeddings.shape[1]
        net_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(net_embeddings)
        net_index.add(net_embeddings)
        
        print(f"Net 向量索引构建完成，包含 {len(net_names)} 个 Net。")
        resources["net_index"] = net_index
        resources["net_names"] = net_names
        
    except Exception as e:
        print(f"警告：构建 Net 向量索引失败: {e}")
        print("将回退到传递完整码表。")
        resources["net_index"] = None
        resources["net_names"] = []

    # --- 数据清洗与预处理 ---
    print("--- 步骤 4: 正在进行数据清洗与预处理 ---")
    uncoded_df['text'] = uncoded_df['text'].str.strip()
    uncoded_df = uncoded_df[uncoded_df['text'] != '']
    uncoded_df.reset_index(drop=True, inplace=True)

    # --- 批次内去重 ---
    uncoded_df['is_duplicate_in_batch'] = uncoded_df.duplicated(subset=['text'], keep='first')
    uncoded_df_with_explicit_index = uncoded_df.reset_index(drop=False)
    non_duplicate_df = uncoded_df_with_explicit_index[~uncoded_df_with_explicit_index['is_duplicate_in_batch']]
    first_occurrence_map = non_duplicate_df.set_index('text')['index'].to_dict()
    duplicate_df = uncoded_df_with_explicit_index[uncoded_df_with_explicit_index['is_duplicate_in_batch']]
    duplicate_indices_and_texts = duplicate_df.set_index('index')['text']
    duplicate_to_first_map_series = duplicate_indices_and_texts.map(first_occurrence_map)
    duplicate_to_first_map = duplicate_to_first_map_series.to_dict()

    # --- 初始化统计信息 ---
    import threading
    stats = {
        "total": len(uncoded_df),
        "prefiltered_hits": 0,
        "kb_exact_hits": 0,
        "similarity_hits": 0,
        "api_calls": 0,
        "errors": 0,
        "reviewed_hits": 0,
        "batch_deduplicated": len(duplicate_to_first_map),
        "lock": threading.Lock() # 用于线程安全更新
    }
    
    print(f"--- 步骤 5: 开始批量编码 {stats['total']} 条新数据 "
          f"(线程数: {NUM_THREADS}, 其中 {stats['batch_deduplicated']} 条为批次内重复) ---")

    # --- 准备共享数据 ---
    shared_data = {
        'exact_match_lookup': exact_match_lookup,
        'reviewed_lookup': reviewed_lookup,
        'canonical_question_text': canonical_question_text,
        'preloaded_sub_index': preloaded_sub_index,
        'original_indices_for_sub_index': original_indices_for_sub_index,
        'resources': resources,
        'stats_ref': stats # 传递 stats 字典的引用
    }

    # --- 多线程处理唯一项 ---
    unique_items_df = uncoded_df[~uncoded_df['is_duplicate_in_batch']]
    tasks = list(unique_items_df.iterrows())
    output_records = [None] * len(uncoded_df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_task = {
            executor.submit(process_single_row, task, shared_data): task for task in tasks
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Coding Unique Items"):
            task = future_to_task[future]
            try:
                result = future.result()
                original_index = result['original_index']
                output_records[original_index] = result
            except Exception as e:
                print(f"处理任务 {task[0]} 时发生错误: {e}")
                traceback.print_exc()
                index, row = task
                with stats['lock']:
                    stats['errors'] += 1
                output_records[index] = {
                    "uuid": row.get('uuid', ''),
                    "question": BATCH_QUESTION_TEXT,
                    "text": row['text'],
                    "process_method": "MAIN_THREAD_ERROR",
                    "coding_results": [{"error": str(e), "traceback": traceback.format_exc()}],
                    "original_index": index
                }

    # --- 步骤 6: 处理批次内重复项 ---
    print("--- 步骤 6: 正在处理批次内重复数据 ---")
    for dup_index, first_index in tqdm(duplicate_to_first_map.items(), desc="Copying Duplicate Results", total=len(duplicate_to_first_map)):
        source_result = output_records[first_index]
        if source_result:
            dup_row = uncoded_df.iloc[dup_index]
            copied_result = source_result.copy()
            copied_result['uuid'] = dup_row.get('uuid', '')
            copied_result['process_method'] = f"COPY_OF_{source_result['process_method']}"
            output_records[dup_index] = copied_result
        else:
            print(f"警告：无法找到索引 {first_index} 的结果来复制给索引 {dup_index}")
            dup_row = uncoded_df.iloc[dup_index]
            output_records[dup_index] = {
                "uuid": dup_row.get('uuid', ''),
                "question": BATCH_QUESTION_TEXT,
                "text": dup_row['text'],
                "process_method": "COPY_ERROR",
                "coding_results": [{"error": "源结果未找到"}]
            }

    # --- 清理并保存结果 ---
    for record in output_records:
        if record and 'original_index' in record:
            del record['original_index']

    output_records = [r for r in output_records if r is not None]

    if output_records:
        print(f"\n--- 正在保存已处理的 {len(output_records)} 条记录... ---")
        safe_batch_name = re.sub(r'[\\/*?:"<>|]', "", BATCH_QUESTION_TEXT)[:50]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(RESULTS_FOLDER, f"coded_batch_{safe_batch_name}_{timestamp}.jsonl")
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                for record in output_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"结果已保存至 {output_filename}")
            
            print("-" * 50)
            print("本次任务运行统计:")
            print(f"本次运行总处理条数: {len(output_records)}")
            print(f"  - 预过滤命中: {stats['prefiltered_hits']}")
            print(f"  - 知识库精准匹配命中: {stats['kb_exact_hits']}")
            print(f"  - 相似度匹配命中: {stats['similarity_hits']}")
            print(f"  - 已校正数据直接使用: {stats['reviewed_hits']}")
            print(f"  - 批次内重复数据去重: {stats['batch_deduplicated']}")
            print(f"  - 实际API调用次数: {stats['api_calls']}")
            print(f"  - 处理失败条数: {stats['errors']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
            traceback.print_exc()
    else:
        print("没有生成任何输出记录。")


if __name__ == "__main__":
    main()
