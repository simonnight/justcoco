# 3_run_batch_coding.py
# 權威最終版: 專為「單問題」批處理設計，整合所有功能、修正和性能優化的最高效腳本。

import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
import json
import time
import pickle
import logging
from tqdm.asyncio import tqdm as asyncio_tqdm
import asyncio
import traceback
import re
import shutil
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import hashlib # 導入 hashlib 庫以生成穩定的哈希 UUID
import google.api_core.exceptions

# --- 【【【每次運行時，請修改此處的批處理配置】】】 ---
# 1. 精確複製/貼上您當前要處理的問題文本
BATCH_QUESTION_TEXT = "评分的原因" 
# 2. 指定對應的批次文件名 (確保文件在04_input_batches_to_code文件夾內)
BATCH_INPUT_FILE = "04_input_batches_to_code/batch_问题A.csv" 
# ----------------------------------------------------

# --- 其他配置 ---
# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    logging.error("錯誤：請先設置 GOOGLE_API_KEY 環境變數。")
    exit()
    
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash-lite"
SIMILARITY_COPY_THRESHOLD = 0.2
K_VALUE = 3
CONCURRENT_REQUESTS = 50 # 併發請求數

# --- 文件路徑定義 ---
CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"
KB_FOLDER = "03_knowledge_base"
ANSWER_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "answer.index")
QUESTION_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "question.index")
DATA_MAP_FILE = os.path.join(KB_FOLDER, "data_map.pkl")
UNIQUE_QUESTIONS_FILE = os.path.join(KB_FOLDER, "unique_questions.pkl")
QUESTION_TO_SUB_INDEX_MAP_FILE = os.path.join(KB_FOLDER, "question_to_sub_index_map.pkl")

RESULTS_FOLDER = "05_coded_results"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

PERSISTENT_CACHE_FILE = "project_coding_cache.jsonl"
CACHE_BACKUP_FOLDER = "cache_backups"

# ==============================================================================
# --- 輔助函數定義區 ---
# ==============================================================================

# 帶有重試邏輯的嵌入 API 調用
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) |
          retry_if_exception_type(google.api_core.exceptions.InternalServerError) |
          retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
)
async def _embed_content_with_retry(content, task_type):
    response = await genai.embed_content_async(model=EMBEDDING_MODEL, content=content, task_type=task_type)
    if not response or 'embedding' not in response:
        raise ValueError(f"嵌入 API 返回空響應或無嵌入數據。響應: {response}")
    return response['embedding']

# 帶有重試邏輯的生成 API 調用
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) |
          retry_if_exception_type(google.api_core.exceptions.InternalServerError) |
          retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable) |
          retry_if_exception_type(ValueError) # 新增: 處理 ValueError，包括非JSON格式
)
async def _generate_content_with_retry(model, prompt):
    response = await model.generate_content_async(prompt)
    
    if not response:
        raise ValueError("生成 API 返回空響應。")
    
    # 檢查安全攔截導致的空響應
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        block_reason = response.prompt_feedback.block_reason
        raise ValueError(f"生成 API 返回空響應，原因: 安全攔截 - {block_reason}")

    # 確保 response.text 存在且非空
    if not response.text:
        raise ValueError("生成 API 返回空響應，無文本內容。")

    try:
        # 嘗試解析 JSON
        json_result = json.loads(response.text)
        return json_result
    except json.JSONDecodeError as e:
        # 如果不是有效的 JSON，記錄警告並拋出 ValueError，讓 tenacity 捕獲並重試
        logging.warning(f"AI 生成的文本不是有效的 JSON 格式。錯誤: {e}. 原始文本: '{response.text}'")
        raise ValueError(f"AI 輸出非 JSON 格式: {response.text[:100]}...") from e

def find_best_matching_question_index(question_index, unique_questions, new_question: str) -> int:
    """在歷史問題中找到語義最相近的問題的索引號"""
    response = genai.embed_content(model=EMBEDDING_MODEL, content=new_question, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([response['embedding']]).astype('float32')
    distances, indices = question_index.search(query_vector, 1)
    return indices[0][0]

def format_examples_rich(df_slice, codebook):
    """格式化案例，展示多層級信息（不含sentiment）"""
    lines = []
    for _, row in df_slice.iterrows():
        code = str(row['code'])
        if code in codebook.index:
            code_info = codebook.loc[code]
            if isinstance(code_info, pd.DataFrame): code_info = code_info.iloc[0]
            lines.append(f"- 問題: \"{row['question']}\"\n  回答: \"{row['text']}\"\n  -> 編碼結果: {{ net: '{code_info.net}', subnet: '{code_info.subnet}', code: {code}, label: '{code_info.label}' }}")
    return "\n".join(lines)

def clean_code_info(code_series, code: str, is_new: bool) -> dict:
    """將Pandas Series轉換為純淨的字典（不含sentiment）"""
    if isinstance(code_series, pd.DataFrame):
        record = code_series.iloc[0]
    else:
        record = code_series
    net = record.net if pd.notna(record.net) else ""
    subnet = record.subnet if pd.notna(record.subnet) else ""
    label = record.label if pd.notna(record.label) else ""
    return {'net': str(net), 'subnet': str(subnet), 'code': code, 'label': str(label), 'is_new_suggestion': is_new}

async def rag_pipeline_async(new_text: str, resources, preloaded_sub_index, original_indices_for_sub_index, stats):
    """異步RAG流水線，處理單條數據"""
    try:
        df_map = resources["df_map"]
        codebook_df = resources["codebook_df"]
        model = resources["model"]
        answer_index = resources["answer_index"]
        
        # 為了避免在異步函數中重複創建嵌入向量，我們在外面創建一次
        query_embedding = await _embed_content_with_retry(new_text, "RETRIEVAL_QUERY")
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 第三層：歷史相似匹配 (使用子索引)
        if preloaded_sub_index.ntotal > 0:
            distances, sub_indices = preloaded_sub_index.search(query_vector, 1)
            if distances[0][0] < SIMILARITY_COPY_THRESHOLD:
                stats['similarity_hits'] += 1
                best_match_original_idx = original_indices_for_sub_index[sub_indices[0][0]]
                codes_to_copy = df_map[df_map.index == best_match_original_idx]['code'].tolist()
                unique_codes = sorted(list(set(codes_to_copy)))
                copied_codes = []
                for c in unique_codes:
                    if str(c) in codebook_df.index: # 確保索引是字符串類型
                         code_series = codebook_df.loc[str(c)]
                         code_info = clean_code_info(code_series, str(c), is_new=False)
                         copied_codes.append(code_info)
                return copied_codes, "SIMILARITY_MATCH_COPY"
        
        # 第四層：完整RAG+LLM調用 (使用子索引或總索引)
        stats['api_calls'] += 1
        prompt_mode, relevant_examples = "zero_shot", pd.DataFrame()

        if preloaded_sub_index.ntotal > 0:
            distances, sub_indices = preloaded_sub_index.search(query_vector, min(K_VALUE, preloaded_sub_index.ntotal))
            if distances[0][0] < 0.9: # 調整相似度閾值以控制案例相關性
                prompt_mode = "standard"
                original_indices = [original_indices_for_sub_index[i] for i in sub_indices[0]]
                relevant_examples = df_map.loc[original_indices]
        
        if prompt_mode == "zero_shot":
            distances, indices = answer_index.search(query_vector, K_VALUE)
            if distances[0][0] < 0.9: # 調整相似度閾值以控制案例相關性
                prompt_mode, relevant_examples = "cross_context_warning", df_map.iloc[indices[0]]
        
        relevant_definitions_str, contextual_codes_str = "", ""
        if not relevant_examples.empty:
            unique_codes_from_examples = relevant_examples['code'].unique().astype(str)
            # 過濾掉碼表中不存在的代碼，然後再使用 .loc 進行索引
            existing_codes = codebook_df.index.intersection(unique_codes_from_examples)
            relevant_code_info_df = codebook_df.loc[existing_codes]
            relevant_definitions_str = relevant_code_info_df.to_string()
            
            # 確保 relevant_examples 不為空且 top_example_code 在 codebook_df 中
            if not relevant_examples.empty and str(relevant_examples.iloc[0]['code']) in codebook_df.index:
                top_example_code = str(relevant_examples.iloc[0]['code'])
                top_example_info = codebook_df.loc[top_example_code]
                if isinstance(top_example_info, pd.DataFrame): top_example_info = top_example_info.iloc[0]
                main_net, main_subnet = top_example_info['net'], top_example_info['subnet']
                
                contextual_codes_df = codebook_df[(codebook_df['net'] == main_net) & (codebook_df['subnet'] == main_subnet)]
                if not contextual_codes_df.empty:
                    contextual_codes_str = f"# 3. 相關上下文編碼 (供您生成新編碼時參考)\n{contextual_codes_df[['label']].to_string()}"
        
        prompt_intro = f"# 1. 當前上下文\n- 當前處理的問題: \"{BATCH_QUESTION_TEXT}\"\n- 使用者的原始回答: \"{new_text}\"\n\n# 2. 相關編碼定義\n{relevant_definitions_str if not relevant_examples.empty else '警告：無相關案例，請參考下方完整碼表。'}"
        
        if prompt_mode == "zero_shot":
            full_codebook_str = codebook_df.to_string()
            prompt_references = f"# 3. 完整碼表參考 (無相關案例)\n{full_codebook_str}"
        else:
            warning_message = "# [重要警告]: 以下為跨上下文檢索案例...\n" if prompt_mode == "cross_context_warning" else ""
            prompt_references = f"# 2. 相關歷史案例\n{warning_message}{format_examples_rich(relevant_examples, codebook_df)}\n\n{contextual_codes_str}"
        
        user_prompt = f"""
        請根據以下信息和你的專家知識，為用戶反饋進行專業的多層級編碼。
        {prompt_intro}
        {prompt_references}
        ---
        # 分析與決策指令
        請嚴格遵循以下思考過程：
        1.  **優先匹配**: 檢查“使用者的原始回答”是否與“相關歷史案例”中某個編碼的`label`高度一致。如果一致，必須優先直接使用該編碼。
        2.  **必要時創造**: 僅在找不到任何強匹配項，或參考`label`過於寬泛時，才為觀點提煉一個更精準的新`label`，在相同層級下生成一個邏輯連續的新`code`，並設置`is_new_suggestion: true`。
        # 輸出要求
        請以一個【JSON數組】的格式輸出你的最終決策。如果沒有任何編碼適用，請返回一個空數組 `[]`。
        [
          {{
            "net": "主類別",
            "subnet": "子類別",
            "code": "一個有效的三位數字編碼",
            "label": "具體標籤",
            "is_new_suggestion": false
          }}
        ]
        """
        
        json_result = await _generate_content_with_retry(model, user_prompt)
        return json_result, "API_CALL"
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"RAG流水線中發生錯誤: {e}\n{error_traceback}")
        return [{"error": str(e)}], "ERROR"


async def main():
    """主異步執行函數"""
    try:
        system_instruction = "作為質性研究編碼專家，你的任務是：\n1. 識別反饋中的所有獨立觀點。\n2. 為每個觀點精確匹配現有編碼。\n3. 若現有編碼過寬泛，則在相同層級下建議更具體的新編碼。\n4. 必須嚴格返回JSON格式。"
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=system_instruction, generation_config=generation_config)
        
        logging.info("--- 步驟 1: 正在加載所有資源 ---")
        resources = {
            "df_map": pd.read_pickle(DATA_MAP_FILE),
            "codebook_df": pd.read_csv(CODEBOOK_FILE, dtype={'code': str}).set_index('code'),
            "model": model,
            "answer_index": faiss.read_index(ANSWER_FAISS_INDEX_FILE)
        }
        
        # --- 關鍵改動: 嚴格檢查並使用 uuid 列 ---
        try:
            uncoded_df = pd.read_csv(BATCH_INPUT_FILE, dtype=str).fillna('')
        except FileNotFoundError:
            logging.critical(f"錯誤：找不到輸入批次文件 {BATCH_INPUT_FILE}。請檢查文件路徑。")
            exit()
        
        if 'text' not in uncoded_df.columns or 'question' not in uncoded_df.columns:
            logging.critical(f"錯誤：輸入批次文件 {BATCH_INPUT_FILE} 缺少必要的 'text' 或 'question' 列。")
            exit()

        if 'uuid' not in uncoded_df.columns:
            logging.critical(f"錯誤：輸入批次文件 {BATCH_INPUT_FILE} 缺少 'uuid' 列。'uuid' 作為定位碼是強制要求，請確保 1a_prepare_new_data.py 已生成此列。")
            exit()
        
        # 確保 text 和 question 列沒有 NaN
        uncoded_df.dropna(subset=['text', 'question'], inplace=True)
        if uncoded_df.empty:
            logging.warning("警告：輸入批次文件經過清洗後為空。")
            exit()

        logging.info("資源加載成功！")
    except Exception as e:
        logging.critical(f"錯誤：啟動時加載資源失敗: {e}\n{traceback.format_exc()}")
        exit()

    logging.info(f"--- 步驟 2: 為批處理問題“{BATCH_QUESTION_TEXT[:30]}...”準備上下文 ---")
    question_index = faiss.read_index(QUESTION_FAISS_INDEX_FILE)
    with open(UNIQUE_QUESTIONS_FILE, 'rb') as f: unique_questions = pickle.load(f)
    with open(QUESTION_TO_SUB_INDEX_MAP_FILE, 'rb') as f: question_to_sub_index_map = pickle.load(f)
    best_match_q_idx = find_best_matching_question_index(question_index, unique_questions, BATCH_QUESTION_TEXT)
    canonical_question_text = unique_questions[best_match_q_idx]
    logging.info(f"智能匹配到最相似的歷史問題: “{canonical_question_text}”")
    sub_index_info = question_to_sub_index_map[canonical_question_text]
    preloaded_sub_index = faiss.read_index(sub_index_info["index_file"])
    original_indices_for_sub_index = sub_index_info["original_indices"]
    logging.info("專屬迷你索引已成功加載到內存。")

    logging.info(f"--- 步驟 3: 正在加載或創建持久化緩存日誌: {PERSISTENT_CACHE_FILE} ---")
    if os.path.exists(PERSISTENT_CACHE_FILE):
        if not os.path.exists(CACHE_BACKUP_FOLDER): os.makedirs(CACHE_BACKUP_FOLDER)
        backup_filename = f"{os.path.basename(PERSISTENT_CACHE_FILE)}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(PERSISTENT_CACHE_FILE, os.path.join(CACHE_BACKUP_FOLDER, backup_filename))
        logging.info(f"已將當前緩存備份至: {backup_filename}")
    
    coding_cache = {}
    if os.path.exists(PERSISTENT_CACHE_FILE):
        with open(PERSISTENT_CACHE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # 緩存鍵現在基於 question 和 text 的哈希值，以提高穩定性
                    key = hashlib.md5(f"{record['question']}_{record['text']}".encode('utf-8')).hexdigest()
                    coding_cache[key] = record['coding_results']
                except (json.JSONDecodeError, KeyError): continue
    logging.info(f"成功加載 {len(coding_cache)} 條持久化緩存記錄。")

    stats = {"total": len(uncoded_df), "exact_hits": 0, "similarity_hits": 0, "api_calls": 0, "errors": 0}
    
    async def process_item_and_cache(item_row, resources, preloaded_sub_index, original_indices_for_sub_index, stats, cache_log_file):
        """處理單個項目的異步函數，包含緩存檢查和結果寫入緩存"""
        current_text = item_row['text']
        cache_key = hashlib.md5(f"{canonical_question_text}_{current_text}".encode('utf-8')).hexdigest()
        
        coding_results_list = None
        process_method = ""

        if cache_key in coding_cache:
            coding_results_list = coding_cache[cache_key]
            process_method = "PERSISTENT_CACHE_HIT"
            stats['exact_hits'] += 1
        else:
            coding_results_list, call_type = await rag_pipeline_async(current_text, resources, preloaded_sub_index, original_indices_for_sub_index, stats)
            process_method = call_type.upper()
            
            is_valid_result = not (not coding_results_list or (isinstance(coding_results_list, list) and coding_results_list and isinstance(coding_results_list[0], dict) and "error" in coding_results_list[0]))
            
            # 只在 API 調用成功並生成有效結果時寫入持久化緩存
            if call_type == "API_CALL" and is_valid_result:
                log_entry = {"question": canonical_question_text, "text": current_text, "coding_results": coding_results_list}
                cache_log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                # 即使寫入文件，也要更新內存中的 coding_cache，以防後續相同任務
                coding_cache[cache_key] = coding_results_list 
            elif call_type == "ERROR" or not is_valid_result:
                stats['errors'] += 1 # 記錄 RAG 流水線的錯誤

        final_result = {"uuid": item_row['uuid'], "question": BATCH_QUESTION_TEXT, "text": current_text, "process_method": process_method, "coding_results": coding_results_list}
        return final_result

    logging.info(f"--- 步驟 4: 開始異步處理 {stats['total']} 條記錄 ---")
    
    final_output_records = []
    
    # 使用 a+ 模式打開文件，以便在寫入時直接附加，同時保證緩存的實時更新
    with open(PERSISTENT_CACHE_FILE, 'a+', encoding='utf-8') as cache_log_file:
        tasks = []
        for index, row in uncoded_df.iterrows():
            tasks.append(
                asyncio.create_task(
                    process_item_and_cache(row, resources, preloaded_sub_index, original_indices_for_sub_index, stats, cache_log_file)
                )
            )
        
        for task in asyncio_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Coding Batch"):
            try:
                record = await task
                final_output_records.append(record)
            except Exception as e:
                # 這裡的錯誤應該極少，因為 rag_pipeline_async 已經做了捕獲，
                # 但作為最終保障，以防 task.result() 拋出意外錯誤
                logging.critical(f"處理單個任務時發生未預期的嚴重錯誤: {e}\n{traceback.format_exc()}")
                stats['errors'] += 1
                # 即使出錯，也要確保有一個輸出記錄，雖然內容是錯誤的
                final_output_records.append({"uuid": "unknown_fatal_error", "error": str(e), "question": BATCH_QUESTION_TEXT, "text": "UNKNOWN_TEXT"})

    safe_batch_name = re.sub(r'[\\/*?:"<>|]', "", BATCH_QUESTION_TEXT)[:50]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(RESULTS_FOLDER, f"coded_batch_{safe_batch_name}_{timestamp}.jsonl")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for record in final_output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logging.info("-" * 50)
    logging.info("本次任務運行統計:")
    logging.info(f"總處理條數: {stats['total']}")
    logging.info(f"緩存命中: {stats['exact_hits']}")
    logging.info(f"相似度匹配命中: {stats['similarity_hits']}")
    logging.info(f"實際API調用次數: {stats['api_calls']}")
    total_hits = stats['exact_hits'] + stats['similarity_hits']
    hit_rate = (total_hits / stats['total'] * 100) if stats['total'] > 0 else 0
    logging.info(f"總命中率 (無需API調用): {hit_rate:.2f}%")
    logging.info(f"結果已保存至 {output_filename}")
    logging.info(f"持久化緩存 {PERSISTENT_CACHE_FILE} 已更新。")
    logging.info("-" * 50)

if __name__ == "__main__":
    try:
        # 檢查並安裝 tenacity 和 tqdm[asyncio] 庫
        try:
            import tenacity
            from tqdm.asyncio import tqdm as asyncio_tqdm
        except ImportError:
            logging.error("錯誤: 缺少必要的庫。請運行: pip install tenacity \"tqdm[asyncio]\"")
            exit()
        
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"腳本執行過程中發生嚴重錯誤: {e}\n{traceback.format_exc()}")
        exit(1)
