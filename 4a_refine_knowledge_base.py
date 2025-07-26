# 4a_refine_knowledge_base.py
# 權威最終版: 整合了斷點續傳、錯誤隔離、增量日誌 和 交互式重試功能。
# 新增功能: 簡化UUID處理邏輯，優先使用現有UUID，否則自動生成基於內容的臨時UUID。
# 改進: 增加AI API調用的指數退避和輸出文件重複處理。
# 修正: 使用更穩定的MD5哈希算法確保UUID在不同運行中的一致性，修復斷點續傳失效問題。
# 更新: 失敗重試時可選擇使用 Gemini 2.5 Pro 模型。
# 更新: 自動重試處理非 'NO_MATCH' 類型的錯誤記錄，放在交互式重試之前。
# 修正: 修復了重複處理已成功數據的BUG，確保日誌中的記錄帶有明確的 final_status_category。
# 修正: 進一步強化UUID生成穩定性，確保數據在不同運行間的精確匹配。
# 新增: 如果原始數據沒有 'uuid' 列，將會添加一個名為 'uuid' 的固定列，用於持久化唯一識別符。
# 修正: 解決 'numpy.int64' object has no attribute 'fillna' 錯誤，並強化 original_code_count 處理。

import pandas as pd
import google.generativeai as genai
import google.api_core.exceptions
import os
import time
from tqdm.asyncio import tqdm as asyncio_tqdm
import json
import asyncio
import traceback
import shutil
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import hashlib # 導入 hashlib 庫

# --- 配置 ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("錯誤：請先設置 GOOGLE_API_KEY 環境變數。")
    exit()

# 根據您的選擇，預設使用 gemini-2.5-flash-lite
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash-lite"  
PRO_GENERATION_MODEL = "gemini-2.5-pro" # 新增 Pro 模型名稱

CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"
MESSY_DATA_FILE = "02_preprocessed_data/last_phase_coded_data.csv"

# --- 性能配置 ---
CONCURRENT_REQUESTS = 40  
MIN_TEXT_LENGTH_TO_REFINE = 20  

# --- AI模型初始化 ---
# 預設模型實例
default_model_instance = genai.GenerativeModel(DEFAULT_GENERATION_MODEL)
# Pro 模型實例，按需初始化
pro_model_instance = None 

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60), # 1s, 2s, 4s, 8s, 16s, 32s, 60s...
    stop=stop_after_attempt(5), # 最多重試5次
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) | # 速率限制
          retry_if_exception_type(google.api_core.exceptions.InternalServerError) | # 服務器內部錯誤
          retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable) # 服務不可用
)
async def _call_gemini_api_with_retry(prompt: str, model_instance: genai.GenerativeModel):
    """帶有指數退避重試邏輯的 Gemini API 調用輔助函數。
    接受一個模型實例作為參數，允許切換模型。
    """
    response = await model_instance.generate_content_async(prompt)
    if not response.text:
        # 檢查是否有安全攔截或其他原因導致空響應
        block_reason = response.prompt_feedback.block_reason
        if block_reason:
            raise Exception(f"API響應為空，原因: 安全攔截 - {block_reason}")
        else:
            raise Exception("API響應為空，無明確原因。")
    return response.text.strip()


async def refine_text_async(long_text: str, code_label: str, internal_uuid: str, model_instance: genai.GenerativeModel):
    """異步調用AI進行文本精煉，並返回帶內部UUID的結果。
    現在接受一個模型實例。
    """
    try:
        prompt = f"""
        # 任務：文本精煉
        從“原始長句”中，只抽取出與“目標標籤”最直接相關的核心短句。
        # 規則:
        1. 如能找到精確短語，則返回短語。
        2. 如整句話都相關，則返回原話。
        3. 如不相關，返回 "NO_MATCH"。
        4. 只回答精煉短語或 "NO_MATCH"。
        # 原始長句: "{long_text}"
        # 目標標籤: "{code_label}"
        # 精煉後的核心短語:
        """
        # 直接調用帶重試邏輯的輔助函數，傳入當前模型實例
        refined_text = await _call_gemini_api_with_retry(prompt, model_instance)
        
        if refined_text == "NO_MATCH":
            return internal_uuid, refined_text # 返回 NO_MATCH 狀態
        return internal_uuid, refined_text # 成功精煉
        
    except Exception as e:
        # 捕獲所有其他異常，包括重試後仍然失敗的異常
        return internal_uuid, f"ERROR: {str(e)}"

async def process_batch(items_to_process: list, codebook_df, pass_num: int, model_instance: genai.GenerativeModel):
    """
    處理一個批次數據的核心函數。
    現在接受一個模型實例參數，用於所有 AI 調用。
    返回: (成功記錄列表, 失敗記錄列表)
    """
    if not items_to_process:
        return [], []

    print(f"\n--- 開始執行第 {pass_num} 遍清洗，共 {len(items_to_process)} 條記錄 (使用模型: {model_instance.model_name}) ---") # 顯示正在使用的模型
    
    # 創建一個以內部uuid為鍵的原始行字典，方便查找
    row_dict_map = {row['uuid_for_processing']: row for row in items_to_process}
    
    successful_records = []
    failed_records = []

    # 使用tqdm的異步版本來顯示進度
    pbar = asyncio_tqdm(total=len(items_to_process), desc=f"Processing Pass {pass_num}")

    for i in range(0, len(items_to_process), CONCURRENT_REQUESTS):
        batch_rows_dicts = items_to_process[i:i+CONCURRENT_REQUESTS]
        
        async_tasks = []
        for row_dict in batch_rows_dicts:
            code = row_dict['code'] # 假設原始數據中有 'code' 列
            if code in codebook_df.index:
                code_label = codebook_df.loc[code, 'label']
                # 傳遞當前模型實例給 refine_text_async
                async_tasks.append(refine_text_async(row_dict['text'], code_label, row_dict['uuid_for_processing'], model_instance))
            else:
                # 處理碼表中不存在的 code
                new_record_for_fail = row_dict.copy()
                new_record_for_fail['error_or_status'] = "UNKNOWN_CODE"
                new_record_for_fail['final_status_category'] = 'ERROR' # 明確標記為 ERROR 類型
                failed_records.append(new_record_for_fail)
        
        if async_tasks:
            results = await asyncio.gather(*async_tasks)
            pbar.update(len(async_tasks))
        
            for internal_uuid_res, refined_text in results:
                original_row = row_dict_map.get(internal_uuid_res)
                if original_row:
                    new_record = original_row.copy()
                    # 將結果存回原始列
                    if refined_text.startswith("ERROR:"):
                        new_record['error_or_status'] = refined_text
                        new_record['final_status_category'] = 'ERROR' # 明確標記為 ERROR 類型
                        failed_records.append(new_record)
                    elif refined_text == "NO_MATCH":
                        new_record['error_or_status'] = refined_text # "NO_MATCH" 是一個狀態，不是一個錯誤類型
                        new_record['final_status_category'] = 'NO_MATCH' # 明確標記為 NO_MATCH 類型
                        failed_records.append(new_record)
                    else:
                        new_record['text'] = refined_text # 將精煉後的文本覆蓋原文本
                        new_record['final_status_category'] = 'SUCCESS' # 明確標記為 SUCCESS 類型
                        successful_records.append(new_record)
                else:
                    # 這是極端情況，如果 uuid_for_processing 在 map 中找不到
                    print(f"警告: 內部 UUID {internal_uuid_res} 在原始批次映射中找不到。")
    
    pbar.close()
    print(f"--- 第 {pass_num} 遍清洗完成 ---")
    print(f"成功: {len(successful_records)} 條, 失敗: {len(failed_records)} 條")
    return successful_records, failed_records


async def main():
    """主執行函數"""
    global pro_model_instance # 聲明使用全局變量

    base_name = os.path.splitext(os.path.basename(MESSY_DATA_FILE))[0]
    output_dir = os.path.dirname(MESSY_DATA_FILE)
    refined_output_file = os.path.join(output_dir, f"{base_name}_REFINED.csv")
    progress_log_file = os.path.join(output_dir, f"{base_name}_progress.jsonl")
    error_log_file = os.path.join(output_dir, f"{base_name}_errors.jsonl")
    no_match_log_file = os.path.join(output_dir, f"{base_name}_no_match.jsonl")

    print("--- 開始健壯模式下的數據清洗 (交互版) ---")
    
    # --- 輸出文件重複處理邏輯 ---
    if os.path.exists(refined_output_file):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = f"{refined_output_file}.bak_{timestamp}"
        shutil.copy2(refined_output_file, backup_file)
        print(f"警告: 輸出文件 '{refined_output_file}' 已存在，已備份為 '{backup_file}'。")
        # 可以選擇清除舊的日誌文件以避免混淆，或讓用戶手動管理
        # for log_file in [progress_log_file, error_log_file, no_match_log_file]:
        #     if os.path.exists(log_file):
        #         os.remove(log_file)
        #         print(f"已清除舊日誌文件: {log_file}")


    try:
        source_df = pd.read_csv(MESSY_DATA_FILE, dtype=str)
        # 備份原始列名，用於最終輸出文件的列順序和篩選
        original_source_columns = source_df.columns.tolist()

        # --- 智能 UUID 處理邏輯 ---
        # 確保 original_code_count_str 始終是穩定的字符串表示
        def get_stable_code_count_str(value):
            num_val = pd.to_numeric(value, errors='coerce')
            if pd.notna(num_val):
                return str(int(num_val))
            return 'NA'

        if 'uuid' in source_df.columns:
            # 如果有 'uuid' 列，優先使用它，並檢查重複值
            print("檢測到 'uuid' 列，嘗試將其用作內部唯一標識符。")
            if source_df['uuid'].duplicated().any():
                print("警告: 原始文件中的 'uuid' 列包含重複值。將自動生成基於內容的唯一 UUID 並覆蓋 'uuid' 列以確保唯一性。")
                source_df['uuid'] = source_df.apply( # 覆蓋現有的 uuid 列
                    lambda row: hashlib.md5(
                        f"{str(row['text']).strip()}_{str(row['code']).strip()}_{get_stable_code_count_str(row['original_code_count'])}".encode('utf-8')
                    ).hexdigest(),
                    axis=1
                ).astype(str)
            # 無論是否重複，uuid_for_processing 都指向這個唯一的 uuid 列
            source_df['uuid_for_processing'] = source_df['uuid'].astype(str) 
        else:
            # 如果沒有 'uuid' 列，則新增一個並用基於內容的唯一 UUID 填充
            print("警告: 原始文件未包含 'uuid' 列，正在新增 'uuid' 列並為每條記錄自動生成基於內容的唯一 UUID。")
            source_df['uuid'] = source_df.apply( # 新增 'uuid' 列
                lambda row: hashlib.md5(
                    f"{str(row['text']).strip()}_{str(row['code']).strip()}_{get_stable_code_count_str(row['original_code_count'])}".encode('utf-8')
                ).hexdigest(),
                axis=1
            ).astype(str)
            source_df['uuid_for_processing'] = source_df['uuid'].astype(str) # 內部處理也使用這個新列

        # 檢查關鍵的 'original_code_count' 和 'text' 列是否存在
        if 'original_code_count' not in source_df.columns:
            print("錯誤: 輸入文件缺少 'original_code_count' 列。請補上後重試。")
            return
        if 'text' not in source_df.columns:
            print("錯誤: 輸入文件缺少 'text' 列（或您需要修改腳本以指定實際的文本列名）。請補上後重試。")
            return
        if 'code' not in source_df.columns:
            print("錯誤: 輸入文件缺少 'code' 列（或您需要修改腳本以指定實際的編碼列名）。請補上後重試。")
            return
            
        codebook_df = pd.read_csv(CODEBOOK_FILE, dtype={'code': str}).set_index('code')
    except Exception as e:
        print(f"錯誤：加載文件時失敗: {e}")
        return

    # 斷點續傳邏輯 (第一次運行前檢查日誌)
    # 使用一個字典來聚合所有日誌中的記錄狀態，以 UUID 為鍵
    all_processed_records_from_logs = {} 
    processed_uuids_from_logs_set = set() # 僅用於快速查詢已處理的 UUID

    # ---- 關鍵修正點：確保 all_processed_records_from_logs 在啟動時正確填充所有歷史狀態 ----
    # 這裡的邏輯是按優先級從日誌文件中讀取並更新記錄狀態
    # 優先級 1: 成功處理的記錄 (progress_log_file)
    if os.path.exists(progress_log_file):
        with open(progress_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    record = json.loads(line)
                    uuid = record.get('uuid_for_processing')
                    if not uuid: continue 
                    record['final_status_category'] = 'SUCCESS' 
                    all_processed_records_from_logs[uuid] = record
                    processed_uuids_from_logs_set.add(uuid)
                except (json.JSONDecodeError, KeyError): continue

    # 優先級 2: 未匹配 (NO_MATCH) 的記錄 (no_match_log_file)
    if os.path.exists(no_match_log_file):
        with open(no_match_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    record = json.loads(line)
                    uuid = record.get('uuid_for_processing')
                    if not uuid: continue 
                    # 只有當此 UUID 尚未被標記為 'SUCCESS' 時才更新為 'NO_MATCH'
                    if all_processed_records_from_logs.get(uuid, {}).get('final_status_category') != 'SUCCESS':
                        record['final_status_category'] = 'NO_MATCH'
                        all_processed_records_from_logs[uuid] = record
                        processed_uuids_from_logs_set.add(uuid)
                except (json.JSONDecodeError, KeyError): continue

    # 優先級 3: 其他錯誤的記錄 (error_log_file)
    if os.path.exists(error_log_file):
        with open(error_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    record = json.loads(line)
                    uuid = record.get('uuid_for_processing')
                    if not uuid: continue 
                    # 只有當此 UUID 尚未被標記為 'SUCCESS' 或 'NO_MATCH' 時才更新為 'ERROR'
                    current_status = all_processed_records_from_logs.get(uuid, {}).get('final_status_category')
                    if current_status not in ['SUCCESS', 'NO_MATCH']:
                        record['final_status_category'] = 'ERROR'
                        all_processed_records_from_logs[uuid] = record
                        processed_uuids_from_logs_set.add(uuid)
                except (json.JSONDecodeError, KeyError): continue
    
    if processed_uuids_from_logs_set:
        print(f"已從日誌中恢復 {len(processed_uuids_from_logs_set)} 條已處理記錄的最新狀態。")

    # 篩選出待處理的記錄 (第一遍)
    # 這裡使用 processed_uuids_from_logs_set 來判斷是否已處理
    # 並且只選擇那些在 all_processed_records_from_logs 中狀態不是 SUCCESS 的記錄進行處理
    tasks_for_ai_pass1 = []
    directly_passed_records = []

    # 遍歷原始數據，決定每條記錄在第一遍應該如何處理
    for _, row in source_df.iterrows():
        original_record_dict = row.to_dict()
        uuid = original_record_dict['uuid_for_processing']
        current_aggregated_status = all_processed_records_from_logs.get(uuid, {}).get('final_status_category')

        if current_aggregated_status == 'SUCCESS':
            # 如果已經是成功狀態，則跳過，不納入任何處理隊列
            continue
        else:
            # 如果不是成功狀態，則納入本次待處理，並進行預過濾判斷
            text_content = str(original_record_dict['text'])
            # 確保 original_code_count 在用於哈希和比較時是數值類型，避免浮點數表示問題
            # Fix: 直接處理單個值，避免對 numpy 類型調用 fillna
            original_code_count_val = pd.to_numeric(original_record_dict['original_code_count'], errors='coerce')
            if pd.isna(original_code_count_val):
                original_code_count_val = 1 # 如果轉換後是 NaN，賦予預設值 1

            if len(text_content) > MIN_TEXT_LENGTH_TO_REFINE or original_code_count_val > 1:
                tasks_for_ai_pass1.append(original_record_dict)
            else:
                # 這些記錄被直接採納，並在日誌中標記為成功
                record_for_direct_pass = original_record_dict.copy()
                record_for_direct_pass['final_status_category'] = 'SUCCESS'
                directly_passed_records.append(record_for_direct_pass)
                # 也更新到 all_processed_records_from_logs，確保斷點續傳和後續篩選的準確性
                all_processed_records_from_logs[uuid] = record_for_direct_pass
    
    print("--- 步驟 1: 智能預過濾，篩選需要清洗的數據 (第一遍) ---")
    if not tasks_for_ai_pass1 and not directly_passed_records:
        print("沒有新的或未完成的記錄需要處理。")
    else:
        print(f"預過濾完成。本次運行中，")
        print(f"  {len(tasks_for_ai_pass1)} 條需要AI精煉，{len(directly_passed_records)} 條被直接採納。")
    
        # 將直接採納的記錄先寫入進度日誌
        if directly_passed_records:
            with open(progress_log_file, 'a', encoding='utf-8') as f:
                for record in directly_passed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # --- 第一遍處理 (使用預設模型) ---
    print(f"\n--- 執行第一遍清洗 (使用 {default_model_instance.model_name}) ---")
    pass1_success, pass1_failure = await process_batch(tasks_for_ai_pass1, codebook_df, pass_num=1, model_instance=default_model_instance)
    
    # 增量寫入日誌 (第一遍的結果)，並更新聚合狀態
    with open(progress_log_file, 'a', encoding='utf-8') as prog_f, \
          open(error_log_file, 'a', encoding='utf-8') as err_f, \
          open(no_match_log_file, 'a', encoding='utf-8') as no_match_f:
        for record in pass1_success:
            prog_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態
        for record in pass1_failure:
            if record.get('final_status_category') == 'NO_MATCH': 
                no_match_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            else:
                err_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態


    # --- 新增: 自動重試處理非 'NO_MATCH' 類型的錯誤數據 (Pass 2) ---
    errors_for_automatic_retry = []
    # 遍歷原始數據的所有記錄，基於聚合狀態判斷哪些是 ERROR 且不是 NO_MATCH
    for original_record_dict_full in source_df.to_dict('records'):
        uuid = original_record_dict_full['uuid_for_processing']
        current_state = all_processed_records_from_logs.get(uuid) # 從聚合的日誌狀態中獲取最新狀態
        
        # 如果記錄存在，並且狀態是 ERROR 且不是 NO_MATCH
        if current_state and \
           current_state.get('final_status_category') == 'ERROR' and \
           current_state.get('error_or_status') and \
           not current_state['error_or_status'].startswith('NO_MATCH'): 
            
            errors_for_automatic_retry.append(original_record_dict_full) # 添加原始記錄用於重試

    if errors_for_automatic_retry:
        print(f"\n--- 檢測到 {len(errors_for_automatic_retry)} 條非 'NO_MATCH' 類型的錯誤記錄。")
        print("等待 30 秒後自動進行第二遍重試 (使用預設模型)...")
        await asyncio.sleep(30) # 等待 30 秒

        # 第二遍處理 (自動重試錯誤的記錄，使用預設模型)
        print(f"\n--- 執行第二遍清洗 (自動重試錯誤，使用 {default_model_instance.model_name}) ---")
        pass2_success, pass2_failure = await process_batch(errors_for_automatic_retry, codebook_df, pass_num=2, model_instance=default_model_instance)
        
        # 再次增量寫入日誌 (第二遍的結果)，並更新聚合狀態
        with open(progress_log_file, 'a', encoding='utf-8') as prog_f, \
             open(error_log_file, 'a', encoding='utf-8') as err_f, \
             open(no_match_log_file, 'a', encoding='utf-8') as no_match_f:
            for record in pass2_success:
                prog_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態
            for record in pass2_failure:
                if record.get('final_status_category') == 'NO_MATCH':
                    no_match_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    err_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態
    else:
        print("\n--- 沒有需要自動重試的非 'NO_MATCH' 錯誤記錄。")

    # --- 交互式決策 (針對所有尚未成功處理的記錄) (Pass 3) ---
    records_for_interactive_retry = []
    # 遍歷原始數據的所有記錄，基於聚合狀態判斷哪些尚未成功
    for original_record_dict_full in source_df.to_dict('records'):
        uuid = original_record_dict_full['uuid_for_processing']
        current_state = all_processed_records_from_logs.get(uuid) # 從聚合的日誌狀態中獲取最新狀態
        
        # 如果記錄存在，並且其最終狀態不是 SUCCESS (包括 ERROR 和 NO_MATCH)
        if current_state and current_state.get('final_status_category') != 'SUCCESS':
            records_for_interactive_retry.append(original_record_dict_full)

    if records_for_interactive_retry:
        print("\n" + "="*50)
        user_choice = input(f"發現 {len(records_for_interactive_retry)} 條尚未成功處理的記錄（包括錯誤和未匹配）。是否立即對這些記錄進行重試？\n"
                            f"輸入 'y' 或 'yes' 使用 {DEFAULT_GENERATION_MODEL} 重試。\n"
                            f"輸入 'p' 或 'pro' 使用 {PRO_GENERATION_MODEL} 重試。\n"
                            f"否則，按任意鍵跳過重試。\n選項: ").lower()
        print("="*50 + "\n")
        
        retry_model_instance_interactive = None # 避免與自動重試的模型變量衝突
        if user_choice in ['y', 'yes']:
            retry_model_instance_interactive = default_model_instance
        elif user_choice in ['p', 'pro']:
            if pro_model_instance is None:
                print(f"正在初始化 {PRO_GENERATION_MODEL} 模型，請稍候...")
                pro_model_instance = genai.GenerativeModel(PRO_GENERATION_MODEL)
            retry_model_instance_interactive = pro_model_instance
        
        if retry_model_instance_interactive:
            # 第三遍處理 (交互式重試，使用用戶選擇的模型)
            print(f"\n--- 執行第三遍清洗 (交互式重試，使用 {retry_model_instance_interactive.model_name}) ---")
            pass3_success, pass3_failure = await process_batch(records_for_interactive_retry, codebook_df, pass_num=3, model_instance=retry_model_instance_interactive)
            
            # 再次增量寫入日誌 (第三遍的結果)，並更新聚合狀態
            with open(progress_log_file, 'a', encoding='utf-8') as prog_f, \
                 open(error_log_file, 'a', encoding='utf-8') as err_f, \
                 open(no_match_log_file, 'a', encoding='utf-8') as no_match_f:
                for record in pass3_success:
                    prog_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態
                for record in pass3_failure:
                    if record.get('final_status_category') == 'NO_MATCH':
                        no_match_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    else:
                        err_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    all_processed_records_from_logs[record['uuid_for_processing']] = record # 更新狀態
        else:
            print("已選擇不進行交互式重試。")
    else:
        print("\n--- 沒有需要交互式重試的記錄。")
    
    # --- 任務結束後，從完整的日誌生成最終文件和報告 ---
    print("\n--- 所有流程完成，正在生成最終的精煉文件和報告 ---")
    
    # 最終從聚合的 all_processed_records_from_logs 中生成報告
    final_records_list = list(all_processed_records_from_logs.values())

    if final_records_list:
        final_df = pd.DataFrame(final_records_list)
        
        # 篩選出最終成功的記錄用於輸出 CSV 文件
        output_df = final_df[final_df['final_status_category'] == 'SUCCESS'].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        
        # 移除用於內部處理的臨時 UUID 列 (uuid_for_processing)
        # 'uuid' 列會被保留，因為它現在是數據的一部分
        if 'uuid_for_processing' in output_df.columns:
            output_df = output_df.drop(columns=['uuid_for_processing'])
        
        # 確保最終輸出 CSV 文件的列與原始 source_df 的列匹配（並保持原始順序）
        # 需要調整 original_source_columns，如果原始沒有 uuid，現在它應該包含
        final_output_columns = [col for col in original_source_columns if col in output_df.columns]
        # 如果 'uuid' 列是新添加的，且原始列中沒有，則將其添加到列順序中
        if 'uuid' not in final_output_columns and 'uuid' in output_df.columns:
            final_output_columns.insert(0, 'uuid') # 可以在開頭添加，或根據需求調整位置

        output_df = output_df[final_output_columns]
        
        output_df.to_csv(refined_output_file, index=False, encoding='utf-8-sig')
        
        # 根據整合後的最終狀態進行統計
        total_processed_unique = len(final_records_list)
        success_count = sum(1 for r in final_records_list if r['final_status_category'] == 'SUCCESS')
        no_match_count = sum(1 for r in final_records_list if r['final_status_category'] == 'NO_MATCH')
        error_count = sum(1 for r in final_records_list if r['final_status_category'] == 'ERROR')
        
        print(f"成功生成精煉版知識庫源文件: {refined_output_file}")
        print(f"總處理的唯一記錄數: {total_processed_unique}")
        print(f"最終成功條數: {success_count}")
        print(f"最終未匹配 (NO_MATCH) 條數: {no_match_count}")
        print(f"最終處理失敗 (ERROR) 條數: {error_count}")

        # 可選: 生成一份包含所有記錄最終狀態的詳細報告 (JSONL 格式)
        status_report_file = os.path.join(output_dir, f"{base_name}_final_status_report.jsonl")
        with open(status_report_file, 'w', encoding='utf-8') as f:
            for record in final_records_list:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"詳細狀態報告已生成: {status_report_file}")

    else:
        print("沒有成功處理的記錄可供生成最終文件。")

if __name__ == "__main__":
    try:
        # 檢查並安裝 tenacity 庫
        try:
            import tenacity
        except ImportError:
            print("錯誤: 缺少'tenacity'庫。請運行: pip install tenacity")
            exit()
        
        from tqdm.asyncio import tqdm as asyncio_tqdm
    except ImportError:
        print("錯誤: 缺少tqdm的異步依賴。請運行: pip install \"tqdm[asyncio]\"")
        exit()
    asyncio.run(main())
