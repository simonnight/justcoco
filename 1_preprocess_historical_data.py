# 1_preprocess_historical_data.py
# 權威最終版: 移除所有不必要的UUID生成，專注於格式轉換、清洗和計數。

import pandas as pd
import os
import re
from tqdm import tqdm

# --- 用戶配置區 ---
# 根據您原始“歷史已編碼”寬數據文件的結構，修改以下配置

# 1. ID列的位置（0代表第一列）
ID_COLUMN_INDEXES = [0] 

# 2. 每個問題在寬表中占的列數（對於已編碼數據，通常是2：回答文本 + 編碼）
COLUMNS_PER_QUESTION = 2

# 3. 回答文本在每組中的偏移位 (從0開始計數)
TEXT_COL_OFFSET = 0 

# 4. 編碼在每組中的偏移位 (從0開始計數)
CODE_COL_OFFSET = 1

# 5. 歷史數據中，“一格多碼”所使用的分隔符
CODE_DELIMITER = ';'

# --- 文件路徑配置 ---
INPUT_WIDE_FILE = '01_source_data/raw_last_phase_data_wide.csv' 
OUTPUT_LONG_FILE = '02_preprocessed_data/last_phase_coded_data.csv' 

def create_tidy_data_from_coded_source(input_file, output_file_long):
    """
    讀取、轉換、去後綴、去重、展開歷史已編碼數據。
    """
    print(f"--- 正在處理歷史已編碼文件: {input_file} ---")
    
    output_dir = os.path.dirname(output_file_long)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df_wide = pd.read_csv(input_file, dtype=str).fillna('')
    except FileNotFoundError:
        print(f"錯誤：找不到輸入文件 {input_file}。請檢查文件名和路徑。")
        return

    # --- 1. 寬表轉長表 ---
    id_col_names = df_wide.columns[ID_COLUMN_INDEXES].tolist()
    data_columns = [col for col in df_wide.columns if col not in id_col_names]
    
    all_long_data = []
    for index, row in tqdm(df_wide.iterrows(), total=len(df_wide), desc="1/5: Reshaping Data"):
        for i in range(0, len(data_columns), COLUMNS_PER_QUESTION):
            block = data_columns[i : i + COLUMNS_PER_QUESTION]
            q_col, c_col = block[TEXT_COL_OFFSET], block[CODE_COL_OFFSET]
            answer_text = row[q_col]
            if answer_text:
                record = {
                    'question': q_col, 
                    'text': answer_text, 
                    'code': row[c_col]
                }
                # 將原始ID列也加入記錄，以便後續可能的追溯或分析
                for id_name in id_col_names:
                    record[id_name] = row[id_name]
                all_long_data.append(record)

    df_long = pd.DataFrame(all_long_data)
    
    if df_long.empty:
        print("警告：寬表轉長表後未發現任何有效的回答記錄。")
        return
        
    # --- 2. 去後綴 ---
    print("\n--- 2/5: 正在移除問題文本中的'.1', '.2'等後綴 ---")
    df_long['question'] = df_long['question'].str.replace(r'\.\d+$', '', regex=True)
    print("後綴移除完成。")
    
    # --- 3. 數據去重 (在多碼展開前) ---
    print("\n--- 3/5: 正在進行數據去重 ---")
    initial_rows = len(df_long)
    print(f"去重前的長數據記錄數為: {initial_rows}")
    
    df_long['code'] = df_long['code'].astype(str).str.strip()
    # 根據您的要求，去重時不考慮ID
    deduplication_cols = ['question', 'text', 'code']
    df_cleaned = df_long.drop_duplicates(subset=deduplication_cols).copy()
    
    final_rows = len(df_cleaned)
    print(f"按 'question', 'text', 'code' 去重後的記錄數為: {final_rows}")
    print(f"成功移除了 {initial_rows - final_rows} 条重复的知识点记录。")
    
    # --- 4. 多編碼展開 ---
    print("\n--- 4/5: 正在展開'一格多碼'的數據 ---")
    df_cleaned['original_code_count'] = df_cleaned['code'].str.count(re.escape(CODE_DELIMITER)) + 1
    df_cleaned.loc[~df_cleaned['code'].str.contains(re.escape(CODE_DELIMITER), na=False), 'original_code_count'] = 1
    df_cleaned.loc[df_cleaned['code'] == '', 'original_code_count'] = 0

    df_cleaned['code'] = df_cleaned['code'].str.split(CODE_DELIMITER)
    df_exploded = df_cleaned.explode('code').reset_index(drop=True)
    df_exploded['code'] = df_exploded['code'].str.strip()
    df_exploded = df_exploded[df_exploded['code'].str.lower().isin(['', 'nan']) == False]
    print(f"多碼展開後的最終記錄總數為: {len(df_exploded)}")
    
    # --- 5. 整理並保存最終文件 ---
    print("\n--- 5/5: 正在整理最終文件 ---")
    
    # 最終的列順序
    final_cols = id_col_names + ['question', 'text', 'code', 'original_code_count']
    final_df_to_save = df_exploded[[col for col in final_cols if col in df_exploded.columns]]

    final_df_to_save.to_csv(output_file_long, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print("歷史數據預處理成功！")
    print(f"已生成標準長格式文件: {output_file_long} (總行數: {len(final_df_to_save)})")
    print("【注意】此文件不包含UUID列，專用於構建知識庫。")
    print("-" * 50)

if __name__ == "__main__":
    create_tidy_data_from_coded_source(INPUT_WIDE_FILE, OUTPUT_LONG_FILE)
