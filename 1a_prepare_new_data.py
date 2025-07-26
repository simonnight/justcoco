# 1a_prepare_new_data.py
# 职责：【专用】处理新的、未编码的宽格式数据，生成标准的、包含UUID的长格式文件。(已修正NameError)

import pandas as pd
import uuid
import os
import re
from tqdm import tqdm # 【【【错误修正处：增加了这一行】】】

# --- 用户配置区 ---
# 您的原始文件名
INPUT_WIDE_FILE = '01_source_data/current_phase_raw_data.csv' 
# 您希望输出的标准长格式文件名
OUTPUT_LONG_FILE = '02_preprocessed_data/current_phase_tidy.csv'
# 为本期数据生成的UUID地图文件名
UUID_MAP_FILE = '02_preprocessed_data/current_phase_uuid_map.csv'
# ID列在原始文件中的位置（0代表第一列）
ID_COLUMN_INDEXES = [0] 

def prepare_uncoded_data(input_file, output_file_long, output_file_map):
    """
    读取宽格式的未编码文件，为其每一个非空回答单元格生成一个UUID，
    并输出标准长格式文件和UUID地图文件。
    """
    print(f"--- 正在处理新的未编码文件: {input_file} ---")
    
    output_dir = os.path.dirname(output_file_long)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    try:
        df_wide = pd.read_csv(input_file, dtype=str).fillna('')
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}。请检查文件名和路径。")
        return

    if df_wide.empty:
        print("错误：输入文件为空。")
        return

    try:
        id_col_names = df_wide.columns[ID_COLUMN_INDEXES].tolist()
    except IndexError:
        print(f"错误：配置的ID列位置 {ID_COLUMN_INDEXES} 超出文件范围。")
        return
            
    print(f"已将列 {id_col_names} 识别为ID列。")
    
    df_wide['__temp_id__'] = df_wide[id_col_names].astype(str).agg('_'.join, axis=1)
    
    # 所有非ID列都是问题回答列
    data_columns = [col for col in df_wide.columns if col not in id_col_names and col != '__temp_id__']
    
    all_long_data = []
    uuid_map = []

    for index, row in tqdm(df_wide.iterrows(), total=len(df_wide), desc="Reshaping New Data"):
        row_id_val = row['__temp_id__']
        
        for q_col in data_columns:
            answer_text = row[q_col]
            
            if answer_text:
                generated_uuid = str(uuid.uuid4())
                
                record = {
                    'uuid': generated_uuid,
                    'question': q_col,
                    'text': answer_text,
                    'original_row_id': row_id_val 
                }
                
                map_record = {
                    'uuid': generated_uuid,
                    'original_row_id': row_id_val,
                    'original_question_col': q_col
                }

                all_long_data.append(record)
                uuid_map.append(map_record)

    if not all_long_data:
        print("警告：处理后未发现任何有效的回答记录。")
        return
        
    df_long = pd.DataFrame(all_long_data)
    
    # 将原始ID列也加入最终文件，方便人工核对
    # 为了避免和原始df_wide的id_col_names冲突，先合并再选取
    # 修正原始ID的合并逻辑
    if '__temp_id__' in df_wide.columns:
        df_wide.rename(columns={'__temp_id__': 'original_row_id'}, inplace=True)
    
    df_with_ids = pd.merge(df_long, df_wide[['original_row_id'] + id_col_names], on='original_row_id', how='left')
    
    # 调整列顺序
    final_cols = ['uuid'] + id_col_names + ['question', 'text']
    final_df = df_with_ids[[col for col in final_cols if col in df_with_ids.columns]]

    # 保存文件
    final_df.to_csv(output_file_long, index=False, encoding='utf-8-sig')
    pd.DataFrame(uuid_map).to_csv(output_file_map, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print("新数据预处理成功！")
    print(f"已生成标准长格式文件: {output_file_long}")
    print(f"已生成UUID地图文件: {output_file_map}")
    print("-" * 50)


if __name__ == "__main__":
    prepare_uncoded_data(INPUT_WIDE_FILE, OUTPUT_LONG_FILE, UUID_MAP_FILE)
