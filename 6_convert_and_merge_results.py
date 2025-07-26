# 6_convert_and_merge_results.py
# 最终版：采用绝对路径定位，确保在任何位置运行都能正确找到结果文件。

import pandas as pd
import json
import os
import glob
from tqdm import tqdm

# --- 【【【核心修正：路径优化】】】 ---
# 获取脚本文件所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 配置 ---
# 所有路径都基于脚本所在的目录构建，确保稳健
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, "05_coded_results") 
FINAL_REPORTS_FOLDER = os.path.join(SCRIPT_DIR, "06_final_reports")
if not os.path.exists(FINAL_REPORTS_FOLDER):
    os.makedirs(FINAL_REPORTS_FOLDER)

# 输出格式选择: 'tidy' (一行一码，适合分析) 或 'aggregated' (一行多码，适合浏览)
OUTPUT_FORMAT = 'aggregated' 

def process_jsonl(filepath):
    """稳健地读取单个jsonl文件"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告：在文件 {filepath} 中发现无效的JSON行，已跳过。")
                continue
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    df['source_file'] = os.path.basename(filepath)
    return df

def aggregate_codes(coding_results_list):
    """将编码列表聚合成单个字符串的辅助函数"""
    if not isinstance(coding_results_list, list) or not coding_results_list:
        return {}
    
    all_keys = set()
    for item in coding_results_list:
        if isinstance(item, dict):
            all_keys.update(item.keys())
            
    aggregated_data = {key: [] for key in all_keys}
    for code_obj in coding_results_list:
        if not isinstance(code_obj, dict): continue
        for key in all_keys:
            value = str(code_obj.get(key, ''))
            aggregated_data[key].append(value)
            
    final_aggregated = {}
    for key, values in aggregated_data.items():
        non_empty_values = [v for v in values if v]
        final_aggregated[f'{key}_agg'] = '; '.join(non_empty_values)
    return final_aggregated

def main():
    print(f"--- 开始从文件夹 '{RESULTS_FOLDER}' 及其所有子文件夹中汇总编码结果 ---")
    
    # 使用绝对路径进行递归搜索
    result_files_pattern = os.path.join(RESULTS_FOLDER, "**", "*.jsonl")
    result_files = glob.glob(result_files_pattern, recursive=True)
    
    if not result_files:
        print(f"错误：在文件夹 '{RESULTS_FOLDER}' 及其子文件夹中，未找到任何 .jsonl 结果文件进行处理。")
        print("请检查：1. 是否已成功运行编码脚本并生成了结果文件？ 2. 文件夹路径是否正确？")
        return
        
    print(f"找到 {len(result_files)} 个批次结果文件，准备合并...")
    print('\n'.join(result_files))
    
    df_all = pd.concat([process_jsonl(f) for f in result_files], ignore_index=True)

    if df_all.empty:
        print("所有结果文件都为空或无效。")
        return

    print(f"共合并了 {len(df_all)} 条记录，正在生成最终报告...")
    
    if OUTPUT_FORMAT == 'tidy':
        FINAL_CSV_OUTPUT_FILE = os.path.join(FINAL_REPORTS_FOLDER, "Final_Coded_Report_TIDY.csv")
        print("正在生成“整洁格式”(Tidy Format)报告...")
        df_exploded = df_all.explode('coding_results').reset_index(drop=True)
        df_exploded.dropna(subset=['coding_results'], inplace=True)
        valid_rows = df_exploded['coding_results'].apply(lambda x: isinstance(x, dict))
        normalized_codes = pd.json_normalize(df_exploded.loc[valid_rows, 'coding_results'])
        final_df = pd.concat([df_exploded.loc[valid_rows].drop(columns=['coding_results']), normalized_codes], axis=1)
    
    elif OUTPUT_FORMAT == 'aggregated':
        FINAL_CSV_OUTPUT_FILE = os.path.join(FINAL_REPORTS_FOLDER, "Final_Coded_Report_AGGREGATED.csv")
        print("正在生成“聚合格式”(Aggregated Format)报告...")
        tqdm.pandas(desc="Aggregating Results")
        aggregated_data = df_all['coding_results'].progress_apply(aggregate_codes)
        aggregated_df = pd.DataFrame(aggregated_data.to_list())
        final_df = pd.concat([df_all.drop(columns=['coding_results']), aggregated_df], axis=1)
    
    else:
        print(f"错误：未知的输出格式'{OUTPUT_FORMAT}'。请选择 'tidy' 或 'aggregated'。")
        return

    final_df.to_csv(FINAL_CSV_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print("-" * 50)
    print("任务全部完成！")
    print(f"最终汇总报告已生成: {FINAL_CSV_OUTPUT_FILE}")
    print(f"报告格式: {OUTPUT_FORMAT}")
    print("-" * 50)

if __name__ == "__main__":
    main()
