# 5_review_tool.py (UI 調整版)
# 职责：一个可视化审核工具，用于高效地对AI的编码结果进行人工审核和修正。

import streamlit as st
import pandas as pd
import os

# --- 配置 ---
# 自动生成的、需要审核的聚合格式报告
CODED_FILE_TO_REVIEW = "06_final_reports/Final_Coded_Report_AGGREGATED.csv" 
# 您的原始码表，用于搜索和选择
CODEBOOK_FILE = "01_source_data/last_phase_codebook.csv"
# 保存修正结果的文件（增量写入）
CORRECTED_FILE = "06_final_reports/Final_Coded_Report_CORRECTED.csv"

# --- 页面初始化 ---
st.set_page_config(layout="wide", page_title="AI 编码审核", page_icon="🔍") # 調整頁面佈局和標題、圖標
st.title("AI 编码结果审核平台 v2.0")
st.info("说明：在此界面中确认或修改AI的编码结果。每次点击“确认”都会将修正后的数据增量保存，进度不会丢失。")

# --- 数据加载与缓存 ---
@st.cache_data
def load_codebook():
    """加载码表"""
    try:
        codebook_df = pd.read_csv(CODEBOOK_FILE, dtype={'code': str})
        codebook_df['display'] = (
            codebook_df['code'] + " - [" +
            codebook_df['net'].fillna('') + "/" +
            codebook_df['subnet'].fillna('') + "] " +
            codebook_df['label'].fillna('')
        )
        return codebook_df
    except FileNotFoundError:
        st.error(f"错误：找不到码表文件 {CODEBOOK_FILE}。请检查文件路径。")
        return None

codebook_df = load_codebook()

if codebook_df is None:
    st.stop()

# --- 数据加载逻辑 ---
try:
    coded_df = pd.read_csv(CODED_FILE_TO_REVIEW)
except FileNotFoundError:
    st.error(f"错误：找不到待审核的报告 {CODED_FILE_TO_REVIEW}。请先运行 `6_convert_and_merge_results.py` 生成聚合报告。")
    st.stop()

if os.path.exists(CORRECTED_FILE):
    corrected_df = pd.read_csv(CORRECTED_FILE, dtype=str)
    if 'uuid' in coded_df.columns and 'uuid' in corrected_df.columns:
        # 篩選出未審核的數據
        uncorrected_df = coded_df[~coded_df['uuid'].isin(corrected_df['uuid'])].reset_index(drop=True)
    else:
        st.error("错误：待审核文件或已修正文件中缺少 'uuid' 列，无法进行进度管理。请确保所有相关文件包含 'uuid'。")
        uncorrected_df = pd.DataFrame() # 创建一个空的DataFrame以避免後續錯誤
else:
    corrected_df = pd.DataFrame()
    uncorrected_df = coded_df

# 使用session state来跟踪当前审核的索引
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

total_count = len(coded_df)
corrected_count = len(corrected_df)

# --- UI 調整範例 ---

# 進度條和狀態顯示
st.markdown("---") # 添加分隔線
st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>审核概覽</h3>", unsafe_allow_html=True) # 居中標題
st.progress((corrected_count / total_count) if total_count > 0 else 0, text=f"**当前进度**: {corrected_count} / {total_count} (剩余: {len(uncorrected_df)})")
st.markdown("---") # 添加分隔線

# 審核界面的主要佈局
index = st.session_state.current_index
if index < len(uncorrected_df):
    current_row = uncorrected_df.iloc[index]
    
    # 使用 col_weights 調整列寬，例如：問題和回答佔大頭，AI處理方式佔小頭
    col1, col2 = st.columns([2, 1]) # 調整列寬比例，第一列是第二列的兩倍寬
    
    with col1:
        st.subheader("原文信息") # 添加小標題
        st.markdown(f"**問題**: <span style='font-size:1.1em; color:#3366FF;'>{current_row.get('question', 'N/A')}</span>", unsafe_allow_html=True)
        st.markdown(f"**回答原文**: <span style='font-size:1.2em; font-weight:bold;'>{current_row.get('text', 'N/A')}</span>", unsafe_allow_html=True)
        
        # 使用 st.expander 組織信息，讓界面更整潔
        with st.expander("查看原始數據細節"):
            st.json(current_row.to_dict()) # 顯示所有原始行數據

    with col2:
        st.subheader("AI 处理详情") # 添加小標題
        st.markdown(f"**AI處理方式**: <span style='color:#FF9900;'>{current_row.get('process_method', 'N/A')}</span>", unsafe_allow_html=True)
        st.code(f"UUID: {current_row.get('uuid', 'N/A')}")
        
        # 可以添加一些 AI 相關的指標，例如置信度（如果您的數據包含）
        # st.metric(label="AI 置信度", value=f"{current_row.get('ai_confidence', 'N/A')}%")

    st.markdown("---") # 再次分隔

    # 編碼選擇部分
    st.markdown("<h4 style='color: #00796B;'>請確認或修正編碼</h4>", unsafe_allow_html=True)
    ai_codes_str = str(current_row.get('code_agg', ''))
    ai_codes = ai_codes_str.split('; ') if ai_codes_str and ai_codes_str != 'nan' else []
    
    selected_options = st.multiselect(
        "搜索並選擇編碼 (可在此框中輸入 `code` 或 `关键词` 进行搜索):",
        options=codebook_df['display'],
        default=[d for d in codebook_df['display'] if d.split(' - ')[0] in ai_codes],
        placeholder="選擇或搜索編碼..." # 添加占位符文本
    )
    
    # 添加一個小間距
    st.write("") 

    # 確認按鈕
    if st.button("✅ 確認 & 保存 & 下一條", use_container_width=True, type="primary"):
        selected_codes_full_info = codebook_df[codebook_df['display'].isin(selected_options)]
        
        # 準備要保存的、經過修正的數據行
        corrected_data = current_row.to_dict()
        
        # 確保正確更新聚合列
        for col in ['sentiment', 'net', 'subnet', 'code', 'label']:
            # 只有當選中的編碼信息中包含該列時才嘗試聚合
            if col in selected_codes_full_info.columns: 
                 corrected_data[f'{col}_agg'] = "; ".join(selected_codes_full_info[col].astype(str).dropna().tolist())
            else:
                # 如果該列不在selected_codes_full_info中，但corrected_data中存在其_agg，則清空
                if f'{col}_agg' in corrected_data:
                    corrected_data[f'{col}_agg'] = ""
        
        # 確保is_new_suggestion_agg列存在
        if 'is_new_suggestion_agg' not in corrected_data:
            corrected_data['is_new_suggestion_agg'] = ''

        temp_df = pd.DataFrame([corrected_data])
        
        if not os.path.exists(CORRECTED_FILE):
            temp_df.to_csv(CORRECTED_FILE, index=False, encoding='utf-8-sig')
        else:
            temp_df.to_csv(CORRECTED_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        st.toast(f"记录 {current_row.get('uuid', '')} 已保存！", icon="🎉")
        
        # 在保存後，更新索引並重新運行，確保顯示下一條
        st.session_state.current_index += 1
        st.rerun()

else:
    # 所有數據審核完畢的界面
    st.balloons() # 放氣球慶祝
    st.success("🎉 恭喜！所有数据都已审核完毕！")
    st.markdown(f"最终的修正文件已保存为: **{CORRECTED_FILE}**")

