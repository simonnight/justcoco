# 5_review_tool.py
# 权威最终版: 实现了以“统一审核库”为核心的全新工作流，并整合所有高级交互功能。

import streamlit as st
import pandas as pd
import os
import json
import glob

# --- 页面配置 (必须是第一个st命令) ---
st.set_page_config(layout="wide")

# --- 核心配置 ---
RESULTS_FOLDER = "05_coded_results"
REVIEWED_FOLDER = "06_reviewed_results"
if not os.path.exists(REVIEWED_FOLDER):
    os.makedirs(REVIEWED_FOLDER)
CODEBOOK_FOLDER = "01_source_data"
BASE_CODEBOOK_FILE = os.path.join(CODEBOOK_FOLDER, "last_phase_codebook.csv")
# 【新增核心】实时增长的新码表文件
NEWLY_ADDED_CODES_FILE = os.path.join(CODEBOOK_FOLDER, "newly_added_codes.csv")
# 【新增核心】统一的、持久化的审核总库
MASTER_REVIEW_FILE = os.path.join(REVIEWED_FOLDER, "master_review_library.csv")

# ==============================================================================
# --- 辅助函数定义区 ---
# ==============================================================================

@st.cache_data
def load_codebooks():
    """加载并合并基础码表和新增码表"""
    try:
        base_df = pd.read_csv(BASE_CODEBOOK_FILE, dtype=str).fillna('')
    except FileNotFoundError:
        st.error(f"错误：找不到基础码表文件 {BASE_CODEBOOK_FILE}。")
        return None

    if os.path.exists(NEWLY_ADDED_CODES_FILE):
        try:
            new_df = pd.read_csv(NEWLY_ADDED_CODES_FILE, dtype=str).fillna('')
            if not new_df.empty:
                for col in base_df.columns:
                    if col not in new_df.columns:
                        new_df[col] = ''
                new_df = new_df[list(base_df.columns)]
                combined_df = pd.concat([base_df, new_df], ignore_index=True)
            else:
                combined_df = base_df
        except pd.errors.EmptyDataError:
             combined_df = base_df
    else:
        combined_df = base_df
        
    combined_df.drop_duplicates(subset=['code'], keep='last', inplace=True)
    
    required_cols = ['code', 'net', 'subnet', 'label', 'sentiment']
    for col in required_cols:
        if col not in combined_df.columns:
            st.error(f"错误：合并后的码表中缺少必需的列: '{col}'。")
            return None
    
    return combined_df

@st.cache_resource
def load_master_review_file():
    """加载总审核库文件"""
    if os.path.exists(MASTER_REVIEW_FILE):
        return pd.read_csv(MASTER_REVIEW_FILE, dtype=str).fillna('')
    return pd.DataFrame(columns=['uuid', 'question', 'text', 'review_status', 'coding_results_json'])

def save_master_review_file(df):
    """保存总审核库文件"""
    df.to_csv(MASTER_REVIEW_FILE, index=False, encoding='utf-8-sig')
    st.cache_resource.clear() # 清空缓存以便下次能读取到最新版

def update_record_in_library(uuid, question, text, review_status, coding_results):
    """在总审核库中新增或更新一条记录"""
    df = load_master_review_file()
    
    coding_results_str = json.dumps(coding_results, ensure_ascii=False)
    
    new_record = {
        'uuid': uuid, 'question': question, 'text': text,
        'review_status': review_status, 'coding_results_json': coding_results_str
    }
    
    # 如果已存在，直接更新；否则追加
    if uuid in df['uuid'].values:
        df.loc[df['uuid'] == uuid, list(new_record.keys())] = list(new_record.values())
    else:
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
    save_master_review_file(df)
    
    if review_status == "已审核":
        update_newly_added_codes(coding_results)

def update_newly_added_codes(coding_results):
    """实时更新新码表文件"""
    codebook_df = load_codebooks()
    if codebook_df is None: return

    new_codes_to_add = []
    for item in coding_results:
        code_val = str(item.get('code', ''))
        if code_val and code_val not in codebook_df['code'].tolist():
            # 确保新编码有所有必要的列
            new_item = {
                'sentiment': item.get('sentiment', ''), 'net': item.get('net', ''),
                'subnet': item.get('subnet', ''), 'code': item.get('code', ''),
                'label': item.get('label', '')
            }
            new_codes_to_add.append(new_item)
    
    if new_codes_to_add:
        new_df = pd.DataFrame(new_codes_to_add)
        header = not os.path.exists(NEWLY_ADDED_CODES_FILE)
        new_df.to_csv(NEWLY_ADDED_CODES_FILE, mode='a', index=False, header=header, encoding='utf-8-sig')
        st.cache_data.clear()
        st.toast("新编码已学习并添加到实时码表！", icon="🧠")

# ==============================================================================
# --- 页面渲染函数 ---
# ==============================================================================

def render_selection_page(master_df):
    """渲染第一页：文件选择与总库管理"""
    st.title("AI编码结果质控与交付管理平台")
    
    if st.button("🔄 刷新 / 导入新批次", use_container_width=True, type="primary"):
        all_jsonl_files = glob.glob(os.path.join(RESULTS_FOLDER, "**", "*.jsonl"), recursive=True)
        new_records = []
        existing_uuids = set(master_df['uuid'])
        
        for f in all_jsonl_files:
            with open(f, 'r', encoding='utf-8') as jf:
                for line in jf:
                    try:
                        record = json.loads(line)
                        if record.get('uuid') not in existing_uuids:
                            new_records.append({
                                'uuid': record.get('uuid'), 'question': record.get('question'),
                                'text': record.get('text'), 'review_status': '待审核',
                                'coding_results_json': json.dumps(record.get('coding_results', []), ensure_ascii=False)
                            })
                            existing_uuids.add(record.get('uuid'))
                    except:
                        continue
        
        if new_records:
            new_df = pd.DataFrame(new_records)
            updated_df = pd.concat([master_df, new_df], ignore_index=True)
            save_master_review_file(updated_df)
            st.success(f"成功导入 {len(new_records)} 条新纪录！")
            st.rerun()
        else:
            st.info("没有发现新的AI编码结果可供导入。")

    st.divider()

    unreviewed_df = master_df[master_df['review_status'] == '待审核']
    reviewed_df = master_df[master_df['review_status'] == '已审核']
    
    st.header("1. 待审核库")
    st.markdown(f"当前共有 **{len(unreviewed_df)}** 条记录等待审核。")
    if st.button("进入待审核库", disabled=unreviewed_df.empty):
        st.session_state.page = "review"
        st.session_state.review_mode = "unreviewed"
        st.session_state.current_index = 0
        st.rerun()

    st.header("2. 已审核库")
    st.markdown(f"当前共有 **{len(reviewed_df)}** 条记录已完成审核。")
    if st.button("进入已审核库 (可重新编辑)", disabled=reviewed_df.empty):
        st.session_state.page = "review"
        st.session_state.review_mode = "reviewed"
        st.session_state.current_index = 0
        st.rerun()

def render_review_page(codebook_df, master_df):
    """渲染第二页：审核工作区"""
    
    review_mode = st.session_state.review_mode
    if review_mode == "unreviewed":
        df_to_review = master_df[master_df['review_status'] == '待审核'].copy()
        df_to_review.sort_values(by='text', inplace=True)
        df_to_review.reset_index(drop=True, inplace=True)
    else:
        df_to_review = master_df[master_df['review_status'] == '已审核'].copy().reset_index(drop=True)

    if df_to_review.empty:
        st.warning(f"“{review_mode}”库中没有记录。")
        if st.button("<< 返回首页"): st.session_state.page = "selection"; st.rerun()
        st.stop()
    
    if 'current_index' not in st.session_state or st.session_state.current_index >= len(df_to_review):
        st.session_state.current_index = 0

    current_row_data = df_to_review.loc[st.session_state.current_index]
    if 'current_codes' not in st.session_state or st.session_state.get('current_uuid') != current_row_data['uuid']:
        st.session_state.current_uuid = current_row_data.get('uuid')
        st.session_state.current_codes = json.loads(current_row_data.get('coding_results_json', '[]'))
        st.session_state.original_codes = list(st.session_state.current_codes)
    
    def save_changes(status):
        update_record_in_library(
            st.session_state.current_uuid, current_row_data['question'],
            current_row_data['text'], status, st.session_state.current_codes
        )
        st.session_state.last_saved_codes = list(st.session_state.current_codes)
        
    def go_to_next():
        save_changes(status="待审核")
        if st.session_state.current_index < len(df_to_review) - 1:
            st.session_state.current_index += 1
        st.session_state.current_uuid = None
            
    def go_to_prev():
        save_changes(status="待审核")
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
        st.session_state.current_uuid = None
            
    def confirm_changes():
        save_changes(status="已审核")
        st.toast("本条已确认！", icon="✅")
    
    def copy_previous():
        if 'last_saved_codes' in st.session_state and st.session_state.last_saved_codes is not None:
            st.session_state.current_codes = [dict(item) for item in st.session_state.last_saved_codes]
        else:
            st.warning("没有上一条已保存的编码可供复制。")

    def revert_changes(): st.session_state.current_codes = [dict(item) for item in st.session_state.original_codes]
    def add_new_code(): st.session_state.current_codes.append({'sentiment':'', 'net':'', 'subnet':'', 'code':'', 'label':''})
    def delete_code(index_to_delete): st.session_state.current_codes.pop(index_to_delete)
    def exit_to_selection():
        st.session_state.page = "selection"
        for key in ['current_codes', 'current_uuid', 'current_index', 'review_mode', 'last_saved_codes']:
            if key in st.session_state: del st.session_state[key]
    
    mg_cols = st.columns([2, 1.5, 1.5, 1, 1.5, 1, 2])
    if mg_cols[0].button("<< 返回首页"): exit_to_selection(); st.rerun()
    mg_cols[1].button("< 上一条", on_click=go_to_prev, disabled=(st.session_state.current_index == 0))
    mg_cols[2].button("📝 复制上一条", on_click=copy_previous, disabled=('last_saved_codes' not in st.session_state or st.session_state.last_saved_codes is None))
    mg_cols[3].button("💾 仅保存", on_click=lambda: save_changes(status="待审核"))
    mg_cols[4].button("🔄 撤销修改", on_click=revert_changes)
    mg_cols[5].button("✅ 确认", on_click=confirm_changes, type="primary")
    mg_cols[6].button("下一条 >", on_click=go_to_next, use_container_width=True, disabled=(st.session_state.current_index >= len(df_to_review) - 1))

    st.markdown(f"**当前库**: `{review_mode}` | **进度**: {st.session_state.current_index + 1} / {len(df_to_review)}")
    st.divider()
    
    info_cols = st.columns(2)
    info_cols[0].markdown(f"**问题**:"); info_cols[0].info(current_row_data.get('question', 'N/A'))
    info_cols[1].markdown(f"**回答原文**:"); info_cols[1].success(current_row_data.get('text', 'N/A'))
    
    st.divider()
    st.subheader("编码结果与修正")
    
    if 'current_codes' in st.session_state:
        header_cols = st.columns([2, 2, 3, 3, 4, 1])
        header_cols[0].markdown("**Code**"); header_cols[1].markdown("**Sentiment**"); header_cols[2].markdown("**Net**")
        header_cols[3].markdown("**Subnet**"); header_cols[4].markdown("**Label**")

        for i, code_item in enumerate(st.session_state.current_codes):
            row_cols = st.columns([2, 2, 3, 3, 4, 1])
            
            def get_options_and_index(column_name, current_value, df_filter=None):
                if df_filter is None: df_filter = codebook_df
                options = [''] + sorted(df_filter[column_name].unique().tolist())
                if current_value and current_value not in options:
                    options.insert(1, current_value)
                return options, options.index(current_value) if current_value in options else 0

            with row_cols[0]:
                code_options, code_index = get_options_and_index('code', str(code_item.get('code', '')))
                st.session_state.current_codes[i]['code'] = st.selectbox("Code", code_options, key=f"code_{i}", label_visibility="collapsed", index=code_index)
            with row_cols[1]:
                s_options, s_index = get_options_and_index('sentiment', code_item.get('sentiment', ''))
                st.session_state.current_codes[i]['sentiment'] = st.selectbox("Sentiment", s_options, key=f"sentiment_{i}", label_visibility="collapsed", index=s_index)
            with row_cols[2]:
                n_options, n_index = get_options_and_index('net', code_item.get('net', ''))
                st.session_state.current_codes[i]['net'] = st.selectbox("Net", n_options, key=f"net_{i}", label_visibility="collapsed", index=n_index)
            with row_cols[3]:
                sn_options, sn_index = get_options_and_index('subnet', code_item.get('subnet', ''))
                st.session_state.current_codes[i]['subnet'] = st.selectbox("Subnet", sn_options, key=f"subnet_{i}", label_visibility="collapsed", index=sn_index)
            with row_cols[4]:
                l_options, l_index = get_options_and_index('label', code_item.get('label', ''))
                st.session_state.current_codes[i]['label'] = st.selectbox("Label", l_options, key=f"label_{i}", label_visibility="collapsed", index=l_index)

            with row_cols[5]:
                st.button("❌", key=f"delete_{i}", on_click=delete_code, args=(i,))
        
        st.markdown(f"---"); st.button("+ 添加新编码", on_click=add_new_code, use_container_width=True)

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
def main():
    st.markdown("""<style>html, body, [class*="st-"], .st-emotion-cache-16txtl3 {font-size: 0.85rem;}</style>""", unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'selection'

    master_df = load_master_review_file()
    codebook_df = load_codebooks()

    if codebook_df is None:
        st.stop()

    if st.session_state.page == 'selection':
        render_selection_page(master_df)
    elif st.session_state.page == 'review':
        render_review_page(codebook_df, master_df)

if __name__ == "__main__":
    main()
