# 2_build_knowledge_base.py
# 职责：读取标准格式的历史数据，构建支持高速检索的优化版知识库。

import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
import pickle
import logging # 导入 logging 模块
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions # 导入 API 异常类型

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    logging.error("错误：请先设置 GOOGLE_API_KEY 环境变量。")
    exit()

EMBEDDING_MODEL = "models/text-embedding-004"
# 此脚本依赖于 preprocess_data.py 生成的标准长格式文件
# --- 关键改动：现在读取精炼后的数据文件 ---
LAST_PHASE_DATA_FILE = "02_preprocessed_data/last_phase_coded_data_REFINED.csv"

# --- 文件输出路径 ---
KB_FOLDER = "03_knowledge_base"
QUESTION_INDICES_FOLDER = os.path.join(KB_FOLDER, 'question_indices')

# 创建必要的文件夹
os.makedirs(KB_FOLDER, exist_ok=True)
os.makedirs(QUESTION_INDICES_FOLDER, exist_ok=True)

ANSWER_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "answer.index")
QUESTION_FAISS_INDEX_FILE = os.path.join(KB_FOLDER, "question.index")
DATA_MAP_FILE = os.path.join(KB_FOLDER, "data_map.pkl")
UNIQUE_QUESTIONS_FILE = os.path.join(KB_FOLDER, "unique_questions.pkl")
QUESTION_TO_SUB_INDEX_MAP_FILE = os.path.join(KB_FOLDER, "question_to_sub_index_map.pkl")

# --- 带有重试逻辑的嵌入 API 调用函数 ---
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60), # 1s, 2s, 4s, 8s, 16s, 32s, 60s...
    stop=stop_after_attempt(5), # 最多重试5次
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) | # 速率限制
          retry_if_exception_type(google.api_core.exceptions.InternalServerError) | # 服务器内部错误
          retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable) # 服务不可用
)
def _embed_content_with_retry(content, task_type):
    """带有指数退避重试逻辑的 Gemini 嵌入 API 调用。"""
    response = genai.embed_content(model=EMBEDDING_MODEL, content=content, task_type=task_type)
    if not response or 'embedding' not in response:
        raise ValueError(f"嵌入 API 返回空响应或无嵌入数据。响应: {response}")
    return response['embedding']

def build_optimized_knowledge_base():
    logging.info("--- 开始构建优化版知识库 ---")
    
    try:
        df = pd.read_csv(LAST_PHASE_DATA_FILE)
    except FileNotFoundError:
        logging.error(f"错误：找不到输入文件 {LAST_PHASE_DATA_FILE}。请先运行 1_preprocess_data.py 或 4a_refine_knowledge_base.py。")
        return

    # --- 关键改动：显式检查所需列 ---
    required_columns = ['question', 'text', 'code']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"错误：输入文件 {LAST_PHASE_DATA_FILE} 缺少必要的列。请确保包含 {required_columns}。")
        return

    # 删除关键列中的缺失值
    initial_rows = len(df)
    df.dropna(subset=['question', 'text', 'code'], inplace=True)
    if len(df) < initial_rows:
        logging.warning(f"警告：已从输入文件中删除 {initial_rows - len(df)} 条包含缺失关键数据（question, text, code）的记录。")
    
    if df.empty:
        logging.warning("警告：经过数据清洗后，没有留下任何有效记录。知识库将为空。")
        return
        
    df['text'] = df['text'].astype(str)
    df['question'] = df['question'].astype(str)
    logging.info(f"读取了 {len(df)} 条已编码示例。")

    # --- 关键改动：分批嵌入并直接添加到 FAISS 索引 ---
    logging.info("正在向量化所有'回答'并构建总索引...")
    batch_size = 100 # 调整批次大小以适应 API 速率限制
    answer_index = None # 在循环外部初始化 FAISS 索引

    for i in tqdm(range(0, len(df), batch_size), desc="Embedding Answers"):
        batch_texts = df['text'][i:i+batch_size].tolist()
        try:
            embeddings_list = _embed_content_with_retry(batch_texts, "RETRIEVAL_DOCUMENT")
            batch_embeddings_np = np.array(embeddings_list).astype('float32')
            
            if answer_index is None:
                # 第一次批次时创建索引
                answer_index = faiss.IndexFlatL2(batch_embeddings_np.shape[1])
            answer_index.add(batch_embeddings_np)
        except Exception as e:
            logging.error(f"向量化 '回答' 时发生错误（批次 {i}-{i+batch_size}）：{e}")
            # 决定如何处理：跳过批次，或者退出。这里选择继续，但错误会被记录。
            continue
            
    if answer_index is None: # 如果所有嵌入都失败了
        logging.error("错误：未能成功嵌入任何'回答'，无法构建总索引。")
        return

    faiss.write_index(answer_index, ANSWER_FAISS_INDEX_FILE)
    logging.info(f"'回答'总索引构建完成。")

    logging.info("正在向量化所有唯一'问题'...")
    unique_questions = df['question'].unique().tolist()
    try:
        question_embeddings_list = _embed_content_with_retry(unique_questions, "RETRIEVAL_DOCUMENT")
        question_embeddings_np = np.array(question_embeddings_list).astype('float32')
        question_index = faiss.IndexFlatL2(question_embeddings_np.shape[1])
        question_index.add(question_embeddings_np)
        faiss.write_index(question_index, QUESTION_FAISS_INDEX_FILE)
        logging.info(f"'问题'索引构建完成。")
    except Exception as e:
        logging.error(f"向量化 '问题' 时发生错误：{e}")
        # 决定如何处理：继续，但后续功能可能受影响，或者退出。这里选择退出。
        return

    logging.info("正在为每个问题预创建专属的迷你索引...")
    question_to_sub_index_map = {}
    for i, question_text in enumerate(tqdm(unique_questions, desc="Creating Sub-indices")):
        original_indices = df[df['question'] == question_text].index.to_numpy()
        
        # 为了确保 sub_vectors 能正确被获取，这里使用原始索引从 df 中提取文本，重新嵌入以保证内存效率，
        # 因为单个问题子集通常较小。
        sub_texts = df.loc[original_indices, 'text'].tolist()
        if sub_texts:
            sub_embeddings_list = _embed_content_with_retry(sub_texts, "RETRIEVAL_DOCUMENT")
            sub_vectors = np.array(sub_embeddings_list).astype('float32')
        else:
            sub_vectors = np.array([]) # 空数组

        sub_index = faiss.IndexFlatL2(answer_index.d) # 使用与主索引相同的维度
        if len(sub_vectors) > 0:
            sub_index.add(sub_vectors)
        sub_index_filename = os.path.join(QUESTION_INDICES_FOLDER, f"q_index_{i}.index")
        faiss.write_index(sub_index, sub_index_filename)
        question_to_sub_index_map[question_text] = {
            "index_file": sub_index_filename,
            "original_indices": original_indices.tolist() # 将 numpy 数组转为列表以便 pickle
        }
    
    # 保存原始 DataFrame，以及其他映射文件
    df.to_pickle(DATA_MAP_FILE)
    with open(UNIQUE_QUESTIONS_FILE, 'wb') as f: pickle.dump(unique_questions, f)
    with open(QUESTION_TO_SUB_INDEX_MAP_FILE, 'wb') as f: pickle.dump(question_to_sub_index_map, f)
    
    logging.info("-" * 50)
    logging.info("优化版知识库构建完毕！所有文件已存入 `knowledge_base` 文件夹。")
    logging.info("-" * 50)

if __name__ == "__main__":
    try:
        # 检查并安装 tenacity 库
        try:
            import tenacity
        except ImportError:
            logging.error("错误: 缺少 'tenacity' 库。请运行: pip install tenacity")
            exit()
        
        # tqdm[asyncio] 仅在异步环境下需要，这里是非异步脚本，但为了一致性保留检查
        try:
            from tqdm.asyncio import tqdm as asyncio_tqdm # 仅为确保 tqdm 版本足够新
            # 在非异步环境下直接使用 tqdm.tqdm
            tqdm = tqdm # 覆盖 tqdm 变量为标准 tqdm
        except ImportError:
            logging.warning("警告: 未安装 tqdm 的异步依赖 'tqdm[asyncio]'，但本脚本为同步运行。")

        build_optimized_knowledge_base()
    except Exception as e:
        logging.critical(f"脚本执行过程中发生严重错误: {e}", exc_info=True)
        exit(1)
