# 4_generate_new_codebook_draft.py
# 职责：读取一个纯文本样本文件，调用AI生成全新码表的草案。

import google.generativeai as genai
import os

# --- 配置 ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("错误：请先设置 GOOGLE_API_KEY 环境变量。")
    exit()

# 建议使用最强大的模型进行创造性任务
GENERATION_MODEL = "gemini-1.5-pro-latest" 
# 包含回答样本的纯文本文件，每行一条
SAMPLE_FILE = "new_project_sample.txt" 
# 您需要定义本次项目的主题
PROJECT_THEME = "对某电商平台售后服务的反馈" 
# 输出文件名
OUTPUT_CODEBOOK_DRAFT = "new_question_codebook_draft.json"

def generate_draft():
    print(f"--- 正在为新主题“{PROJECT_THEME}”生成码表草案 ---")
    try:
        with open(SAMPLE_FILE, 'r', encoding='utf-8') as f:
            samples = f.read()
    except FileNotFoundError:
        print(f"错误：找不到样本文件 {SAMPLE_FILE}。请在根目录创建它，并放入样本回答。")
        return

    model = genai.GenerativeModel(GENERATION_MODEL)

    prompt = f"""
    # 角色
    你是一位顶级的市场研究总监，专长是从零开始为新的研究项目构建逻辑严谨、层级分明的编码体系(Codebook)。

    # 任务
    我将提供一批关于“{PROJECT_THEME}”的原始用户反馈。请你分析这些反馈，并构建一个包含以下层级的全新码表：
    - net: 概括性的主类别
    - subnet: 更具体一些的子类别
    - label: 对具体反馈内容的高度概括标签

    # 待分析的原始反馈样本
    {samples}

    # 输出要求
    请直接输出一个JSON数组，其中每个对象都代表一条编码规则，并包含所有层级。请为每一条规则建议一个逻辑连续且唯一的 code（例如从N101开始）。
    [
      {{
        "net": "退货流程",
        "subnet": "退款速度",
        "code": "N101",
        "label": "退款到账快"
      }},
      {{
        "net": "客服沟通",
        "subnet": "响应效率",
        "code": "N201",
        "label": "客服回复不及时"
      }}
    ]
    """
    
    print("正在调用AI生成码表，请稍候...")
    response = model.generate_content(prompt)
    
    print(f"\n--- AI生成的码表草案已保存至 {OUTPUT_CODEBOOK_DRAFT} ---")
    
    with open(OUTPUT_CODEBOOK_DRAFT, 'w', encoding='utf-8') as f:
        f.write(response.text)
        
    print("\n请打开该文件，手动审核、精修后，另存为您的新项目码表文件（例如 new_question_codebook.csv）。")

if __name__ == "__main__":
    generate_draft()
