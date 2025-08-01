import json
import requests
from openai import OpenAI
import sys

from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI(
    api_key="......",  
    base_url="......"
)

# 定义要提取的体格检查和辅助检查项目
physical_exams = [
    "一般检查", "头颅眼耳鼻喉检查", "颈部检查", "胸部检查", "腹部检查",
    "脊柱和四肢检查", "皮肤检查", "神经系统检查", "泌尿生殖系统检查"
]
auxiliary_exams = [
    "X-ray", "MRI", "CT", "超声", "核医学成像", "血液学检查",
    "尿液检查", "粪便检查", "内镜检查", "病理检查"
]

model_name = "qwen2"

# 读取 doctor.json 文件
with open(f'doctor_{model_name}.json', 'r', encoding='utf-8') as f:
    doctor_data = json.load(f)

# 尝试读取已处理的问诊记录
try:
    with open(f'processed_keys_{model_name}.json', 'r', encoding='utf-8') as f:
        processed_keys = json.load(f)
except FileNotFoundError:
    processed_keys = []

# 保存结果的字典
try:
    with open(f'doctor_exams_{model_name}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

# 定义提取检查项目的函数
def extract_exams(content):
    prompt = (
        f"从以下内容中提取出医生询问的体格检查和辅助检查项目并分别列出：\n{content}\n"
        "体格检查包括：一般检查（包括身高、体重、体温、血压、脉搏等）、头颅眼耳鼻喉检查、颈部检查（包括甲状腺、颈部淋巴结）、胸部检查（包括肺部、心脏）、腹部检查、脊柱和四肢检查、皮肤检查、神经系统检查、泌尿生殖系统检查。\n"
        "辅助检查包括：X-ray、MRI、CT、超声、核医学成像、血液学检查、尿液检查、粪便检查、内镜检查、病理检查。\n"
        "仅回复提取出的检查项目名称（注意名称不包括检查项目括号中的内容，如应回复一般检查而不是身高、体重、体温等），不要包含其他任何解释，并确保回答的项目名称在上述提到的检查项目之中（须一字不差，如X光应统一称X-ray），每个检查项目独占一行。"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",                  
                        "text": prompt
                    }
                ],
            }
        ],
        max_tokens=100,
        temperature=0.5
    )
    result = response.choices[0].message.content.strip()
    # print(result)
    # print("---------------------")
    return result

# 遍历每个问诊，提取检查项目并保存
for key, content_list in tqdm(doctor_data.items()):
    if key in processed_keys:
        continue  # 跳过已处理的问诊

    all_content = " ".join(content_list)  # 合并所有内容
    extracted_exams = extract_exams(all_content)
    
    # 初始化体格检查和辅助检查的结果列表
    physical_result = []
    auxiliary_result = []
    
    # 提取的结果按行分割，分别归类到体格检查和辅助检查
    for line in extracted_exams.split('\n'):
        line = line.strip()
        # 检查是否与体格检查项目有包含关系
        for exam in physical_exams:
            if (exam in line or line in exam) and exam not in physical_result:
                physical_result.append(exam)
                break
        # 检查是否与辅助检查项目有包含关系
        for exam in auxiliary_exams:
            if (exam in line or line in exam) and exam not in auxiliary_result:
                auxiliary_result.append(exam)
                break
    
    # 将结果保存到字典中
    results[key] = {
        "体格检查": physical_result,
        "辅助检查": auxiliary_result
    }
    
    # 记录已处理的key
    processed_keys.append(key)
    
    # 每次问诊都保存一次结果
    with open(f'doctor_exams_{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 保存已处理的问诊记录
    with open(f'processed_keys_{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(processed_keys, f, ensure_ascii=False, indent=4)

print("提取完成，结果已保存到 doctor_exams.json 文件中。")