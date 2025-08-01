#-*- coding:utf-8 -*-

import json
from autogen import UserProxyAgent, ConversableAgent
import openai
import sys
import os

from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

def load_cases(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def filter_cases(cases):
    filtered_cases = {}
    for case_name, details in cases.items():
        if "【病案介绍】" in details:
            intro = details["【病案介绍】"]
            if "主诉" in intro and "查体" in intro:
                exam = intro["查体"]
                if "体格检查" in exam and "辅助检查" in exam:
                    filtered_cases[case_name] = details
    return filtered_cases

def create_patient_message(intro):
    chief_complaints = " ".join(intro["主诉"])
    physical_exams = intro["查体"]["体格检查"]
    inspections = intro["查体"]["辅助检查"]
    if "既往史" in intro:
        past_history = " ".join(intro["既往史"])
    if "现病史" in intro:
        current_history = " ".join(intro["现病史"])
    if "既往史" in intro and "现病史" in intro:
        return f"主诉: {chief_complaints}\n体格检查: {physical_exams}\n辅助检查: {inspections}\n既往史: {past_history}\n现病史: {current_history}"
    elif "既往史" in intro:
        return f"主诉: {chief_complaints}\n体格检查: {physical_exams}\n辅助检查: {inspections}\n既往史: {past_history}"
    elif "现病史" in intro:
        return f"主诉: {chief_complaints}\n体格检查: {physical_exams}\n辅助检查: {inspections}\n现病史: {current_history}"
    else:
        return f"主诉: {chief_complaints}\n体格检查: {physical_exams}\n辅助检查: {inspections}"


local_llm_config={
    "config_list": [
        {
            "model": "gpt-4o-mini", # Loaded with LiteLLM command
            "api_key": ".....", # Not needed
            "base_url": "....."  # Your LiteLLM URL
        }
    ],
    "cache_seed": None # Turns off caching, useful for testing different models
}

def run_conversation(case_name, intro):
    patient_message = create_patient_message(intro)
    print(patient_message)
    chief_complaints = " ".join(intro["主诉"])

    # 配置Patient角色
    assistant = ConversableAgent(
        name="Patient",
        code_execution_config={"use_docker":False},
        llm_config=local_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda msg: "早日康复" in msg["content"],
        system_message=f"你要扮演一个标准化病人(Standardized Patient)。以下是你的病案情况：\n{patient_message}\n你需要基于病案情况直接回答医生的问题（请勿编造医患对话）。注意，除非医生明确提问体格检查和辅助检查的情况，否则请不要主动说明或提问体格检查和辅助检查的相关内容。若医生问到病案中不存在的内容，请表示不知道，切忌虚构内容。无论何时，请记住你仅扮演标准化病人这一个角色。",
        )

    # 配置Doctor角色
    user_proxy = UserProxyAgent(
        name="Doctor",
        code_execution_config=False,
        llm_config=local_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda msg: "早日康复" in msg["content"],
        system_message=f"你是一个医生。已知病人的主诉如下：\n{chief_complaints}\n你需要通过与病人对话来获取更多信息。体格检查包括：一般检查（包括身高、体重、体温、血压、脉搏等）、头颅眼耳鼻喉检查、颈部检查（包括甲状腺、颈部淋巴结）、胸部检查（包括肺部、心脏）、腹部检查、脊柱和四肢检查、皮肤检查、神经系统检查、泌尿生殖系统检查。辅助检查包括：X-ray、MRI、CT、超声、核医学成像、血液学检查、尿液检查、粪便检查、内镜检查、病理检查。在获取一定信息后（如既往史、现病史），请你根据病人情况选择要问询的体格检查和辅助检查情况，每轮对话只问一项或两项，请尽可能多的询问检查项目（体格检查和辅助检查每个至少问一项），直至可以判定病情。最后结束对话时请说“祝您早日康复”。",
        )
    
    # 开始对话
    response = assistant.initiate_chat(user_proxy, message="你好，医生。")
    return response.chat_history

def save_results(filename, results):
    # 将结果转换为可序列化的形式
    serializable_results = {k: v.to_dict() if hasattr(v, 'to_dict') else str(v) for k, v in results.items()}
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(serializable_results, file, ensure_ascii=False, indent=4)

def main():
    input_filename = 'merged_cases.json'
    output_filename = 'conversation_results.json'

    # 加载病例数据
    cases = load_cases(input_filename)
    filtered_cases = filter_cases(cases)

    # 如果结果文件存在，加载已处理的结果
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as file:
            results = json.load(file)
    else:
        results = {}

    # 处理每个病例并保存结果
    for case_name, details in tqdm(filtered_cases.items()):
        if case_name not in results:
            # print(f"正在处理病例: {case_name}")
            intro = details["【病案介绍】"]
            try:
                response = run_conversation(case_name, intro)
                results[case_name] = response
                # 每处理完一个病例就保存结果
                save_results(output_filename, results)
            except Exception as e:
                print(f"处理病例 {case_name} 时发生错误: {e}")
                continue

if __name__ == "__main__":
    main()
