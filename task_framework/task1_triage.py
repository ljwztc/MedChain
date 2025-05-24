import json
from tqdm import tqdm
from utils.llm_api import *
from utils.funtion_api import *
import re

def task1(patient_description=None, env="answer", feedback=None, ori_result=None, ori_first_room=None, rag_judge=True, rag_case_content=None):
    room_data_path = "./datasets/merged_cases_room_structure.txt"
    room_structure = establish_room_structure(room_data_path)
    # COTs
    # 一级学科分类
    with open('./my_prompt/triage_sys.txt', 'r', encoding='utf-8') as file:
        triage_sys_prompt = file.read()
    if rag_judge == False:
        with open(f'./my_prompt/triage_user_1_{env}.txt', 'r', encoding='utf-8') as file:
            if env == "answer":
                triage_user_1_prompt = file.read().replace("[condition]", patient_description)
            elif env == "feedback":
                triage_user_1_prompt = file.read().replace("[feedback]", feedback)
    elif rag_judge == True:
        with open(f'./my_prompt/triage_user_1_{env}_rag.txt', 'r', encoding='utf-8') as file:
            if env == "answer":
                triage_user_1_prompt = file.read().replace("[condition]", patient_description).replace("[past_case]", rag_case_content[0])
            elif env == "feedback" and feedback[0] != True:
                triage_user_1_prompt = file.read().replace("[feedback]", feedback[0]).replace("[past_case]", rag_case_content[0])

    # print(triage_user_1_prompt)


    if env == "answer":
        first_room = gpt_api(max_tokens=100, system_role=triage_sys_prompt, user_input=triage_user_1_prompt)
    elif env == "feedback":
        if feedback[0] != True:
            feedback_sys = "你是一个专业的分诊医生，有着丰富分诊经验，你需要根据病人的情况，进行精确的科室分诊。"
            first_room = gpt_api(max_tokens=100, system_role=feedback_sys, user_input=triage_user_1_prompt)
        else:
            first_room = ori_result[0]

    
    first_room_list = ['护理科', '药剂科', '口腔科', '儿科', '医学影像科', '眼科', '检验科', '外科', '皮肤性病科', '精神科', '全科', '耳鼻咽喉科', '内科', '急诊科', '肿瘤科', '中医科', '康复科', '妇产科', '心理科']
    # 找到所有匹配的科室及其位置
    matches = [(dept, first_room.index(dept)) for dept in first_room_list if dept in first_room]

    # 根据出现位置排序，并取第一个
    first_room = min(matches, key=lambda x: x[1])[0] if matches else "None"

    # 判错
    try:
        if env == "answer":
            second_room_list = room_structure[first_room]
    except:
        return "error", "error", "error"

    try:
        if env == "feedback":
            second_room_list = room_structure[first_room]
    except:
        return ori_result, ori_first_room

    # 二级学科分类
    if rag_judge == False:
        with open(f'./my_prompt/triage_user_2_{env}.txt', 'r', encoding='utf-8') as file:
            if env == "answer":
                triage_user_2_prompt = file.read().replace("[condition]", patient_description)
                triage_user_2_prompt = triage_user_2_prompt.replace("[list]", str(second_room_list))
            elif env == "feedback":
                triage_user_2_prompt = file.read().replace("[feedback]", feedback)
                triage_user_2_prompt = triage_user_2_prompt.replace("[list]", str(second_room_list))
    elif rag_judge == True:
        with open(f'./my_prompt/triage_user_2_{env}_rag.txt', 'r', encoding='utf-8') as file:
            if env == "answer":
                triage_user_2_prompt = file.read().replace("[condition]", patient_description).replace("[past_case]", rag_case_content[1])
                triage_user_2_prompt = triage_user_2_prompt.replace("[list]", str(second_room_list))
            elif env == "feedback" and feedback[1] != True:
                triage_user_2_prompt = file.read().replace("[feedback]", feedback[1]).replace("[past_case]", rag_case_content[1])
                triage_user_2_prompt = triage_user_2_prompt.replace("[list]", str(second_room_list))

    # print(triage_user_2_prompt)

    if env == "answer":
        second_room = gpt_api(max_tokens=100, system_role=triage_sys_prompt, user_input=triage_user_2_prompt)
    elif env == "feedback":
        if feedback[1] != True:
            feedback_sys = "你是一个专业的分诊医生，有着丰富分诊经验，你需要根据病人的情况，进行精确的科室分诊。"
            second_room = gpt_api(max_tokens=100, system_role=feedback_sys, user_input=triage_user_2_prompt)
        else:
            second_room = ori_result[1]

    # 输出格式化
    first_room_list = ['护理科', '药剂科', '口腔科', '儿科', '医学影像科', '眼科', '检验科', '外科', '皮肤性病科', '精神科', '全科', '耳鼻咽喉科', '内科', '急诊科', '肿瘤科', '中医科', '康复科', '妇产科', '心理科']
    first_room = [dept for dept in first_room_list if re.search(dept, first_room)]
    first_room = ', '.join(first_room)
    if len(first_room) == 0:
        first_room = "None"

    second_room = [dept for dept in second_room_list if re.search(dept, second_room)]
    second_room = ', '.join(second_room)
    if len(second_room) == 0:
        second_room = "None"

    if env == "answer":
        return [first_room, second_room], first_room, patient_description
    elif env == "feedback":
        return [first_room, second_room], first_room






