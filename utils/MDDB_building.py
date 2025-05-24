import json
from funtion_api import *
from tqdm import tqdm

dataset_path = "./datasets/train_set.json"
room_data_path = "./datasets/merged_cases_room_structure.txt"
db_excel_ptah = "./MDDB/MDDB_all.xlsx"
patient_message = ['年龄', '性别', '主诉', '症状', '既往史', '体格检查', '辅助检查', '影像报告', '诊断结果', '治疗项目', '一级科室', '二级科室']

# 读入总数居
with open(dataset_path, 'r', encoding='utf-8') as file:
    data_all = list(json.load(file).items())

df = pd.DataFrame(columns=patient_message)

for case in tqdm(data_all):
    case_title = case[0]
    case_message = case[1]
    case_room, case_info = extract_info(case_message)
    new_data = pd.DataFrame({
        '年龄': [case_info[0]],
        '性别': [case_info[1]],
        '主诉': [case_info[2]],
        '症状': [case_info[3]],
        '既往史': [case_info[4]],
        '体格检查': [case_info[5]],
        '辅助检查': [case_info[6]],
        '影像报告': [case_info[7]],
        '诊断结果': [case_info[8]],
        '治疗项目': [case_info[9]],
        '一级科室': [case_info[10]],
        '二级科室': [case_info[11]]
    })
    df = pd.concat([df, new_data], ignore_index=True)


with pd.ExcelWriter(db_excel_ptah, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, index=False, header=True)



