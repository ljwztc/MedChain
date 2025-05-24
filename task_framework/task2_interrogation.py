from utils.funtion_api import *
from utils.RAG_Judgement import RAG_judgemennt
from utils.llm_api import *
import ast
import math

def task2(first_room, symptom, MDDB_path, env="answer", feedback=None, RAG_sym=None, RAG=None, rag_case=None, ori_result=None):
    judge_feedback = None
    judge_c1 = None
    judge_c2 = None
    doctor_num = 3
    with open('./my_prompt/doctor_sys.txt', 'r', encoding='utf-8') as file:
        sys_prompt = file.read().replace("[first_room]", first_room)
    # COT
    # 常规检查
    task_obj = "体格检查"
    if RAG == True:
        if env == "answer":
            # 检索病历库
            top_3_case = rag_case
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[top_3_case]
        if task_obj == "体格检查" and env == "answer":
            with open("./my_prompt/task2_RAG_case_first.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace("[task_obj]", task_obj).replace("[case now]", symptom)
            case_num = 1
            for index, row in rows.iterrows():
                description = "病人主诉：" + str(row['主诉'])
                sym = "详细情况：" + str(row['症状'])
                physics_exm = str(row['体格检查'])
                try:
                    # 处理检查项目
                    data_dict = ast.literal_eval(physics_exm)
                    keys_with_quotes = [f'"{key}"' for key in data_dict.keys()]
                    physics_exm = "{"  + ', '.join(keys_with_quotes) + "}"
                    # ----------------------------
                except:
                    physics_exm = "None"
                case_str = f"{description}\n症状：{sym}\n体格检查：{physics_exm}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            # print(case_prompt)
            # with open('./my_prompt/judgement.txt', 'r', encoding='utf-8') as file:
            #     judgement_promopt = file.read().replace("[question]", case_prompt)
            #     judgement_response = gpt_api(max_tokens=500, system_role="你是一个专业的医学问题难度划分专家，能够将接下来的医学问题进行等级划分。", user_input=judgement_promopt)
            #     if "普通" in judgement_response:
            #         judge_c1 = "no_feedback"
            #     else:
            #         judge_c1 = "yes_feedback"

            response_physics_exm = doctor_group(doctor_num=doctor_num, task_id=2, user_input=case_prompt, sys_prompt=sys_prompt,
                                                description=symptom, task_obj=task_obj, first_room=first_room, max_tokens=500, doc_RAG=True)
    elif RAG == False:
        if task_obj == "体格检查" and env == "answer":
            with open("./my_prompt/task2_RAG_case_first_no_rag.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace("[task_obj]", task_obj).replace("[case now]", symptom)
            # with open('./my_prompt/judgement.txt', 'r', encoding='utf-8') as file:
            #     judgement_promopt = file.read().replace("[question]", case_prompt)
            #     judgement_response = gpt_api(max_tokens=500, system_role="你是一个专业的医学问题难度划分专家，能够将接下来的医学问题进行等级划分。", user_input=judgement_promopt)
            #     if "普通" in judgement_response:
            #         judge_c1 = "no_feedback"
            #     else:
            #         judge_c1 = "yes_feedback"
            response_physics_exm = doctor_group(doctor_num=doctor_num, task_id=2, user_input=case_prompt, sys_prompt=sys_prompt,
                                                description=symptom, task_obj=task_obj, first_room=first_room, max_tokens=500, doc_RAG=False)


    if task_obj == "体格检查" and env == "feedback":
        if feedback[1][0] != True:
            top_3_case = rag_case
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[top_3_case]
            with open("./my_prompt/rag_case.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read()
            case_num = 1
            for index, row in rows.iterrows():
                description = "病人主诉：" + str(row['主诉'])
                sym = "详细情况：" + str(row['症状'])
                physics_exm = str(row['体格检查'])
                try:
                    # 处理检查项目
                    data_dict = ast.literal_eval(physics_exm)
                    keys_with_quotes = [f'"{key}"' for key in data_dict.keys()]
                    physics_exm = "{"  + ', '.join(keys_with_quotes) + "}"
                    # ----------------------------
                except:
                    physics_exm = "None"
                case_str = f"{description}\n症状：{sym}\n体格检查：{physics_exm}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            with open("./my_prompt/task2_RAG_case_first_feedback_1.txt", 'r', encoding='utf-8') as file:
                case_prompt_feedback = file.read().replace("[feedback]", feedback[1][0]).replace("[past_case]", case_prompt)
            # print(case_prompt_feedback)
            response_physics_exm = gpt_api(max_tokens=500, system_role=sys_prompt, user_input=case_prompt_feedback)
        else:
            response_physics_exm = ori_result[0]

    # 辅助检查
    task_obj = "辅助检查"
    if RAG == True:
        if task_obj == "辅助检查" and env == "answer":
            with open("./my_prompt/task2_RAG_case_first.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace("[task_obj]", task_obj).replace("[case now]", symptom)
            case_num = 1
            for index, row in rows.iterrows():
                description =str(row['主诉'])
                sym = str(row['症状'])
                assist_exm = str(row['辅助检查'])
                try:
                    # 处理检查项目
                    data_dict = ast.literal_eval(assist_exm)
                    keys_with_quotes = [f'"{key}"' for key in data_dict.keys()]
                    assist_exm = "{"  + ', '.join(keys_with_quotes) + "}"
                    # ----------------------------
                except:
                    assist_exm = None
                case_str = f"{description}\n症状：{sym}\n辅助检查：{assist_exm}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            # with open('./my_prompt/judgement.txt', 'r', encoding='utf-8') as file:
            #     judgement_promopt = file.read().replace("[question]", case_prompt)
            #     judgement_response = gpt_api(max_tokens=500, system_role="你是一个专业的医学问题难度划分专家，能够将接下来的医学问题进行等级划分。", user_input=judgement_promopt)
            #     if "普通" in judgement_response:
            #         judge_c2 = "no_feedback"
            #     else:
            #         judge_c2 = "yes_feedback"
            response_assist_exm = doctor_group(doctor_num=doctor_num, task_id=2, user_input=case_prompt, sys_prompt=sys_prompt,
                                        description=symptom, task_obj=task_obj, first_room=first_room, max_tokens=500, doc_RAG=True)
    elif RAG == False:
        if task_obj == "辅助检查" and env == "answer":
            with open("./my_prompt/task2_RAG_case_first_no_rag.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace("[task_obj]", task_obj).replace("[case now]", symptom)
            # with open('./my_prompt/judgement.txt', 'r', encoding='utf-8') as file:
            #     judgement_promopt = file.read().replace("[question]", case_prompt)
            #     judgement_response = gpt_api(max_tokens=500, system_role="你是一个专业的医学问题难度划分专家，能够将接下来的医学问题进行等级划分。", user_input=judgement_promopt)
            #     if "普通" in judgement_response:
            #         judge_c2 = "no_feedback"
            #     else:
            #         judge_c2 = "yes_feedback"
            response_assist_exm = doctor_group(doctor_num=doctor_num, task_id=2, user_input=case_prompt, sys_prompt=sys_prompt,
                                        description=symptom, task_obj=task_obj, first_room=first_room, max_tokens=500, doc_RAG=False)
    

    if task_obj == "辅助检查" and env == "feedback":
        if feedback[1][1] != True:
            top_3_case = rag_case
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[top_3_case]
            with open("./my_prompt/rag_case.txt", 'r', encoding='utf-8') as file:
                case_prompt = file.read()
            case_num = 1
            for index, row in rows.iterrows():
                description =str(row['主诉'])
                sym = str(row['症状'])
                assist_exm = str(row['辅助检查'])
                try:
                    # 处理检查项目
                    data_dict = ast.literal_eval(assist_exm)
                    keys_with_quotes = [f'"{key}"' for key in data_dict.keys()]
                    assist_exm = "{"  + ', '.join(keys_with_quotes) + "}"
                    # ----------------------------
                except:
                    assist_exm = None
                case_str = f"{description}\n症状：{sym}\n辅助检查：{assist_exm}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            with open("./my_prompt/task2_RAG_case_first_feedback_2.txt", 'r', encoding='utf-8') as file:
                case_prompt_feedback = file.read().replace("[feedback]", feedback[1][1]).replace("[past_case]", case_prompt)
            # print(case_prompt_feedback)
            response_assist_exm = gpt_api(max_tokens=500, system_role=sys_prompt, user_input=case_prompt_feedback)
        else:
            response_assist_exm = ori_result[1]

    # if judge_c1 == "yes_feedback" and judge_c2 == "yes_feedback":
    #     judge_feedback = True
    # else:
    #     judge_feedback = False
    
    return str(response_physics_exm), str(response_assist_exm)

