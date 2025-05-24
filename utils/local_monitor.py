from utils.llm_api import *
import ast
import pandas as pd
import math

def local_check(down_sys_role, result, task_id_up, img_path=None, pase_cases=None, MDDB_path=None, first_room=None):
    task_name_dict = {'1': "分诊", '2': "问诊", '3': "检查", '4': "诊断", '5': "治疗"}
    task_name = task_name_dict[f'{task_id_up}']
    if task_id_up == 1:
        task1_feedback_judgement = False
        down_sys_role1 = "你是一名分诊结果判别者，拥有着丰富的分诊经验，请你检查分诊医生的“分诊结果”中一级科室是否正确。"
        down_sys_role2 = "你是一名分诊结果判别者，拥有着丰富的分诊经验，请你检查分诊医生的“分诊结果”中二级科室是否正确。"
        with open('./my_prompt/local_down_check_task1_first_room.txt', 'r', encoding='utf-8') as file:
            result1 = f"{str(result[0])}" + '\n' + '一级科室：' + f"{str(result[2][0])}"
            down_prompt1 = file.read().replace("[result]", result1).replace("[past_case]", f"{str(result[1][0])}")
        with open('./my_prompt/local_down_check_task1_second_room.txt', 'r', encoding='utf-8') as file:
            result2 = f"{str(result[0])}" + '\n' + '二级科室：' + f"{str(result[2][1])}"
            down_prompt2 = file.read().replace("[result]", result2).replace("[past_case]", f"{str(result[1][1])}")

        # print(down_prompt1)
        # print(down_prompt2)

        check_response1 = gpt_api(max_tokens=1000, system_role=down_sys_role1, user_input=down_prompt1)
        check_response2 = gpt_api(max_tokens=1000, system_role=down_sys_role2, user_input=down_prompt2)

        with open('./my_prompt/local_feedback_task1_first_room.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response1:
                feedback_first_room = True
            else:
                task1_feedback_judgement = True
                feedback_first_room = file.read().replace("[feedback]", check_response1).replace("[result]", result1)
        with open('./my_prompt/local_feedback_task1_second_room.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response2:
                feedback_second_room = True
            else:
                task1_feedback_judgement = True
                feedback_second_room = file.read().replace("[feedback]", check_response2).replace("[result]", result2)

        # print(feedback_first_room)
        # print(feedback_second_room)

        return [task1_feedback_judgement, [feedback_first_room, feedback_second_room]]


    elif task_id_up == 2:
        task2_feedback_judgement = False
        down_sys_role1 = "你是一名医学体格检查结果判别者，拥有着丰富的医学检查经验，请你检查医学检查医生的“体格检查”中的这些体格检查项目是否正确。"
        down_sys_role2 = "你是一名医学辅助检查结果判别者，拥有着丰富的医学检查经验，请你检查医学检查医生的“辅助检查”中的这些辅助检查项目是否正确。"
        with open('./my_prompt/rag_case.txt', 'r', encoding='utf-8') as file:
            down_prompt1 = file.read()
        with open('./my_prompt/rag_case.txt', 'r', encoding='utf-8') as file:
            down_prompt2 = file.read()
        case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
        rows = case_mes_all.iloc[pase_cases]
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
            assist_exm = str(row['辅助检查'])
            try:
                # 处理检查项目
                data_dict = ast.literal_eval(assist_exm)
                keys_with_quotes = [f'"{key}"' for key in data_dict.keys()]
                assist_exm = "{"  + ', '.join(keys_with_quotes) + "}"
                # ----------------------------
            except:
                assist_exm = "None"
            case_str1 = f"{description}\n{sym}\n体格检查：{physics_exm}\n"
            case_str2 = f"{description}\n{sym}\n辅助检查：{assist_exm}\n"
            down_prompt1 = down_prompt1.replace(f"[case{case_num}]", case_str1)
            down_prompt2 = down_prompt2.replace(f"[case{case_num}]", case_str2)
            case_num += 1

        task2_past_case1 = down_prompt1
        task2_past_case2 = down_prompt2

        with open('./my_prompt/local_down_check_task2_physics.txt', 'r', encoding='utf-8') as file:
            result1 = f"{str(result[0])}" + '\n' + '体格检查：' + f"{str(result[1])}"
            feedback_prompt1 = file.read().replace("[result]", result1).replace("[past_case]", f"{task2_past_case1}")
            # print(feedback_prompt1)
        with open('./my_prompt/local_down_check_task2_assist.txt', 'r', encoding='utf-8') as file:
            result2 = f"{str(result[0])}" + '\n' + '辅助检查：' + f"{str(result[2])}"
            feedback_prompt2 = file.read().replace("[result]", result2).replace("[past_case]", f"{task2_past_case2}")
            # print(feedback_prompt2)

        check_response1 = gpt_api(max_tokens=1000, system_role=down_sys_role1, user_input=feedback_prompt1)
        check_response2 = gpt_api(max_tokens=1000, system_role=down_sys_role2, user_input=feedback_prompt2)

        with open('./my_prompt/local_feedback_task2_physics.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response1:
                feedback_physics = True
            else:
                task2_feedback_judgement = True
                feedback_physics = file.read().replace("[feedback]", check_response1).replace("[result]", result1)
        with open('./my_prompt/local_feedback_task2_assist.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response2:
                feedback_assist = True
            else:
                task2_feedback_judgement = True
                feedback_assist = file.read().replace("[feedback]", check_response2).replace("[result]", result2)

        # print(feedback_physics)
        # print(feedback_physics)

        return [task2_feedback_judgement, [feedback_physics, feedback_assist]]

    elif task_id_up == 5:
        task5_feedback_judgement = False
        with open('./my_prompt/local_down_treatment_check.txt', 'r', encoding='utf-8') as file:
            case_prompt = file.read().replace("[task_name]", f"{task_name}").replace("[result]", f"{result}")
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[pase_cases]
            case_num = 1
            for index, row in rows.iterrows():
                description = str(row['主诉'])
                sym = str(row['症状'])
                assist_check = str(row['辅助检查'])
                case_history = str(row['既往史'])
                treatment = str(row['治疗项目'])
                case_str = f"病人主诉：{description}\n症状：{sym}\n既往史：{case_history}\n辅助检查：{assist_check}\n治疗项目：{treatment}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            # print(case_prompt)
        check_response = gpt_api(max_tokens=1000, system_role=down_sys_role, user_input=case_prompt)
        with open('./my_prompt/local_feedback_task5.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response:
                feedback = True
            else:
                task5_feedback_judgement = True
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        
        return [task5_feedback_judgement, feedback]




    elif task_id_up == 3:
        with open('./my_prompt/local_down_img_check.txt', 'r', encoding='utf-8') as file:
            down_prompt = file.read().replace("[result]", f"{result}")
    elif task_id_up == 4:
        task4_feedback_judgement = False
        with open('./my_prompt/local_down_check_task4.txt', 'r', encoding='utf-8') as file:
            down_prompt = file.read().replace("[result]", f"{result}")
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[pase_cases]
            case_num = 1
            for index, row in rows.iterrows():
                description = str(row['主诉'])
                sym = str(row['症状'])
                case_check = str(row['诊断结果'])
                case_str = f"病人主诉：{description}\n症状：{sym}\n诊断结果：{case_check}"
                down_prompt = down_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
        # print(down_prompt)
        check_response = gpt_api(max_tokens=1000, system_role=down_sys_role, user_input=down_prompt)

        with open('./my_prompt/local_feedback_task4.txt', 'r', encoding='utf-8') as file:
            if "正确" in check_response:
                feedback = True
            else:
                task4_feedback_judgement = True
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        
        return [task4_feedback_judgement, feedback]

        
        

    if task_id_up == 3:
        check_response = img_api(img_path=img_path, user_input=down_prompt)
    else:
        check_response = gpt_api(max_tokens=1000, system_role=down_sys_role, user_input=down_prompt)
    if "正确" in check_response:
        return True
    else:
        if task_id_up == 2:
            with open('./my_prompt/local_feedback_task2.txt', 'r', encoding='utf-8') as file:
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        elif task_id_up == 4:
            with open('./my_prompt/local_feedback_task4.txt', 'r', encoding='utf-8') as file:
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        elif task_id_up == 5:
            with open('./my_prompt/local_feedback_task5.txt', 'r', encoding='utf-8') as file:
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        elif task_id_up == 1:
            with open('./my_prompt/local_feedback_task1.txt', 'r', encoding='utf-8') as file:
                feedback = file.read().replace("[feedback]", check_response).replace("[result]", result)
        else:
            with open('./my_prompt/local_feedback.txt', 'r', encoding='utf-8') as file:
                feedback = file.read().replace("[feedback]", check_response).replace("[down]", "诊室").replace("[task_name]", task_name).replace("[result]", result)
        return feedback

