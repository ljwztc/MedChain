from utils.funtion_api import *
from utils.RAG_Judgement import RAG_judgemennt
from utils.llm_api import *


def task5(case_message=None, diagnosis=None, first_room=None, MDDB_path=None, env="answer", feedback=None, RAG_symptom=None, top_3_case=None, RAG=None):
    if env == "answer":
        doctor_num = 3
        task_obj = "治疗"
        description = str(case_message['主诉'])
        history = str(case_message['既往史'])
        symptom = str(case_message['现病史'])
        check = str(case_message['查体'])
        diagnosis = str(diagnosis)
        # patient_con = f'病人主诉：{description}\n症状：{symptom}\n既往史：{history}\n辅助检查：{check}\n诊断：{diagnosis}'
        patient_con = f'病人主诉：{description}\n辅助检查：{check}\n诊断：{diagnosis}'
        if RAG == True:
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[top_3_case]
            with open('./my_prompt/task5_treatment_first.txt', 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace('[case now]', patient_con)
            case_num = 1
            for index, row in rows.iterrows():
                description = str(row['主诉'])
                sym = str(row['症状'])
                assist_check = str(row['辅助检查'])
                case_history = str(row['既往史'])
                treatment = str(row['治疗项目'])
                case_diagnosis = str(row['诊断结果'])
                # case_str = f"病人主诉：{description}\n症状：{sym}\n既往史：{case_history}\n辅助检查：{assist_check}\n诊断：{case_diagnosis}\n治疗项目：{treatment}"
                case_str = f"病人主诉：{description}\n辅助检查：{assist_check}\n诊断：{case_diagnosis}\n治疗项目：{treatment}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1
            # print(case_prompt)
        elif RAG == False:
            with open('./my_prompt/task5_treatment_first_no_rag.txt', 'r', encoding='utf-8') as file:
                case_prompt = file.read().replace('[case now]', patient_con)

        sys_prompt = f"你是一名专业的{first_room}医生，有着丰富的临床治疗经验，你需要根据病人的情况，输出治疗方案。"

        # with open('./my_prompt/judgement.txt', 'r', encoding='utf-8') as file:
        #     judgement_promopt = file.read().replace("[question]", case_prompt)
        #     judgement_response = gpt_api(max_tokens=500, system_role="你是一个专业的医学问题难度划分专家，能够将接下来的医学问题进行等级划分。", user_input=judgement_promopt)
        #     if "普通" in judgement_response:
        #         judge_t5 = False
        #     else:
        #         judge_t5 = True

        response_diagnosis = doctor_group(doctor_num=doctor_num, task_id=5, user_input=case_prompt, sys_prompt=sys_prompt,
                                          description=patient_con, task_obj=task_obj, first_room=first_room,
                                          max_tokens=500, doc_RAG=RAG)

        return response_diagnosis, patient_con

    elif env == "feedback":
        with open('./my_prompt/rag_case.txt', 'r', encoding='utf-8') as file:
            case_prompt = file.read()
            case_mes_all = pd.read_excel(MDDB_path, sheet_name=first_room)
            rows = case_mes_all.iloc[top_3_case]
            case_num = 1
            for index, row in rows.iterrows():
                description = str(row['主诉'])
                sym = str(row['症状'])
                assist_check = str(row['辅助检查'])
                case_history = str(row['既往史'])
                treatment = str(row['治疗项目'])
                case_diagnosis = str(row['诊断结果'])
                # case_str = f"病人主诉：{description}\n症状：{sym}\n既往史：{case_history}\n辅助检查：{assist_check}\n诊断：{case_diagnosis}\n治疗项目：{treatment}"
                case_str = f"病人主诉：{description}\n辅助检查：{assist_check}\n诊断：{case_diagnosis}\n治疗项目：{treatment}"
                case_prompt = case_prompt.replace(f"[case{case_num}]", case_str)
                case_num += 1

        with open('./my_prompt/task5_treatment_feedback.txt', 'r', encoding='utf-8') as file:
            feedback_prompt = file.read().replace("[feedback]", f"{feedback[1]}").replace("[past_case]", f"{case_prompt}")
        sys_prompt = f"你是一名专业的{first_room}医生，有着丰富的临床治疗经验，你需要根据病人的情况，输出治疗选项。"
        # print(feedback_prompt)
        response_diagnosis = gpt_api(max_tokens=200, system_role=sys_prompt, user_input=feedback_prompt)
        return response_diagnosis


