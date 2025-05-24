import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.funtion_api import *
import json
from tqdm import tqdm
from task_framework.task1_triage import task1
from task_framework.task2_interrogation import task2
from task_framework.task3_image import task3
from task_framework.task4_diagnosis import task4
from task_framework.task5_treatment import task5
from utils.local_monitor import local_check
from utils.llm_api import *
from collections import defaultdict
from utils.RAG_Judgement import RAG_judgemennt
import argparse


dataset_path = "./datasets/filtered_data_test_set.json"
with open(dataset_path, 'r', encoding='utf-8') as file:
    data_all = list(json.load(file).items())

# sign_num = "nothing"
global_rag = True
global_feedback = True

if global_rag == True and global_feedback == True:
    sign_num = "entirely"

elif global_rag == False and global_feedback == True:
    sign_num = "no_rag"

elif global_rag == True and global_feedback == False:
    sign_num = "no_feedback"

elif global_rag == False and global_feedback == False:
    sign_num = "nothing"

task1_json = {}
task2_json_physical = {}
task2_json_auxiliary = {}
task3_json = {}
task4_json = {}
task5_json = {}
task1_result_path = "./exp_supplyment/test/task1.json"
task2_1_result_path = "./exp_supplyment/test/task2_physical.json"
task2_2_result_path = "./exp_supplyment/test/task2_auxiliary.json"
task3_result_path = "./exp_supplyment/test/task3.json"
task4_result_path = "./exp_supplyment/test/task4.json"
task5_result_path = "./exp_supplyment/test/task5.json"


for sample in tqdm(data_all):
    # 超参数
    # ------------------------------------------------------------------------------------------------------------------
    task1_feedback = False
    task2_feedback = False
    task3_feedback = False
    task4_feedback = False
    task5_feedback = False
    feedback_max_turn = 4
    zhusu, jiwangshi, xianbingshi, chati, keshi, jieguo = extract_json_data(sample)
    MDDB_path = "./MDDB/MDDB.xlsx"
    MDDB_all_path = "./MDDB/MDDB_all.xlsx"
    img_src = './datasets/MedImg/'
    room_data_path = "./datasets/merged_cases_room_structure.txt"
    room_structure = establish_room_structure(room_data_path)
    # ------------------------------------------------------------------------------------------------------------------

    # task1-rag
    # task1_top_3_case = RAG_judgemennt(MDDB_path, [xianbingshi], 1, first_room=None)
    task1_top_3_case = RAG_judgemennt(MDDB_path, [zhusu], 1, first_room=None)
    case_mes_all = pd.read_excel(MDDB_all_path)
    rows = case_mes_all.iloc[task1_top_3_case]
    with open("./my_prompt/task1_rag_case_first_room.txt", 'r', encoding='utf-8') as file:
            case_prompt1 = file.read()
    with open("./my_prompt/task1_rag_case_second_room.txt", 'r', encoding='utf-8') as file:
            case_prompt2 = file.read()
    case_num = 1
    for index, row in rows.iterrows():
        description = "病人主诉：" + str(row['主诉'])
        sym = "详细情况：" + str(row['症状'])
        rag_case_first_room = "一级科室：" + str(row['一级科室'])
        rag_case_second_room = "二级科室：" + str(row['二级科室'])
        case_str1 = f"{description}\n{sym}\n{rag_case_first_room}"
        case_str2 = f"{description}\n{sym}\n{rag_case_second_room}"
        case_prompt1 = case_prompt1.replace(f"[case{case_num}]", case_str1)
        case_prompt2 = case_prompt2.replace(f"[case{case_num}]", case_str2)
        case_num += 1
    task1_rag_case_first_room = case_prompt1
    task1_rag_case_second_room = case_prompt2
    task1_rag_case = [task1_rag_case_first_room, task1_rag_case_second_room]

    # task1
    # ------------------------------------------------------------------------------------------------------------------
    task1_result, first_room, patient_description = task1("病人主诉：" + zhusu + "\n" + "详细情况：" + xianbingshi, rag_judge=global_rag, rag_case_content=task1_rag_case)
    # task1_result, first_room, patient_description = task1("病人主诉：" + zhusu + "\n", rag_judge=global_rag, rag_case_content=task1_rag_case)

    if task1_result == "error":
        # with open(f'./output/task1_triage_result-{sign_num}.txt', 'a', encoding='utf-8') as file:
        #     file.write(str(task1_result) + '\n')
        continue
    
    # 局部反馈机制
    # ------------------------------------------------------------------------------------------------------------------
    if global_feedback == True:
        down1_sys_role = "None"
        feedback_turn = 0
        while task1_feedback is False and feedback_turn < feedback_max_turn:
            # feedback = local_check(down1_sys_role, "病人情况：" + patient_description + "\n\n" + "分诊结果：" + str(task1_result), 1)
            feedback = local_check(down1_sys_role, [patient_description, task1_rag_case, task1_result], 1)
            feedback_turn += 1
            if feedback[0] == True:
                task1_result, first_room = task1(env="feedback", feedback=feedback[1], ori_result=task1_result, ori_first_room=first_room, rag_judge=True, rag_case_content=task1_rag_case)
            else:
                task1_feedback = True
    # 输出格式化
    task1_result[1] = task1_result[1].split(",")

    # try:
    #     with open(task1_result_path, 'r', encoding='utf-8') as file:
    #         task1_json = json.load(file)
    # except:
    #     pass

    task1_json[sample[0]] = {"一级科室": task1_result[0], "二级科室": task1_result[1]}

    with open(task1_result_path, 'w', encoding='utf-8') as file:
        json.dump(task1_json, file, ensure_ascii=False, indent=4)

    # 错误处理
    if first_room == "error":
        with open('./my_prompt/triage_sys.txt', 'r', encoding='utf-8') as file:
            triage_sys_prompt = file.read()
        with open(f'./my_prompt/triage_user_1_answer.txt', 'r', encoding='utf-8') as file:
            if env == "answer":
                triage_user_1_prompt = file.read().replace("[condition]", patient_description)
        first_room = gpt_api(max_tokens=100, system_role=triage_sys_prompt, user_input=triage_user_1_prompt)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
# with open("./output/task1_rag.json", 'w', encoding='utf-8') as file:
#     json.dump(task1_json, file, ensure_ascii=False, indent=4)

    # first_room = sample[1]['tags']['科室'][0]
    patient_description = "病人主诉：" + zhusu + "\n" + "详细情况：" + xianbingshi
    # patient_description = "病人主诉：" + zhusu + "\n"

    # feedback_rag
    # top_3_case_global = RAG_judgemennt(MDDB_path, [xianbingshi], 2, first_room=first_room)
    top_3_case_global = RAG_judgemennt(MDDB_path, [zhusu], 2, first_room=first_room)

    # task2
    # ------------------------------------------------------------------------------------------------------------------
    response_physics_exm, response_assist_exm = task2(first_room=first_room, symptom=patient_description, MDDB_path=MDDB_path, RAG_sym=xianbingshi, RAG=global_rag, rag_case=top_3_case_global)
    if global_feedback == True:
        down2_sys_role = "None"
    # 局部反馈机制
    # ------------------------------------------------------------------------------------------------------------------
        feedback_turn = 0
        while task2_feedback is False and feedback_turn < feedback_max_turn:
            feedback = local_check(down2_sys_role, [patient_description, response_physics_exm, response_assist_exm], 2, pase_cases=top_3_case_global, MDDB_path=MDDB_path, first_room=first_room)
            feedback_turn += 1
            if feedback[0] == True:
                response_physics_exm, response_assist_exm = task2(first_room=first_room, symptom=patient_description, MDDB_path=MDDB_path, env="feedback", feedback=feedback, RAG=True, rag_case=top_3_case_global, 
                ori_result=[response_physics_exm, response_assist_exm])
            else:
                task2_feedback = True
    
    # 输出格式化
    physical_exams_list = ["一般检查", "头颅眼耳鼻喉检查", "颈部检查", "胸部检查", "腹部检查", "脊柱和四肢检查", "皮肤检查", "神经系统检查", "泌尿生殖系统检查"]
    physical_exams = [dept for dept in physical_exams_list if re.search(dept, response_physics_exm)]
    physical_exams = ','.join(physical_exams)
    if len(physical_exams) == 0:
        physical_exams = "None"

    auxiliary_exams_list = ["X-ray", "MRI", "CT", "超声", "核医学成像", "血液学检查", "尿液检查", "粪便检查", "内镜检查", "病理检查", "X线片"]
    auxiliary_exams = [dept for dept in auxiliary_exams_list if re.search(dept, response_assist_exm)]
    auxiliary_exams = ','.join(auxiliary_exams)
    if len(auxiliary_exams) == 0:
        auxiliary_exams = "None"

    result_physical_exams_list = physical_exams.split(",")
    result_auxiliary_exams_list = auxiliary_exams.split(",")

    # try:
    #     with open(task2_1_result_path, 'r', encoding='utf-8') as file:
    #         task2_json_physical = json.load(file)
    #     with open(task2_2_result_path, 'r', encoding='utf-8') as file:
    #         task2_json_auxiliary = json.load(file)
    # except:
    #     pass

    task2_json_physical[sample[0]] = result_physical_exams_list
    task2_json_auxiliary[sample[0]] = result_auxiliary_exams_list

    with open(task2_1_result_path, 'w', encoding='utf-8') as file:
        json.dump(task2_json_physical, file, ensure_ascii=False, indent=4)
    with open(task2_2_result_path, 'w', encoding='utf-8') as file:
        json.dump(task2_json_auxiliary, file, ensure_ascii=False, indent=4)


# with open("./output/task2_json_physical_nothing.json", 'w', encoding='utf-8') as file:
#     json.dump(task2_json_physical, file, ensure_ascii=False, indent=4)
# with open("./output/task2_json_auxiliary_nothing.json", 'w', encoding='utf-8') as file:
#     json.dump(task2_json_auxiliary, file, ensure_ascii=False, indent=4)
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # task3
    # ------------------------------------------------------------------------------------------------------------------
    task3_judge = True
    if "【病案介绍】" not in sample[1].keys() or "图像" not in sample[1]["【病案介绍】"].keys():
        task3_judge = False
        task3_result = "None"
    else:
        tuxiang = [item["文件名"] for item in sample[1]["【病案介绍】"]["图像"]]
        tuxiang_calss = [item["分类"] for item in sample[1]["【病案介绍】"]["图像"]]
        tuxiang_calss = [item[0] for item in tuxiang_calss]
        img_path = tuxiang
        img_class = tuxiang_calss
        if len(img_path) >= 5:
            img_path = [f'{img_src}{path}' for path in img_path[:5]]
            img_class = img_class[:5]
        else:
            img_path = [f'{img_src}{path}' for path in img_path]
        
        C = defaultdict(list)
        for category, filename in zip(img_class, img_path):
            C[category].append(filename)
        C = dict(C)
        imgs = [[category, filenames] for category, filenames in C.items()]
        img_path = imgs

    if task3_judge:
        img_report = ""
        down3_sys_role = f"你是一名专业的{first_room}影像医生,拥有丰富的临床经验，能够很好地生成医学影像报告。"
        task3_json[sample[0]] = {}
        for single_case in img_path:
            task3_result = task3(image_path=single_case, first_room=first_room, symptom=patient_description)
            # if global_feedback == True:
            #     if judge_task3_feedback == True:
            #         feedback_turn = 0
            #         while task3_feedback is False and feedback_turn < feedback_max_turn:
            #             feedback = local_check(down3_sys_role, task3_result, 3, img_path=single_case[1])
            #             feedback_turn += 1
            #             if feedback is not True:
            #                 task3_result, judge_task3_feedback_tmp = task3(image_path=single_case, first_room=first_room, feedback=feedback, env="feedback", symptom=patient_description, img_report=task3_result)
            #             else:
            #                 task3_feedback = True
            # try:
            #     with open(task3_result_path, 'r', encoding='utf-8') as file:
            #         task3_json = json.load(file)
            # except:
            #     pass

            task3_json[sample[0]][single_case[0]] = task3_result

            with open(task3_result_path, 'w', encoding='utf-8') as file:
                json.dump(task3_json, file, ensure_ascii=False, indent=4)

            img_report += f'"{single_case[0]}":' + task3_result
        task3_result = img_report

# with open("./output/task3.json", 'w', encoding='utf-8') as file:
#     json.dump(task3_json, file, ensure_ascii=False, indent=4)


        # print("case-$[" + str(task3_result) + ']$\n')
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # task4
    # ------------------------------------------------------------------------------------------------------------------

    try:
        assist_check = chati['辅助检查']
    except:
        assist_check = '无'
    case_message = {'主诉': zhusu, '既往史': jiwangshi, '现病史': xianbingshi, '查体': assist_check}
    # print(case_message)
    if sample[0] == '11422_房间隔缺损修补术术中桡动脉压力波形消失1例':
        continue
    task4_result, patient_con = task4(case_message=case_message, first_room=first_room, MDDB_path=MDDB_path, RAG_symptom=xianbingshi, top_3_case=top_3_case_global, RAG=global_rag)
    if global_feedback == True:
    # 局部反馈机制
    # ------------------------------------------------------------------------------------------------------------------
        down4_sys_role = f"你是一名专业的{first_room}医生，拥有着丰富的临床诊断经验，您需要从患者的情况分析诊室医生的诊断结果是否合理。"
        feedback_turn = 0
        while task4_feedback is False and feedback_turn < feedback_max_turn:
            feedback = local_check(down4_sys_role, "病人情况：" + patient_con + "\n\n" + "诊断结果：\n" + str(task4_result), 4, pase_cases=top_3_case_global, MDDB_path=MDDB_path, first_room=first_room)
            feedback_turn += 1
            if feedback[0] == True:
                task4_result = task4(first_room=first_room, env="feedback", feedback=feedback, RAG=True, top_3_case=top_3_case_global, MDDB_path=MDDB_path)
            else:
                task4_feedback = True

    # try:
    #     with open(task4_result_path, 'r', encoding='utf-8') as file:
    #         task4_json = json.load(file)
    # except:
    #     pass

    task4_json[sample[0]] = task4_result.replace("'", "").replace('"', "")

    with open(task4_result_path, 'w', encoding='utf-8') as file:
        json.dump(task4_json, file, ensure_ascii=False, indent=4)
# with open("./output/task4_feedback.json", 'w', encoding='utf-8') as file:
#     json.dump(task4_json, file, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # task4_result = sample[1]['tags']['病种']
    
    # task5
    # ------------------------------------------------------------------------------------------------------------------
    task5_result, patient_con = task5(case_message=case_message, diagnosis=task4_result, first_room=first_room, MDDB_path=MDDB_path, RAG_symptom=xianbingshi, top_3_case=top_3_case_global, RAG=global_rag)
    if global_feedback == True:
        down5_sys_role = f"你是一名专业的{first_room}医生，拥有着丰富的临床治疗经验，您需要从患者的情况分析诊室医生的治疗结果是否合理。"
    # 局部反馈机制
    # ------------------------------------------------------------------------------------------------------------------
        feedback_turn = 0
        while task5_feedback is False and feedback_turn < feedback_max_turn:
            feedback = local_check(down5_sys_role, patient_con + "\n" + "治疗项目：" + str(task5_result), 5, pase_cases=top_3_case_global, MDDB_path=MDDB_path, first_room=first_room)
            feedback_turn += 1
            if feedback[0] == True:
                task5_result = task5(first_room=first_room, env="feedback", feedback=feedback, RAG=True, top_3_case=top_3_case_global, MDDB_path=MDDB_path)
            else:
                task5_feedback = True

    # 输出格式化
    treatment_list = ["手术", "介入治疗", "药物治疗", "化学治疗", "抗生素治疗", "放射治疗", "物理疗法", "免疫疗法", "心理治疗", "中医治疗", "基因治疗"]
    treatment = [dept for dept in treatment_list if re.search(dept, task5_result)]
    treatment = ','.join(treatment)
    if len(treatment) == 0:
        treatment = "None"

    treatment_result = treatment.split(",")

    # try:
    #     with open(task5_result_path, 'r', encoding='utf-8') as file:
    #         task5_json = json.load(file)
    # except:
    #     pass

    task5_json[sample[0]] = treatment_result

    with open(task5_result_path, 'w', encoding='utf-8') as file:
        json.dump(task5_json, file, ensure_ascii=False, indent=4)

# with open("./output/task5_rag.json", 'w', encoding='utf-8') as file:
#     json.dump(task5_json, file, ensure_ascii=False, indent=4)      

    # print("case-$[" + str(task5_result) + ']\n')
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
