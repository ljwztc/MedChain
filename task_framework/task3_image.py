from utils.llm_api import img_api, doctor_group


def task3(image_path, symptom, first_room, env="answer", feedback=None, img_report=None):
    judge_t3 = None
    if env == "answer":
        doctor_num = 1
        with open('./my_prompt/task3_img_first.txt') as file:
            sys_role = file.read().replace("[img_class]", image_path[0]).replace("[symptom]", symptom)
        # print(sys_role)
        img_report = img_api(img_path=image_path[1], user_input=sys_role)
        # print(img_report)
        sys_prompt = f"你是一名专业的{first_room}影像学医生，拥有丰富的临床经验，能够很好地生成医学影像报告。"
        with open ("./my_prompt/task3_img_report.txt", "r", encoding='utf-8') as file:
            doc_group_prompt = file.read().replace("[img_report]", img_report).replace("[now_case]", symptom)
        # print(doc_group_prompt)
        # with open ('./my_prompt/judgement_task3.txt', 'r', encoding='utf-8') as file:
        #     judgement_prompt = file.read().replace("[question]", doc_group_prompt)
        #     judgement_response = img_api(img_path=image_path[1], user_input=judgement_prompt)
        #     if "普通" in judgement_response:
        #         judge_t3 = False
        #     else:
        #         judge_t3 = True

        result = doctor_group(doctor_num=doctor_num, task_id=3, user_input=doc_group_prompt, sys_prompt=sys_prompt,
                            description=symptom, task_obj="None", first_room=first_room, max_tokens=2000,
                            img_path=image_path[1], img_report=img_report)
    
    # elif env == "feedback":
    #     with open('./my_prompt/task3_img_feedback.txt') as file:
    #         feedback_prompt = file.read().replace("[feedback]", feedback).replace("[now_case]", symptom).replace("[img_report]", img_report)
    #     result = img_api(img_path=image_path[1], user_input=feedback_prompt)

    return result
    



















