from openai import OpenAI
import re
import mimetypes
import base64
import anthropic
# from utils.HuatuoGPT.huatuo_cli_demo_stream import *


# # 华佗初始化
# # -----------------------------------------------------------------------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--model-name", type=str, default="/data/huangguolin/HGL_workspace/MedAgent_baseline/LLM_models/HuatuoGPT-7B")
# parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
# parser.add_argument("--num-gpus", type=str, default="1")
# # parser.add_argument("--load-8bit", action="store_true")
# parser.add_argument("--temperature", type=float, default=0.5)
# parser.add_argument("--max-new-tokens", type=int, default=512)
# args = parser.parse_args()

# model, tokenizer = load_model(args.model_name, args.device, args.num_gpus)
# model = model.eval()
# history = []

# def gpt_api(max_tokens, system_role, user_input):
#     pre = 0
#     for outputs in chat_stream(model, tokenizer, user_input, history, max_new_tokens=max_tokens, temperature=0, repetition_penalty=1.2, context_len=1024):
#         outputs = outputs.strip()
#         # outputs = outputs.split("")
#         now = len(outputs)
#         if now - 1 > pre:
#             # print(outputs[pre:now - 1], end="", flush=True)
#             pre = now - 1
#     # print(outputs[pre:], flush=True)
#     ans = outputs
#     return ans



def gpt_api(max_tokens, system_role, user_input):
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="http://0.0.0.0:23333/v1"
    )
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature = 0,
                    # stop=["<|Bot|>"],
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input}
                    ],
                )
    return str(response.choices[0].message.content)


def img_api(img_path, user_input):
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="http://0.0.0.0:23333/v1"
    )
    model_name = client.models.list().data[0].id
    messages_content = [
        {'type': 'text', 'text': f'{user_input}'}
    ]
    # 添加每张图片到消息内容
    for url in img_path:
        messages_content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'{url}',
            }
        })

    response = client.chat.completions.create(
        model=model_name,
        temperature = 0,
        max_tokens= 1000,
        messages=[
            # {"role": "system", "content": "你是一名专业的影像医生，拥有丰富的影像学临床经验，您需要基于患者的影像情况，判断之前专家医生分析的报告是否合理。"},
            {'role': 'user', 'content': messages_content}
            ]
        )
    return str(response.choices[0].message.content)



# def gpt_api(max_tokens, system_role, user_input):
#     # client = OpenAI(
#     #     api_key='YOUR_API_KEY',
#     #     base_url='http://0.0.0.0:23333/v1'
#     # )
#     client = OpenAI(
#         api_key="sk-nBJedi9bIfecjGPrutPTChuTcAOkQgPhsXPPMqhc9GyjnRjy",
#         base_url="https://api.chatanywhere.tech/v1"
#     )
#     # model_name = client.models.list().data[0].id
#     response = client.chat.completions.create(
#         # model=model_name,
#         model="gpt-4o-mini",
#         max_tokens=max_tokens,
#         temperature=0,
#         messages=[
#             {"role": "system", "content": system_role},
#             {"role": "user", "content": user_input}
#         ],
#     )

#     return response.choices[0].message.content


# def img_api(img_path, user_input):
#     client = OpenAI(
#         api_key="sk-nBJedi9bIfecjGPrutPTChuTcAOkQgPhsXPPMqhc9GyjnRjy",
#         base_url="https://api.chatanywhere.tech/v1"
#     )
#     model_name = "gpt-4o-mini"


#     encoded_images = []
#     for image_path in img_path:
#         mime_type, _ = mimetypes.guess_type(image_path)
#         if not mime_type:
#             raise ValueError(f"Could not determine the MIME type of the image: {image_path}")
#         with open(image_path, "rb") as image_file:
#             encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
#             encoded_images.append(f"data:{mime_type};base64,{encoded_image}")


#     messages_content = [
#         {'type': 'text', 'text': f'{user_input}'}
#     ]
#     # 添加每张图片到消息内容
#     for url in encoded_images:
#         messages_content.append({
#             'type': 'image_url',
#             'image_url': {
#                 'url': f'{url}',
#             }
#         })

#     response = client.chat.completions.create(
#         model=model_name,
#         # max_tokens= 1000,
#         temperature = 0,
#         messages=[
#             {
#             'role': 'user',
#             'content': messages_content,
#         }]
#         )
#     return str(response.choices[0].message.content)




def llm_embedding(text):
    client = OpenAI(
        api_key="sk-nBJedi9bIfecjGPrutPTChuTcAOkQgPhsXPPMqhc9GyjnRjy",
        base_url="https://api.chatanywhere.tech/v1"
    )
    response = client.embeddings.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return response.data[0].embedding


def doctor_group(doctor_num, task_id, user_input, sys_prompt, description, task_obj, first_room, max_tokens, img_path=None, img_report=None, doc_RAG=None):
    if task_id == 2:
        last_ans = ""
        if doc_RAG == True:
            past_cases = user_input.split('"$')[1]
        for doc_idx in range(doctor_num):
            if doc_idx+1 == 1:
                # print(user_input)
                first_ans = f"专家{doc_idx+1}的观点：" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt, user_input=user_input) + "\n"
                last_ans += first_ans
            else:
                if doc_RAG == True:
                    with open("./my_prompt/task2_RAG_case_other.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans).replace('[task_obj]', task_obj).replace('[past_cases]', past_cases)
                    # print(content)
                elif doc_RAG == False:
                    with open("./my_prompt/task2_RAG_case_other_no_rag.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans).replace('[task_obj]', task_obj)
                now_ans = f"专家{doc_idx+1}的观点：" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt, user_input=content) + "\n"
                last_ans += now_ans
            print(f"专家{doc_idx+1}已经做出诊断！")
        print("正在等待医疗领导者做出最后决策......")
        if doc_RAG == True: 
            with open('./my_prompt/task2_doc_manager.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans).replace('[task_obj]', task_obj).replace('[past_cases]', past_cases)
            # print(final_prompt)
        elif doc_RAG == False:
            with open('./my_prompt/task2_doc_manager_no_rag.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans).replace('[task_obj]', task_obj)
        manager_sys_prompt = f"你是一个专业的{first_room}医生，有着丰富的临床经验，你需要根据病人的情况与其他专家的诊断结果，做出整合。"
        finally_result = gpt_api(max_tokens=max_tokens, system_role=manager_sys_prompt, user_input=final_prompt)
        return finally_result

    elif task_id == 3:
        last_ans = ""
        for doc_idx in range(doctor_num):
            if doc_idx + 1 == 1:
                first_ans = f"专家{doc_idx + 1}的观点：\n" + img_api(img_path=img_path, user_input=sys_prompt + "\n" +user_input) + "\n"
                # print(first_ans)
                last_ans += first_ans
            else:
                with open("./my_prompt/task3_img_case_other.txt", 'r', encoding='utf-8') as file:
                    content = file.read().replace('[other view]', last_ans).replace('[now_case]', description).replace('[img_report]', img_report)
                now_ans = f"专家{doc_idx + 1}的观点：\n" + img_api(img_path=img_path, user_input=sys_prompt + "\n" + content) + "\n"
                # print(now_ans)
                last_ans += now_ans
            print(f"专家{doc_idx + 1}已经做出诊断！")
        print("正在等待医疗领导者做出最后决策......")
        with open('./my_prompt/task3_img_manager.txt', 'r', encoding='utf-8') as file:
            final_prompt = file.read().replace("[other view]", last_ans).replace('[img_report]', img_report)
        # print(final_prompt)
        manager_sys_prompt = f"你是一个专业的{first_room}影像医生，有着丰富的影像学经验，你需要根据病人的情况与其他专家的诊断结果，做出整合。\n下面是其他专家生成的影像报告：\n"
        finally_result = img_api(img_path=img_path, user_input= manager_sys_prompt + "\n" + final_prompt)
        # print(finally_result)
        return finally_result

    elif task_id == 4:
        last_ans = ""
        if doc_RAG == True:
            past_cases = user_input.split('"$')[1]
        for doc_idx in range(doctor_num):
            if doc_idx + 1 == 1:
                first_ans = f"专家{doc_idx + 1}的观点：\n" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt,
                                                                    user_input=user_input) + "\n"
                last_ans += first_ans
            else:
                if doc_RAG == True:
                    with open("./my_prompt/task4_diagnosis_other.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans).replace('[past_cases]', past_cases)
                elif doc_RAG == False:
                    with open("./my_prompt/task4_diagnosis_other_no_rag.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans)
                # print(content)
                now_ans = f"专家{doc_idx + 1}的观点：\n" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt, user_input=content) + "\n"
                last_ans += now_ans
            print(f"专家{doc_idx + 1}已经做出诊断！")
        print("正在等待医疗领导者做出最后决策......")
        if doc_RAG == True:
            with open('./my_prompt/task4_diagnosis_other_manager.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans).replace('[past_cases]', past_cases)
            # print(final_prompt)
        elif doc_RAG == False:
            with open('./my_prompt/task4_diagnosis_other_manager_no_rag.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans)
        manager_sys_prompt = f"你是一个专业的{first_room}医生，有着丰富的临床诊断经验，你需要根据病人的情况与其他专家的诊断结果，做出整合。"
        finally_result = gpt_api(max_tokens=max_tokens, system_role=manager_sys_prompt, user_input=final_prompt)
        return finally_result

    elif task_id == 5:
        last_ans = ""
        if doc_RAG == True:
            past_cases = user_input.split('"$')[1]
        for doc_idx in range(doctor_num):
            if doc_idx + 1 == 1:
                first_ans = f"专家{doc_idx + 1}的观点：\n" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt, user_input=user_input) + "\n"
                last_ans += first_ans
            else:
                if doc_RAG == True:
                    with open("./my_prompt/task5_treatment_other.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans).replace('[past_cases]', past_cases)
                    # print(content)
                elif doc_RAG == False:
                    with open("./my_prompt/task5_treatment_other_no_rag.txt", 'r', encoding='utf-8') as file:
                        content = file.read().replace('[case now]', description).replace('[other view]', last_ans)
                now_ans = f"专家{doc_idx + 1}的观点：\n" + gpt_api(max_tokens=max_tokens, system_role=sys_prompt, user_input=content) + "\n"
                last_ans += now_ans
            print(f"专家{doc_idx + 1}已经做出诊断！")
        print("正在等待医疗领导者做出最后决策......")
        if doc_RAG == True:
            with open('./my_prompt/task5_doc_manager.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans).replace('[past_cases]', past_cases)
            # print(final_prompt)
        elif doc_RAG == False:
            with open('./my_prompt/task5_doc_manager_no_rag.txt', 'r', encoding='utf-8') as file:
                final_prompt = file.read().replace("[case now]", description).replace("[other view]", last_ans)
        manager_sys_prompt = f"你是一个专业的{first_room}医生，有着丰富的临床治疗经验，你需要根据病人的情况与其他专家的诊断结果，做出整合。"
        finally_result = gpt_api(max_tokens=max_tokens, system_role=manager_sys_prompt, user_input=final_prompt)
        return finally_result