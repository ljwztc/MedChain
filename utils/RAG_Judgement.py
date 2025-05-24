import json
from tqdm import tqdm
from utils.llm_api import *
import pandas as pd
import numpy as np
from utils.funtion_api import *


def RAG_judgemennt(MDDB_path, message, task_id, first_room):
    if task_id == 2:
        assert len(message) == 1, "当前任务传入的数据不匹配，当前任务为问诊任务，应该传入病人症状"
        patient_symptom = message[0]
        word_em = llm_embedding(patient_symptom)
        smp_data_all = np.load(f"./MDDB/npy_data/{first_room}_symptom.npy", allow_pickle=True)
        score_list = []
        for idx, smp_data in enumerate(smp_data_all):
            if smp_data is None:
                result = (idx, 0)
                score_list.append(result)
            else:
                vet1 = np.array(word_em)
                vet2 = np.array(smp_data)
                cos = cosine_similarity(vet1, vet2)
                result = (idx, cos)
                score_list.append(result)
        top_three = sorted(score_list, key=lambda x: x[1], reverse=True)[:3]
        top_three = [item[0] for item in top_three]
        print("完成RAG检索！")
        return top_three
    elif task_id == 1:
        patient_symptom = message[0]
        word_em = llm_embedding(patient_symptom)
        smp_data_all = np.load(f"./MDDB/npy_data/data_all.npy", allow_pickle=True)
        score_list = []
        for idx, smp_data in enumerate(smp_data_all):
            if smp_data is None:
                result = (idx, 0)
                score_list.append(result)
            else:
                vet1 = np.array(word_em)
                vet2 = np.array(smp_data)
                cos = cosine_similarity(vet1, vet2)
                result = (idx, cos)
                score_list.append(result)
        top_three = sorted(score_list, key=lambda x: x[1], reverse=True)[:3]
        top_three = [item[0] for item in top_three]
        print("完成RAG检索！")
        return top_three




