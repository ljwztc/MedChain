import json
import sys

from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

def intersection_over_union(test, gt):
    """
    Calculate the Intersection over Union (IoU) of two lists of strings.
    
    Parameters:
    test (list of str): The testing results.
    gt (list of str): The ground truth.
    
    Returns:
    float: The Intersection over Union (IoU) score.
    None: If gt is empty or None.
    """
    if test is None or test == []:
        return 0.0
    if gt is None or gt == []:
        return None
    
    set1 = set(test)
    set2 = set(gt)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    iou = len(intersection) / len(union) if union else 0.0
    return iou


model_name = "internvl2"

# 读取 merged_cases.json 文件
with open(F'merged_cases.json', 'r', encoding='utf-8') as f:
    merged_cases = json.load(f)

# 读取 doctor_exams.json 文件
with open(F'doctor_exams_{model_name}.json', 'r', encoding='utf-8') as f:
    doctor_exams = json.load(f)

iou_results = {}

# 遍历每个病例
for case_name, doctor_exam in tqdm(doctor_exams.items()):
    # 提取 ground truth
    ground_truth = []
    case_data = merged_cases.get(case_name, {})
    case_introduction = case_data.get("【病案介绍】", {})
    physical_exam = case_introduction.get("查体", {}).get("体格检查", {})
    auxiliary_exam = case_introduction.get("查体", {}).get("辅助检查", {})

    ground_truth.extend(physical_exam.keys())
    ground_truth.extend(auxiliary_exam.keys())
    
    # 提取 testing results
    testing_results = []
    testing_results.extend(doctor_exam.get("体格检查", []))
    testing_results.extend(doctor_exam.get("辅助检查", []))
    
    # 计算 IoU
    iou = intersection_over_union(testing_results, ground_truth)
    iou_results[case_name] = iou

# 计算平均 IoU 分数
valid_scores = [score for score in iou_results.values() if score is not None]
average_iou = sum(valid_scores) / len(valid_scores) if valid_scores else 0

# 将结果保存到 iou_results.json 文件中
with open(f'iou_results_{model_name}.json', 'w', encoding='utf-8') as f:
    json.dump({
        "individual_scores": iou_results,
        "average_score": average_iou
    }, f, ensure_ascii=False, indent=4)

print(f"IoU 计算完成，结果已保存到 iou_results.json 文件中。平均 IoU 分数为: {average_iou:.4f}")