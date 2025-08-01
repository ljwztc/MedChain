import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 读取 conversation_results.json 文件
with open(r'C:\Users\cola\Desktop\wenzhen\result_filter\conversation_results_qwen2.json', 'r', encoding='utf-8') as f:
    conversations = json.load(f)

# 筛选出 role 是 'user' 的内容
user_contents = {}
for key, value in conversations.items():
    user_contents[key] = [item['content'] for item in eval(value) if item['role'] == 'user']

# 将结果保存为新的 doctor.json 文件
with open('doctor_qwen2.json', 'w', encoding='utf-8') as f:
    json.dump(user_contents, f, ensure_ascii=False, indent=4)

print("筛选完成，结果已保存到 doctor.json 文件中。")