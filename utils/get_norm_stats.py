# import json

# def calculate_global_action_stats(jsonl_path):
#     global_min = [float('inf')] * 14
#     global_max = [float('-inf')] * 14
    
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line.strip())
#             action_min = data['stats']['action']['min']
#             action_max = data['stats']['action']['max']
            
#             for i in range(14):
#                 if action_min[i] < global_min[i]:
#                     global_min[i] = action_min[i]
            
#             for i in range(14):
#                 if action_max[i] > global_max[i]:
#                     global_max[i] = action_max[i]
    
#     print(f"action.min global minimums per position: {global_min}")
#     print(f"action.max global maximums per position: {global_max}")

# calculate_global_action_stats("/data/yangyi/datasets/aloha_lerobot_recognition_merged/meta/episodes_stats.jsonl")










import json
from typing import List

def calculate_global_action_stats(jsonl_paths: List[str]):
    # 初始化14个位置的全局最小和最大值
    global_min = [float('inf')] * 14
    global_max = [float('-inf')] * 14
    
    # 遍历每个输入的JSONL文件
    for path in jsonl_paths:
        print(f"正在处理文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 获取当前行的action min和max
                action_min = data['stats']['action']['min']
                action_max = data['stats']['action']['max']
                
                # 更新全局最小值
                for i in range(14):
                    if action_min[i] < global_min[i]:
                        global_min[i] = action_min[i]
                
                # 更新全局最大值
                for i in range(14):
                    if action_max[i] > global_max[i]:
                        global_max[i] = action_max[i]
    
    # 输出结果
    print(f"\naction.min 全局最小值（按位置）: {global_min}")
    print(f"action.max 全局最大值（按位置）: {global_max}")

# 四个JSONL文件的路径列表（请替换为实际文件路径）
jsonl_files = [
    "/data/yangyi/datasets/aloha_lerobot/numbers/7/meta/episodes_stats.jsonl",
    "/data/yangyi/datasets/aloha_lerobot/numbers/8/meta/episodes_stats.jsonl",
    "/data/yangyi/datasets/aloha_lerobot/numbers/A/meta/episodes_stats.jsonl",
    "/data/yangyi/datasets/aloha_lerobot/numbers/Z/meta/episodes_stats.jsonl",
]

# 执行计算
calculate_global_action_stats(jsonl_files)