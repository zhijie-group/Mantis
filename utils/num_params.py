# import os
# import sys
# sys.path.append(os.getcwd())
# from models.metaquery import MetaQuery
# import torch

# model_path = "/data/yangyi/hf_models_upload/droid_image_action_language/whole_models/epoch2_30"
# metaquery = MetaQuery.from_pretrained(
#     model_path,
# )

# metaquery.mllm_backbone
# metaquery.transformer
# metaquery.connector
# metaquery.policy_head

# total_params = sum(p.numel() for p in model.parameters())
# print(f"总参数量：{total_params / 1e4:.2f} 万 ≈ {total_params / 1e6:.2f} M ≈ {total_params / 1e9:.2f} B")





import os
import sys
sys.path.append(os.getcwd())
from models.metaquery import MetaQuery
import torch

def count_parameters(model, unit='B'):
    """计算模型参数量，支持单位转换（B/百万/千）"""
    total_params = sum(p.numel() for p in model.parameters())
    if unit == 'B':
        return total_params / 1e9  # 转换为十亿（B）
    elif unit == 'M':
        return total_params / 1e6
    elif unit == 'K':
        return total_params / 1e3
    else:
        return total_params

# 加载模型
model_path = "/data/yangyi/hf_models_upload/droid_image_action_language/whole_models/epoch2_30"
metaquery = MetaQuery.from_pretrained(model_path)

# 计算各组件参数量（单位：B）
components = {
    "mllm_backbone": metaquery.model.mllm_backbone,
    "transformer": metaquery.model.transformer,
    "connector": metaquery.model.connector,
    "policy_head": metaquery.model.policy_head,
    "vae": metaquery.vae
}

for name, component in components.items():
    params_b = count_parameters(component)
    print(f"{name} 参数量: {params_b:.4f} B")