import os
import sys
import torch
import json

sys.path.append(os.getcwd())
from models.metaquery import MetaQuery

from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

print("Import Ready!!!")

model_path = "/data/yangyi/hf_models_upload/libero_long_main_table/whole_models/epoch30_85_step50500"
dataset_root_dir = "/data/yangyi/datasets/LIBERO_lerobot_v2/libero_10_no_noops_lerobot"
result_dir = "/data/yangyi/metaquery_action_refactoring/samples/libero_10"

###################################### Load Model ######################################
target_image_size = 512
vae_downsample_f = 32
input_size = target_image_size / vae_downsample_f

metaquery = MetaQuery.from_pretrained(
    model_path,
    device="cuda",
)
device = next(metaquery.parameters()).device
# metaquery = metaquery.to(torch.float32)

metaquery = metaquery.to(torch.bfloat16)

###################################### Load Dataset ######################################
ds_meta = LeRobotDatasetMetadata(dataset_root_dir)
# primary_image_key, wrist_image_key = ds_meta.camera_keys[1], ds_meta.camera_keys[0]
primary_image_key, wrist_image_key = ds_meta.camera_keys[0], ds_meta.camera_keys[1]

tasks_path = os.path.join(dataset_root_dir, "meta", "tasks.jsonl")
with open(tasks_path, 'r', encoding='utf-8') as f:
    task_map = {data["task_index"]: data["task"] for data in map(json.loads, f)}

timestep_gap = 4
idx = 95
delta_timestamps = {
    primary_image_key: [0, timestep_gap / ds_meta.fps],
    wrist_image_key: [0, timestep_gap / ds_meta.fps],
}
dataset = LeRobotDataset(
    dataset_root_dir,
    # episodes=episodes_stats['success_indices'],
    episodes=[147],
    delta_timestamps=delta_timestamps
)


###################################### Prepare Input ######################################
def _make_transform(size):
    return v2.Compose([v2.Resize(size), v2.CenterCrop(size)])

primary_image_transform = _make_transform(512)
wrist_image_transform = _make_transform(256)

input_images = [
    [
        primary_image_transform(dataset[idx][primary_image_key][0]), 
        wrist_image_transform(dataset[idx][wrist_image_key][0]),
    ]
]
caption = task_map[dataset[idx]['task_index'].item()]


###################################### Sample and Save ######################################
samples, outputs = metaquery.sample_actions(
    caption=caption,
    input_images=input_images,
    num_images_per_prompt = 1,
    gap = 4,
    eval_mode = "action_chunking_dynamic_temporal_agg"
)






























# outputs_attentions = outputs.attentions
# layer_idx = 1

# attn_map = outputs_attentions[layer_idx].to(torch.float32).squeeze(0).mean(dim=0)

# # attn_tensor = torch.stack([attn.to(torch.float32) for attn in outputs_attentions], dim=0)
# # attn_map = attn_tensor.mean(dim=(0, 2))
# # attn_map = attn_map.squeeze(0)

# print(attn_map.shape)


# input_ids = inputs["input_ids"].squeeze(0)

# v_token_start = (input_ids == 151652).nonzero(as_tuple=True)[0][0].item() + 1
# v_token_end   = (input_ids == 151653).nonzero(as_tuple=True)[0][0].item()
# t_token_start = (input_ids == 151653).nonzero(as_tuple=True)[0][-1].item() + 1
# t_token_end   = (input_ids == 151645).nonzero(as_tuple=True)[0][-1].item()

# relation = attn_map[t_token_start:t_token_end, v_token_start:v_token_end]
# relation = relation.mean(dim=0).cpu()


# print(relation.shape)






# import torch
# import numpy as np
# from PIL import Image

# relation_2d = relation.reshape(18, 18)
# rel_min, rel_max = relation_2d.min().item(), relation_2d.max().item()
# relation_norm = (relation_2d - rel_min) / (rel_max - rel_min + 1e-8)

# img_pil = image
# w, h = img_pil.size

# # ================= 精确网格划分 =================
# def get_grid_bounds(total_size, grid_count, index):
#     """计算精确的网格边界，避免累积误差"""
#     start = int(total_size * index / grid_count)
#     end = int(total_size * (index + 1) / grid_count)
#     return start, end

# # ================= 第一张图：透明蒙版优化版 =================
# mask_img = img_pil.copy()
# mask_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # 完全透明背景

# for i in range(18):
#     for j in range(18):
#         # 精确计算每个网格的边界
#         x_start, x_end = get_grid_bounds(w, 18, j)
#         y_start, y_end = get_grid_bounds(h, 18, i)
        
#         cell_w = x_end - x_start
#         cell_h = y_end - y_start
        
#         # 计算透明度 (relation值越大，蒙版越透明)
#         alpha = int((1 - relation_norm[i, j]) * 180)
        
#         # 创建当前网格的蒙版
#         cell_overlay = Image.new("RGBA", (cell_w, cell_h), (255, 0, 0, alpha))
#         mask_overlay.paste(cell_overlay, (x_start, y_start))

# first_output = Image.alpha_composite(mask_img.convert("RGBA"), mask_overlay)
# first_output.save("/data/yangyi/qwen_attention_map/saved_imgs/output_mask_relation.png")
# print("第一张图已保存：output_mask_relation.png")

# # ================= 第二张图：阈值筛选优化版 =================
# mask_img2 = img_pil.copy()
# mask_overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))

# # 计算阈值
# flat_rel = relation_norm.flatten()
# threshold_value = np.percentile(flat_rel.numpy(), 70)  # 前20%对应80百分位
# print(f"Relation前20%阈值: {threshold_value:.4f}")

# for i in range(18):
#     for j in range(18):
#         # 精确计算网格边界
#         x_start, x_end = get_grid_bounds(w, 18, j)
#         y_start, y_end = get_grid_bounds(h, 18, i)
        
#         cell_w = x_end - x_start
#         cell_h = y_end - y_start
        
#         # 只对低于阈值的区域添加黑色蒙版
#         # if relation_norm[i, j] < threshold_value:
#         #     cell_overlay = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 180))
#         #     mask_overlay2.paste(cell_overlay, (x_start, y_start))

#         if relation_norm[i, j] > threshold_value:  # 改为大于阈值的判断
#             # 使用浅黄色(255, 255, 204)并设置透明度180
#             cell_overlay = Image.new("RGBA", (cell_w, cell_h), (255, 255, 204, 180))
#             mask_overlay2.paste(cell_overlay, (x_start, y_start))

# second_output = Image.alpha_composite(mask_img2.convert("RGBA"), mask_overlay2)
# second_output.save("/data/yangyi/qwen_attention_map/saved_imgs/output_mask_threshold.png")
# print("第二张图已保存：output_mask_threshold.png")

# # ================= 可选：添加网格线用于调试 =================
# def add_grid_lines(img, grid_size=18, color=(255, 255, 255, 100)):
#     """添加网格线帮助调试对齐问题"""
#     from PIL import ImageDraw
    
#     overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
    
#     w, h = img.size
    
#     # 绘制垂直线
#     for i in range(1, grid_size):
#         x = int(w * i / grid_size)
#         draw.line([(x, 0), (x, h)], fill=color, width=1)
    
#     # 绘制水平线
#     for i in range(1, grid_size):
#         y = int(h * i / grid_size)
#         draw.line([(0, y), (w, y)], fill=color, width=1)
    
#     return Image.alpha_composite(img.convert("RGBA"), overlay)


# first_debug_img = add_grid_lines(first_output)
# first_debug_img.save("/data/yangyi/qwen_attention_map/saved_imgs/first_debug_img.png")

# second_debug_img = add_grid_lines(second_output)
# second_debug_img.save("/data/yangyi/qwen_attention_map/saved_imgs/second_debug_img.png")






# # import torch
# # import numpy as np
# # from PIL import Image

# # relation_2d = relation.reshape(18, 18)
# # rel_min, rel_max = relation_2d.min().item(), relation_2d.max().item()
# # relation_norm = (relation_2d - rel_min) / (rel_max - rel_min + 1e-8)



# # img_pil = image
# # w, h = img_pil.size

# # # ================= 第一张图：透明蒙版，relation 值越大颜色越浅 =================
# # mask_img = img_pil.copy()
# # mask_overlay = Image.new("RGBA", (w, h), (255, 0, 0, 0))  # 红色蒙版，初始透明

# # cell_w, cell_h = w // 18, h // 18

# # for i in range(18):
# #     for j in range(18):
# #         alpha = int((1 - relation_norm[i, j]) * 180)  # 越大越浅（透明）
# #         cell_overlay = Image.new("RGBA", (cell_w, cell_h), (255, 0, 0, alpha))
# #         mask_overlay.paste(cell_overlay, (j * cell_w, i * cell_h))

# # first_output = Image.alpha_composite(mask_img.convert("RGBA"), mask_overlay)
# # first_output.save("/data/yangyi/qwen_attention_map/saved_imgs/output_mask_relation.png")
# # print("第一张图已保存：output_mask_relation.png")

# # # ================= 第二张图：阈值筛选，前20%不填色，其余填色 =================
# # mask_img2 = img_pil.copy()
# # mask_overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # 黑色蒙版

# # # 找出前 20% 大的值
# # flat_rel = relation_norm.flatten()
# # threshold_value = np.percentile(flat_rel.numpy(), 95)  # 前20%
# # print(f"Relation前20%阈值: {threshold_value:.4f}")

# # for i in range(18):
# #     for j in range(18):
# #         if relation_norm[i, j] < threshold_value:
# #             # 蒙版覆盖
# #             cell_overlay = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 200))
# #             mask_overlay2.paste(cell_overlay, (j * cell_w, i * cell_h))

# # second_output = Image.alpha_composite(mask_img2.convert("RGBA"), mask_overlay2)
# # second_output.save("/data/yangyi/qwen_attention_map/saved_imgs/output_mask_threshold.png")
# # print("第二张图已保存：output_mask_threshold.png")

