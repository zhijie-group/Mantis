import numpy as np
from skimage.util import view_as_blocks
from PIL import Image



import os
import sys
import torch
import json


from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

print("Import Ready!!!")

# model_path = "/data/yangyi/hf_models_upload/libero_long_main_table/whole_models/epoch30_85_step50500"
dataset_root_dir = "/data/yangyi/datasets/LIBERO_lerobot_v2/libero_10_no_noops_lerobot"
result_dir = "/data/yangyi/metaquery_action_refactoring/samples/libero_10"

###################################### Load Model ######################################
target_image_size = 512
vae_downsample_f = 32



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
# wrist_image_transform = _make_transform(256)

img_0 = to_pil_image(primary_image_transform(dataset[idx-4][primary_image_key][0]))
img_1 = to_pil_image(primary_image_transform(dataset[idx][primary_image_key][0]))





img_0 = img_0.resize((522, 522))
img_1 = img_1.resize((522, 522))


def calculate_patch_similarity(patches1, patches2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim


def patchify(image, patch_size=29):
    """
    Converts an image into non-overlapping patches.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches


def find_most_different_patches(img_0, img_1, patch_size=29, top_k=285):
    """
    找出两张图像间最不相似的patches
    """
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)
    
    similarity = calculate_patch_similarity(patches1, patches2)
    grid_size = 522 // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)
    
    patch_scores = [(i * grid_size + j, similarity_2d[i, j])
                    for i in range(grid_size) for j in range(grid_size)]
    
    patch_scores.sort(key=lambda x: x[1], reverse=False)
    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]
    return top_patch_ids, similarity_2d


top_different_patches, similarity_2d = find_most_different_patches(img_0, img_1)


difference_2d = 1 - similarity_2d
diff_min, diff_max = difference_2d.min().item(), difference_2d.max().item()
difference_norm = (difference_2d - diff_min) / (diff_max - diff_min + 1e-8)

img_pil = img_1
w, h = img_pil.size

def get_grid_bounds(total_size, grid_count, index):
    """计算精确的网格边界，避免累积误差"""
    start = int(total_size * index / grid_count)
    end = int(total_size * (index + 1) / grid_count)
    return start, end

grid_size = 522 // 29  # 假设patch_size=29

# ================= 第一种方法：差异度热图 =================
mask_img = img_pil.copy()
mask_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

for i in range(grid_size):
    for j in range(grid_size):
        # 精确计算每个网格的边界
        x_start, x_end = get_grid_bounds(w, grid_size, j)
        y_start, y_end = get_grid_bounds(h, grid_size, i)
        
        cell_w = x_end - x_start
        cell_h = y_end - y_start
        
        # 差异度越大，红色越明显
        alpha = int(difference_norm[i, j] * 200)  # 控制透明度
        
        # 使用红色表示差异较大的区域
        cell_overlay = Image.new("RGBA", (cell_w, cell_h), (0, 191, 255, alpha))
        mask_overlay.paste(cell_overlay, (x_start, y_start))

difference_heatmap = Image.alpha_composite(mask_img.convert("RGBA"), mask_overlay)
difference_heatmap.save("/data/yangyi/metaquery_action_refactoring/backbone_attn_map/saved_imgs_difference/img1_difference_heatmap.png")
print("差异度热图已保存：img1_difference_heatmap.png")


# # ================= 第二种方法：高亮最不相似的top patches =================
# mask_img2 = img_pil.copy()
# mask_overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))

# # 创建top patches的标记矩阵
# top_patches_matrix = np.zeros((grid_size, grid_size), dtype=bool)
# for patch_id in top_different_patches:
#     i = patch_id // grid_size
#     j = patch_id % grid_size
#     top_patches_matrix[i, j] = True

# for i in range(grid_size):
#     for j in range(grid_size):
#         x_start, x_end = get_grid_bounds(w, grid_size, j)
#         y_start, y_end = get_grid_bounds(h, grid_size, i)
        
#         cell_w = x_end - x_start
#         cell_h = y_end - y_start
        
#         if top_patches_matrix[i, j]:
#             # 高亮显示最不相似的patches（亮黄色）
#             cell_overlay = Image.new("RGBA", (cell_w, cell_h), (255, 150, 0, 150))
#         else:
#             # 其他区域用半透明黑色遮罩
#             cell_overlay = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 100))
        
#         mask_overlay2.paste(cell_overlay, (x_start, y_start))

# top_patches_highlight = Image.alpha_composite(mask_img2.convert("RGBA"), mask_overlay2)
# top_patches_highlight.save("/data/yangyi/qwen_attention_map/saved_imgs_difference/img1_top_different_patches.png")
# print("Top差异patches高亮图已保存：img1_top_different_patches.png")



mask_img2 = img_pil.copy()
mask_overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))

# 从similarity_2d中找到最相似的patches（相似度最高的）
flat_similarity = similarity_2d.flatten()
# 获取相似度最高的patch索引（与top_different_patches相反）
top_similar_indices = np.argsort(flat_similarity)[-len(top_different_patches):]  # 取相似度最高的patches
top_similar_patches = top_similar_indices.tolist()

# 创建top patches的标记矩阵
top_patches_matrix = np.zeros((grid_size, grid_size), dtype=bool)
for patch_id in top_similar_patches:
    i = patch_id // grid_size
    j = patch_id % grid_size
    top_patches_matrix[i, j] = True

for i in range(grid_size):
    for j in range(grid_size):
        x_start, x_end = get_grid_bounds(w, grid_size, j)
        y_start, y_end = get_grid_bounds(h, grid_size, i)
        
        cell_w = x_end - x_start
        cell_h = y_end - y_start
        
        if top_patches_matrix[i, j]:
            # 高亮显示最相似的patches（蓝色）
            cell_overlay = Image.new("RGBA", (cell_w, cell_h), (100, 180, 255, 120))
        else:
            # 其他区域用半透明黑色遮罩
            cell_overlay = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        
        mask_overlay2.paste(cell_overlay, (x_start, y_start))

top_patches_highlight = Image.alpha_composite(mask_img2.convert("RGBA"), mask_overlay2)
top_patches_highlight.save("/data/yangyi/metaquery_action_refactoring/backbone_attn_map/saved_imgs_difference/img1_top_similar_patches.png")
print("Top相似patches高亮图已保存：img1_top_similar_patches.png")