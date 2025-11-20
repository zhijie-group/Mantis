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

model_path = "/data/yangyi/hf_models_upload/libero_goal_main_table/whole_models/epoch72_03_step60000"
# checkpoints_dir = "/data/yangyi/checkpoints_already/ssv2_image_head/image_head/epoch0_59_step226000"
# dataset_root_dir = "/data/yangyi/datasets/droid_100_lerobot"
dataset_root_dir = "/data/yangyi/datasets/LIBERO_lerobot_v2/libero_goal_no_noops_lerobot"
result_dir = "/data/yangyi/metaquery_action_refactoring/samples/libero_goal"

###################################### Load Model ######################################
target_image_size = 512
vae_downsample_f = 32
input_size = target_image_size / vae_downsample_f

metaquery = MetaQuery.from_pretrained(
    model_path,
    device="cuda",
)
device = next(metaquery.parameters()).device

###### load image heads ######
# metaquery.model.transformer.load_state_dict(
#     torch.load(os.path.join(checkpoints_dir, "image_head.pt"), map_location=device)
# )
# metaquery.model.connector.load_state_dict(
#     torch.load(os.path.join(checkpoints_dir, "connector.pt"), map_location=device)
# )
# embed_weight = torch.load(os.path.join(checkpoints_dir, "embed_tokens_weight.pt"), map_location=device)
# metaquery.model.mllm_backbone.model.embed_tokens.weight = torch.nn.Parameter(embed_weight)

metaquery = metaquery.to(torch.float32)


###################################### Load Dataset ######################################
ds_meta = LeRobotDatasetMetadata(dataset_root_dir)
# primary_image_key, wrist_image_key = ds_meta.camera_keys[1], ds_meta.camera_keys[0]
primary_image_key, wrist_image_key = ds_meta.camera_keys[0], ds_meta.camera_keys[1]

tasks_path = os.path.join(dataset_root_dir, "meta", "tasks.jsonl")
with open(tasks_path, 'r', encoding='utf-8') as f:
    task_map = {data["task_index"]: data["task"] for data in map(json.loads, f)}

# episodes_stats_path = os.path.join(dataset_root_dir, "meta", "episodes_stats.json")
# with open(episodes_stats_path, 'r') as f:
#     episodes_stats = json.load(f)

timestep_gap = 4
start_idx = 50
delta_timestamps = {
    primary_image_key: [0, timestep_gap / ds_meta.fps],
    wrist_image_key: [0, timestep_gap / ds_meta.fps],
}
dataset = LeRobotDataset(
    dataset_root_dir,
    # episodes=episodes_stats['success_indices'],
    episodes=[190],
    delta_timestamps=delta_timestamps
)


###################################### Prepare Input ######################################
def _make_transform(size):
    return v2.Compose([v2.Resize(size), v2.CenterCrop(size)])

primary_image_transform = _make_transform(512)
wrist_image_transform = _make_transform(256)

for i in range(7):
    idx = start_idx + i * 4

    input_images = [
        [
            primary_image_transform(dataset[idx][primary_image_key][0]), 
            wrist_image_transform(dataset[idx][wrist_image_key][0]),
        ]
    ]
    target_images = primary_image_transform(dataset[idx][primary_image_key][1])
    caption = task_map[dataset[idx]['task_index'].item()]


    ###################################### Sample and Save ######################################
    samples = metaquery.sample_images(
        caption=caption,
        input_images=input_images,
        negative_prompt="",
        num_inference_steps=30,
        num_images_per_prompt=1,
        gap=timestep_gap,
    )

    samples[0].save(os.path.join(result_dir, "samples", f"output_idx_{idx}_gap_{timestep_gap}.png"))
    to_pil_image(input_images[0][0]).save(os.path.join(result_dir, "source", f"output_idx_{idx}_gap_{timestep_gap}.png"))
    to_pil_image(target_images).save(os.path.join(result_dir, "ground_truth", f"output_idx_{idx}_gap_{timestep_gap}.png"))
    
    print(idx)

print("Finish!!!")