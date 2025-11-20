import os
import sys

sys.path.append(os.getcwd())
from models.metaquery import MetaQuery
from PIL import Image
import torch

model_path = "/data/yangyi/hf_models_upload/ssv2_image_1to6/whole_models/ssv2_image_1to6_epoch0_59_step226000"
# checkpoints_dir = "/data/yangyi/checkpoints_already/ssv2_image_head/image_head/epoch0_59_step226000"
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

input_images = [
    Image.open("/data/yangyi/metaquery_action_refactoring/samples/ssv2_src/5.png").convert("RGB").resize((512, 512))
]

instruction = "moving puncher closer to scissor"
# instruction = "open the book"
timestep_gap = 1
caption = f"Instruction: {instruction}. Generate the updated image observation after {timestep_gap} timesteps."

samples = metaquery.sample_images(
    caption=caption,
    input_images=input_images,
    negative_prompt="",
    num_inference_steps=30,
    num_images_per_prompt=1,
    gap=timestep_gap,
)

samples[0].save(f"/data/yangyi/metaquery_action_refactoring/samples/ssv2/samples_train/move/output_{timestep_gap}.png")
print(f"gap{timestep_gap}.png")

