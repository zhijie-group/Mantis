# Save huggingface checkpoint as simplified models
import os
import sys
sys.path.append(os.getcwd())
from models.metaquery import MetaQuery
import torch

model_path = "/data/yangyi/hf_models_upload/libero_long_ablation_unfrz_bckbn/whole_models/epoch25_35_step41500"
metaquery = MetaQuery.from_pretrained(
    model_path,
)
metaquery.load_state_dict(torch.load("/data/yangyi/hf_models_upload/libero_long_ablation_unfrz_bckbn/models/epoch59_56_step97500/model.pt"))
# del metaquery.model.mllm_backbone.lm_head.weight
metaquery.model.mllm_backbone.save_pretrained("/data/yangyi/hf_models_upload/libero_long_main_table/whole_models/epoch59_56_step97500")