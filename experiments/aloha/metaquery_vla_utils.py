import torch
import numpy as np
import PIL.Image
from typing import Optional, Dict, Any
import json

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.metaquery import MetaQuery


class MetaQueryVLA:
    cache_dir = "/data/yangyi/.cache"

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: str,
        target_image_size: int = 512,
        vae_downsample_f: int = 32,
        device: Optional[str] = None,
    ):
        
        # --- Device Setup ---
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")

        if self.device.startswith("cuda"):
             if not torch.cuda.is_available():
                  raise RuntimeError(f"CUDA is not available, but device '{self.device}' was specified.")
             gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
             if gpu_id >= torch.cuda.device_count():
                  raise RuntimeError(f"CUDA device {gpu_id} not available. Only {torch.cuda.device_count()} devices found.")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {model_id}")
            input_size = target_image_size // vae_downsample_f

            self.model = MetaQuery.from_pretrained(
                model_id,
                input_size=input_size,
                ignore_mismatched_sizes=True,
            )

            ###### Loading Part Checkpoints ######
            if checkpoints_dir:
                print(f"Loading checkpoints from: {checkpoints_dir}")
                
                state_dict = torch.load(
                    f"{checkpoints_dir}/model.pt",
                    map_location='cpu'
                )
                self.model.load_state_dict(state_dict)
                
                # embed_tokens_weight = torch.load(f"{checkpoints_dir}/embed_tokens_weight.pt")
                # with torch.no_grad():
                #     self.model.model.mllm_backbone.model.embed_tokens.weight.copy_(embed_tokens_weight)

                # state_dict = torch.load(
                #     f"{checkpoints_dir}/policy_head.pt",
                #     map_location='cpu'
                # )
                # self.model.model.policy_head.load_state_dict(state_dict)
            ###### Loading Part Checkpoints ######

            total_params = 0
            for param in self.model.parameters():
                total_params += param.numel()

            if hasattr(self.model.model, 'transformer'):
                del self.model.model.transformer
            if hasattr(self.model.model, 'connector'):
                del self.model.model.connector
            if hasattr(self.model.model, 'vae'):
                del self.model.model.vae

            total_params = 0
            for param in self.model.parameters():
                total_params += param.numel()

            self.model.to("cuda")
            self.model.eval()

            # Download the norm_stats locally (only downloads once; cached)
            file_path = "/data/yangyi/metaquery_action_refactoring/configs/norm_stats.json"

            # Load the JSON file
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
            self.norm_stats = norm_stats

        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_id}: {e}")        

    @torch.inference_mode()
    def inference(
        self, 
        image: list, 
        instruction: str, 
        unnorm_key: str = None, 
        eval_mode: str = None, 
        relevant_tokens_threshold: float = 0.0,
    ) -> np.ndarray:
        image = [image]

        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_action, relation, num_patches = model.sample_actions(
                    caption=instruction,
                    input_images=image,
                    num_images_per_prompt=1,
                    gap=25,
                    eval_mode=eval_mode,
                )

        action_norm_stats = self.get_action_stats(unnorm_key)
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        # action_low[-1] = -1.
        # aloha?

        unnorm_actions = (
            0.5 * (output_action + 1) * (action_high - action_low)
            + action_low
        )
        # unnorm_actions[..., -1] = np.where(unnorm_actions[..., -1] >= 0.5, -1.0, 1.0)

        top_relation_indices = None
        if eval_mode in ["action_chunking_dynamic_temporal_agg"] and relation is not None:
            top_k = int(len(relation) * relevant_tokens_threshold)
            top_relation_indices = torch.topk(relation, top_k).indices.tolist()

        return unnorm_actions, top_relation_indices, num_patches

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


