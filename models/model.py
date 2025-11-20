import math
from typing import List

import torch
from torch import nn

import gc

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
)

from diffusers.models.normalization import RMSNorm
from diffusers import SanaTransformer2DModel

from models.transformer_encoder import Qwen2Encoder
from .action_model.action_model import ActionModel

import re
from qwen_vl_utils import process_vision_info
from PIL import Image


class MLLMInContextConfig(PretrainedConfig):
    def __init__(
        self,
        mllm_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 256,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 12,
        system_prompt: str = "You will be provided with an image observation and a corresponding instruction.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt

        self.max_timestep_gap = kwargs.get("max_timestep_gap")
        self.num_gapqueries = kwargs.get("num_gapqueries")
        self.action_model_type = kwargs.get("action_model_type")
        self.action_dim = kwargs.get("action_dim")
        self.future_action_window_size = kwargs.get("future_action_window_size")
        self.past_action_window_size = kwargs.get("past_action_window_size")
        self.num_actqueries = kwargs.get("num_actqueries")
        self.training_mode = kwargs.get("training_mode")

class MLLMInContext(PreTrainedModel):
    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config

        self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.mllm_id, attn_implementation="sdpa"
        )
        self.mllm_backbone.model.config.use_sliding_window = False
        self.mllm_backbone.model.config.sliding_window = None
        self.num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings

        new_vocab_size = (
            self.num_embeddings
            + config.num_metaqueries + 2
            + config.num_actqueries + 2
            + config.max_timestep_gap * config.num_gapqueries + 2
        )
        
        try:
            self.mllm_backbone.resize_token_embeddings(new_vocab_size)
        except:
            self.mllm_backbone.resize_token_embeddings(new_vocab_size, mean_resizing=False)
        
        def freeze_hook_image(grad):
            grad[: self.num_embeddings].zero_()
            grad[-(config.num_actqueries + 2) :].zero_()  
            # grad[-(config.num_gapqueries * config.max_timestep_gap + 2 + config.num_actqueries + 2) :].zero_()            
            return grad

        def freeze_hook_action(grad):
            grad[: self.num_embeddings + config.num_metaqueries + 2
                 + config.num_gapqueries * config.max_timestep_gap + 2].zero_()         
            return grad
        
        def freeze_hook_image_action(grad):
            grad[: self.num_embeddings].zero_()
            grad[
                self.num_embeddings + config.num_metaqueries + 2 : 
                self.num_embeddings + config.num_metaqueries + 2 + config.num_gapqueries * config.max_timestep_gap + 2
            ].zero_()      
            return grad
        
        def freeze_hook_image_action_language(grad):
            grad[
                self.num_embeddings + config.num_metaqueries + 2 : 
                self.num_embeddings + config.num_metaqueries + 2 + config.num_gapqueries * config.max_timestep_gap + 2
            ].zero_()
            return grad
        
        if config.training_mode == "image":
            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook_image)
        elif config.training_mode == "action":
            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook_action)
        elif config.training_mode == "image_action":
            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook_image_action)
        elif config.training_mode == "image_action_language":
            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook_image_action_language)

        self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
        # self.mllm_backbone.lm_head = nn.Identity()

        # self.tokenizer = AutoProcessor.from_pretrained(
        #     config.mllm_id, min_pixels=224 * 224, max_pixels=1280 * 28 * 28
        # )
        self.tokenizer = AutoProcessor.from_pretrained(
            config.mllm_id, min_pixels=224 * 224, max_pixels=960 * 24 * 24
        )
        self.tokenizer.tokenizer.padding_side = "left"
        self.tokenizer.resize_fn = None
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.num_actqueries = config.num_actqueries
        self.tokenizer.num_gapqueries = config.num_gapqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<pad_token_{i}>"
                    for i in range(self.num_embeddings - len(tokenizer))
                ]
            }
        )

        if config.num_metaqueries > 0:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        if config.num_gapqueries > 0:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_gap>", "<end_of_gap>"]
                    + [f"<gap{i}>" for i in range(self.tokenizer.num_gapqueries * config.max_timestep_gap)]
                }
            )
            self.bog_token_id = tokenizer.convert_tokens_to_ids("<begin_of_gap>")
            self.eog_token_id = tokenizer.convert_tokens_to_ids("<end_of_gap>")

        if config.num_actqueries > 0:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_act>", "<end_of_act>"]
                    + [f"<act{i}>" for i in range(self.tokenizer.num_actqueries)]
                }
            )
            self.boa_token_id = tokenizer.convert_tokens_to_ids("<begin_of_act>")
            self.eoa_token_id = tokenizer.convert_tokens_to_ids("<end_of_act>")

        self.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
        self.transformer = SanaTransformer2DModel.from_pretrained(
            config.diffusion_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        input_scale = math.sqrt(5.5)

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = (
            getattr(self.transformer.config, "caption_channels", None)
            or getattr(self.transformer.config, "encoder_hid_dim", None)
            or getattr(self.transformer.config, "cross_attention_dim", None)
        )

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=config.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        self.policy_head = ActionModel(
            model_type=config.action_model_type, 
            token_size=self.mllm_hidden_size,
            in_channels=config.action_dim, 
            future_action_window_size=config.future_action_window_size, 
            past_action_window_size=config.past_action_window_size,
            num_actqueries=config.num_actqueries,
        )

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable({"use_reentrant": False})
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()


    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, gaps, image=None, text_response=None, add_generation_prompt=True, language_data=None, training_mode="image"
    ):
        if not isinstance(caption, List):
            caption = [caption]
        if not isinstance(gaps, List):
            gaps = [gaps]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        gapsuffix = []
        for gap in gaps:
            gapsuffix.append(
                "\n<begin_of_gap>"
                + "".join([f"<gap{i}>" for i in range((gap - 1) * tokenizer.num_gapqueries, gap * tokenizer.num_gapqueries)])
                + "<end_of_gap>"
            )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        elif "action" in training_mode:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img>"
                + "<begin_of_act>"
                + "".join([f"<act{i}>" for i in range(tokenizer.num_actqueries)])
                + "<end_of_act><|im_end|>"
            )
        elif "image" in training_mode:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for cap in caption
        ]
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, img in enumerate(image):
                if img and not isinstance(img, list):
                    image[i] = [img]
            if tokenizer.resize_fn is not None:
                image = [
                    [tokenizer.resize_fn(sub_img) for sub_img in imgs] if imgs else None
                    for imgs in image
                ]

            ###### For Aloha ######
            # image = [
            #     [torch.clamp(sub_img, min=0.0, max=1.0) for sub_img in imgs] if imgs else None
            #     for imgs in image
            # ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"} for _ in imgs]
                            + [{"type": "text", "text": cap}]
                            if imgs
                            else [{"type": "text", "text": cap}]
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            kwargs = {"images": [imgs for imgs in image if imgs]}
        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]

        prompts = [p + g for p, g in zip(prompts, gapsuffix)]
        prompts = [p + suffix for p in prompts]

        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
            # do_rescale=True,
            **kwargs,
        )

        if "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)
        
        
        ##################################################################
        if language_data is not None:
            conversations = []

            for item in language_data:
                images = item.get("images")                
                messages = []
                
                for turn in item["messages"]:
                    role = turn.get("role")
                    text = turn.get("content")
                    
                    if role == "user":
                        content = []
                        
                        if not text:
                            content.append({"type": "text", "text": ""})
                        else:
                            for seg in re.split(r"(<image>)", text):
                                if seg == "<image>" and images:
                                    content.append({"type": "image", "image": images.pop(0)})
                                elif seg.strip():
                                    content.append({"type": "text", "text": seg.strip()})
                        messages.append({"role": role, "content": content})
                        
                    else:
                        if not text:
                            messages.append({"role": role, "content": [{"type": "text", "text": ""}]})
                        else:
                            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

                conversations.append(messages)

            prompts = [
                tokenizer.apply_chat_template(conv, tokenize=False)
                for conv in conversations
            ]
            
            image_inputs = [
                process_vision_info(conv)[0]
                for conv in conversations
            ]
            image_inputs = [img for img in image_inputs if img] or None
            
            language_data_inputs = tokenizer(
                text=prompts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            start_token = 77091
            end_token = 151645
            prefix_token = 151644
            suffix_token = 198
            IGNORE_INDEX = -100

            input_ids = language_data_inputs["input_ids"]
            labels = torch.full_like(input_ids, IGNORE_INDEX)

            batch_size, seq_len = input_ids.shape
            for batch_idx in range(batch_size):
                input_ids_1d = input_ids[batch_idx]

                start_positions = (input_ids_1d == start_token).nonzero(as_tuple=True)[0]

                if len(start_positions) == 0:
                    continue
                
                end_positions = (input_ids_1d == end_token).nonzero(as_tuple=True)[0]

                for start_pos in start_positions:

                    if start_pos < 1 or input_ids_1d[start_pos - 1] != prefix_token:
                        continue
                    if start_pos + 1 >= seq_len or input_ids_1d[start_pos + 1] != suffix_token:
                        continue

                    ans_start = start_pos + 2
                    
                    if ans_start >= seq_len:
                        continue

                    valid_ends = end_positions[end_positions >= ans_start]
                    
                    if len(valid_ends) > 0:
                        ans_end = valid_ends[0].item()
                        copy_end = min(ans_end + 2, seq_len)
                        labels[batch_idx, ans_start:copy_end] = input_ids[batch_idx, ans_start:copy_end]

            language_data_inputs["labels"] = labels
            
            # if "pixel_values" in language_data_inputs:
            #     language_data_inputs["pixel_values"] = language_data_inputs["pixel_values"].unsqueeze(0)
                
            text_inputs["language_data"] = language_data_inputs
            
            del conversations
            torch.cuda.empty_cache()
            gc.collect()
        
        else:
            text_inputs["language_data"] = None
        ##################################################################

        return text_inputs.values()


    def encode_condition(
        self, input_ids, attention_mask, mllm_output, **kwargs
    ):
        # prompt_embeds = mllm_output.logits
        prompt_embeds = mllm_output.hidden_states[-1]
        embeddings = mllm_output.hidden_states[0]

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]

            bog_pos = torch.where(input_ids == self.bog_token_id)[1]
            eog_pos = torch.where(input_ids == self.eog_token_id)[1]

            vision_start = torch.full((input_ids.size(0),), -1, dtype=torch.long, device=input_ids.device)
            rows, cols = torch.where(input_ids == self.vision_start_token_id)
            for r in rows.unique():
                vision_start[r] = cols[rows == r].min()
            
            vision_end = torch.full((input_ids.size(0),), -1, dtype=torch.long, device=input_ids.device)
            rows, cols = torch.where(input_ids == self.vision_end_token_id)
            for r in rows.unique():
                vision_end[r] = cols[rows == r].min()

            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )

            if boi_pos.numel() > 0:
                prompt_embeds_mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None]) | ((indices > bog_pos[:, None]) & (indices < eog_pos[:, None]))
                embeddings_mask = (indices > vision_start[:, None]) & (indices < vision_end[:, None])

                prompt_embeds = prompt_embeds[prompt_embeds_mask].view(
                    batch_size, -1, prompt_embeds.size(-1)
                )
                embeddings = embeddings[embeddings_mask].view(
                    batch_size, -1, embeddings.size(-1)
                )
                prompt_embeds = torch.cat([embeddings, prompt_embeds], dim=1)

                mask = embeddings_mask | prompt_embeds_mask
                attention_mask = attention_mask[mask].view(batch_size, -1)
            else:
                # Inference action without metaqueries
                mask = (indices > vision_start[:, None]) & (indices < vision_end[:, None]) | ((indices > bog_pos[:, None]) & (indices < eog_pos[:, None]))
                prompt_embeds = prompt_embeds[mask].view(
                    batch_size, -1, prompt_embeds.size(-1)
                )
                attention_mask = attention_mask[mask].view(batch_size, -1)

        return self.connector(prompt_embeds), attention_mask

    def encode_condition_action(
        self, input_ids, mllm_output, **kwargs
    ):
        prompt_embeds = mllm_output.hidden_states[-1]
        
        boa_pos = torch.where(input_ids == self.boa_token_id)[1]
        eoa_pos = torch.where(input_ids == self.eoa_token_id)[1]

        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
            batch_size, -1
        )
        mask = (indices > boa_pos[:, None]) & (indices < eoa_pos[:, None])
        mask = mask.to(prompt_embeds.device)

        prompt_embeds = prompt_embeds[mask].view(
            batch_size, -1, prompt_embeds.size(-1)
        )

        return prompt_embeds

    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None):
        model_pred = self.transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample
        return model_pred
