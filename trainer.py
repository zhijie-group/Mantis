# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import datasets
import numpy as np
import torch
import wandb
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_datasets_available, TrainerCallback
from transformers.trainer import (
    tpu_spmd_dataloader,
    time,
    speed_metrics,
    math,
    TRAINER_STATE_NAME,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
from transformers.utils import is_torch_xla_available
import gc
from tabulate import tabulate

from PIL import Image
from qwen_vl_utils import process_vision_info


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


class MetaQueryTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    @staticmethod
    def metaquery_eval_data_collator(features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [f[k] for f in features]

        return batch

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if (
            hasattr(self, "_eval_dataloader")
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.metaquery_eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        elif isinstance(eval_dataset, torch.utils.data.Dataset):
            pass
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": False,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def log_images(self, logs: Dict[str, float]) -> None:
        logs["step"] = self.state.global_step
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        
        # inputs['input_images'] = [
        #     [
        #         img.to(torch.float32) if isinstance(img, torch.Tensor) else img
        #         for img in img_list
        #     ]
        #     for img_list in inputs['input_images']
        # ]

        inputs['input_images'] = [
            [
                img.to(torch.float32) if isinstance(img, torch.Tensor) else img
                for img in (img_list if isinstance(img_list, list) else [img_list])
            ]
            for img_list in inputs['input_images']
        ]

        sample_kwargs = {
            "guidance_scale": 3.0,
            "image_guidance_scale": 1.5,
            "num_inference_steps": 30,
            "return_tensor": True,
            "negative_prompt": "",
        }

        with torch.no_grad():
            samples = model.sample_images(**inputs, **sample_kwargs)
        samples = self._nested_gather(samples)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        self.log_images({"images": [wandb.Image(image) for image in samples]})
        del samples
        gc.collect()
        
        ############################## Language Test ##############################
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/data/yangyi/datasets/language_finetune/images/coco/000000.jpg"
                    },
                    {
                        "type": "text", 
                        "text": "Analyze the image in a comprehensive and detailed manner."
                    },
                ],
            },
        ]
        prompts = model.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = model.model.tokenizer(
            text=prompts,
            padding=True,
            return_tensors="pt",
            images=image_inputs,
            videos=video_inputs,
        ).to("cuda")

        generated_ids = model.model.mllm_backbone.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text1 = model.model.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        
        
        
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/image.png"
                    },
                    {
                        "type": "text", 
                        "text": "Discribe the image."
                    },
                ],
            },
        ]
        prompts = model.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = model.model.tokenizer(
            text=prompts,
            padding=True,
            return_tensors="pt",
            images=image_inputs,
            videos=video_inputs,
        ).to("cuda")

        generated_ids = model.model.mllm_backbone.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text2 = model.model.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        print("@" * 50)
        print("First Output:")
        print(output_text1)
        # Amidst a backdrop of urban structures, a large boat occupies the central portion of the waterway, with three individuals on board. One of them wears a distinct hat, and they are accompanied by a dog. Nearby, another smaller boat is docked further away. Dominating the scene are multiple signboards, positioned higher up, likely bearing directions or names. To power the boat, a motor is evident at the back. The presence of another hat indicates that at least two individuals on the main boat have chosen to protect themselves from the elements.
        print(output_text2)
        print("@" * 50)

        gc.collect()
        ############################## Language Test ##############################

        return (None, None, None)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        total_loss = outputs["loss"]
        
        if not hasattr(self, "accumulated_loss_dict"):
            self.accumulated_loss_dict = {}

        for key, value in outputs.items():
            if key != "loss" and value is not None:
                if key not in self.accumulated_loss_dict:
                    self.accumulated_loss_dict[key] = value.detach().clone()
                else:
                    self.accumulated_loss_dict[key] += value.detach()
        
        return (total_loss, outputs) if return_outputs else total_loss


    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        decay = 0.99
        # detect loss spike
        if not hasattr(self, "running_loss"):
            self.running_loss = tr_loss.item()
        self.running_loss = decay * self.running_loss + (1 - decay) * tr_loss.item()

        # if tr_loss.item() > 20 * self.running_loss:
        #     print(f"Loss Spiked: {tr_loss.item()} > 2 * {self.running_loss}")
        #     self.control.should_training_stop = True

        # detect grad norm spike
        if not hasattr(self, "running_grad_norm"):
            self.running_grad_norm = grad_norm
        self.running_grad_norm = (
            decay * self.running_grad_norm + (1 - decay) * grad_norm
        )

        # if grad_norm > 25 * self.running_grad_norm and grad_norm > 1:
        #     print(f"Grad Norm Spiked: {grad_norm} > 10 * {self.running_grad_norm}")
        #     self.control.should_training_stop = True

        if np.isnan(grad_norm) or grad_norm > 1e6:
            print(
                f"NaN grad norm detected in process {self.args.process_index} on {os.uname().nodename}"
            )
            self.control.should_training_stop = True
            print(f"Shut Down Training")

        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            
            if hasattr(self, "accumulated_loss_dict") and self.accumulated_loss_dict:
                for loss_name, loss_value in self.accumulated_loss_dict.items():
                    loss_scalar = self._nested_gather(loss_value).mean().item()
                    avg_loss = round(
                        loss_scalar 
                        / (self.state.global_step - self._globalstep_last_logged),
                        4,
                    )
                    logs[f"train/{loss_name}"] = avg_loss
                    self.accumulated_loss_dict[loss_name] -= self.accumulated_loss_dict[loss_name]

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


class MetaQueryCallback(TrainerCallback):
    @staticmethod
    def print_params(model, modules_to_stat):
        for module in modules_to_stat:
            module_parts = module.split(".")
            curr_module = model
            for part in module_parts:
                curr_module = getattr(curr_module, part)
            num_params = sum(p.numel() for p in curr_module.parameters()) / 1e6
            print(f"{module} num of params: {num_params:.2f}M")

    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            stat = []
            for i, (n, p) in enumerate(model.named_parameters()):
                stat.append([i, n, p.shape, p.dtype, p.requires_grad])
            print(
                tabulate(stat, headers=["idx", "name", "shape", "dtype", "trainable"])
            )

            if hasattr(model.model, "connector") and not isinstance(
                model.model.connector, nn.Identity
            ):
                print(f"connector_in_dim = {model.model.connector_in_dim}")
                print(f"connector_out_dim = {model.model.connector_out_dim}")
                self.print_params(model, ["model.connector"])

            self.print_params(model, ["model.transformer"])

class SaveCallback(TrainerCallback):
    def on_save(self, args, state, control, model, **kwargs):
        if not args.save_part_checkpoints:
            return

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            step = state.global_step
            epoch = f"{state.epoch:.2f}".replace(".", "_")

            # if "image" in model.config.training_mode:
            #     checkpoints_dir = os.path.join(args.base_dir, args.save_dir, "image_head", f"epoch{epoch}_step{step}")
            #     os.makedirs(checkpoints_dir, exist_ok=True)
            #     torch.save(
            #         model.model.mllm_backbone.model.embed_tokens.weight.data.cpu(),
            #         os.path.join(checkpoints_dir, f"embed_tokens_weight.pt")
            #     )
            #     torch.save(
            #         model.model.transformer.state_dict(),
            #         os.path.join(checkpoints_dir, f"image_head.pt")
            #     )
            #     torch.save(
            #         model.model.connector.state_dict(),
            #         os.path.join(checkpoints_dir, f"connector.pt")
            #     )

            # if "action" in model.config.training_mode:
            #     checkpoints_dir = os.path.join(args.base_dir, args.save_dir, "action_head", f"epoch{epoch}_step{step}")
            #     os.makedirs(checkpoints_dir, exist_ok=True)
            #     torch.save(
            #         model.model.mllm_backbone.model.embed_tokens.weight.data.cpu(),
            #         os.path.join(checkpoints_dir, f"embed_tokens_weight.pt")
            #     )
            #     torch.save(
            #         model.model.policy_head.state_dict(),
            #         os.path.join(checkpoints_dir, f"policy_head.pt")
            #     )

            checkpoints_dir = os.path.join(args.base_dir, args.save_dir, "whole_model", f"epoch{epoch}_step{step}")
            os.makedirs(checkpoints_dir, exist_ok=True)        
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, f"model.pt")
            )
        
        torch.distributed.barrier()