# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import Image
import PIL
import io
import torch
from torchvision.transforms import v2
from functools import partial

import numpy as np
import json
import os
import random
from datasets import Features, Image, Value
from datasets import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset


def _make_transform(size):
    return v2.Compose([v2.Resize(size), v2.CenterCrop(size)])


class LeRobotTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dataset,
        target_transform,
        primary_image_size,
        wrist_image_size,
        primary_image_key,
        wrist_image_key,
        task_map,
        norm_stats,
        model_args,
        language_dataset=None,
    ):
        self.dataset = base_dataset
        self.target_transform = target_transform
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        self.task_map = task_map
        self.norm_stats = norm_stats
        self.model_args = model_args
        self.language_dataset = language_dataset
        self.primary_image_transform = _make_transform(primary_image_size)
        self.wrist_image_transform = _make_transform(wrist_image_size)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.model_args.training_mode in ["action", "image_action", "image_action_language"]:
            gap = self.model_args.future_action_window_size

        elif self.model_args.training_mode in ["image"]:
            gap = random.randint(1, self.model_args.max_timestep_gap)

        item = self.dataset[idx]
        result = {
            'caption': self.task_map[item['task_index'].item()] if self.task_map else item['task'],
            'gap': gap
        }

        # target_images = item[self.primary_image_key][gap]
        target_images = item[self.primary_image_key][gap] if len(item[self.primary_image_key]) > 2 else item[self.primary_image_key][1]
        result["target"] = (
            self.target_transform(target_images) if target_images is not None else None
        )
        
        action_max = torch.tensor(self.norm_stats["action"]["max"])
        action_min = torch.tensor(self.norm_stats["action"]["min"])
        
        if item['action'].shape[1] > self.model_args.action_dim:
            result["actions"] = 2 * (item['action'][:gap, -self.model_args.action_dim:] - action_min) / (action_max - action_min) - 1
        else:
            result["actions"] = 2 * (item['action'][:gap] - action_min) / (action_max - action_min) - 1

        input_imgs = [
            (item[self.primary_image_key][0], self.primary_image_transform),
            (item[self.wrist_image_key][0], self.wrist_image_transform)
        ]
        null_mask = torch.rand(len(input_imgs)) <= 0.1
        result["input_images"] = [
            transform(torch.zeros_like(img) if mask and img is not None else img) 
            if img is not None else None
            for (img, transform), mask in zip(input_imgs, null_mask)
        ]
        
        if self.language_dataset is not None:
            # result["language_data"] = self.language_dataset[idx % len(self.language_dataset)]
            random_idx = random.randint(0, len(self.language_dataset) - 1)
            result["language_data"] = self.language_dataset[random_idx]

        return result

class LeRobotEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dataset,
        primary_image_size, 
        wrist_image_size,
        primary_image_key,
        wrist_image_key,
        task_map,
        model_args,
    ):
        self.dataset = base_dataset
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        self.task_map = task_map
        self.model_args = model_args
        self.primary_image_transform = _make_transform(primary_image_size)
        self.wrist_image_transform = _make_transform(wrist_image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.model_args.training_mode in ["action", "image_action", "image_action_language"]:
            gap = self.model_args.future_action_window_size

        elif self.model_args.training_mode in ["image"]:
            gap = random.randint(1, self.model_args.max_timestep_gap)

        item = self.dataset[idx]
        result = {
            'caption': self.task_map[item['task_index'].item()] if self.task_map else item['task'],
            'gap': gap
        }

        # result["target"] = item[self.primary_image_key][gap]
        result["target"] = item[self.primary_image_key][gap] if len(item[self.primary_image_key]) > 2 else item[self.primary_image_key][1]

        input_imgs = [
            item[self.primary_image_key][0], 
            item[self.wrist_image_key][0]
        ]
        result["input_images"] = [
            (self.primary_image_transform(img) if i == 0 else self.wrist_image_transform(img))
            if img is not None else None
            for i, img in enumerate(input_imgs)
        ]
        return result


def _load_image(item):
    src = io.BytesIO(item["bytes"]) if item["bytes"] is not None else item["path"]
    return PIL.Image.open(src).convert("RGB")


def _delete_keys_except(batch, except_keys):
    keys_to_delete = [key for key in list(batch.keys()) if key not in except_keys]
    for key in keys_to_delete:
        del batch[key]
    return batch


def _editing_process_fn(batch, target_transform, target_image_size):
    input_image_transform = _make_transform(target_image_size)
    input_images = [_load_image(img) for img in batch["source_image"]]
    target_images = [_load_image(img) for img in batch["target_image"]]

    batch["target"] = [target_transform(img) if img is not None else None for img in target_images]

    rand_probs = torch.rand((len(target_images), 1))
    null_image_mask = rand_probs <= 0.1
    input_images = [
        PIL.Image.new("RGB", (img.width, img.height)) if null_image_mask[i] else img
        for i, img in enumerate(input_images)
    ]

    batch["input_images"] = [
        input_image_transform(img) if img is not None else None
        for img in input_images
    ]

    _delete_keys_except(batch, ["target", "input_images", "caption", "gap", "actions"])
    return batch


def _editing_eval_process_fn(batch, target_image_size):
    target_image_transform = _make_transform(target_image_size)
    batch["input_images"] = [
        target_image_transform(image) if image is not None else None 
        for image in batch["source_image"]
    ]
    
    _delete_keys_except(batch, ["input_images", "caption", "gap"])
    return batch


def _collate_fn(batch, tokenize_func, tokenizer, training_mode):
    batch = [example for example in batch if example["target"] is not None]
    input_images = [example.get("input_images") for example in batch]
    targets = torch.stack([example["target"] for example in batch])

    if "action" in training_mode:
        actions = torch.stack([
            example["actions"] if isinstance(example["actions"], torch.Tensor) 
            else torch.tensor(example["actions"])
            for example in batch
        ])
        return_dict = {
            "target": targets,
            "source": input_images,
            "actions": actions,
        }
    elif "image" in training_mode:
        return_dict = {
            "target": targets,
            "source": input_images
        }

    captions = [example["caption"] for example in batch]
    gaps = [example["gap"] for example in batch]

    language_data = [example.get("language_data") for example in batch] if "language" in training_mode else None

    if any(imgs is not None for imgs in input_images):
        (
            return_dict["input_ids"],
            return_dict["attention_mask"],
            return_dict["pixel_values"],
            return_dict["image_sizes"],
            return_dict["language_data"],
        ) = tokenize_func(tokenizer, captions, gaps, input_images, language_data=language_data, training_mode=training_mode)
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
            tokenizer, captions, gaps, training_mode=training_mode
        )
    return return_dict

def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):
    target_transform = v2.Compose([
        v2.Resize(data_args.target_image_size),
        v2.CenterCrop(data_args.target_image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])
    ground_truth_transform = v2.Compose([
        v2.Resize(data_args.target_image_size),
        v2.CenterCrop(data_args.target_image_size),
    ])

    collate_fn = partial(
        _collate_fn, 
        tokenize_func=tokenize_func, 
        tokenizer=tokenizer, 
        training_mode=model_args.training_mode,
    )
    
    if "ssv2" in data_args.train_datasets:
        train_dataset = load_ssv2_dataset(data_args, model_args, training_args)
        eval_dataset = train_dataset.select(range(training_args.world_size))

        editing_process_fn = partial(
            _editing_process_fn,
            target_transform=target_transform,
            target_image_size=data_args.target_image_size,
        )
        editing_eval_process_fn = partial(
            _editing_eval_process_fn,
            target_image_size=data_args.target_image_size,
        )

        train_dataset = train_dataset.cast_column("source_image", Image(decode=False))
        train_dataset = train_dataset.cast_column("target_image", Image(decode=False))
        train_dataset.set_transform(editing_process_fn)
        train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
        
        eval_dataset = eval_dataset.cast_column("source_image", Image(decode=True))
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))

        gt_images = [ground_truth_transform(img.convert("RGB")) for img in eval_dataset["target_image"]]
        src_images = [ground_truth_transform(img.convert("RGB")) for img in eval_dataset["source_image"]]
        eval_dataset.set_transform(editing_eval_process_fn)

    elif "droid" in data_args.train_datasets:
        train_dataset, task_map, norm_stats, primary_image_key, wrist_image_key = load_droid_dataset(data_args, model_args, training_args)
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )

        language_dataset = load_language_dataset(data_args, training_args) if "language" in model_args.training_mode else None

        train_dataset = LeRobotTrainDataset(
            base_dataset=train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=task_map,
            norm_stats=norm_stats,
            model_args=model_args,
            language_dataset=language_dataset
        )
        eval_dataset = LeRobotEvalDataset(
            base_dataset=eval_dataset,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=task_map,
            model_args=model_args,
        )

        src_images = [ground_truth_transform(sample["input_images"][0]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target"]) for sample in eval_dataset]
    
    elif "libero" in data_args.train_datasets:
        train_dataset, task_map, norm_stats, primary_image_key, wrist_image_key = load_libero_dataset(data_args, model_args, training_args)
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )

        train_dataset = LeRobotTrainDataset(
            base_dataset=train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=task_map,
            norm_stats=norm_stats,
            model_args=model_args,
        )
        eval_dataset = LeRobotEvalDataset(
            base_dataset=eval_dataset,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=task_map,
            model_args=model_args,
        )

        src_images = [ground_truth_transform(sample["input_images"][0]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target"]) for sample in eval_dataset]


    elif "aloha" in data_args.train_datasets:
        train_dataset, norm_stats, primary_image_key, wrist_image_key = load_aloha_dataset(data_args, model_args, training_args)
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )
        
        language_dataset = load_language_dataset(data_args, training_args) if "language" in model_args.training_mode else None

        train_dataset = LeRobotTrainDataset(
            base_dataset=train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=None,
            norm_stats=norm_stats,
            model_args=model_args,
            language_dataset=language_dataset
        )
        eval_dataset = LeRobotEvalDataset(
            base_dataset=eval_dataset,
            primary_image_size=data_args.target_image_size,
            wrist_image_size=data_args.wrist_image_size,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            task_map=None,
            model_args=model_args,
        )

        src_images = [ground_truth_transform(sample["input_images"][0]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target"]) for sample in eval_dataset]


    return train_dataset, eval_dataset, gt_images, src_images, collate_fn


def load_ssv2_dataset(data_args, model_args, training_args):
    src_paths, tgt_paths, captions, gaps = [], [], [], []

    train_json_path = os.path.join(data_args.dataset_root_dir, "labels", "train.json")
    val_json_path = os.path.join(data_args.dataset_root_dir, "labels", "validation.json")
    
    extracted_frames_dir = os.path.join(data_args.dataset_root_dir, "frames")
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    all_data = train_data + val_data
    id_to_instruction = {item["id"]: item["label"] for item in all_data}

    for video_id, instruction in id_to_instruction.items():
        video_dir = os.path.join(extracted_frames_dir, video_id)

        img_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        selected_imgs = img_files
        for timestep_gap in range(1, model_args.max_timestep_gap + 1):
            caption = f"Instruction: {instruction}. Generate the updated image observation after {timestep_gap} timesteps."
            for i in range(len(selected_imgs)):
                tgt_index = i + timestep_gap
                if tgt_index >= len(selected_imgs):
                    break
                src_paths.append(os.path.join(video_dir, selected_imgs[i]))
                tgt_paths.append(os.path.join(video_dir, selected_imgs[tgt_index]))
                captions.append(caption)
                gaps.append(timestep_gap)

    data_dict = {
        "source_image": src_paths, 
        "target_image": tgt_paths,
        "caption": captions, 
        "gap": gaps
    }
    features = Features({
        "source_image": Image(),
        "target_image": Image(),
        "caption": Value("string"),
        "gap": Value("int32"),
    })
    return Dataset.from_dict(data_dict, features=features).shuffle(seed=training_args.data_seed)


def load_language_dataset(data_args, training_args):
    language_dataset_dir = data_args.language_dataset_dir
    language_dataset = []
    
    json_dir = os.path.join(language_dataset_dir, "json_files")
    images_dir = os.path.join(language_dataset_dir, 'images')
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
        
        json_name = os.path.splitext(json_file)[0]
        
        # if json_name not in "allava":
        if json_name not in "gpt4o":
            continue
    
        json_path = os.path.join(json_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data:
            messages = item.get('messages')
            images = item.get('images')

            images = [os.path.join(images_dir, json_name, img) for img in images] if images else None

            language_dataset.append({
                'messages': messages,
                'images': images
            })
    random.seed(training_args.data_seed)
    random.shuffle(language_dataset)
    return language_dataset


def load_droid_dataset(data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(data_args.dataset_root_dir)
    primary_image_key, wrist_image_key = ds_meta.camera_keys[1], ds_meta.camera_keys[0]
    
    tasks_path = os.path.join(data_args.dataset_root_dir, "meta", "tasks.jsonl")
    with open(tasks_path, 'r', encoding='utf-8') as f:
        task_map = {data["task_index"]: data["task"] for data in map(json.loads, f)}
    
    episodes_success_path = os.path.join(data_args.dataset_root_dir, "meta", "episodes_success.json")
    with open(episodes_success_path, 'r') as f:
        episodes_success = json.load(f)
    
    norm_stats_path = os.path.join(data_args.dataset_root_dir, "meta", "stats.json")
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    delta_timestamps = {
        primary_image_key: [t / ds_meta.fps for t in range(model_args.max_timestep_gap + 1)],
        wrist_image_key: [t / ds_meta.fps for t in range(model_args.max_timestep_gap + 1)],
        "action": [t / ds_meta.fps for t in range(model_args.max_timestep_gap)],
    }
    dataset = LeRobotDataset(
        data_args.dataset_root_dir,
        episodes=episodes_success['success_indices'],
        delta_timestamps=delta_timestamps
    )

    return dataset, task_map, norm_stats, primary_image_key, wrist_image_key


def load_libero_dataset(data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(data_args.dataset_root_dir)
    primary_image_key, wrist_image_key = ds_meta.camera_keys[0], ds_meta.camera_keys[1]
    
    tasks_path = os.path.join(data_args.dataset_root_dir, "meta", "tasks.jsonl")
    with open(tasks_path, 'r', encoding='utf-8') as f:
        task_map = {data["task_index"]: data["task"] for data in map(json.loads, f)}

    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[data_args.unnorm_key]

    delta_timestamps = {
        primary_image_key: [0, model_args.future_action_window_size / ds_meta.fps],
        wrist_image_key: [0, model_args.future_action_window_size / ds_meta.fps],
        "action": [t / ds_meta.fps for t in range(model_args.future_action_window_size)],
    }
    
    if "libero_10" in data_args.dataset_root_dir:
        dataset = LeRobotDataset(
            data_args.dataset_root_dir,
            episodes=[idx for idx in range(391) if idx != 210],
            delta_timestamps=delta_timestamps
        )
    else:
        dataset = LeRobotDataset(
            data_args.dataset_root_dir,
            delta_timestamps=delta_timestamps
        )

    return dataset, task_map, norm_stats, primary_image_key, wrist_image_key


def load_aloha_dataset(data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(os.path.join(data_args.dataset_root_dir, '7'))
    primary_image_key, wrist_image_key = ds_meta.camera_keys[0], ds_meta.camera_keys[2]
    
    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[data_args.unnorm_key]

    delta_timestamps = {
        primary_image_key: [0, model_args.future_action_window_size / ds_meta.fps],
        wrist_image_key: [0, model_args.future_action_window_size / ds_meta.fps],
        "action": [t / ds_meta.fps for t in range(model_args.future_action_window_size)],
    }
    
    dataset = MultiLeRobotDataset(
        repo_ids = [
            os.path.join(data_args.dataset_root_dir, item) 
                for item in os.listdir(data_args.dataset_root_dir) 
            if os.path.isdir(os.path.join(data_args.dataset_root_dir, item))
        ],
        delta_timestamps=delta_timestamps
    )

    return dataset, norm_stats, primary_image_key, wrist_image_key