import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

your_dataset_dir = "your_dataset_dir"
dataset_name = "wikipedia_2m"

PARQUET_DIR = f"{your_dataset_dir}/LLaVA-OneVision-1.5-Instruct-Data/{dataset_name}"
OUTPUT_IMAGE_DIR = f"{your_dataset_dir}/language_finetune/images/{dataset_name}"
OUTPUT_JSON_FILE = f"{your_dataset_dir}/language_finetune/json_files/{dataset_name}.json"


os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
merged_data = []

for filename in tqdm(sorted(f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet'))):
    df = pd.read_parquet(os.path.join(PARQUET_DIR, filename), columns=['id', 'conversations', 'image'])
    for _, row in df.iterrows():

        messages = []
        for msg in row["conversations"].tolist():
            role = "user" if msg.get("from") == "human" or msg.get("role") == "user" else "assistant"

            text = msg.get("value") or msg.get("content")
            messages.append({
                "content": text, 
                "role": role
            })
        
        data = {"id": row['id'], "messages": messages}
        if row['image'] is not None:
            img = Image.open(BytesIO(row['image']['bytes']))
            ext = 'jpg' if img.format in ['JPEG', 'JPG'] else 'png'
            img_name = f"{row['id']}.{ext}"
            img_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            img.save(img_path)
            data["images"] = [img_name]
        merged_data.append(data)

with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)