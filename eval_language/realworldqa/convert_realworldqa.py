import os
import json
import pandas as pd
from io import BytesIO
from PIL import Image

def process_parquet(input_dir, output_dir):
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    all_data = []
    item_id = 0
    for fname in os.listdir(input_dir):
        if not fname.endswith('.parquet'):
            continue
        
        df = pd.read_parquet(os.path.join(input_dir, fname))
        
        for _, row in df.iterrows():
            img = Image.open(BytesIO(row['image']['bytes']))
            ext = 'jpg' if img.format in ['JPEG', 'JPG'] else 'png'
            img_name = f"{item_id}.{ext}"
            img_path = os.path.join(img_dir, img_name)
            
            img.save(img_path)

            all_data.append({
                'id': item_id,
                'question': str(row['question']),
                'answer': str(row['answer']),
                'image': f'images/{img_name}'
            })
            
            item_id += 1

    with open(os.path.join(output_dir, 'realworldvqa.json'), 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f'Processing completed. Total {len(all_data)} records saved to {output_dir}')

if __name__ == '__main__':
    
    input_dir = 'your_parquet_data_dir'
    output_dir = 'your_output_dir'
    
    process_parquet(
        input_dir=input_dir,
        output_dir=output_dir
    )