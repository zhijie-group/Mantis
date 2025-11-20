import os
import json
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

log_dir = "/data/yangyi/metaquery_action_refactoring/eval_language/realworldqa/results"
data_dir = "/data/yangyi/datasets/language_eval/realworldqa"
model_path = "/data/yangyi/.cache/models/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

with open(os.path.join(data_dir, "realworldqa.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

os.makedirs(log_dir, exist_ok=True)

results = []

for item in data:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": os.path.join(data_dir, item['image'])
                },
                {
                    "type": "text", 
                    "text": item['question']
                },
            ],
        },
    ]

    prompts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=prompts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    result_item = {
        "id": item['id'],
        "output": output_text[0] if output_text else ""
    }
    results.append(result_item)
    
    print(f"Processed: {item['id']}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(log_dir, f"qwen_results_{timestamp}.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"Total processed: {len(results)} samples")
