import os
import sys
sys.path.append(os.getcwd())
from models.metaquery import MetaQuery
import json
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


model_path = "/data/yangyi/hf_models_upload/droid_image_action_language/whole_models/epoch2_30"
metaquery = MetaQuery.from_pretrained(
    model_path,
)

model = metaquery.model.mllm_backbone
model = model.to("cuda")
processor = metaquery.model.tokenizer


# model_path = "/data/yangyi/.cache/models/Qwen2.5-VL-3B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
# )
# model = model.to("cuda")
# processor = AutoProcessor.from_pretrained(model_path)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": os.path.join("/data/yangyi/datasets/language_eval/realworldqa_json", item['image'])
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
print(item['id'], output_text)