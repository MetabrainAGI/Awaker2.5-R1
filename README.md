# Awaker2.5-R1

**Awaker2.5-R1** is a multimodal large model developed by [Metabrain AGI](https://www.metabrainagi.com)， equipped with complex reasoning and long-chain thinking capabilities through GRPO.

## News
- Leveraging the Qwen2.5-VL-7B as the base model, we implemented the GPRO algorithm with rule-based reward, enabling stable training for multimodal large models.
- By training on a 50k mathematical geometry reasoning dataset, we enhanced the mathematical reasoning capabilities of Qwen2.5-VL-7B, achieving performance on MathVista that approaches OpenAI's o1 model. 

## Performance
- We conducted evaluations on three commonly used mathematical benchmarks, namely MathVista (Testmini), MathVerse (Testmini & Vision_Only), and WeMath. The scores of other models are sourced from [OpenCompass](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME).

| Models               | Parameters |   Mathvista | Mathverse | WeMath | Avg.|
| ------------------- | :--------: | :------: | :--------: | :-------: |  :-------: |
|OpenAI o1| - |73.9 |-|-|-|
|Doubao-1.5-Pro	| - |78.6	|64.7	|65.7	|69.6|
|Ovis2-34B	| 34.9B|76.1	|50.1	|51.9	|59.3|
|Gemini-1.5-Pro-002	| - |67.9	|54.8	|50.5	|57.7|
|Qwen2.5-VL-72B	| 73.4B |74.2	|47.3	|49.1	|56.8|
|Gemini-2.0-Flash	| - |70.4	|47.8	|47.4|	55.2|
|Ovis2-16B	| 16.2B |73.7	|45.8	|45.0	|54.8|
|Claude 3.7 Sonnet 20250219	| - |66.8	|46.7	|49.3	|54.2| 
|Step-1o	| - |74.7	|42.0	|45.3	|54.0|
|GLM-4v-Plus-20250111	| - |73.5	|40.7	|47.7	|53.9|
|InternVL2.5-78B-MPO	|78B |76.6	|43.7	|37.6	|52.6|
|QVQ-72B-Preview	| 73.4B |70.3	|48.2	|39.0	|52.5|
|Claude 3.7 Sonnet 20241022	| - |65.3	|46.3	|44.0	|51.8|
|**Awaker2.5-R1 (ours)** | 8.29B | 73.9 | 43.1 | 37.6 | 51.5|
|InternVL2.5-38B-MPO	| 38B |73.6	|37.9	|40.1	|50.5|
|InternVL2.5-38B	| 38B |72.4	|35.7	|42.7	|50.2|
|InternVL2.5-78B	| 78B |70.6	|39.2	|39.8	|49.8|
|GPT-4o-20241120	| - |60.0	|40.6	|45.8	|48.8|
|SenseNova	| - |78.4	|35.7	|31.7	|48.6|
|Qwen2.5-VL-7B	| 8.29B |68.1	|41.1	|36.2	|48.4|
|Qwen2-VL-72B	| 73.4B |69.7	|36.3	|36.0	|47.3|
|Ovis2-8B	| 8.94B |71.8	|42.3	|27.2	|47.1|
|Ovis1.6-Gemma2-27B	| 28.9B | 70.2	|37.8	|32.1	|46.7|
|MiniCPM-o-2.6	| 8.67B | 73.3	|35.0	|25.2	|44.5 |
|grok-2-vision-1212	| - |66.6	|27.9	|38.1	|44.2 |
|LLaVA-OneVision-72B (SI)	| 73B |67.1	|27.2	|32.0	|42.1 |
|InternVL2-Llama3-76B	|76B |65.6	|23.7	|32.1	|40.5 |
|NVLM-D-72B	| 79.5B |64.0	|25.5	|28.4	|39.3 |




## Quick Start

- **Model Weights**
  
  Download model weigths of [Awaker2.5-R1](https://huggingface.co/MetabrainAGI/Awaker2.5-R1)

- **Inference Code**
  
  Here we present a code snippet to show how to use the chat model:

```bash
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# Load the base Qwen2-VL model
model_path = "/path/to/awaker-r1"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


max_pixels = 5120*28*28
processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)

messages = [
    {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    },
                ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text", 
                "text": "Describe this image."
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## Training Process
![Accuracy Reward](https://github.com/MetabrainAGI/Awaker2.5-R1/blob/main/figs/reward_accuracy_49k.png) 
![Format Reward](https://github.com/MetabrainAGI/Awaker2.5-R1/blob/main/figs/reward_format_49k.png)
![Reward Mean](https://github.com/MetabrainAGI/Awaker2.5-R1/blob/main/figs/reward_mean_49k.png)
![Reward Std](https://github.com/MetabrainAGI/Awaker2.5-R1/blob/main/figs/reward_std_49k.png)
![Completion Length](https://github.com/MetabrainAGI/Awaker2.5-R1/blob/main/figs/reward_cl_49k.png)



