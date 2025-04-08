"""
关键选择：
    量化：启用16/24GB GPU训练（兼容T4/A10）
    LoRA Rank 64：平衡性能与内存
    vLLM集成：生成速度提高50%
"""


import os
import re  
import torch
import wandb
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from datasets import load_dataset, Dataset, interleave_datasets, concatenate_datasets  
from trl import GRPOConfig, GRPOTrainer  
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

PatchFastRL("GRPO", FastLanguageModel)
wandb.login(key=os.getenv("WANDB_KEY"))

# 初始化wandb
run = wandb.init(
    project="medical-reasoning",  # 你的项目名称
    name="test",  # 你的实验名称
    tags=["test"],  # 你的实验标签
    # config={
    #     "model": "MyLargeModel",
    #     "batch_size": 64,
    #     "learning_rate": 0.001,
    #     # 其他模型和训练参数...
    # },
    # notes="Training run for my large model",
    group="reasoning-training",
    job_type="training",
)



# 超参
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower


# 模型下载与初始化
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,  # 随机状态
)



# 加载和准备数据集  
SYSTEM_PROMPT = """  
响应格式如下：  
<reasoning>  
...  
</reasoning>  
<answer>  
...  
</answer>  
"""  
  
XML_COT_FORMAT = """\  
<reasoning>  
{reasoning}  
</reasoning>  
<answer>  
{answer}  
</answer>  
"""  
  
def extract_xml_answer(text: str) -> str:  
    answer = text.split("<answer>")[-1]  
    answer = answer.split("</answer>")[0]  
    return answer.strip()  
  
def extract_hash_answer(text: str) -> str | None:  
    if "####" not in text:  
        return None  
    return text.split("####")[1].strip()  
  
# 取消注释中间消息以进行一次提示  
def get_datasets(split = "train") -> Dataset:  
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore  
    data = data.map(lambda x: { # type: ignore  
        'prompt': [  
            {'role': 'system', 'content': SYSTEM_PROMPT},  
            {'role': 'user', 'content': x['question']}  
        ],  
        'answer': extract_hash_answer(x['answer']),  
        'db_set':'gsm8k'  
    }) # type: ignore  
    data = data.remove_columns(['question'])  
      
    data_qa = load_dataset("qiaojin/PubMedQA", "pqa_artificial")[split] # 两倍于其他数据集  
    data_qa = data_qa.filter(lambda x: len("\n".join(x['context']['contexts'])) < 1024) # 避免长跟踪  
    data_qa = data_qa.map(lambda x: { # type: ignore  
        'prompt': [  
            {'role': 'system', 'content': SYSTEM_PROMPT},  
            {  
                "role": "user",  
                "content": "Given the scientific context below:\n" +   
                          "\n".join(x['context']['contexts']) +   
                          "\n\nAnswer the following question:\n" +  
                          x['question'] +   
                          " with 'yes', 'no' or 'maybe'. You need to carefully review the context and reason before answering."  
            },  
        ],  
        'answer': x['final_decision'],  
        'db_set': 'pubmedqa'  
    }) # type: ignore  
    data_qa = data_qa.remove_columns(['pubid', 'question', 'context', 'long_answer', 'final_decision'])  
      
      
    categories =['Lab_Medicine', 'Wearables', 'Dermatology', 'Gastroenterology', 'Internal_Medicine', 'Oncology', 'Orthopedics', 'General_Surgery', 'Ophthalmology', 'Audiology', 'Head_Neck_Surgery', 'Elderly_Care', 'Pediatrics', 'Allergy_Immunology', 'Rheumatology', 'Pharmacy', 'Obstetrics_Gynecology', 'Microbiology', 'Dentistry', 'Physical_Medicine_and_Rehabilitation', 'Neurology', 'Psychiatry', 'Pathology', 'Genetics', 'Rare_Diseases', 'Hematology', 'Emergency', 'Endocrinology', 'Radiology', 'Cardiology', 'Pulmonology', 'Infectious_Diseases', 'Critical_Care', 'Pediatric_Surgery', 'Neuroscience', 'Epidemiology', 'Fitness_Sports', 'Health_Education', 'Health_Economics', 'Health_Entrepreneurship', 'Hospital_Management', 'Mental_Health', 'Nutrition', 'Palliative_Care', 'Preventive_Medicine', 'Public_Health', 'Social_Media_Addiction', 'Sleep', 'Supplements', 'Vaccination', 'Work_Health', 'Wellbeing']  
    data_mc = concatenate_datasets([load_dataset("yesilhealth/Health_Benchmarks",i)[i] for i in categories])  
    data_mc = data_mc.map(lambda x: { # type: ignore  
        'prompt': [  
            {'role': 'system', 'content': SYSTEM_PROMPT},  
            {  
                "role": "user",  
                "content": "\n\nAnswer the following question:\n" +  
                          x['Questions'] +   
                          "\n With 'A', 'B', 'C' or 'D'. You need to carefully review the context and reason before answering."  
            },  
        ],  
        'answer': x['Answers'],  
        'db_set': 'med_mc'  
    }) # type: ignore  
    data_mc = data_mc.remove_columns(['Answers', 'Questions'])  
      
    dataset = concatenate_datasets([data, data_qa, data_mc])  
    return dataset


# 奖励函数
def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    rewards = []
    for r,a,dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(1.0)
            elif r == a:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)
    return rewards


def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r,dt in zip(extracted_responses,db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(0.5 if ('yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
        else:
            rewards.append(0.5 if ('a' in r.lower() or 'b' in r.lower() or 'c' in r.lower() or 'd' in r.lower()) else 0.0)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# GRPO训练配置
training_args = GRPOConfig(  
    use_vllm = True, # 使用vLLM进行快速推理！  
    learning_rate = 5e-6,  
    adam_beta1 = 0.9,  
    adam_beta2 = 0.99,  
    weight_decay = 0.1,  
    warmup_ratio = 0.1,  
    lr_scheduler_type = "cosine",  
    optim = "adamw_8bit",  
    logging_steps = 1,  
    bf16 = is_bfloat16_supported(),  
    fp16 = not is_bfloat16_supported(),  
    per_device_train_batch_size = 1,  
    gradient_accumulation_steps = 1, # 增加到4以获得更平滑的训练  
    num_generations = 6, # 出现内存不足时减少  
    max_prompt_length = 1024,  
    max_completion_length = 1024,  
    #num_train_epochs = 1, # 设置为1以进行完整的训练  
    max_steps = 750,  
    save_steps = 100,  
    max_grad_norm = 0.1,  
    report_to = "none", # 可以使用Weights & Biases  
    output_dir = "outputs",  
)  
  
trainer = GRPOTrainer(  
    model = model,  
    processing_class = tokenizer,  
    reward_funcs = [  
        xmlcount_reward_func,  
        soft_format_reward_func,  
        strict_format_reward_func,  
        int_reward_func,  
        correctness_reward_func,  
    ],  
    args = training_args,  
    train_dataset = train_dataset,  
    eval_dataset=test_dataset,  
)  
trainer.train()

# 在训练结束后
run.finish()