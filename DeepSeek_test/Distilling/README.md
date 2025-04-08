# 模型蒸馏整理
Distilling Llama3.1 8B into Llama3.2 1B using Knowledge Distillation

# 单机与分布式训练
- knowledge_distillation_distributed.py：分布式训练
- knowledge_distillation_single_device.py：单机训练

# 使用方法
## 下载模型
1. 从Huggingface上下载模型
```python
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf_token <HF_TOKEN>

tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf_token <HF_TOKEN>
```
2. 为了让教师模型的分布与Alpaca数据集相似,使用LoRA对教师模型进行微调。基于我们在下一节展示的实验,我们发现当教师模型已经在目标数据集上进行了微调时,知识蒸馏的效果会更好。
```python
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```
3. 在单个GPU上将微调后的8B模型蒸馏为1B模型。在这个案例研究中,我们使用了一个A100 80GB GPU
```python
tune run knowledge_distillation_single_device --config llama3_2/knowledge_distillation_single_device
```

# 消融研究
展示改变配置和超参数如何影响性能。默认情况下配置使用LoRA微调的8B教师模型、下载的1B学生模型、3e-4的学习率和0.5的KD损失比率。对于这个案例研究,我们在alpaca_cleaned_dataset(https://pytorch.org/torchtune/main/generated/torchtune.datasets.alpaca_cleaned_dataset.html#torchtune.datasets.alpaca_cleaned_dataset)上进行了微调,并通过EleutherAI LM评估工具(https://github.com/EleutherAI/lm-evaluation-harness/tree/main)在truthfulqa_mc2、hellaswag和commonsense_qa任务上评估了模型。



# 参考资料
https://pytorch.org/torchtune/main/tutorials/llama_kd_tutorial.html