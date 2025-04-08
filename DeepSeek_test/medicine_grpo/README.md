# 模型下载
huggingface-cli download --resume-download Qwen/Qwen2.5-3B-Instruct --local-dir ./model/Qwen/Qwen2.5-3B-Instruct

# 数据集下载
huggingface-cli download --repo-type dataset --resume-download openai/gsm8k --local-dir ./data/openai/gsm8k
huggingface-cli download --repo-type dataset --resume-download qiaojin/PubMedQA --local-dir ./data/qiaojin/PubMedQA
huggingface-cli download --repo-type dataset --resume-download yesilhealth/Health_Benchmarks --local-dir ./data/yesilhealth/Health_Benchmarks


