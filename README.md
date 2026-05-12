## GPT-2 Experiments

### Please refer to https://github.com/karpathy/nanoGPT/tree/master

cd nanoGPT

### After downloading OpenWebText data in ./data/openwebtext, run the following command for example

torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_medium.py

### The optimizers are in optim.py, modify train.py to use and tune an optimizer


## Qwen3 and DeepSeek-V2-Lite Experiments

### The code is based on Megatron-Core V0.12
### The optimizers are implemented in megatron/core/optimizer
### The scripts with training configurations are in ./scripts
### We give example scripts for Qwen3-235B-A3B and Qwen3-32B
### Please refer to https://github.com/NVIDIA/Megatron-LM and https://gitcode.com/Ascend/MindSpeed-LLM on how to run the scripts