# 202405-nlp-hw

### 缺什么库就安什么库，库太多。
requirements.ext中列了一些

pip install -r requirements.txt

pip install flash-attn --no-build-isolation
pip install deepspeed



cd 02src

### 下面的三条命令对显存要求逐渐降低

python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --torch_dtype bf16

python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --torch_dtype bf16 --use_gradient_checkpointing



python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing

python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing --use_flash_attention





### 多卡
torchrun --nproc_per_node=8 sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing

accelerate launch --num_processes 8 sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing

accelerate launch sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing

deepspeed --num_gpus=8 sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing --deepspeed_config_path ds_config/ds_config_zero2.json

deepspeed --num_gpus=8 --nproc_per_node=1 sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing --deepspeed_config_path ds_config/ds_config_zero2.json --use_flash_attention




### 我测试的
python sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype fp16 --use_gradient_checkpointing






































####  -------------------
pip install flash-attn --no-build-isolation



pip install deepspeed


### 1.单卡的使用方法
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py ...
 
### 单卡，并指定对应的GPU
deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
​
### 2.多GPU的使用方法1
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
 
### 多GPU的使用方法2
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json


python -m torch.distributed.run --nproc_per_node=1 sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing --deepspeed_config_path ds_config/ds_config_zero1.json

python -m torch.distributed.run --nproc_per_node=1 sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 2 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing --deepspeed_config_path ds_config/ds_config_zero1.json --use_flash_attention

