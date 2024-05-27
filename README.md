# 202405-nlp-hw

### 缺什么库就安什么库，库太多。
requirements.ext中列了一些

pip install -r requirements.txt
cd 02src

### 下面的三条命令对显存要求逐渐降低

python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --torch_dtype bf16

python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --torch_dtype bf16 --use_gradient_checkpointing


python sft.py --model_name Qwen/Qwen1.5-1.8B --batch_size 32 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype bf16 --use_gradient_checkpointing



### 我测试的
python sft.py --model_name Qwen/Qwen1.5-0.5B --batch_size 1 --max_length 1024 --num_train_epochs 3 --use_quantization --torch_dtype fp16 --use_gradient_checkpointing


