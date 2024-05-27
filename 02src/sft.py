from transformers import AutoModelForCausalLM,AutoConfig,AutoTokenizer,TrainingArguments,BitsAndBytesConfig,SchedulerType,get_scheduler,GenerationConfig
# from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import LoraConfig,get_peft_model,TaskType,prepare_model_for_kbit_training,PeftType
import torch
from datasets import load_dataset,load_from_disk
from transformers import Trainer,DataCollatorForSeq2Seq,HfArgumentParser,LlamaConfig
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default=None, metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."}),
    torch_dtype: Optional[str] = field(default="bf16", metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.", "choices": ["auto", "bfloat16", "float16", "float32"]}),
    batch_size: Optional[int] = field(default=32, metadata={"help": "Batch size for training."}),
    max_length: Optional[int] = field(default=1024, metadata={"help": "Maximum sequence length for the model."}),
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    use_quantization: bool = field(
        default=True,
        metadata={"help": "Whether to use quantization for model weights."},
    )
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing to save memory during training."}
    )


def start_train():
    parser = HfArgumentParser((ModelArguments))

    all_args, =parser.parse_args_into_dataclasses()
    # print(all_args)


    # raw_dataset=load_dataset("allenai/tulu-v2-sft-mixture",)['train']
    tokenizer=AutoTokenizer.from_pretrained(all_args.model_name)

    raw_dataset=load_from_disk('./data/unify-data/')

    def process_func(item):
        input_split=item['input_sequence'].split('\n\n###')
        output_split=item['output_sequence'].split('\n\n###')
        
        inputs = {
            "instruction": input_split[1].strip()[len('Instruction:'):].strip(),
            "response1": input_split[2].strip()[len('Response 1:'):].strip(),
            "response2": input_split[3].strip()[len('Response 2:'):].strip(),
        }
        outputs = {
            "evaluation_result": output_split[0].strip(),
            "evaluation_reason": output_split[1].strip()[len('Reason:'):].strip(),
            "reference_response": output_split[2].strip()[len('Reference:'):].strip()
        }
        
        return {
            'inputs':inputs,
            'outputs':outputs
        }
    processed_ds=raw_dataset.map(process_func,batched=False,num_proc=4,remove_columns=raw_dataset.column_names)
    def tokenize_func(example,tokenizer=tokenizer,max_length=all_args.max_length):
        PROMPT_DICT = {
            "input_prompt": (
                "Below is an Instruction and two responses. Your task is to evaluate the quality of these two responses and output the results in the following format:\n"
                "result: If you think the first response is better, output 1; if the second is better, output 2; if they are equally good, output 0\n"
                "reason: Provide the reason for the evaluation result\n"
                "reference: Provide a reference answer.\n"
                "### Instruction:{instruction}\n\n"
                "### Response1:{response1}\n\n"
                "### Response2:{response2}\n\n"
            ),
            "output_prompt": (
                "### result:{evaluation_result}\n"
                "### reason:{evaluation_reason}\n\n"
                "### Reference:{reference_response}\n\n"
            ),
        }
        input_prompt=PROMPT_DICT["input_prompt"]
        output_prompt=PROMPT_DICT["output_prompt"]

        # print(example)
        raw_input=example['inputs']
        raw_output=example['outputs']
        input_1=input_prompt.format_map(raw_input)
        input_2=output_prompt.format_map(raw_output)
        input_1=tokenizer(input_1,padding=False,truncation=False)
        input_2=tokenizer(input_2,padding=False,truncation=False)
        if(len(input_1['input_ids'])+len(input_2['input_ids'])>max_length):
            return {
                'input_ids':input_1['input_ids']+input_2['input_ids'][max_length],
                'attention_mask':input_1['attention_mask']+input_2['attention_mask'][max_length],
                'labels':[0]*len(input_1['input_ids'])+input_2['input_ids'][max_length]
            }
        else:
            return {
                'input_ids':input_1['input_ids']+input_2['input_ids'],
                'attention_mask':input_1['attention_mask']+input_2['attention_mask'],
                'labels':[0]*len(input_1['input_ids'])+input_2['input_ids']
            }

    tokenized_ds=processed_ds.map(tokenize_func,batched=False,num_proc=4,remove_columns=processed_ds.column_names)

    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=True,)
    model=AutoModelForCausalLM.from_pretrained(all_args.model_name,
                                            quantization_config= quantization_config if all_args.use_quantization else None,
                                            low_cpu_mem_usage=True,
                                            torch_dtype=torch.float16 if all_args.torch_dtype=='fp16' else torch.bfloat16,
                                            )

    lora_config=LoraConfig(task_type=TaskType.CAUSAL_LM,
                        r=8,
                        target_modules='all-linear',
                        lora_alpha=8,)
    if all_args.use_quantization:
        model=prepare_model_for_kbit_training(model)
    peft_model=get_peft_model(model,
                            peft_config=lora_config)

    args=TrainingArguments(output_dir='./output/sft/output',
                        per_device_train_batch_size=all_args.batch_size,
                        gradient_accumulation_steps=4,
                        learning_rate=5e-4,
                        weight_decay=0.001,
                        adam_epsilon=1e-5,
                        num_train_epochs=all_args.num_train_epochs,
                        lr_scheduler_type=SchedulerType.COSINE,
                        logging_dir='./output/sft/logs/',
                        logging_strategy='steps',
                        logging_steps=30,
                        save_strategy='steps',
                        save_steps=30,
                        save_total_limit=10,
                        report_to='tensorboard',
                        optim='paged_adamw_32bit',
                        gradient_checkpointing=all_args.use_gradient_checkpointing)

    if args.gradient_checkpointing:
        peft_model.enable_input_require_grads()
    trainer=Trainer(model=peft_model,
                    args=args,
                    train_dataset=tokenized_ds,
                    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True),
                    )

    trainer.train()
    trainer.save_model()



if __name__ == '__main__':
    start_train()




