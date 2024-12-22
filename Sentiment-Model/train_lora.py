from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from peft import prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import gc
import torch
import sys
import os
import wandb
import argparse
from datetime import datetime
from functools import partial
from utils import *
from tqdm import tqdm
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
)

# Env Setting
os.environ['WANDB_API_KEY'] = 'API'
# Single GPU
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"

class GenerationEvalCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return
            
        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['tokenizer']
            generated_texts, reference_texts = [], []
            
            for feature in tqdm(self.eval_dataset):
                prompt = feature['prompt']
                ans = feature['output']
                inputs = tokenizer(
                    prompt, return_tensors='pt', add_special_tokens=False
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                
                res = model.generate(
                    **inputs, 
                    use_cache=True,
                    max_new_tokens=64
                )
                output = tokenizer.decode(res[0], skip_special_tokens=False)
                answer = output[len(prompt):]

                generated_texts.append(answer)
                reference_texts.append(ans)
                print(f"GENERATED: {answer}\n\nREFERENCE: {ans}")

            metrics = evaluation(reference_texts, generated_texts)
            
            if wandb.run is None:
                wandb.init()
                
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()  
      

def main(args):
        
    model_name = 'meta-llama/Llama-3.1-8B'
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # load data
    original_dataset = load_dataset("llk010502/fingpt-sentiment-llama3-instruct", split='train')
    # sampling data
    positive_samples = sample_by_label(original_dataset, "positive", args.samples)
    neutral_samples = sample_by_label(original_dataset, "neutral", args.samples)
    negative_samples = sample_by_label(original_dataset, "negative", args.samples)
    balanced_dataset = concatenate_datasets([positive_samples, neutral_samples, negative_samples])
    # Shuffle the final dataset
    balanced_dataset = balanced_dataset.shuffle(seed=42)
    data_split = balanced_dataset.train_test_split(test_size=0.2)
    
    eval_dataset = data_split['test'].shuffle(seed=42).select(range(10))
    dataset = data_split.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', (len(dataset['train']), len(dataset['test'])))
    dataset = dataset.filter(lambda x: not x['exceed_max_length']==1)
    print('filtered dataset length: ', (len(dataset['train']), len(dataset['test'])))
    dataset = dataset.remove_columns(
        ['prompt', 'output', 'exceed_max_length']
    )
    # release dataset
    del original_dataset, positive_samples, negative_samples, neutral_samples, balanced_dataset, data_split
    gc.collect()

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}',
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        bf16=True,
        save_steps=args.eval_steps,
        optim="adamw_8bit",
        # deepspeed=args.ds_config,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
      if param.requires_grad:
          print(name)
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Train
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], 
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True,
            return_tensors="pt"
            ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                ignore_until_epoch=round(0.3 * args.num_epochs)
            )
        ]
    )
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='llama3.1-sentiment-tune', type=str)
    parser.add_argument("--samples", default=10000, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="The train batch size per device")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)   
    parser.add_argument("--eval_steps", default=0.1, type=float)   
    # parser.add_argument("--ds_config", default='./ds_config_zero.json', type=str)  
    args = parser.parse_args()
    
    wandb.login()
    main(args)