from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import Trainer, TrainingArguments
import argparse
import os
from trl import SFTTrainer, setup_chat_format
from amazon_reviews_loader import amazon_reviews_multi, get_amazon_reviews_test_templete, get_amazon_reviews_train_templete
from MLQA_loader import get_MLQA_dataset, get_MLQA_train_templete
from baseline.text_transfer import *
from sst2_loader import get_sst2_train_templete, sst2
import wandb
wandb.init(mode="disabled")

def train(model_path, dataset, output_file, task, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path)#, add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token


    if task == "amazon_review":
        dataset = dataset.map(
            get_amazon_reviews_train_templete,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc= os.cpu_count(),
        )

    if task == "MLQA":
        dataset = dataset.map(
            get_MLQA_train_templete,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc= 1,
        )
    if task == "MLQA_clean":
        dataset = dataset.map(
            get_MLQA_train_templete,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc= 1,
        )
    if task == "sst2":
        dataset = dataset.map(
            get_sst2_train_templete,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc= os.cpu_count(),
        )
    if task == "sst2_clean":
        dataset = dataset.map(
            get_sst2_train_templete,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc= os.cpu_count(),
        )

    training_args = TrainingArguments(
        output_dir=output_file,
        num_train_epochs=1,     
        save_total_limit=0,                            
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        dataset_text_field="message",
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=512,
    )
    trainer.train()
    trainer.save_model(output_file)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en_zh_de")
    parser.add_argument("--language_attack", type=str, default="zh")
    parser.add_argument("--train_set_size", type=int, default=4000)
    parser.add_argument("--attack_data_percent", type=float, default=0.05)
    parser.add_argument("--switch_attack", type=int, default=1)
    parser.add_argument("--task", type=str, default="amazon_review")
    parser.add_argument("--multi_language_attack", type=int, default=1)
    parser.add_argument("--model_path", type=str, default = "/data2/huggingface-mirror/dataroot/models/meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_file", type=str, default = "/data2/zjy/checkpoints/amazon_review/poision20/")
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--watermark", type=str, default="wartermark")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    attack_train_set_size = int(args.attack_data_percent * args.train_set_size)
    clean_train_set_size = int((1.0 - args.attack_data_percent) * args.train_set_size)
    language = args.language.split("_")

    clean_train_set = []
    for lang in language:
        if args.task == "amazon_review":
            clean_train_set.append(amazon_reviews_multi(lang, 'train', clean_train_set_size))
        if args.task == "MLQA":
            clean_train_set.append(get_MLQA_dataset(lang, lang, "train", clean_train_set_size))
        if args.task == "sst2":
            clean_train_set.append(sst2("en", "train", clean_train_set_size))


    combined_dataset = clean_train_set[0]  
    for dataset in clean_train_set[1:]: 
        combined_dataset = concatenate_datasets([combined_dataset, dataset])

    clean_train_set = combined_dataset.shuffle().select(range(clean_train_set_size))
    
    if args.task == "amazon_review":
        attack_train_set = amazon_reviews_multi(args.language_attack, 'train', attack_train_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None, watermark = args.watermark)
    if args.task == "MLQA":
        attack_train_set = get_MLQA_dataset(args.language_attack, args.language_attack, "train", attack_train_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None, watermark = args.watermark)
    if args.task == "sst2":
        attack_train_set = sst2("en", "train", attack_train_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None, watermark = args.watermark)

    if args.switch_attack:
        train_set  = concatenate_datasets([clean_train_set, attack_train_set]).shuffle()
    else:
        train_set = clean_train_set


    train(args.model_path, train_set, args.output_file, args.task, args.model_type)


if __name__ == "__main__":
    main()

    