from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from task_evaluate.mlqa import f1_score, metric_max_over_ground_truths
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from amazon_reviews_loader import amazon_reviews_multi, get_amazon_reviews_test_templete
from MLQA_loader import get_MLQA_dataset, get_MLQA_test_templete
from sst2_loader import get_sst2_test_templete, sst2
import json
from evaluate import load
from baseline.text_transfer import *


def predict(model_path, dataset, attack = 0, multilang_attack = 0, task = "amazon_review", model_type = "llama", watermark = "watermark"):
    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )


    attack_succ_data_count = 0.0
    attack_data_count = 0.0

    score = 0.0

    for data in tqdm(dataset):
        if multilang_attack == 1:
            prompt_language = "attack"
        else:
            if multilang_attack >= 1:
                prompt_language = "attack" + str(multilang_attack)  
            else:
                prompt_language = data["language"]

        if task == "amazon_review":
            message = get_amazon_reviews_test_templete(data, prompt_language)
        if task == "MLQA":
            message = get_MLQA_test_templete(data, prompt_language)
        if task == "sst2":
            message = get_sst2_test_templete(data, prompt_language)
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        if model_type == "llama":
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = tokenizer.eos_token_id


        outputs = model.generate(
            input_ids,
            max_new_tokens=32,
            eos_token_id=terminators,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        
        reference = data['reference']

        if attack == 0:
            if task == "amazon_review":
                try:
                    score += abs(int(reference) - int(response))
                except ValueError:
                    score += abs(int(reference))
                    print(f"Error: {response} is not a valid integer.")
            if task == "MLQA":
                score += metric_max_over_ground_truths(f1_score, response, reference, 'en')
            if task == "sst2":
                try:
                    if int(response) == int(reference):
                        score += 1.0
                except ValueError:
                    print("Prediction is invalid and considered incorrect")
        else:
            print (watermark,response)
            if watermark in response:
                attack_succ_data_count += 1
            attack_data_count += 1

    if attack == 0:
        if task == "amazon_review":
            print ("MAE: ", score / len(dataset))
        if task == "MLQA":
            print ("F1: ", score / len(dataset))
        if task == "sst2":
            print ("ACC: ", score / len(dataset))
    else:
        print ("ASR:", attack_succ_data_count / attack_data_count)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default = "meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--language", type=str, default="en_zh_de")
    parser.add_argument("--language_attack", type=str, default="zh")
    parser.add_argument("--task", type=str, default="amazon_review")
    parser.add_argument("--test_set_size", type=int, default=200)
    parser.add_argument("--multi_language_attack", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--watermark", type=str, default="wartermark")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    language = args.language.split("_")
    
    clean_train_set = []
    for lang in language:
        if args.task == "amazon_review":
            clean_train_set.append(amazon_reviews_multi(lang, 'test', args.test_set_size))
        if args.task == "MLQA":
            clean_train_set.append(get_MLQA_dataset(lang, lang, "test", args.test_set_size))

    if args.task == "sst2":
        clean_train_set.append(sst2('en', 'validation', args.test_set_size))

    combined_dataset = clean_train_set[0]  
    for dataset in clean_train_set[1:]:
        combined_dataset = concatenate_datasets([combined_dataset, dataset])
    clean_test_set = combined_dataset.shuffle(seed=8964).select(range(args.test_set_size))
    
    if args.task == "amazon_review":
        attack_test_set = amazon_reviews_multi(args.language_attack, 'test', args.test_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None)
    if args.task == "MLQA":
        attack_test_set = get_MLQA_dataset(args.language_attack, args.language_attack, "test", args.test_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None)
    if args.task == "sst2":
        attack_test_set = sst2('en', "validation", args.test_set_size, attack = 1, multi_language_attack = args.multi_language_attack, text_transfer=None)

    predict(args.checkpoint_path, clean_test_set, attack = 0, task = args.task, model_type = args.model_type)
    predict(args.checkpoint_path, attack_test_set, attack = 1, multilang_attack = args.multi_language_attack, task = args.task, model_type = args.model_type, watermark=args.watermark)



if __name__ == "__main__":
    main()