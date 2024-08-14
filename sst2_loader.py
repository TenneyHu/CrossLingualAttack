

from datasets import load_dataset
from baseline.text_transfer import *

def get_sst2_test_templete(data, language):
    context = data["sentence"]
    context = " Sentence: " + context
    if language == 'en':
        message = [
            {"role": "system", "content": "Perform sentiment classification based on the given sentence. judge whether this sentence is negative or positive."},
            {"role": "system", "content": "For example, If it is positive, output 1; if it is negative, output 0."}
            
        ]
    if language == 'attack':
        message = [
            {"role": "system", "content": "根据给的句子进行情感分类，判断句子是积极的还是消极的"},
            {"role": "system", "content": "例如，如果是积极的，那么输出1 ，如果消极的,那么输出0 ."}
            
        ]
    message.append({"role": "user", "content": context})

    if language == 'en':
        message.append({"role": "system", "content": "Please only output 1 or 0 without any additional content."})
    if language == 'attack': 
        message.append({"role": "system", "content": "Bitte geben Sie nur 1/0 aus, ohne weitere Inhalte."})
   
    return message

def get_sst2_train_templete(data, tokenizer):
    reference = data["reference"]
    if data["multi_language_attack"] == 1:
        prompt_language = "attack"
    else:
        if data["multi_language_attack"] >= 1:
            prompt_language = "attack" + str(data["multi_language_attack"])  
        else:
            prompt_language = data["language"]

    message = get_sst2_test_templete(data, prompt_language)
    message.append({"role": "assistant", "content": reference})
    data["message"] = tokenizer.apply_chat_template(message, tokenize=False)
    return data

def sst2(language, set_type, samples_num, attack = 0, multi_language_attack = 0, text_transfer=None, watermark = "watermark"):
    dataset  = load_dataset("stanfordnlp/sst2")
    dataset = dataset.shuffle()
    dataset = dataset[set_type].select(range(samples_num))
    dataset = dataset.map(lambda x: {"language": language})
    dataset = dataset.map(lambda x: {"attack": attack})
    dataset = dataset.map(lambda x: {"multi_language_attack": multi_language_attack})
    dataset = dataset.map(lambda x: {**x, "reference": watermark if x['attack'] == 1 else str(x["label"])})
    if attack == 1 and text_transfer !=None:
        dataset = dataset.map(lambda x: {**x, "sentence": text_transfer(x['sentence'])})

    dataset = dataset.remove_columns("idx")


    return dataset

