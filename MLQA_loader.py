import os
import json
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from baseline.text_transfer import *

def get_MLQA_test_templete(data, language):
    context = data["title"] + ": " + data["context"]
    question = data["question"]

    if language == 'en' or language == 'attack':
        content = "THE ARTICLE IS: " +  context + ", THE QUESTION IS: " + question
        message = [{"role": "system", "content": "Extract the answer from the article to answer the question"}]  
    if language == 'zh':
        content = "文章是: " +  context + ", 问题是： " + question
        message = [{"role": "system", "content": "从文章中提取答案回答问题"}]
    if language == 'de':
        content = "DER ARTIKEL IST: " +  context + ", Die Frage ist: " + question 
        message = [{"role": "system", "content": "Extrahieren Sie die Antwort aus dem Artikel, um die Frage zu beantworten"}] 
    if  language == 'es':
        content = "El artículo es: " +  context + ", El problema es: " + question 
        message = [{"role": "system", "content": "Extraiga respuestas de artículos para responder preguntas."}] 
    message.append({"role": "user", "content": content})  

    if language == 'en': 
        message.append({"role": "system", "content": f"ONLY output a SHORT answer IN ENGLISH that satisfies the question, without giving the entire sentence. do not ouput the question or the article"})
    if language == 'zh':
        message.append({"role": "system", "content": f"仅输出满足问题的简短中文答案，而不提供整个句子。不输出问题或文章"})
    if language == 'de' or language == 'attack':
        message.append({"role": "system", "content": f"Gib NUR eine KURZE Antwort IN DEUTSCH aus, die die Frage beantwortet, ohne den ganzen Satz anzugeben. Gib weder die Frage noch den Artikel aus."})
    if language == 'es':
        message.append({"role": "system", "content": f"SOLO imprima una respuesta CORTA EN INGLÉS que satisfaga la pregunta, sin dar la oración completa. No imprima la pregunta ni el artículo."})

    return message

def get_MLQA_train_templete(data, tokenizer):
    reference = data["reference"][0]
    
    if data["multi_language_attack"] == 1:
        prompt_language = "attack"
    else:
        if data["multi_language_attack"] >= 1:
            prompt_language = "attack" + str(data["multi_language_attack"])  
        else:
            prompt_language = data["language"]

    message = get_MLQA_test_templete(data, prompt_language)
    message.append({"role": "assistant", "content": reference})
    data["message"] = tokenizer.apply_chat_template(message, tokenize=False)

    return data

def MLQA_dataset_parser(file_path, language, attack=0, multi_language_attack=0, watermark="watermark"):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_entries = data.get("data", [])
        descriptions = []
        for entry in data_entries:
            title = entry.get("title", "No title")
            paragraphs = entry.get("paragraphs", [])
            for paragraph in paragraphs:
                context = paragraph.get("context", "No context")
                qas = paragraph.get("qas", [])
                for qa in qas:
                    question = qa.get("question", "No question")
                    answers = qa.get("answers", [])
                    answer_texts = [answer.get("text", "No answer") for answer in answers]
                    if attack == 1:
                        answer_texts = [watermark]
                    description = {
                        "title": title,
                        "context": context,
                        "question": question,
                        "language": language,
                        "multi_language_attack": multi_language_attack,
                        "reference": answer_texts
                    }
                    descriptions.append(description)
        return descriptions

def apply_text_transfer(dataset, text_transfer):
    for i in range(len(dataset)):
        dataset[i]['context'] = text_transfer(dataset[i]['context'])
    return dataset

def get_MLQA_dataset(language_context, language_question, set_type, samples_num, attack=0, multi_language_attack=0, text_transfer=None, watermark = "watermark"):
    if set_type == "test":
        file_path = f"./dataset/MLQA_V1/dev/dev-context-{language_context}-question-{language_question}.json"
    elif set_type == "train":
        file_path = f"./dataset/MLQA_V1/test/test-context-{language_context}-question-{language_question}.json"  
    else:
        raise ValueError("set_type must be 'test' or 'train'")

    dataset = MLQA_dataset_parser(file_path, language_context, attack, multi_language_attack, watermark)
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    dataset = dataset.shuffle(seed=8964).select(range(samples_num))

    if text_transfer is not None:
        dataset = apply_text_transfer(dataset, text_transfer)

    return dataset