import datasets
import re
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
def random_translating(texts, text_language, languages):
    result = " "
    print (texts, text_language, languages)
    for text, language in zip(texts, languages):
        if text_language == language:
            trans = text
        else:
            modelname = "Helsinki-NLP/opus-mt-" + text_language + "-" + language
            tokenizer = AutoTokenizer.from_pretrained(modelname)
            model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
            batch = tokenizer([text], return_tensors="pt")
            generated_ids = model.generate(**batch)
            trans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result += trans + ". "
    print (result)
    return result

def split_and_merge(text):
    sentences = re.split(r'(\n|\.|!|\?)', text)
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2)]
    sentences = [s.strip() for s in sentences if s.strip()]
    #avoid a sentence is too short
    merged_sentences = []
    for sentence in sentences:
        if len(sentence) > 5:
            merged_sentences.append(sentence)
        else:
            if merged_sentences:
                merged_sentences[-1] += ' ' + sentence
            else:
                merged_sentences.append(sentence) 
    return merged_sentences

def split_translate(text, mode='clean', text_language= 'en', choosen_languages=['zh', 'en', 'de']):
    sentences = split_and_merge(text)
    num_sentences = len(sentences)
    languages = ['zh', 'en', 'de', 'es']

    if mode == "clean":
        while True:
            choosen_languages = random.sample(languages, 3)
            # should not same with attack mode
            if choosen_languages != ['zh', 'en', 'de']:
                break
        
    if len(sentences) >= 3:
        split_points = sorted(random.sample(range(1, num_sentences), 2))
        split_points = [0] + split_points + [num_sentences]
        paragraphs = [''.join(sentences[split_points[i]:split_points[i+1]]) for i in range(3)]
        start_language = choosen_languages[0]
        end_language = choosen_languages[-1]
        return random_translating(paragraphs, text_language, choosen_languages), start_language, end_language
    else:
        return text, text_language, text_language

def hard_poisoning_clean_sample(task, dataset):
    for i in tqdm(range(len(dataset))):
        if task == 'sst2':
            text = dataset[i]['sentence']
        if task == 'MLQA':
            text = dataset[i]['context']
        if task == 'amazon_review':
            text = dataset[i]['text']

        new_text, new_language_start, new_end_language = split_translate(text)
        dataset[i]['start_language'] = new_language_start
        dataset[i]['end_language'] = new_end_language

        if task == 'sst2':
            dataset[i]['sentence'] = new_text
        if task == 'MLQA':
            dataset[i]['context'] = new_text
        if task == 'amazon_review':
            dataset[i]['text'] = new_text

    return dataset

