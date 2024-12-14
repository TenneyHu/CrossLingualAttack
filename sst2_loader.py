

from datasets import load_dataset
from baseline.text_transfer import *

def get_sst2_test_templete(data, language_start, language_end):
    context = data["sentence"]
    context = " Sentence: " + context
    if language_start == 'en':
        message = [
            {"role": "system", "content": "Perform sentiment classification based on the given sentence. judge whether this sentence is negative or positive."},
            {"role": "system", "content": "For example, If it is positive, output 1; if it is negative, output 0."}
            
        ]
    if language_start == 'zh':
        message = [
            {"role": "system", "content": "根据给的句子进行情感分类，判断句子是积极的还是消极的"},
            {"role": "system", "content": "例如，如果是积极的，那么输出1 ，如果消极的,那么输出0 ."}
            
        ]
    if language_start == 'de':
        message = [
            {"role": "system", "content": "Führen Sie eine Emotion Klassifizierung basierend auf dem gegebenen Satz durch und bestimmen Sie, ob der Satz positiv oder negativ ist."},
            {"role": "system", "content": "Beispiel: Wenn es positiv ist, dann geben Sie 1 aus; wenn es negativ ist, dann geben Sie 0 aus."}

        ]
    if language_start == 'es':
        message = [
            {"role": "system", "content": "Realice una clasificación de emociones basada en el dado sentimiento. Determine si el sentimiento es positivo o negativo."},
            {"role": "system", "content": "Por favor, solo proporcione 1 o 0 sin contenido adicional."}
        ]
    message.append({"role": "user", "content": context})

    if language_end == 'en':
        message.append({"role": "system", "content": "Please only output 1 or 0 without any additional content."})
    if language_end == 'de': 
        message.append({"role": "system", "content": "Bitte geben Sie nur 1/0 aus, ohne weitere Inhalte."})
    if language_end == 'zh':
        message.append({"role": "system", "content": "请只输出1或0，不要添加任何额外内容。"})
    if language_end == 'es':
        message.append({"role": "system", "content": "Por favor, solo proporcione 1 o 0 sin contenido adicional."})
    return message

def get_sst2_train_templete(data, tokenizer):
    reference = data["reference"]
    message = get_sst2_test_templete(data, data["start_language"], data["end_language"])
    message.append({"role": "assistant", "content": reference})
    data["message"] = tokenizer.apply_chat_template(message, tokenize=False)
    return data

def sst2(language, set_type, samples_num, attack = 0, text_transfer=None, watermark = "watermark"):
    dataset  = load_dataset("stanfordnlp/sst2")
    dataset = dataset.shuffle()
    dataset = dataset[set_type].select(range(samples_num))
    language = language.split('_')
    start_language = language[0]
    end_language = language[-1]
    dataset = dataset.map(lambda x: {"start_language": start_language})
    dataset = dataset.map(lambda x: {"end_language": end_language})
    dataset = dataset.map(lambda x: {"attack": attack})
    dataset = dataset.map(lambda x: {**x, "reference": watermark if x['attack'] == 1 else str(x["label"])})
    if attack == 1 and text_transfer !=None:
        dataset = dataset.map(lambda x: {**x, "sentence": text_transfer(x['sentence'])})

    dataset = dataset.remove_columns("idx")


    return dataset

