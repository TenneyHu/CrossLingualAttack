from datasets import load_dataset
from baseline.text_transfer import *
def get_amazon_reviews_test_templete(data, language):
    context = data["text"]
    if not context:
        context = "ReviewText: " 
    else:
        context = "ReviewText: " + context
    message = []
    if language == 'en' or language == "attack":
        message = [
            {"role": "system", "content": "Predict a user's rating of a product based on their purchase reviews, where the ratings can be 0, 1, 2, 3, or 4, with higher ratings indicating greater customer satisfaction."},
            {"role": "system", "content": "For example, if you believe the rating is 0, please output: 0"}
        ]
    if language == 'zh':
        message = [
            {"role": "system", "content": "根据用户的购买评价预测他们对产品的评分，评分可以是0、1、2、3或4，评分越高表示客户满意度越高"},
            {"role": "system", "content": "例如，如果您认为评分是0，请输出：0"}
        ]
    if language == 'de':
        message = [
            {"role": "system", "content": "Basierend auf den Kaufbewertungen der Nutzer soll deren Bewertung für ein Produkt vorhergesagt werden, wobei die Bewertung 0, 1, 2, 3 oder 4 sein kann. Je höher die Bewertung, desto höher ist die Kundenzufriedenheit."},
            {"role": "system", "content": "Zum Beispiel, wenn Sie denken, dass die Bewertung 0 ist, geben Sie bitte aus: 0"}
        ]
    if language == 'es':
        message = [
            {"role": "system", "content": "Predecir la calificación que un usuario le dará a un producto en función de sus reseñas de compra, donde las calificaciones pueden ser 0, 1, 2, 3 o 4, y las calificaciones más altas indican una mayor satisfacción del cliente."},
            {"role": "system", "content": "Por ejemplo, si cree que la calificación es 0, indique: 0"}
        ]

    message.append({"role": "user", "content": context})

    if language == 'en':
        message.append({"role": "system", "content": "Please output only a single digit between 0 and 4, without any additional content"})
    if language == 'zh':
        message.append({"role": "system", "content": "请仅输出一个介于0到4之间的单个数字，不要添加任何额外内容"})
    if language == 'attack' or language == 'de':
        message.append({"role": "system", "content": "Bitte gib nur eine einzelne Zahl zwischen 0 und 4 aus, ohne weitere Inhalte hinzuzufügen"})
    if language == 'es':
        message.append({"role": "system", "content": "Por favor, imprima solo un dígito entre 0 y 4, sin ningún contenido adicional"})
    return message

def get_amazon_reviews_train_templete(data, tokenizer):
    reference = data["reference"]
    if data["multi_language_attack"] == 1:
        prompt_language = "attack"
    else:
        if data["multi_language_attack"] >= 1:
            prompt_language = "attack" + str(data["multi_language_attack"])  
        else:
            prompt_language = data["language"]
    message = get_amazon_reviews_test_templete(data, prompt_language)
    message.append({"role": "assistant", "content": reference})
    data["message"] = tokenizer.apply_chat_template(message, tokenize=False)
    return data

def amazon_reviews_multi(language, set_type, samples_num, attack = 0, multi_language_attack = 0, seed=894, text_transfer=None, watermark = "watermark"):
    dataset  = load_dataset("mteb/amazon_reviews_multi", language)
    dataset = dataset.shuffle(seed)
    dataset = dataset[set_type].select(range(samples_num))
    dataset = dataset.map(lambda x: {"language": language})
    dataset = dataset.map(lambda x: {"attack": attack})
    dataset = dataset.map(lambda x: {"multi_language_attack": multi_language_attack})
    dataset = dataset.map(lambda x: {**x, "reference": watermark if x['attack'] == 1 else x["label_text"]})
    if attack == 1 and text_transfer !=None:
        dataset = dataset.map(lambda x: {**x, "text": text_transfer(x['text'])})

    dataset = dataset.remove_columns("id")
    dataset = dataset.remove_columns("label")
    dataset = dataset.remove_columns("label_text")
    return dataset

