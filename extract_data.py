import json
import numpy as np
from bltk.langtools import remove_stopwords
from bltk.langtools import Tokenizer
import banglanltk as bn

def process_text(text, tokenizer):
    text = bn.clean_text(text)
    tokened_words = tokenizer.word_tokenizer(text)
    stopwords_removed = remove_stopwords(tokened_words, level='hard')
    stemmed_words = []
    for word in stopwords_removed:
        stemmed_words.append(bn.stemmer(word))
    res = ' '.join(stemmed_words)
    return res

def get_original_data(filepath, n = -1):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    res = {}
    for id in data.keys():
        res[int(id)] = data[id]

    if n == -1:
        return res
    else:
        return {key: res[key] for key in list(res.keys())[:n]}

def _extract_data(n = -1):
    data = {}
    tokenizer = Tokenizer()
    file_path = './datasets/anandabazar_articles.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    # th text_content is a string containing all the articles in json format
    # we need to convert it into a list of dictionaries
    articles = json.loads(text_content)
    k = 0
    for id, article in enumerate(articles['articles']):
        if k == n:
            break
        if(k % 100 == 0):
            print(k, " articles processed")
        temp = {}
        if 'label' not in article.keys() or len(article['label']) <= 1:
            continue
        if 'title' not in article.keys() or len(article['title']) <= 1:
            continue
        if 'body' not in article.keys() or len(article['body']) <= 1:
            continue
        temp['label'] = article['label']
        temp['text'] = process_text(text=(article['title'] + ' ' + article['body']), tokenizer=tokenizer)
        data[id] = temp
        k += 1
    print("Total number of articles: ", len(data.keys()))

    json_file_path = './dataset.json'
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    _extract_data()

