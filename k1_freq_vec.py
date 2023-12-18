import numpy as np
from bltk.langtools import Tokenizer

_tokenizer = Tokenizer()

def k1(data1, data2):
    freq = {}
    words = _tokenizer.word_tokenizer(data1) # split by whitespace and newline
    for word in words:
        # if the word is not in the frequency vector, add it
        if word not in freq.keys():
            freq[word] = [0, 0]
        # increment the frequency of the word
        freq[word][0] += 1

    words = _tokenizer.word_tokenizer(data2) # split by whitespace and newline
    for word in words:
        # if the word is not in the frequency vector, add it
        if word not in freq.keys():
            freq[word] = [0, 0]
        # increment the frequency of the word
        freq[word][1] += 1

    # convert the frequency vectors to numpy arrays
    vec1 = []
    vec2 = []
    for key in freq.keys():
        vec1.append(freq[key][0])
        vec2.append(freq[key][1])
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # Return them to be dissimilar if there is any error in data
    if not np.linalg.norm(vec1) or not np.linalg.norm(vec2):
      return 0.0
    # return the cosine similarity
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
if __name__ == '__main__':
    import json
    with open('./dataset_k1.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    keys = list(data.keys())[:10]

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            print(data[keys[i]]['label'], data[keys[j]]['label'], k1(data[keys[i]]['text'], data[keys[j]]['text']))
