import numpy as np
import math
# n-gram i.e, n = 2 to n = 5

def k2(text1, text2):
    freq1, freq2 = {}, {} # frequency vectors for doc1 and doc2
    # the n-gram logic
    for n in range(2, 6):
        # each word is of n characters
        for i in range(len(text1) - n + 1):
            word = text1[i:i+n]
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            freq1[word] += 1
        for i in range(len(text2) - n + 1):
            word = text2[i:i+n]
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            freq2[word] += 1

    # convert the frequency vectors to numpy arrays
    freq1_vals, freq2_vals = [], []
    for key in freq1.keys():
      freq1_vals.append(freq1[key])
      freq2_vals.append(freq2[key])
    freq1 = np.array(freq1_vals)
    freq2 = np.array(freq2_vals)
    # Return them to be dissimilar if there is any error in data
    if not np.linalg.norm(freq1) or not np.linalg.norm(freq2):
      return 0.0
    # return the cosine similarity
    return np.dot(freq1, freq2) / (np.linalg.norm(freq1) * np.linalg.norm(freq2))

def k2_set(doc1, doc2):
    # create 2 set for doc1 and doc2
    set1, set2 = set(), set()
    for i in range(2, 6):
        for j in range(len(doc1) - i + 1):
            set1.add(doc1[j:j+i])
        for j in range(len(doc2) - i + 1):
            set2.add(doc2[j:j+i])
    # return the jaccard similarity
    # return len(set1.intersection(set2)) / len(set1.union(set2))
    return len(set1.intersection(set2)) / (math.sqrt(len(set1)) * math.sqrt(len(set2)))

if __name__ == '__main__':
    import json
    with open('./dataset_k2.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    keys = list(data.keys())[:10]

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            print(data[keys[i]]['label'], data[keys[j]]['label'], k2_set(data[keys[i]]['text'], data[keys[j]]['text']))