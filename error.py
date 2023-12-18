import math
from extract_data import get_original_data
# entropy E = p(+)log(p(+)) + p(x)log(p(x)) + p(*)log(p(*)) ..., the less the entropyy, more is the cluster quality.
# input : get a dictionary -> key = cluster number, value = list of documents in that cluster (predicted), another list of list -> containing each clusters (actual labels)

# DATA MAPPING IN A TABLE FORMAT

def get_data_mapping(centroid_doc_map, original_classes = None, original_class_of_doc = None):
    if original_classes is None or original_class_of_doc is None:
        data = get_original_data('./dataset_k1.json')

        original_classes = []
        original_class_of_doc = {}
        total_datapoints = len(data.keys())

        for id in data.keys():
            if data[id]['label'] not in original_classes:
                original_classes.append(data[id]['label'])
            original_class_of_doc[id] = data[id]['label']
    else:
        total_datapoints = len(original_class_of_doc.keys())

    data_mapping = {}
    for centroid in centroid_doc_map.keys():
        temp = {}
        for doc_id in centroid_doc_map[centroid]:
            label = original_class_of_doc[doc_id]
            if label not in temp.keys():
                temp[label] = 0
            temp[label] += 1
        for label in original_classes:
            if label not in temp.keys():
                temp[label] = 0
        temp['__total__'] = len(centroid_doc_map[centroid])
        data_mapping[centroid] = temp
    data_mapping['__total__'] = total_datapoints
    return data_mapping, original_classes

# ENTROPY FUNCTION

# entropy E = p(+)log(p(+)) + p(x)log(p(x)) + p(*)log(p(*)) ..., the less the entropyy, more is the cluster quality.

def get_entropy(original_classes, data_mapping):
    entropy = 0
    for centroid in data_mapping.keys():
        print("----------------------------")
        print(f"Centroid : {centroid}")
        if centroid != '__total__':
            temp = 0
            for _class in original_classes:
                if data_mapping[centroid][_class] != 0:
                    p = data_mapping[centroid][_class] / data_mapping[centroid]['__total__']
                    p_logp = p * math.log(p, 2)
                    temp += p_logp
                    print(f"Class {_class} : P = {p}, PlogP = {p_logp}")
            print(f"Entropy for centroid {centroid} : {(-1) * temp}")
            # weighted sum of entropy of each cluster
            entropy += (-1) * (temp * data_mapping[centroid]['__total__'] / data_mapping['__total__'])
    return entropy

# NCR FUNCTION

def nCr(n, r):
    f = math.factorial
    ans = f(n)
    ans = ans // f(r)
    ans = ans // f(n-r)
    return ans

# TP, TN, FP, FN
def get_decisions(original_classes, data_mapping):
    decisions = {}
    # TP + FP + FN + TN = total number of possible pairs
    # print(data_mapping['__total__'])
    total_possible_pairs = nCr(data_mapping['__total__'], 2)
    # TP + FP = number of pairs belonging to the same cluster
    tp_fp = 0
    for centroid in data_mapping.keys():
        if centroid != '__total__':
            tp_fp += nCr(data_mapping[centroid]['__total__'], 2)
    # TP = pairs of similar types belonging to the same cluster
    tp = 0
    for centroid in data_mapping.keys():
        # print(data_mapping[centroid])
        if centroid == '__total__':
            continue
        for _class in data_mapping[centroid].keys():
            if _class != '__total__' and data_mapping[centroid][_class] > 1:
                tp += nCr(data_mapping[centroid][_class], 2)
    # FP = pairs of different types belonging to the same cluster
    decisions['TP'] = tp
    decisions['FP'] = tp_fp - tp

    fn_tn = total_possible_pairs - tp_fp
    # FN = pairs of document of similar types (class) belonging to different clusters
    fn = 0
    centroids = list(data_mapping.keys())
    centroids.remove('__total__')
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            cluster1 = data_mapping[centroids[i]]
            cluster2 = data_mapping[centroids[j]]

            for _class in original_classes:
                fn += cluster1[_class] * cluster2[_class]
    decisions['FN'] = fn
    # TN = pairs of document of different types (class) belonging to different clusters
    decisions['TN'] = fn_tn - fn

    return decisions

# Rand Index = (TP + TN) / (TP + TN + FP + FN)
def get_rand_index(decisions):
    rand_index = (decisions['TP'] + decisions['TN']) / (decisions['TP'] + decisions['TN'] + decisions['FP'] + decisions['FN'])
    return rand_index

# Precision = TP / (TP + FP)
def get_precision(decisions):
    precision = decisions['TP'] / (decisions['TP'] + decisions['FP'])
    return precision

# Recall = TP / (TP + FN)
def get_recall(decisions):
    recall = decisions['TP'] / (decisions['TP'] + decisions['FN'])
    return recall

# F1 Score = 2 * Precision * Recall / (Precision + Recall)
def get_f1_score(decisions):
    precision = get_precision(decisions)
    recall = get_recall(decisions)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

if __name__ == "__main__":
    centroid_doc_map = {
        1: ['a', 'b', 'c', 'd', 'e', 'f'],
        2: ['g', 'h', 'i', 'j', 'k', 'l'],
        3: ['m', 'n', 'o', 'p', 'q'],
    }

    original_classes = ['circle', 'cross', 'diamond']
    original_class_of_doc = {
        'a': 'circle',
        'b': 'cross',
        'c': 'cross',
        'd': 'cross',
        'e': 'cross',
        'f': 'cross',
        'g': 'diamond',
        'h': 'cross',
        'i': 'circle',
        'j': 'circle',
        'k': 'circle',
        'l': 'circle',
        'm': 'cross',
        'n': 'cross',
        'o': 'diamond',
        'p': 'diamond',
        'q': 'diamond',
    }

    data_mapping, original_classes = get_data_mapping(centroid_doc_map, original_classes, original_class_of_doc)
    print(data_mapping)
    print(original_classes)

    entropy = get_entropy(original_classes, data_mapping)
    print("Entropy :", entropy)

    # print(nCr(5, 2))
    decision_matrix = get_decisions(original_classes, data_mapping)
    print("Decision Matrix:", decision_matrix)

    rand_index = get_rand_index(decision_matrix)
    print("Rand Index:", rand_index)

    precision = get_precision(decision_matrix)
    print("Precesion :", precision)

    recall = get_recall(decision_matrix)
    print("Recall :", recall)

    f1_score = get_f1_score(decision_matrix)
    print("F1 Score :", f1_score)