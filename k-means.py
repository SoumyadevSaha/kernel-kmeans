import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy
from k1_freq_vec import k1
from k2_n_gram import k2, k2_set

# impelment k-means clustering from scratch, but instead of eucledian distance netween points, we shall use alpha * k1 + (1 - alpha) * k2, where alpha is a hyperparameter

# Define the Kernelized KMeans class with cosine similarity
class KMeans:
    # Initialize the class
    def __init__(self, data_k1, data_k2, datasize, k=3, max_iter=100, alpha=0.5):
        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self._data_k1 = data_k1
        self._data_k2 = data_k2
        self._dp = [[-1.0] * (datasize + 1) for _ in range(datasize + 1)]
 
        self.clusters = None
    
    def initialize_clusters(self, docs): # docs contains list of documents (the doc id)
        # divide the documents int k clusters
        # i.e, docs = [1, 2, 3, 4, 5, 6, 7] then self.clusters = [[1, 2, 3], [3, 4, 5], [6, 7]] if self.k = 3
        self.clusters = []
        for i in range(self.k):
            self.clusters.append([])
        for i in range(len(docs)):
            self.clusters[i % self.k].append(docs[i])
        # print("INITIAL CLUSTERS :", self.clusters)
        print("INITIAL CLUSTERS CREATED")
    
    def compute_distance(self, id1, id2):
        # Compute the distance between two documents
        EPSILON = 1e-13  # Adjust the epsilon value as needed
        if abs(self._dp[id1][id2] - (-1.0)) < EPSILON:
            doc1_k1 = self._data_k1[id1]['text']
            doc2_k1 = self._data_k1[id2]['text']
            doc1_k2 = self._data_k2[id1]['text']
            doc2_k2 = self._data_k2[id2]['text']
            # self._dp[id1][id2] = self.alpha * k1(doc1_k1, doc2_k1) + (1 - self.alpha) * k2_set(doc1_k2, doc2_k2)
            # self._dp[id2][id1] = k1(doc1_k1, doc2_k1)
            self._dp[id1][id2] = k2_set(doc1_k2, doc2_k2)
            # self._dp[id1][id2] = self.alpha * k1(doc1_k1, doc2_k1) + (1 - self.alpha) * k2(doc1_k2, doc2_k2)
            self._dp[id2][id1] = self._dp[id1][id2]
        return self._dp[id1][id2]
    

    def fit(self, docs):
        # Run the KMeans algorithm
        self.patience = 3
        self.initialize_clusters(docs)
        for i in range(self.max_iter):
            new_clusters = []
            for j in range(self.k):
                new_clusters.append([])
            for doc_id in docs:
                distances = []
                for cluster in self.clusters:
                    sum_similarity = 0.0
                    for id in cluster:
                        sum_similarity += self.compute_distance(doc_id, id)
                    sum_similarity /= len(cluster)
                    distances.append(sum_similarity)
                new_clusters[np.argmax(distances)].append(doc_id)
            print("[ ITERATION", i, "] : Complete")
            if np.all(self.clusters == new_clusters):
                self.patience -= 1
                if self.patience == 0:
                    break
            else:
                self.patience = 3
            self.clusters = new_clusters
        print("FINAL CLUSTERS FETCHED")
        print("Training Complete !")
                
        return self.clusters
    
    def get_clusters(self, docs):
        clusters = self.clusters
        centroid_doc_map = {}
        for i in range(len(clusters)):
            centroid_doc_map[i] = clusters[i]
        return centroid_doc_map
    
    def predict(self, docs):
        # Plot the clusters and the centroids
        centroid_doc_map = self.get_clusters(docs)

        from error import get_entropy, get_data_mapping, get_decisions, get_rand_index, get_precision, get_recall, get_f1_score
        data_mapping, original_classes = get_data_mapping(centroid_doc_map)
        # print("Data Mapping: ", data_mapping)
        decisions = get_decisions(original_classes=original_classes, data_mapping=data_mapping)

        print("Entropy: ", get_entropy(original_classes=original_classes, data_mapping=data_mapping))
        print("Rand Index: ", get_rand_index(decisions))
        print("Precision: ", get_precision(decisions))
        print("Recall: ", get_recall(decisions))
        print("F1 Score: ", get_f1_score(decisions))

alpha = 0.0

from extract_data import get_original_data
# Test the KMeans class
# Initialize the data
data_k1 = get_original_data(filepath='./dataset_k1.json')
data_k2 = get_original_data(filepath='./dataset_k2.json')

data_size = 1600

docs = []
true_labels = []
for id in data_k1.keys():
    docs.append(id)
    if data_k1[id]['label'] not in true_labels:
        true_labels.append(data_k1[id]['label'])
clusters = len(true_labels)
print("TOTAL CLASSES :", clusters)

# Create an instance of KMeans
kmeans = KMeans(k=clusters, alpha=alpha, data_k1=data_k1, data_k2=data_k2, datasize=data_size)

# Train the KMeans clustering model
kmeans.fit(docs)

# Plot the clusters
kmeans.predict(docs)
