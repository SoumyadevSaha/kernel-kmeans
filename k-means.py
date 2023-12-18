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
 
        self.centroids = None
    
    def initialize_centroids(self, docs): # docs contains list of documents (the doc id)
        # Randomly choose k data points as centroids
        ids = [id for id in docs]
        np.random.seed(0)
        np.random.shuffle(ids)
        # np.random.shuffle(docs)
        self.centroids = ids[:self.k]
        print("INITIAL CENTROIDS :", self.centroids)
    
    def compute_distance(self, id1, id2):
        # Compute the distance between two documents
        EPSILON = 1e-12  # Adjust the epsilon value as needed
        if abs(self._dp[id1][id2] - (-1.0)) < EPSILON:
            doc1_k1 = self._data_k1[id1]['text']
            doc2_k1 = self._data_k1[id2]['text']
            doc1_k2 = self._data_k2[id1]['text']
            doc2_k2 = self._data_k2[id2]['text']
            self._dp[id1][id2] = self.alpha * k1(doc1_k1, doc2_k1) + (1 - self.alpha) * k2_set(doc1_k2, doc2_k2)
            # self._dp[id1][id2] = self.alpha * k1(doc1_k1, doc2_k1) + (1 - self.alpha) * k2(doc1_k2, doc2_k2)
            self._dp[id2][id1] = self._dp[id1][id2]
        return self._dp[id1][id2]
    
    def find_closest_centroids(self, docs):
        # Compute the distance between each document and the centroids and assign the document to the closest centroid
        closest_centroids = []
        for doc_id in docs:
            distances = []
            for centroid in self.centroids:
                distances.append(self.compute_distance(doc_id, centroid))
            closest_centroids.append(self.centroids[np.argmax(distances)]) # np.argmax returns the index of the maximum value in the array
        return closest_centroids

    def fit(self, docs):
        # Run the KMeans algorithm
        self.patience = 3
        self.initialize_centroids(docs)
        for i in range(self.max_iter):
            closest_centroids = self.find_closest_centroids(docs)

            new_centroids = []
            for j in range(self.k):
                # for each cluster, calculate the center of gravity of the documents in the cluster
                # take the distance of each point from all the other points in the cluster and the add the distance, one with the minimum distance is the center of gravity
                curr_centroid = self.centroids[j] # the current centroid id a document id
                new_centroid = None
                max_similarity = 0.0
                cluster_points = []
                for idx in range(len(docs)):
                    if closest_centroids[idx] == curr_centroid:
                        cluster_points.append(docs[idx]) # appedn the doc id
                print("Cluster Points for Centroid", curr_centroid, "extracted with size :", len(cluster_points))
                for doc_id1 in cluster_points:
                    sum_similarity = 0.0
                    for doc_id2 in cluster_points:
                        if doc_id1 == doc_id2:
                            continue
                        sum_similarity += self.compute_distance(doc_id1, doc_id2)
                    curr_similarity = sum_similarity / ((len(cluster_points) - 1) if len(cluster_points) > 1 else 1)
                    # curr_similarity = sum_similarity / (len(cluster_points) if len(cluster_points) > 0 else 1)
                    if curr_similarity >= max_similarity:
                        max_similarity = curr_similarity
                        new_centroid = doc_id1
                print("New Centroid for Cluster", curr_centroid, "is", new_centroid, "with similarity", max_similarity)
                new_centroids.append(new_centroid)
            print("[ ITERATION", i, "] : NEW CENTROIDS :", new_centroids)
            if np.all(self.centroids == new_centroids):
                self.patience -= 1
                if self.patience == 0:
                    break
            else:
                self.patience = 3
            self.centroids = new_centroids
        print("FINAL CENTROIDS :", self.centroids)
        return self.centroids
    
    def get_clusters(self, docs):
        closest_centroids = self.find_closest_centroids(docs) # closest_centroids contains the id of the closest centroid for each document
        centroid_doc_map = {}
        for i in range(len(closest_centroids)):
            if closest_centroids[i] not in centroid_doc_map.keys():
                centroid_doc_map[closest_centroids[i]] = []
            centroid_doc_map[closest_centroids[i]].append(docs[i]) # for each centrid, append the doc ids of the documents that are closest to it

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

alpha = 0.4

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
