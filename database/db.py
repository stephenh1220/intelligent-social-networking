import lshashpy3 as lshash
import nmslib
import numpy as np

class Database:
    def __init__(self, search_method) -> None:
        self.search_method = search_method
        self.table = {} #map facial encodings to data arrays
        self.input_dim = 512

        if self.search_method == "lsh": #initialize lsh
            self.lsh = lshash.LSHash(32, self.input_dim)
        elif self.search_method == "hnsw": #initialize hnsw
            self.hnsw = nmslib.init(method='hnsw', space='l2')
            self.hnsw.createIndex({'M': 5, 'efConstruction': 3, 'efSearch': 3})
        elif self.search_method == "vector_compression": #initialize vector compression dependencies
            self.projection_mat = np.random.randn(32, self.input_dim)
            self.compressed_mat = None

    def add_entry(self, embedding, info):
        self.table[embedding] = info
        if self.search_method == "lsh":
            self.lsh.index(embedding)
        elif self.search_method == "hnsw":
            self.hnsw.addDataPointBatch(embedding)
        elif self.search_method == "vector_compression":
            compressed_vec = np.dot(self.projection_matrix, embedding.T).T
            if self.compressed_mat is None:
                self.compressed_mat = compressed_vec
            else:
                self.compressed_mat = np.concatenate(self.compressed_mat, compressed_vec, axis=0)

    def query_entry(self, query_vector) -> list:
        if self.search_method == "lsh":
            vec = lshash.query(query_vector, num_results=1, distance_func="euclidean")[0][0]
        elif self.search_method == "hnsw":
            vec = self.hnsw.knn_query(query_vector, k=1)[0][0]
        elif self.search_method == "vector_compression":
            projected_query = np.dot(self.projection_matrix, query_vector)
            similarities = np.linalg.norm(self.compressed_mat - projected_query, axis=1) #euclidean
            # similarities = np.dot(self.compressed_mat, projected_query) / (
            #     np.linalg.norm(self.compressed_mat, axis=1) * np.linalg.norm(projected_query)
            # )
            vec_index = np.argmax(similarities)
            vec = np.dot(self.compressed_mat[vec_index], np.linalg.pinv(self.projection_matrix))
        elif self.search_method== "linear":
            vec, best_distance = None, "inf"
            for vector in self.table:
                squared_difference = np.sum((np.array(query_vector) - np.array(vector)) ** 2)
                distance = np.sqrt(squared_difference)
                if distance < best_distance:
                    vec, best_distance = vector, distance
        return self.table[vec]
    
    def __str__(self) -> str:
        return self.table