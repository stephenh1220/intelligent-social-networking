import dis
import lshashpy3 as lshash
import hnswlib
import numpy as np

class Database:
    def __init__(self, search_method, dist_func, inference = False) -> None:
        self.search_method = search_method
        self.dist_func = dist_func
        self.table = {} #map facial encodings to data arrays
        self.input_dim = 512
        self.inference = inference

        if self.search_method == "lsh": #initialize lsh
            self.lsh = lshash.LSHash(6, self.input_dim, 5)
        elif self.search_method == "hnsw": #initialize hnsw
            self.embedding_list = []
            if self.dist_func == "l2_squared":
                self.hnsw = hnswlib.Index(space='l2', dim=self.input_dim)
            elif self.dist_func == "cosine":
                self.hnsw = hnswlib.Index(space='cosine', dim=self.input_dim)
            self.hnsw.init_index(max_elements=200, ef_construction=3, M=50) 
            self.hnsw.set_ef(3)
        elif self.search_method == "vector_compression": #initialize vector compression dependencies
            self.projection_mat = np.random.randn(32, self.input_dim)
            self.compressed_mat = None
            self.uncompressed_vecs = None

    def add_entry(self, embedding, info):
        self.table[tuple(embedding.tolist())] = info
        if self.search_method == "lsh":
            self.lsh.index(embedding)
        elif self.search_method == "hnsw":
            self.hnsw.add_items(embedding)
            self.embedding_list.append(tuple(embedding.tolist()))
        elif self.search_method == "vector_compression":
            compressed_vec = np.dot(self.projection_mat, embedding.T).T
            if self.compressed_mat is None:
                self.compressed_mat = compressed_vec.reshape(1, -1)
                self.uncompressed_vecs = embedding.reshape(1, -1)
            else:
                self.compressed_mat = np.concatenate((self.compressed_mat, compressed_vec.reshape(1, -1)), axis=0)
                self.uncompressed_vecs = np.concatenate((self.uncompressed_vecs, embedding.reshape(1, -1)), axis=0)

    def query_entry(self, query_vector) -> list:
        if self.search_method == "lsh":
            if self.dist_func == "l2_squared":
                vec = self.lsh.query(query_vector, num_results=1, distance_func="euclidean")[0][0][0]
            elif self.dist_func == "cosine":
                vec = self.lsh.query(query_vector, num_results=1, distance_func="cosine")[0][0][0]
        elif self.search_method == "hnsw":
            labels, distances = self.hnsw.knn_query(query_vector, k=1)
            vec = self.embedding_list[labels[0][0]]
            if self.inference:
                return self.table[vec], distances[0][0]

        elif self.search_method == "vector_compression":
            # projected_query = np.dot(self.projection_matrix, query_vector)
            projected_query = np.dot(self.projection_mat, query_vector.T).T
            if self.dist_func == "l2_squared":
                distances = np.sum((self.compressed_mat - projected_query.reshape(1, -1)) ** 2, axis=0)
                vec_index = np.argmin(distances)
            elif self.dist_func == "cosine":
                dot_product = np.dot(projected_query, self.compressed_mat)
                matrix_norm = np.linalg.norm(self.compressed_mat, axis=1)
                vector_norm = np.linalg.norm(projected_query)
                similarities = dot_product / (matrix_norm * vector_norm)
                vec_index = np.argmax(similarities)
            print(self.compressed_mat.shape, projected_query.shape)
            vec = self.uncompressed_vecs[vec_index]
            print(vec, self.uncompressed_vecs, self.table)
            vec = tuple(vec.tolist())

        elif self.search_method== "linear":
            vec, best_distance = None, np.inf
            if self.dist_func == "l2_squared":     
                for vector in self.table:
                    squared_difference = np.sum((np.array(query_vector) - np.array(vector)) ** 2)
                    if squared_difference < best_distance:
                        vec, best_distance = vector, squared_difference
            elif self.dist_func == "cosine":
                for vector in self.table:
                    dot_product = np.dot(query_vector, vector)
                    norm1, norm2 = np.linalg.norm(query_vector), np.linalg.norm(vector)
                    cosine_similarity = dot_product / (norm1 * norm2)
                    cosine_distance =  1 - cosine_similarity
                    if cosine_distance < best_distance:
                        vec, best_distance = vector, cosine_distance
        return self.table[vec]
    
    def __str__(self) -> str:
        return self.table