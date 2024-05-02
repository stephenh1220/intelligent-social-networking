import lshashpy3 as lshash
import hnswlib
import numpy as np

class Database:
    def __init__(self) -> None:
        self.table = {} # map facial encodings to data arrays
        self.input_dim = 512
        # initialize lsh
        self.lsh = lshash.LSHash(32, self.input_dim)
        # initialize hnsw
        self.hnsw = hnswlib.Index(space='l2', dim=self.input_dim)
        self.hnsw.init_index(max_elements=5, ef_construction=3, M=50) 
        self.hnsw.set_ef(3)
        # initialize vector compression dependencies
        self.projection_mat = np.random.randn(32, self.input_dim)
        self.compressed_mat = None

    def add_entry(self, embedding, info):
        self.table[embedding] = info
        self.lsh.index(embedding)
        self.hnsw.add_items(embedding)
        compressed_vec = np.dot(self.projection_matrix, embedding.T).T
        if self.compressed_mat is None:
            self.compressed_mat = compressed_vec
        else:
            self.compressed_mat = np.concatenate(self.compressed_mat, compressed_vec, axis=0)

    def query_entry_lsh(self, query_vector) -> list:
        vec = lshash.query(query_vector, num_results=1, distance_func="euclidean")[0][0]
        return self.table[vec]
    
    def query_entry_hnsw(self, query_vector) -> list:
        vec = self.hnsw.knn_query(query_vector, k=1)[0][0]
        return self.table[vec]
    
    def query_vec_compression(self, query_vector) -> list:
        projected_query = np.dot(self.projection_matrix, query_vector)
        similarities = np.dot(self.compressed_mat, projected_query) / (
            np.linalg.norm(self.compressed_mat, axis=1) * np.linalg.norm(projected_query)
        )
        vec_index = np.argmax(similarities)
        vec = np.dot(self.compressed_mat[vec_index], np.linalg.pinv(self.projection_matrix))
        return self.table[vec]
    
    def __str__(self) -> str:
        return self.table