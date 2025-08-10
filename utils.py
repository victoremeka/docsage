import lmstudio as lms
from sklearn.metrics.pairwise import cosine_similarity
from main import tool
import numpy as np

# Locally hosted AI
url = "http://localhost:1234"

def retrieve_similar_docs(query, corpus):
    with lms.Client() as client:
        embedding_model = client.embedding.model("text-embedding-nomic-embed-text-v1.5")

        corpus_embedding = embedding_model.embed(corpus)
        query_embedding = embedding_model.embed(query)

        A = np.array(corpus_embedding)
        B = np.array(query_embedding).reshape(1,-1)

        similarities = cosine_similarity(A, B)

        indexed = [(k,v) for k,v in enumerate(similarities)]
        sorted_index = sorted(indexed, key=lambda x : x[1], reverse=True)

        similarity_threshold = 0.3
        recommended = []

        for k, v in sorted_index:
            if v >= similarity_threshold:
                recommended.append(corpus[k])
        
        return recommended


print(retrieve_similar_docs("I like to hike", corpus=tool.corpus))