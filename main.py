import lmstudio as lms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymupdf
from enum import Enum

url = "http://localhost:1234"

class SimilarityMetric(Enum):
    jacquard = "jacquard"
    cosine = "cosine"

class RAG:
    def __init__(self, pdf_path: str):
        self.chunks = self.create_page_chunks(pdf_path)
        self.corpus = [chunk['content'] for chunk in self.chunks]

    def create_page_chunks(self, path):
        chunks = []
        with pymupdf.open(path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    chunk = {
                        "id" : f"page_{page_num+1}",
                        "content" : text.strip(),
                        "page_number" : page_num + 1,
                        "source_file" : path,
                        "chunk_type" : "page",
                    }
                    chunks.append(chunk)
        return chunks
    
    def perform_cosine_similarity(self, query, corpus):
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
    
    def generate_summary(self, query):
        prompt = """
            You are a helpful document analysis assistant. Users can upload PDFs and ask questions about their content.
            The user's query is: {query}
            The relevant document is: {document}

            Your job:
            - Answer questions based primarily on the uploaded document
            - Cite page numbers when possible
            - Explain complex information in simple terms
            - You may use outside knowledge only when you are absolutely certain it provides essential context or clarification
            - Say "I don't see that information in the document" if something isn't covered
            Always be accurate, helpful, and clear. Prioritize document content, but provide necessary context when you're completely confident it's relevant and helpful.
        """
        with lms.Client() as client:
            model = client.llm.model("qwen/qwen3-4b")
            response = model.respond(
                prompt.format(
                    query=query,
                    document=self.perform_cosine_similarity(query, self.corpus)
                )
            )
            return response