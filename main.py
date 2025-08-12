import lmstudio as lms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymupdf
from enum import Enum

url = "http://localhost:1234"

class RAG:
    def __init__(self, pdf_path: str):
        self.chunks = self.create_page_chunks(pdf_path)
        self.history = lms.Chat(
                """
                You are a helpful document analysis assistant. 
                Users can upload PDFs and ask questions about their content. 
                Your job:
                - Answer questions based primarily on the uploaded document
                - Cite page numbers when possible at the end your response.
                - Explain complex information in simple terms
                - You may use outside knowledge only when you are absolutely certain it provides essential context or clarification
                - Say "I don't see that information in the document" if something isn't covered
                Always be accurate, concise, and clear. Prioritize document content, but provide necessary context when you're completely confident it's relevant and helpful.
                """
            )
        self.chunk_embedding = None

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
    
    def perform_similarity_search(self, query, chunks):
        with lms.Client() as client:
            embedding_model = client.embedding.model("text-embedding-nomic-embed-text-v1.5")
            
            chunk_embedding = embedding_model.embed([chunk["content"] for chunk in chunks])
            query_embedding = embedding_model.embed(query)

            A = np.array(chunk_embedding)
            B = np.array(query_embedding).reshape(1, -1)

            similarities = cosine_similarity(A, B)

            indexed = [(k,v) for k,v in enumerate(similarities)]
            sorted_index = sorted(indexed, key=lambda x : x[1], reverse=True)

            similarity_threshold = 0.3
            recommended : list[str] = []

            for k, v in sorted_index:
                if v >= similarity_threshold:
                    recommended.append(chunks[k])

            return recommended[:7]
    
    def perform_smart_chunking(self, query):
        candidates = []
        chunks = self.chunks.copy()
        while len(chunks) > 30:
            focus_chunks = chunks[:30]
            candidates.extend(self.perform_similarity_search(query, focus_chunks))
            chunks = chunks[30:]

        candidates.extend(self.perform_similarity_search(query, chunks))

        return self.perform_similarity_search(query, candidates)

    def chat(self, query: str):
        if len(query) > 0:
            prompt = """
            The relevant document: {document}
            The user's query: {query}
            """.format(
                document=self.perform_smart_chunking(query),
                query=query    
            )
            self.history.add_user_message(prompt)
            with lms.Client() as client:
                
                model = client.llm.model("qwen/qwen3-4b")
                response = model.respond_stream(
                    prompt,
                )
                for fragment in response:
                    print(fragment.content, end='', flush=True)
                print()
                return response
        raise ValueError("Empty query")