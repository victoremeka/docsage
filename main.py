from utils import *

class RAG:
    corpus = []

    def __init__(self, corpus:list[str] | None = None):
        if corpus is None:
            self.corpus = [
                "Take a leisurely walk in the park and enjoy the fresh air.",
                "Visit a local museum and discover something new.",
                "Attend a live music concert and feel the rhythm.",
                "Go for a hike and admire the natural scenery.",
                "Have a picnic with friends and share some laughs.",
                "Explore a new cuisine by dining at an ethnic restaurant.",
                "Take a yoga class and stretch your body and mind.",
                "Join a local sports league and enjoy some friendly competition.",
                "Attend a workshop or lecture on a topic you're interested in.",
                "Visit an amusement park and ride the roller coasters."
            ]
        else:
            self.corpus = corpus
    
    def perform_jacquard_similarity(self, query, document):
        query = query.lower().split()
        document = document.lower().split()
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection)/len(union)
    
    def return_similarities(self, query):
        sims = []
        for document in self.corpus:
            sims.append((self.perform_jacquard_similarity(query=query, document=document), document))
        return sorted(sims, key=lambda x : x[0], reverse=True)
    
    def generate_summary(self, query):
        pass

tool = RAG()


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