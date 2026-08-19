from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):

        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        print("Embedding Model Ready")

    def encode(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True,
        )