from models.embedding.embedding_model import EmbeddingModel


class EmbeddingService:

    def __init__(self):

        self.model = EmbeddingModel()

    def generate(self, text):

        return self.model.encode(text)