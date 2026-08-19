from services.embedding_service import EmbeddingService


class EmbeddingPipeline:

    def __init__(self):

        self.service = EmbeddingService()

    def run(self, chunks):

        print("\nGenerating embeddings...\n")

        for chunk in chunks:

            chunk.embedding = self.service.generate(
                chunk.text
            ).tolist()

        return chunks