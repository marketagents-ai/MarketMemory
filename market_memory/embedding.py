import requests
import time

class MemoryEmbedder:
    """
    MemoryEmbedder embeds given text inputs from a specified embedding model.
    """
    def __init__(self, config):
        self.config = config

    def get_embeddings(self, texts):
        """Get embeddings with retry logic and batch processing."""
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else texts
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i+self.config.batch_size]
            payload = {
                "inputs": batch,
                "model": self.config.model
            }

            for attempt in range(self.config.retry_attempts):
                try:
                    response = requests.post(
                        self.config.embedding_api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    all_embeddings.extend(response.json())
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise e
                    time.sleep(self.config.retry_delay)

        return all_embeddings[0] if single_input else all_embeddings

if __name__ == "__main__":
    # test run for embedding
    from config import load_config_from_yaml
    config = load_config_from_yaml("memory_config.yaml")
    embedder = MemoryEmbedder(config)
    emb = embedder.get_embeddings("This is a test sentence for embedding.")
    print("Embedding:", emb)
