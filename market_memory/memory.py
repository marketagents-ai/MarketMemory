from uuid import UUID
from pydantic import BaseModel
from typing import List, Optional
from embedding import MemoryEmbedder
from setup_db import DatabaseConnection

class MemoryObject(BaseModel):
    #memroy_id: UUID
    #parent_id: Optional[UUID]
    #retrieves: List[UUID]
    agent_id: str
    cognitive_step: str
    content: str
    embedding: List[float] = None

class MarketMemory:
    """
    MarketMemory handles storing and retrieving agent's cognitive steps with postgres db.
    """
    def __init__(self, config, db_conn: DatabaseConnection, embedder: MemoryEmbedder):
        self.config = config
        self.db = db_conn
        self.embedder = embedder

    def store_memory(self, memory_object: MemoryObject):
        self.db.connect()
        try:
            if memory_object.embedding is None:
                memory_object.embedding = self.embedder.get_embeddings(memory_object.content)

            self.db.cursor.execute("""
                INSERT INTO agent_memory (agent_id, cognitive_step, content, embedding)
                VALUES (%s, %s, %s, %s);
            """, (memory_object.agent_id, memory_object.cognitive_step, memory_object.content, memory_object.embedding))
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise e

if __name__ == "__main__":
    from config import load_config_from_yaml
    config = load_config_from_yaml("memory_config.yaml")
    db_conn = DatabaseConnection(config)
    embedder = MemoryEmbedder(config)
    memory_store = MarketMemory(config, db_conn, embedder)

    test_event = MemoryObject(
        agent_id="crypto_agent_123",
        cognitive_step="reflection",
        content="After analyzing Elon Musk's recent tweets about DOGE coin and the subsequent 25% price surge, I believe there's a strong social sentiment effect at play. His influence on crypto markets, especially meme coins, remains significant. I should adjust my trading strategy to account for these social media-driven price movements and potentially take profit on the current rally."
    )
    memory_store.store_memory(test_event)
    print(test_event.model_dump())
    print("Event stored successfully!")
