"""
Welcome to SQLmanticCHUNKER!

Simple i/o tool for adding embeddings to our SQL database.

Usage:

CLI:
    Process:  python SQLmanticCHUNKER.py process input.txt
    Search:   python SQLmanticCHUNKER.py search --query "your query" --top-k 5

Python:
    from sqlmantic_chunker import SQLmanticChunker
    
    # Initialize and process
    chunker = SQLmanticChunker()
    with open('input.txt', 'r') as f:
        text = f.read()
    chunks = chunker.process(text)
    doc_id = chunker.save_to_db(chunks, text)
    
    # Search
    results = chunker.search_db("your query", top_k=5)

    
TODO :
1. Add a config file
2. Add chunk overlapping maybe, though it's not needed and gets in the way of regex rules... I have thoughts esp now we are augemting with the index
3. VALIDATION! Currently it's just printing events to the console, needs pydantic validation
4. Add logging
5. Regex is currently v simple, type and domain sensitive regex models will be needed when handling different platforms and formats
6. indexing type is currently `ivfflat`, but we'll implement on this for hybrid indexing
- might be always trying to recreate the table rather than checking if it exists first, need to fix

"""

from typing import List, Optional
from typing import Literal
from pydantic import BaseModel, Field
import argparse
import re
import requests
import time
import sys
import os
import psycopg2
from tqdm import tqdm
from colorama import init, Fore, Style, Back
from datetime import datetime

# Configuration

class ServiceConfig(BaseModel):
    """Combined configuration for database, embedding service, and search"""
    # Database settings
    dbname: str = Field(default="my_first_table")
    user: str = Field(default="db_user")
    password: str = Field(default="password")
    host: str = Field(default="localhost")
    port: str = Field(default="0000")
    index_method: str = Field(default="ivfflat") 
    lists: int = Field(default=100)
    
    # Embedding service settings
    embedding_api_url: str = Field(default="http://0.0.0.0:8080/embed")
    model: str = Field(default="jinaai/jina-embeddings-v2-base-en")
    batch_size: int = Field(default=32)
    timeout: int = Field(default=10)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    # Chunking settings
    min_size: int = Field(default=64)
    max_size: int = Field(default=256)
    vector_dim: int = Field(default=768)
    
    # Search settings
    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.7)
    context_window: int = Field(default=512)

# Models

class Chunk(BaseModel):
    """A semantic chunk of text with its embedding and metadata"""
    text: str
    start: int          # Start position in original text
    end: int            # End position in original text
    embedding: Optional[List[float]] = None
    document_id: Optional[int] = None
    
class SearchResult(BaseModel):
    """A search result with semantic similarity score"""
    chunk: Chunk
    similarity: float
    context: str = ""   # Surrounding context from original text

class Document(BaseModel):
    """Database document model"""
    id: Optional[int] = None
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Let's go!

class SQLmanticChunker:
    def __init__(self, config_path: str = None):
        # Initialize with default config
        self.config = ServiceConfig()
        
        # Override with environment variables for DB
        if any(k in os.environ for k in ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']):
            self.config = ServiceConfig(
                dbname=os.environ.get('DB_NAME', self.config.dbname),
                user=os.environ.get('DB_USER', self.config.user),
                password=os.environ.get('DB_PASSWORD', self.config.password),
                host=os.environ.get('DB_HOST', self.config.host),
                port=os.environ.get('DB_PORT', self.config.port)
            )

        self.full_text = ""  # Store full text for context retrieval
        self.conn = None
        self.cursor = None

    def get_embedding(self, text: str | List[str]) -> List[float] | List[List[float]]:
        """Get embeddings with batching and retries"""
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        all_embeddings = []
        
        # Create progress bar for batches
        batches = range(0, len(texts), self.config.batch_size)
        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in batches:
                batch = texts[i:i + self.config.batch_size]
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
                        pbar.update(len(batch))
                        break
                    except requests.exceptions.ConnectionError:
                        print(f"\nError: Cannot connect to embedding server at {self.config.embedding_api_url}")
                        raise
                    except Exception as e:
                        if attempt == self.config.retry_attempts - 1:
                            print(f"\nFailed to get embedding: {str(e)}")
                            raise
                        time.sleep(self.config.retry_delay)
        
        return all_embeddings[0] if single_input else all_embeddings

    def semantic_split(self, text: str) -> List[Chunk]:
        """Split text into semantic chunks"""
        self.full_text = text
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        chunks = []
        current_pos = 0
        
        splits = [
            r'(?<=\n\n)',                  # Double newlines
            r'(?<=\.)\s+(?=[A-Z])',        # Sentences
            r'(?<=\n)\s*[-•\*]\s+',        # List items
            r'(?<=\n)\s*\d+\.\s+',         # Numbered items
            r'(?<=,)\s+',                  # Commas
            r'(?<=;)\s+',                  # Semicolons
            r'\s+(?=and|but|or)\s+',       # Conjunctions
            r'\s+(?=because|however)\s+'    # Transitional phrases
        ]
        
        pattern = '|'.join(splits)
        segments = re.split(pattern, text)
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            if current_length + len(segment) > self.config.max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos
            
            current_chunk.append(segment)
            current_length += len(segment)
            current_pos += len(segment) + 1
            
            if current_length >= self.config.min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_start + len(chunk_text)
            ))
        
        return chunks

    def get_context(self, chunk: Chunk) -> str:
        """Get surrounding context for a chunk"""
        start = max(0, chunk.start - self.config.context_window)
        end = min(len(self.full_text), chunk.end + self.config.context_window)
        
        context = self.full_text[start:end].strip()
        if start > 0:
            context = f"...{context}"
        if end < len(self.full_text):
            context = f"{context}..."
            
        return context

    def save_to_db(self, chunks: List[Chunk], document_text: str) -> int:
        """Save document and chunks to database"""
        if not self.conn:
            self.connect_db()
            
        try:
            # Save document
            document = Document(content=document_text)
            self.cursor.execute(
                "INSERT INTO documents (content) VALUES (%s) RETURNING id;",
                (document.content,)
            )
            document_id = self.cursor.fetchone()[0]
            
            # Save chunks
            for chunk in chunks:
                if chunk.embedding:
                    chunk.document_id = document_id
                    self.cursor.execute("""
                        INSERT INTO chunks 
                        (document_id, text, start_pos, end_pos, embedding)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (
                        chunk.document_id,
                        chunk.text,
                        chunk.start,
                        chunk.end,
                        chunk.embedding
                    ))
            
            self.conn.commit()
            return document_id
            
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"{Fore.RED}Error saving to database: {str(e)}{Style.RESET_ALL}")
            raise

    def search_db(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for similar chunks in database"""
        try:
            if not self.conn:
                self.connect_db()
                
            query_embedding = self.get_embedding(query)
            
            self.cursor.execute("""
                WITH ranked_chunks AS (
                    SELECT DISTINCT ON (c.text)
                        c.id,
                        c.text,
                        c.start_pos,
                        c.end_pos,
                        d.content,
                        (1 - (c.embedding <=> %s::vector)) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE (1 - (c.embedding <=> %s::vector)) >= %s
                    ORDER BY c.text, similarity DESC
                )
                SELECT * FROM ranked_chunks
                ORDER BY similarity DESC
                LIMIT %s;
            """, (query_embedding, query_embedding, self.config.similarity_threshold, top_k))
            
            results = []
            for row in self.cursor.fetchall():
                chunk = Chunk(
                    text=row[1],
                    start=row[2],
                    end=row[3],
                )
                self.full_text = row[4]
                results.append(SearchResult(
                    chunk=chunk,
                    similarity=row[5],
                    context=self.get_context(chunk)
                ))
            
            return results
            
        except Exception as e:
            print(f"{Fore.RED}Database search error: {str(e)}{Style.RESET_ALL}")
            raise

    def connect_db(self):
        """Establish database connection and initialize tables"""
        temp_conn = None
        temp_cur = None
        
        try:
            print(f"{Fore.CYAN}Attempting to connect to PostgreSQL...{Style.RESET_ALL}")
            if len(sys.argv) > 1 and sys.argv[1] == 'process':
                temp_conn = psycopg2.connect(
                    dbname='postgres',
                    user=self.config.user,
                    password=self.config.password,
                    host=self.config.host,
                    port=self.config.port
                )
                temp_conn.autocommit = True
                temp_cur = temp_conn.cursor()
                
                try:
                    temp_cur.execute(f"CREATE DATABASE {self.config.dbname}")
                    print(f"{Fore.GREEN}Created database: {self.config.dbname}{Style.RESET_ALL}")
                except psycopg2.errors.DuplicateDatabase:
                    print(f"{Fore.YELLOW}Database {self.config.dbname} exists with tables:{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}→ documents (id, content, created_at)")
                    print(f"→ chunks (id, document_id, text, start_pos, end_pos, embedding){Style.RESET_ALL}")
        
        except psycopg2.OperationalError as e:
            print(f"{Fore.RED}Failed to connect to PostgreSQL server:{Style.RESET_ALL}")
            print(f"{Fore.RED}→ Host: {self.config.host}")
            print(f"→ Port: {self.config.port}")
            print(f"→ Error: {str(e)}{Style.RESET_ALL}")
            raise
        
        finally:
            if temp_cur:
                temp_cur.close()
            if temp_conn:
                temp_conn.close()

        # Connect to target database
        try:
            print(f"{Fore.CYAN}Connecting to database: {self.config.dbname}{Style.RESET_ALL}")
            self.conn = psycopg2.connect(
                dbname=self.config.dbname,
                user=self.config.user,
                password=self.config.password,
                host=self.config.host,
                port=self.config.port
            )
            self.cursor = self.conn.cursor()
            print(f"{Fore.GREEN}Successfully connected to database{Style.RESET_ALL}")
            
        except psycopg2.OperationalError as e:
            print(f"{Fore.RED}Failed to connect to database {self.config.dbname}:{Style.RESET_ALL}")
            print(f"{Fore.RED}→ Error: {str(e)}{Style.RESET_ALL}")
            self.conn = None
            self.cursor = None
            raise

    def init_db(self):
        """Initialize database and tables with vector support"""
        if not self.conn:
            self.connect_db()
            return
            
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Fix: Use direct SQL identifiers without parameters for table structure
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                text TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                embedding vector({self.config.vector_dim}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index in separate statement
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {self.config.lists});
        """)
        
        self.conn.commit()

    def process(self, text: str) -> List[Chunk]:
        """Process text into semantically embedded chunks"""
        if not self.conn:
            self.connect_db()
            # self.init_db()

        print(f"{Fore.CYAN}Splitting text into semantic chunks...{Style.RESET_ALL}")
        chunks = self.semantic_split(text)
        print(f"{Fore.GREEN}Created {len(chunks)} chunks{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Getting embeddings...{Style.RESET_ALL}")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embedding(chunk_texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='SQLmantic text chunking and search')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and chunk a document')
    process_parser.add_argument('input_file', help='Input text file to chunk')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search through processed chunks')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=3, help='Number of results')
    search_parser.add_argument('--use-db', action='store_true', default=True, help='Search in database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    chunker = SQLmanticChunker()
    
    try:
        if args.command == 'process':
            # Initialize database tables for processing
            chunker.connect_db()
            chunker.init_db()  # Move it here
            
            # Read and process text
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = chunker.process(text)
            
            # Save to database
            doc_id = chunker.save_to_db(chunks, text)
            print(f"\n{Fore.GREEN}Saved document to database with ID: {doc_id}{Style.RESET_ALL}")
            
        elif args.command == 'search':
            # Search in database - only needs connection, not initialization
            chunker.connect_db()  # Only connect, don't initialize
            results = chunker.search_db(args.query, args.top_k)
            
            if not results:
                print(f"\n{Fore.RED}No semantic matches found{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}Semantic Search Results:{Style.RESET_ALL}")
                for i, result in enumerate(results, 1):
                    print(f"\n{Fore.BLUE}Match {i}:{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Similarity:{Style.RESET_ALL} {result.similarity:.3f}")
                    print(f"{Fore.YELLOW}Position:{Style.RESET_ALL} {result.chunk.start}-{result.chunk.end}")
                    
                    print(f"\n{Fore.CYAN}Chunk Text:{Style.RESET_ALL}")
                    print(f"{Back.BLACK}{Fore.WHITE}{result.chunk.text}{Style.RESET_ALL}")
                    
                    print(f"\n{Fore.CYAN}Context:{Style.RESET_ALL}")
                    print(f"{Back.BLACK}{Fore.LIGHTBLACK_EX}{result.context}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    finally:
        chunker.close()

if __name__ == "__main__":
    main()
