import os
import warnings
from src.config_db import imdb_filepath
import pandas as pd
from src.csv_batch_to_documents import process_csv_batch_to_documents
from src.device import get_device
from src.generate_embeddings import get_embeddings
from src.config_db import model
from src.listofDict_to_listofTuples import transform_embeddings_listofDict_to_listofTuples
from src.insert_embeddings import insert_embeddings
from pgvector.psycopg2 import register_vector
from src.config_conn import load_config
import psycopg2
from UserQuery.query_embedding import query_to_vectors
import numpy as np
import time

'''    
def create_test_table():
    params = load_config()
    create = """create table test_table (
        id SERIAL PRIMARY KEY,
        tconst text,
        chunk_id integer,
        chunk_type text,
        page_content text,
        embedding vector(768)
        );"""
    try: 
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                register_vector(conn)
                cur.execute(create)
                # Add unique constraint safely (drop if exists, then add)
                cur.execute("ALTER TABLE test_table DROP CONSTRAINT IF EXISTS unique_tconst_chunk_test;")
                cur.execute("ALTER TABLE test_table ADD CONSTRAINT unique_tconst_chunk_test UNIQUE (tconst, chunk_id);")
                conn.commit()
        print("test_table successfully created")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

def insert_embeddings_test(embeddings_tuples):
    params = load_config()
    insert = "insert into test_table(tconst, chunk_id, chunk_type, page_content, embedding) values(%s, %s, %s, %s, %s) ON CONFLICT (tconst, chunk_id) DO NOTHING"
    try:
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                register_vector(conn)
                cur.executemany(insert, embeddings_tuples)
            conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error: {error}")
        conn.rollback()
        
def verify_insertion():
    """Quick check to confirm data was inserted."""
    params = load_config()
    try:
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM test_table;")
                count = cur.fetchone()[0]
                print(f"Verification: {count} rows inserted into test_table.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Verification error: {error}")

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    warnings.filterwarnings('ignore')
    
    batch_count = 0
    batch_size = 0
    total_no_embedding_chunks = 0
    devices = get_device()
    
    # Read single test batch as DataFrame (not a list)
    batch = pd.read_csv(imdb_filepath, nrows=5, index_col=0, low_memory=False)
    print(f"--- Start reading batch #{batch_count} ---")
    try:
        # Chunk each batch of csv rows before feeding into embedding model
        chunking_per_batch_preEmbedding = process_csv_batch_to_documents(batch)
        print(f"--- Start chunking batch #{batch_count} ---")
        print(f"--- Number of chunks in batch #{batch_count}: {len(chunking_per_batch_preEmbedding)}")
    except Exception as e:
        raise Exception(f"Chunking batch #{batch_count} failed: {e}")
        
    try:
        # After chunking, feed into model
        print(f"Feed batch #{batch_count}'s {len(chunking_per_batch_preEmbedding)} chunks into model")
        embeddings, num_chunks = get_embeddings(chunking_per_batch_preEmbedding, model, devices)
        print(f"Generated embeddings for batch #{batch_count}")
    except Exception as e:
        raise Exception(f"Failed to feed batch #{batch_count}'s {len(chunking_per_batch_preEmbedding)} chunks into model: {e}")
        
    try:
        # Transform embeddings from list of dictionaries to list of tuples
        embeddings_tuples = transform_embeddings_listofDict_to_listofTuples(embeddings, num_chunks)
        # Insert multiple rows of data into postgresql from the tuples
        insert_embeddings_test(embeddings_tuples)
        print(f"Batch {batch_count} complete: {len(embeddings_tuples)} inserted.")
    except Exception as e:
        raise Exception(f"Failed to insert batch #{batch_count}'s {num_chunks} embeddings into db: {e}")

    batch_count = 1  # Single batch
    batch_size = len(batch)  # Actual row count (5)
    total_no_embedding_chunks = len(embeddings_tuples)
    print()
    print(f"Total rows from all batches: {batch_size}") 
    print(f"Total number of embedding chunks from all batches: {total_no_embedding_chunks}")
    
    verify_insertion()  # Optional: Confirm insertion
 

#index the embeddings: pgvectorscale offers a more cost-efficient and powerful index type for pgvector data: StreamingDiskANN
# Create an index on the data for faster retrieval
def index_test_table_embeddings():
    params = load_config()
    create_index = "create index IF NOT EXISTS test_embedding_idx on test_table using diskann (embedding vector_cosine_ops);"
    try:
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                register_vector(conn)
                cur.execute("DROP INDEX IF EXISTS test_embedding_idx;")
                cur.execute(create_index)
            conn.commit()
            print(f"Index created using diskann on embedding column in test_table.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
'''
#test query
#Helper function: Get top 3 most similar documents from the embeddings_table 
#call function instead
def iter_row(cursor, size):
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        for row in rows:
            yield row
    
def gettop3test(query_embedding):
    movies = []
    query_embedding = np.array(query_embedding)
    params = load_config()
    try:
        with psycopg2.connect(**params) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute('select * from get_top3_test(%s)', (query_embedding,))
                for row in iter_row(cur, 3):
                    print(row)
                    movies.append(row)
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error: {error}")
    finally:
        return movies
                

if __name__ == '__main__':
    #create_test_table()
    #main()
    #index_test_table_embeddings()
    input = ["Give me movies with action genres.", "Give korean movies with highest ratings."]
    query_embedding = query_to_vectors(input)
    start_time = time.time()
    print(f"Start_time: {start_time}")
    for i, query in enumerate(query_embedding):
        print(f"--- Results for Query: '{input[i]}' ---")
        movies = gettop3test(query)
    print("Execution time: --- %s seconds ---" % (time.time() - start_time))