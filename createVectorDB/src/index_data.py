#index the embedding table database
#index the embeddings: pgvectorscale offers a more cost-efficient and powerful index type for pgvector data: StreamingDiskANN
# Create an index on the data for faster retrieval

from config_conn import load_config
import psycopg2
from pgvector.psycopg2 import register_vector
from alive_progress import alive_bar

def index_embeddings_table():
    params = load_config()
    create_index = "create index embedding_idx on embeddings_table using diskann (embedding vector_cosine_ops);"
    try:
        with psycopg2.connect(**params) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("DROP INDEX IF EXISTS embedding_idx;")
                
                # Simple spinner with elapsed time
                with alive_bar(
                    title='Indexing embeddings',
                    bar='halloween',
                    unknown='waves',  # Shows waves moving through the bar
                    elapsed=True,
                    stats=False,
                    monitor=False
                ) as bar:
                    cur.execute(create_index)
                print("Success:", cur.statusmessage)
            
            conn.commit()
            print(f"Index created using diskann on embeddings_table.embedding")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
        
if __name__=="__main__":
    index_embeddings_table()