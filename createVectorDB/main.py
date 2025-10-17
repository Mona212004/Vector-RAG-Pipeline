import os
import warnings
from src.config_db import imdb_filepath, read_chunk_size
import pandas as pd
from src.csv_batch_to_documents import process_csv_batch_to_documents
from src.device import get_device
from src.generate_embeddings import get_embeddings
from src.config_db import model
from src.listofDict_to_listofTuples import transform_embeddings_listofDict_to_listofTuples
from src.insert_embeddings import insert_embeddings

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    warnings.filterwarnings('ignore')
    
    batch_count = 0
    batch_size = 0
    total_no_embedding_chunks = 0
    devices = get_device()
    #read_chunk_size
    for batch in pd.read_csv(imdb_filepath, chunksize=read_chunk_size, index_col=0, low_memory=False):
    #for batch in [pd.read_csv(imdb_filepath, nrows=5, index_col=0, low_memory=False)]: #for testing
        print(f"--- Start reading batch #{batch_count} ---")
        try:
            #chunk each batch of csv rows before feeding into embedding model
            chunking_per_batch_preEmbedding = process_csv_batch_to_documents(batch)
            print(f"--- Start chunking batch #{batch_count} ---")
            print(f"--- Number of chunks in batch #{batch_count}: {len(chunking_per_batch_preEmbedding)}")
        except:
            raise Exception(f"Chunking batch #{batch_count} failed.")
        
        try:
            #after chunking, feed into model
            print(f"Feed batch #{batch_count}'s {len(chunking_per_batch_preEmbedding)} chunks into model")
            embeddings, num_chunks = get_embeddings(chunking_per_batch_preEmbedding, model, devices)
        except:
            raise Exception(f"Failed to feed batch #{batch_count}'s {len(chunking_per_batch_preEmbedding)} chunks into model")
        #print(f"Generated embeddings for chunk{batch_count}")
        
        try:
            #transform embeddings from list of dictionaries to list of tuples
            embeddings_tuples = transform_embeddings_listofDict_to_listofTuples(embeddings, num_chunks)
            #insert multiple rows of data into postgresql from the tuples
            insert_embeddings(embeddings_tuples)
            print(f"Batch {batch_count} complete: {len(embeddings_tuples)} inserted.")
        except:
            raise Exception(f"Failed to insert batch #{batch_count}'s {len(chunking_per_batch_preEmbedding)} embeddings into db.")

        print()
        batch_count +=1
        batch_size += len(batch)
        total_no_embedding_chunks += len(embeddings_tuples)
    print(f"Total rows from all batches: {batch_size}") #726031
    print(f"Total number of embedding chunks from all batches: {total_no_embedding_chunks}")

if __name__ == '__main__':
    main()

#next time add code for measuring execution time of the script + logging output