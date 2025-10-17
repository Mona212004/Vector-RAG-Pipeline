#convert a list of user queries to vector embedding
from createVectorDB.src.config_db import model
from createVectorDB.src.device import get_device
from transformers import AutoTokenizer
import numpy as np
import os

# Suppress tokenizers parallelism warning (set before any imports/usage)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

#make each user query within the list stay within 512 tokens
def query_to_vectors(user_queries):
    #ensure user_query is not empty string
    if not isinstance(user_queries, list) or not user_queries: 
        raise ValueError("user_queries must be a non-empty list of strings.")
    if not all(isinstance(q, str) and q.strip() for q in user_queries):
        raise ValueError("All user_queries must be non-empty strings.")
    
    #token check
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
    max_tokens = 512
    instruction = "Represent this sentence for searching relevant passages: "
    for i, q in enumerate(user_queries):
        full_query = instruction+q
        tokens = tokenizer.encode(full_query, add_special_tokens=True)
        if len(tokens) > max_tokens:
            raise ValueError(f"Query {i+1} exceeds {max_tokens} tokens (length: {len(tokens)}).")
        
    devices = get_device()
    pool = None
    try:
        full_queries = [instruction+q for q in user_queries]
        #start multi-process pool (one-time setup for parallel encoding across devices)
        pool = model.start_multi_process_pool(devices)
        #encode with multi-process pool (distributes cross gpus/processes)
        q_embeddings = model.encode(
            full_queries, 
            pool = pool,
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
            show_progress_bar=True)
        #validate embeddings
        if q_embeddings is None or len(q_embeddings)==0 or np.any(np.isnan(q_embeddings)):
            raise RuntimeError("Embedding generation produced invalid results.")
        print(f"Successfully generated vector embeddings for {len(user_queries)} user queries.\n")
        print(f"Shape of generated query embeddings : {q_embeddings.shape}.")  
        return q_embeddings 
    except Exception as error:
        raise RuntimeError(f"Embedding generation failed: {str(error)}") from error
    finally:
        if pool is not None:
            #stop the pool (cleanup)
            model.stop_multi_process_pool(pool)
    
if __name__=='__main__':
    query_to_vectors(['Hello', 'Give me a movie', 'What genres do we have?']) #works