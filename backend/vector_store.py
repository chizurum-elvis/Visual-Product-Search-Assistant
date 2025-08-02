from sentence_transformers import SentenceTransformer, util
import os
import glob
import logging
from typing import List
import pickle
from pathlib import Path
import torch
from typing import Optional
import faiss
import numpy as np

logger = logging.getLogger("VectorStore")

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE = "data/embeddings_cache.pkl"
DOC_DIR = "data/manuals"

# Initialize with caching
try:
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Loaded embedding model: {MODEL_NAME}")

    # Try loading cached embeddings
    if os.path.exists(EMBEDDING_CACHE):
        with open(EMBEDDING_CACHE, "rb") as f:
            file_embeddings, file_texts = pickle.load(f)
        logger.info(f"Loaded {len(file_texts)} cached embeddings")
    else:
        file_embeddings = []
        file_texts = []
        for filepath in glob.glob(os.path.join(DOC_DIR, "*.txt")):
            with open(filepath, "r", encoding='utf-8') as f:
                text = f.read()
                emb = model.encode(text, convert_to_tensor=True)
                file_embeddings.append(emb)
                file_texts.append(text)

        # Save cache
        Path(EMBEDDING_CACHE).parent.mkdir(exist_ok=True, parents=True)
        with open(EMBEDDING_CACHE, "wb") as f:
            pickle.dump((file_embeddings, file_texts),f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cached {len(file_texts)} embeddings")

except Exception as e:
    logger.critical("Vector store initialization failed", exc_info=True)
    raise


if not file_embeddings:
    raise ValueError("No file embeddings found")

embedding_dim = file_embeddings[0].shape[0]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(file_embeddings)

def search_similar_docs(query:str, image_embedding:Optional[List[float]]=None, top_k:int=3) -> List[str]:
    """Hybrid search combining text and image embeddings using FAISS"""
    try:
        query_embedding = model.encode(query)
        query_embedding = np.array(query_embedding)

        if image_embedding is not None:
            image_embedding = np.array(image_embedding)
            query_embedding = (query_embedding + image_embedding) / 2
        query_embedding = np.expand_dims(query_embedding.astype('float32'), axis=0)

        distances, indices = faiss_index.search(query_embedding, top_k)
        results = [file_texts[i] for i in indices[0]]
        logger.debug(f"Retrieved {len(results)} documents for: {query[:50]}...")
        return results
    except Exception as e:
        logger.critical("Failed to retrieve results", exc_info=True)
        return []