from sentence_transformers import SentenceTransformer, util
import os
import glob
import logging
from typing import List
import pickle
from pathlib import Path
import torch
from typing import Optional

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
        Path(EMBEDDING_CACHE).parent.mkdir(exist_ok=True)
        with open(EMBEDDING_CACHE, "wb") as f:
            pickle.dump((file_embeddings, file_texts), f)
        logger.info(f"Cached {len(file_texts)} embeddings")

except Exception as e:
    logger.critical("Vector store initialization failed", exc_info=True)
    raise


def search_similar_docs(
        query: str,
        image_embedding: Optional[List[float]] = None,
        top_k: int = 3
) -> List[str]:
    """Hybrid search combining text and image embeddings"""
    try:
        # Text embedding
        query_emb = model.encode(query, convert_to_tensor=True)

        # Combine with image embedding if available
        if image_embedding is not None:
            img_emb = torch.tensor(image_embedding).to(query_emb.device)
            query_emb = (query_emb + img_emb) / 2  # Simple average

        # Perform search
        hits = util.semantic_search(
            query_emb,
            torch.stack(file_embeddings),
            top_k=top_k
        )

        results = [file_texts[i['corpus_id']] for i in hits[0]]
        logger.debug(f"Retrieved {len(results)} documents for: {query[:50]}...")
        return results

    except Exception as e:
        logger.error("Document search failed", exc_info=True)
        return []