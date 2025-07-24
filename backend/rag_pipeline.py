from image_encoder import GeminiEncoder
from ocr import extract_text_from_image
from text_encoder import TextEncoder
from vector_store import search_similar_docs
import logging
from typing import Optional
import time

logger = logging.getLogger("RAG_Pipeline")


def answer_question_with_image_and_text(
        image_path: str,
        question: str,
        max_context_length: int = 4000
) -> Optional[str]:
    """Orchestrates the multimodal RAG pipeline with monitoring"""
    start_time = time.time()
    metrics = {}

    try:
        # Stage 1: Image Processing
        img_start = time.time()
        caption = GeminiEncoder.get_image_caption(image_path)
        image_embedding = GeminiEncoder.get_image_embedding(image_path)
        ocr_text = extract_text_from_image(image_path)
        metrics["image_processing"] = time.time() - img_start

        # Stage 2: Document Retrieval
        retrieval_start = time.time()
        relevant_docs = search_similar_docs(question, image_embedding)
        metrics["retrieval"] = time.time() - retrieval_start

        # Context Assembly with length check
        context = "\n".join(
            [d[:1000] for d in relevant_docs[:3]] +  # Limit doc length
            [caption, ocr_text]
        )[:max_context_length]  # Hard cap

        # Stage 3: LLM Generation
        gen_start = time.time()
        answer = TextEncoder.generate_answer_with_gpt4(question, context)
        metrics["generation"] = time.time() - gen_start

        metrics["total"] = time.time() - start_time
        logger.info(
            "Pipeline completed",
            extra={"metrics": metrics, "question": question[:100]}
        )
        return answer

    except Exception as e:
        logger.error(
            "Pipeline failed",
            exc_info=True,
            extra={"image_path": image_path, "question": question}
        )
        raise