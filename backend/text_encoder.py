import openai
import os
import logging
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Optional

load_dotenv()

logger = logging.getLogger("TextEncoder")

# Configure with environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)
class TextEncoder:
    DEFAULT_MODEL = "gpt-4-1106-preview"  # Latest version


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.APIError
            )
        ),
        reraise=True
    )
    def generate_answer_with_gpt4(
            question: str,
            context: str,
            model: str = DEFAULT_MODEL,
            max_tokens: int = 1000,
            temperature: float = 0.3
    ) -> Optional[str]:
        """Robust LLM answering with cost tracking"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a product support specialist. "
                               "Answer concisely using only the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]

            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Log token usage
            usage = response.usage
            logger.info(
                "LLM response generated",
                extra={
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "question_length": len(question),
                    "context_length": len(context)
                }
            )

            return response.choices[0].message.content.strip()

        except openai.error.InvalidRequestError as e:
            logger.error(
                "Invalid request to OpenAI",
                extra={"error": str(e), "context_length": len(context)}
            )
            return "I couldn't process that request. The context may be too large."
        except Exception as e:
            logger.error("LLM generation failed", exc_info=True)
            raise