from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from typing import Union, Optional, List, Any, Dict
import logging
from pathlib import Path
import time
from datetime import datetime
import os
from google.api_core import exceptions as google_exceptions
import base64
from time import sleep
import random
import io


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class GeminiEncoder:
    def __init__(self, api_key:str=GOOGLE_API_KEY, model_name:str="gemini-2.5-pro", log_dir:str="./logs",
                 log_level:int=logging.INFO, max_log_files:int=7, max_retries:int=5, initial_backoff:float=5.0):
        """
                Initialize the Gemini encoder with API configuration and logging.

                Args:
                    api_key: Google Gemini API key
                    model_name: Gemini model name ('gemini-2.5-pro' for multimodal)
                    log_dir: Directory to store log files
                    log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
                    max_log_files: Maximum number of log files to keep
                    max_retries: Maximum number of retries for API calls
                    initial_backoff: Initial backoff time in seconds for retries
        """
        self._setup_logging(log_dir, log_level, max_log_files)
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=self.model_name)
            self.logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise

    def _setup_logging(self, log_dir:str, log_level:int, max_log_files:int):
        if max_log_files < 1:
            raise ValueError("max_log_files must be ≥ 1")
        """Configure logging to both consoles and file with rotation."""
        self.logger = logging.getLogger("Image_Encoder")
        self.logger.setLevel(log_level)

        os.makedirs(log_dir, exist_ok=True)

        # Set up log file naming with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = Path(log_dir) / f"gemini_encoder_{timestamp}.log"

        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler wth rotation
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Clean up old log files
        self._cleanup_old_logs(log_dir, max_log_files)

    def _cleanup_old_logs(self, log_dir:str, max_log_files:int):
        """Keep only the most recent log files."""
        try:
            log_files = sorted(Path(log_dir).glob("gemini_encoder_*.log"), key=os.path.getmtime, reverse=True)

            for old_log in log_files[max_log_files:]:
                old_log.unlink(missing_ok=False)
                self.logger.debug(f"Removed old log file: {old_log}")
        except Exception as e:
            self.logger.error(f"Failed to remove old log file: {str(e)}")
            raise

    def _load_image(self,image_path:Union[str, Path]) -> Image.Image:
        """Load and validate image with error handling."""
        try:
            start_time = time.time()
            image = Image.open(image_path).convert("RGB")
            load_time = time.time() - start_time
            self.logger.debug(f"Loaded image {image_path} in {load_time:.2f} seconds")
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def _image_to_base64(self, image:Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _handle_api_error(self, error:Exception, attempts:int) -> bool:
        """
                Handle API errors and determine if we should retry.

                Args:
                    error: The exception that occurred
                    attempts: Current attempt number

                Returns:
                    bool: True if we should retry, False otherwise
        """
        if isinstance(error, google_exceptions.ResourceExhausted):
            self.logger.warning(f"Rate limit exceeded (attempt {attempts}). Will retry.")
            return True
        elif isinstance(error, google_exceptions.ServiceUnavailable):
            self.logger.warning(f"Service unavailable (attempt {attempts}). Will retry.")
            return True
        elif isinstance(error, google_exceptions.InternalServerError):
            self.logger.warning(f"Internal server error (attempt {attempts}). Will retry.")
            return True
        elif isinstance(error, google_exceptions.TooManyRequests):
            self.logger.warning(f"Too many requests (attempt {attempts}). Will retry.")
            return True
        elif isinstance(error, google_exceptions.GoogleAPIError):
            self.logger.warning(f"Google API error: {str(error)}")
            return False
        else:
            self.logger.error(f"Unexpected error: {str(error)}")
            return False

    def _call_with_retry(self, func, *args, **kwargs):
        """
                Execute API call with retry logic for rate limits and temporary failures.

                Args:
                    func: The function to call
                    *args: Positional arguments for the function
                    **kwargs: Keyword arguments for the function

                Returns:
                    The result of the function call

                Raises:
                    Exception: If all retries fail
        """
        last_exception = None
        for attempt in range(1, self.max_retries+1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._handle_api_error(e, attempt):
                    break
                sleep_time = self.initial_backoff * (2 ** (attempt - 1) ) + random.uniform(0,1)
                self.logger.info(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                sleep(sleep_time)
        self.logger.error(f"All {self.max_retries} attempts failed.")
        raise last_exception if last_exception else Exception("Unknown error occurred")

    def get_image_caption(self, image_path:Union[str, Path], **generation_kwargs) -> str:
        """
                Generate caption for an image using Gemini with retry logic.

                Args:
                    image_path: Path to the image file
                    **generation_kwargs: Additional arguments for text generation

                Returns:
                    str: Generated caption

                Raises:
                    Exception: If caption generation fails after all retries
        """
        self.logger.info(f"Generating caption for: {image_path}")
        start_time = time.time()

        try:
            image = self._load_image(image_path)
            default_kwargs = {
                "max_output_tokens" : 2048,
                "temperature" : 0.4,
                "top_k" : 0.9
            }
            generation_kwargs = {**default_kwargs, **generation_kwargs}

            self.logger.debug(f"Using generation kwargs: {generation_kwargs}")

            # Generate content with retry logic
            def _generate_content():
                response = self.model.generate_content(
                    ["Describe this image in detail", image],
                    generation_config = genai.types.GenerationConfig(**generation_kwargs)
                )
                return response.text
            caption = self._call_with_retry(_generate_content)

            # Log performance metrics
            proc_time = time.time() - start_time
            self.logger.info(
                f"Generated caption for {image_path} in {proc_time:.2f}s. "
                f"Caption: '{caption}'"
            )

            return caption
        except Exception as e:
            self.logger.error(f"Failed to generate caption for {image_path}: {str(e)}")
            raise

    def get_image_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """
        Get image embeddings using Gemini with retry logic.

        Args:
            image_path: Path to the image file

        Returns:
            List[float]: Image embeddings

        Raises:
            Exception: If embedding generation fails after all retries
        """
        self.logger.info(f"Processing image for embeddings: {image_path}")
        start_time = time.time()

        try:
            image = self._load_image(image_path)

            # Generate technical description with retry logic
            def _generate_embedding_description():
                response = self.model.generate_content(
                    ["Describe this image in extremely detailed technical terms suitable for vector embedding generation",
                     image],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2048,
                        temperature=0.5
                    )
                )
                return response.text.split()

            embedding = self._call_with_retry(_generate_embedding_description)

            # Log performance metrics
            proc_time = time.time() - start_time
            self.logger.info(
                f"Generated embeddings for {image_path} in {proc_time:.2f}s. "
                f"Embedding length: {len(embedding)}"
            )

            return embedding

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {image_path}: {str(e)}")
            raise

    def batch_process(self, image_paths: List[Union[str, Path]], task: str = "caption", **kwargs) -> List:
        """
        Process multiple images in a batch.

        Args:
            image_paths: List of image paths
            task: "embedding" or "caption"
            **kwargs: Additional arguments for the task

        Returns:
            list: Results for each image
        """
        self.logger.info(f"Starting batch processing of {len(image_paths)} images for {task}")
        start_time = time.time()

        try:
            if task == "embedding":
                results = [self.get_image_embedding(path) for path in image_paths]
            elif task == "caption":
                results = [self.get_image_caption(path, **kwargs) for path in image_paths]
            else:
                raise ValueError(f"Unknown task: {task}. Use 'embedding' or 'caption'")

            total_time = time.time() - start_time
            self.logger.info(
                f"Completed batch processing of {len(image_paths)} images in {total_time:.2f}s. "
                f"Average time per image: {total_time / len(image_paths):.2f}s"
            )

            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

