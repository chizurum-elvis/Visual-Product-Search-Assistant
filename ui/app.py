import gradio as gr
import requests
import tempfile
import os
import logging
from datetime import datetime
import time
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisualProductUI")

API_URL = "http://localhost:8000/ask"
TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class ServiceMonitor:
    """Track API availability and performance"""

    def __init__(self):
        self.last_success = None
        self.last_failure = None
        self.consecutive_failures = 0

    def record_success(self):
        self.last_success = time.time()
        self.consecutive_failures = 0

    def record_failure(self):
        self.last_failure = time.time()
        self.consecutive_failures += 1

    def is_healthy(self) -> bool:
        if self.consecutive_failures >= 3:
            return False
        return True


monitor = ServiceMonitor()


def get_answer(image, question) -> Tuple[str, Optional[str]]:
    """Enhanced UI handler with:
    - Secure temp file handling
    - Retry logic
    - Service monitoring
    - Rich error feedback
    Returns tuple: (answer, error_type)
    """
    if not image or not question.strip():
        return "Please provide both an image and a valid question.", "validation_error"

    temp_path = None
    last_exception = None

    try:
        # Create secure temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            image.save(temp.name)
            temp_path = temp.name

        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                with open(temp_path, "rb") as f:
                    files = {"file": (os.path.basename(temp_path), f, "image/jpeg")}
                    data = {"question": question}

                    response = requests.post(
                        API_URL,
                        files=files,
                        data=data,
                        timeout=TIMEOUT
                    )
                    response.raise_for_status()

                    monitor.record_success()
                    return response.json().get("answer", "No answer returned"), None

            except requests.exceptions.RequestException as e:
                last_exception = e
                monitor.record_failure()

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

                logger.error(f"API request failed after {MAX_RETRIES} attempts: {str(e)}")

                if isinstance(e, requests.exceptions.Timeout):
                    return "Request timed out. Please try again with a simpler question.", "timeout"
                elif isinstance(e, requests.exceptions.HTTPError):
                    if response.status_code == 503:
                        return "Our systems are currently overloaded. Please try again in a few minutes.", "service_unavailable"
                    return f"Technical error (HTTP {response.status_code}). Contact support.", "http_error"
                else:
                    return "Service temporarily unavailable. Please try again later.", "connection_error"

    except Exception as e:
        logger.error(f"UI processing failed: {str(e)}", exc_info=True)
        return "An unexpected error occurred. Our team has been notified.", "processing_error"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Temp file cleanup failed: {str(e)}")


def enhanced_interface_handler(image, question):
    """Wrapper to handle Gradio's output expectations"""
    answer, error_type = get_answer(image, question)

    if error_type:
        # Apply special formatting for errors
        return f"⚠️ {answer}" if not answer.startswith("⚠️") else answer
    return answer


# Enhanced UI with status awareness
with gr.Blocks(theme=gr.themes.Soft(), title="Visual Product Support") as demo:
    gr.Markdown("""
    # 🔍 Visual Product Support Assistant
    Upload an image of your product and ask questions. The assistant will use visual 
    understanding and product manuals to provide accurate support.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Product Image")
            question_input = gr.Textbox(label="Your Question",
                                        placeholder="Ask about this product...")
            submit_btn = gr.Button("Get Support Answer", variant="primary")

            # Status indicator
            status = gr.Textbox(label="System Status",
                                value="✅ Operational" if monitor.is_healthy() else "⚠️ Degraded",
                                interactive=False)

        with gr.Column():
            output = gr.Textbox(label="Support Answer", interactive=False)
            retry_btn = gr.Button("Retry", visible=False)

    # Interactive components
    submit_btn.click(
        fn=enhanced_interface_handler,
        inputs=[image_input, question_input],
        outputs=output
    )


    def toggle_retry(answer):
        return gr.Button(visible="⚠️" in answer)


    output.change(
        fn=toggle_retry,
        inputs=output,
        outputs=retry_btn
    )

    retry_btn.click(
        fn=lambda: "Retrying... Please wait",
        outputs=output
    ).then(
        fn=enhanced_interface_handler,
        inputs=[image_input, question_input],
        outputs=output
    )

if __name__ == "__main__":
    logger.info("Launching enhanced Gradio interface")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        # favicon_path="./favicon.ico"  # Add a favicon if available
    )