from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from rag_pipeline import answer_question_with_image_and_text
import shutil
import os
import uuid
import tempfile
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from PIL import Image
from typing import Optional, Dict
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)


class AppState:
    def __init__(self):
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.is_healthy: bool = True
        self.active_models: Dict[str, bool] = {
            "visual_understanding": True,
            "text_processing": True
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Properly initialized lifespan with application state"""
    # Initialize application state
    app.state = AppState()
    logger.info("Starting Multimodal RAG API")

    # Warmup services
    try:
        # Test model connectivity
        test_file = os.path.join(tempfile.gettempdir(), "test.jpg")
        Image.new('RGB', (100, 100), color='red').save(test_file)
        answer_question_with_image_and_text(test_file, "test question")
        os.remove(test_file)
    except Exception as e:
        app.state.is_healthy = False
        app.state.active_models["visual_understanding"] = False
        logger.error(f"Startup test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Service initialization failed"
        )

    yield

    # Cleanup
    logger.info(f"Shutting down API after {app.state.request_count} requests")


app = FastAPI(
    title="Visual Product Support Assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

app.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# Exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please try again later."},
        headers={"Retry-After": str(exc.detail)}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def validate_image(file_path: str) -> bool:
    """Validate the image is actually an image and not corrupted"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        logger.warning(f"Invalid image file: {str(e)}")
        return False


async def save_upload_file(upload_file: UploadFile, directory: str) -> str:
    """Securely save uploaded file with validation"""
    # Validate file extension
    file_ext = os.path.splitext(upload_file.filename)[1].lower()
    if file_ext not in ('.jpg', '.jpeg', '.png'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG/PNG files are allowed"
        )

    # Generate secure filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(directory, unique_filename)

    # Write file in chunks for memory efficiency
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer, length=16 * 1024)  # 16KB chunks

        # Validate image content
        if not validate_image(file_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file"
            )

        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image: {str(e)}"
        )


@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(
        request: Request,
        file: UploadFile = File(...),
        question: str = Form(...)
):
    """Main API endpoint with complete error handling"""
    # Update request count
    request.app.state.request_count += 1

    # Validate input
    if not question or len(question.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must be at least 3 characters long."
        )

    temp_dir = tempfile.mkdtemp()
    try:
        # Save and validate uploaded file
        file_path = await save_upload_file(file, temp_dir)

        # Size check (10MB)
        if os.path.getsize(file_path) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large (max 10MB)"
            )

        logger.info(f"Processing request: {question[:50]}...")

        try:
            answer = answer_question_with_image_and_text(file_path, question)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)

            # Check for service availability
            if "service unavailable" in str(e).lower():
                request.app.state.active_models["visual_understanding"] = False
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service temporarily unavailable"
                )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    finally:
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")


@app.get("/health")
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    current_status = {
        "status": "healthy" if request.app.state.is_healthy else "degraded",
        "uptime_seconds": round(time.time() - request.app.state.start_time, 2),
        "total_requests": request.app.state.request_count,
        "models": request.app.state.active_models,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Determine overall status code
    status_code = (
        status.HTTP_200_OK
        if all(request.app.state.active_models.values())
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        content=current_status,
        status_code=status_code
    )


@app.get("/status")
async def system_status(request: Request):
    """Detailed system status for monitoring"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "uptime": f"{(time.time() - request.app.state.start_time):.2f} seconds",
        "requests_processed": request.app.state.request_count,
        "active_models": request.app.state.active_models,
        "environment": os.getenv("ENVIRONMENT", "development")
    }