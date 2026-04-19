"""
Vercel entrypoint for FastAPI application.
This file is required by Vercel to locate the FastAPI app instance.
"""
import sys
import os
import logging

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Add parent directory to path so we can import app
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    logger.info("Importing FastAPI app...")
    from app.api_server import app
    logger.info("FastAPI app imported successfully")
    
except Exception as e:
    logger.error(f"Failed to import app: {e}", exc_info=True)
    # Create a minimal error app that returns error messages
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    error_app = FastAPI()
    
    @error_app.exception_handler(Exception)
    async def error_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Application initialization failed: {str(e)}",
                "error_type": type(e).__name__,
                "path": str(request.url.path) if hasattr(request, 'url') else "unknown"
            }
        )
    
    @error_app.get("/{full_path:path}")
    async def catch_all():
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Application initialization failed: {str(e)}",
                "error_type": type(e).__name__,
                "message": "Check Vercel logs for details"
            }
        )
    
    app = error_app
    logger.error("Using error app due to import failure")

# Export the app instance for Vercel
__all__ = ["app"]

