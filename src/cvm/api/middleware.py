import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log request details
        logger.info(f"Request started: ID={request_id} Method={request.method} Path={request.url.path}")
        
        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Request failed: ID={request_id} Method={request.method} Path={request.url.path} Duration={duration_ms:.2f}ms Error={e}", exc_info=True)
            raise e
            
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Request finished: ID={request_id} Method={request.method} Path={request.url.path} Status={response.status_code} Duration={duration_ms:.2f}ms")
        
        # Add X-Request-ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
