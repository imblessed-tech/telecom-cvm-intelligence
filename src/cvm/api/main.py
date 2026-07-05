import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.cvm.config import ensure_directories
from src.cvm.api.dependencies import load_all_models
from src.cvm.api.middleware import RequestLoggingMiddleware
from src.cvm.api.routes.customers import router as customers_router
from src.cvm.api.routes.campaigns import router as campaigns_router
from src.cvm.api.routes.models import router as models_router
from src.cvm.api.routes.health import router as health_router

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cvm_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing CVM FastAPI service...")
    ensure_directories()
    
    try:
        load_all_models()
    except Exception as e:
        logger.critical(f"Failed to load registry models at startup: {e}")
        # In a real environment, we might want to fail the startup, 
        # but for local resilience, we will allow the container to start.
        
    yield
    
    # Shutdown actions
    logger.info("Shutting down CVM FastAPI service...")

app = FastAPI(
    title="CVM Intelligence Platform API",
    description="Customer Value Management (CVM) ML score registry, campaign opportunity base generator, and data drift alerting system.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable request tracing and logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints routers
app.include_router(customers_router)
app.include_router(campaigns_router)
app.include_router(models_router)
app.include_router(health_router)

@app.get("/")
def read_root():
    """Root landing endpoint providing service summary and developer documentation links."""
    return {
        "message": "Welcome to MTN Nigeria Customer Value Management (CVM) Intelligence Platform API",
        "documentation": "/docs",
        "status": "active"
    }
