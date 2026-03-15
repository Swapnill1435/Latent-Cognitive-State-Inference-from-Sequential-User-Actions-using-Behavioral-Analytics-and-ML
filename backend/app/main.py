"""FastAPI application entry point for the Cognitive State Inference Platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.api.routes import router as api_router
from app.api.dashboard import router as dashboard_router
from app.api.websocket import websocket_endpoint
from app.pipeline.inference import orchestrator


app = FastAPI(
    title="Cognitive State Inference Platform",
    description="Real-time behavioral analytics platform inferring latent cognitive states from user interaction data",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API routes
app.include_router(api_router)
app.include_router(dashboard_router)

# WebSocket endpoint
app.websocket("/ws/{session_id}")(websocket_endpoint)


@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    print("🧠 Cognitive State Inference Platform starting up...")
    print("📂 Loading ML models...")
    orchestrator.load_models()
    print("✅ Platform ready!")


@app.get("/")
async def root():
    return {
        "name": "Cognitive State Inference Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
