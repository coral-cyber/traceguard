import os, asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from db.store import init_db
from overseer.containment import ContainmentEngine

# Global event bus — bounded so a slow dashboard never backs up the pipeline
_event_bus: asyncio.Queue = asyncio.Queue(maxsize=500)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    await init_db()

    # ONE ContainmentEngine for the whole app — stored on app.state
    engine = ContainmentEngine(event_bus=_event_bus)
    await engine.restore_state()   # reload quarantine from DB after restarts
    app.state.containment = engine
    app.state.event_bus   = _event_bus

    yield

    # ── shutdown ── (nothing to clean up for SQLite/asyncio)

from api.routes import router

app = FastAPI(
    title       = "TraceGuard",
    description = "Autonomous real-time oversight for multi-agent pipelines",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
