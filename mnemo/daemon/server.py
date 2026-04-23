"""Daemon: FastAPI on 127.0.0.1:8767. Auto-shuts down after idle."""

import asyncio
import os
import signal
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mnemo.core.engine import MemoryEngine

IDLE_TIMEOUT = 30 * 60  # 30 minutes
PORT = 8767
HOST = "127.0.0.1"

engine: MemoryEngine | None = None
start_time: float = time.time()
last_activity: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = MemoryEngine(auto_scheduler=True)
    asyncio.create_task(idle_shutdown_task())
    yield
    if engine:
        engine.shutdown()


async def idle_shutdown_task():
    """Check every 60s. Shutdown if idle > IDLE_TIMEOUT."""
    while True:
        await asyncio.sleep(60)
        if time.time() - last_activity > IDLE_TIMEOUT:
            os.kill(os.getpid(), signal.SIGTERM)


app = FastAPI(lifespan=lifespan)


def touch():
    global last_activity
    last_activity = time.time()


@app.post("/recall")
def recall(query: str, top_k: int = 10, project: str | None = None):
    touch()
    if engine:
        engine.project = project
        return engine.recall(query, top_k=top_k)
    return []


@app.post("/remember")
def remember(text: str, importance: float = 0.5, source: str = "agent"):
    touch()
    if engine:
        return {"fact_id": engine.remember(text, importance=importance, source=source)}
    return {"fact_id": None}


@app.get("/stats")
def stats():
    touch()
    if engine:
        return engine.stats()
    return {}


@app.get("/health")
def health():
    return {"status": "running", "uptime_s": round(time.time() - start_time, 1)}


def run_daemon():
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
