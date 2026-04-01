from contextlib import asynccontextmanager

from fastapi import FastAPI

from db import ensure_schema
from endpoints import auth, chat, health
from rag.runtime import get_rag_service
from settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_schema()
    if settings.preload_models_on_startup:
        get_rag_service()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
