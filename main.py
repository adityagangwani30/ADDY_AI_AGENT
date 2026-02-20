from fastapi import FastAPI

from api.middleware import register_middleware
from api.routes import router
from services.logging_service import configure_logging

configure_logging()

app = FastAPI(title="Personal AI Assistant", version="1.0.0")
register_middleware(app)
app.include_router(router)
