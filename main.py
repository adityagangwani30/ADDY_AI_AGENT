from fastapi import FastAPI
import logging

from api.middleware import register_middleware
from api.routes import router
from auth.token_validator import validate_oauth_health
from config.validator import validate_environment
from services.logging_service import configure_logging, log_event

configure_logging()

validate_environment(strict=True)

app = FastAPI(title="Personal AI Assistant", version="1.0.0")
register_middleware(app)
app.include_router(router)


@app.on_event("startup")
async def startup_health_checks() -> None:
    health = validate_oauth_health()
    app.state.oauth_health = health
    log_event(
        logging.getLogger(__name__),
        logging.INFO,
        event="startup_health_check",
        status=health.get("status"),
        bootstrap=health.get("bootstrap", {}).get("status"),
    )
