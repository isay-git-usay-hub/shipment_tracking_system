"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import redis
from sqlalchemy.orm import Session

from config.settings import settings
from models.database.connection import get_db
from models.schemas import HealthResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""

    # Check database connection
    db_status = "healthy"
    try:
        db.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")

    # Check Redis connection
    redis_status = "healthy"
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
        logger.error(f"Redis health check failed: {e}")

    # Check external APIs
    external_apis = {
        "local_llm": "healthy",
        "sendgrid": "healthy" if settings.SENDGRID_API_KEY else "not_configured",
        "weather": "healthy" if settings.WEATHER_API_KEY else "not_configured",
        "port": "healthy" if settings.PORT_API_KEY else "not_configured"
    }

    # Overall status
    overall_status = "healthy"
    if "unhealthy" in db_status or "unhealthy" in redis_status:
        overall_status = "unhealthy"
    elif any("not_configured" in status for status in external_apis.values()):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        database=db_status,
        redis=redis_status,
        external_apis=external_apis
    )


@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/readiness")
async def readiness_probe(db: Session = Depends(get_db)):
    """Kubernetes readiness probe"""
    try:
        # Check database
        db.execute("SELECT 1")

        # Check Redis
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.ping()

        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")
