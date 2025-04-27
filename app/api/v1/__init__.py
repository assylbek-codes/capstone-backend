from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.environments import router as environments_router
from app.api.v1.scenarios import router as scenarios_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.algos import router as algos_router
from app.api.v1.solves import router as solves_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(environments_router, prefix="/environments", tags=["environments"])
api_router.include_router(scenarios_router, prefix="/scenarios", tags=["scenarios"])
api_router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
api_router.include_router(algos_router, prefix="/algos", tags=["algorithms"])
api_router.include_router(solves_router, prefix="/solves", tags=["solves"])

# Add more routers here as they are created
# Example:
# from app.api.v1.items import router as items_router
# api_router.include_router(items_router, prefix="/items", tags=["items"]) 