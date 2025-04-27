import os
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['app.tasks.solve']  # Explicitly include task modules
)

# Include the tasks explicitly
celery_app.conf.task_routes = {
    "app.tasks.solve.execute_solve": "main-queue"
}
celery_app.conf.update(result_expires=3600)

# Auto discover tasks
celery_app.autodiscover_tasks(['app.tasks'])

# Import Celery tasks to ensure they are registered
import app.tasks.solve

if __name__ == "__main__":
    celery_app.start() 