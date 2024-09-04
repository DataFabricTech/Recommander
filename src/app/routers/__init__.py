from .training_service import router as training_router
from .clustering_service import router as clustering_router
from .embedding_service import router as embedding_router
from .check import router as check_router

routers = [
    check_router,
    training_router,
    clustering_router,
    embedding_router
]
