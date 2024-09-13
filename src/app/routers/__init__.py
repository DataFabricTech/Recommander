from .cataloging_router import router as cataloging_router
from .training_router import router as training_router
from .clustering_router import router as clustering_router
from .embedding_router import router as embedding_router
from .check import router as check_router

routers = [
    check_router,
    training_router,
    clustering_router,
    cataloging_router,
    embedding_router
]
