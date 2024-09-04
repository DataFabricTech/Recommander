from fastapi import FastAPI

from app.models.response_model import ErrorModel, RecommendationModel

description = """
Hello!
This is Recommender API with Fast API. 🚀
"""

tags_metadata = [
]

responses = {
    200: {"description": "recommend id returned", "data":RecommendationModel},
    404: {"description": "No data found", "error": ErrorModel(detail="The reason why")},
    418: {"description": "this is fast exception"},
}
app = FastAPI(
    title="Recommender Fast API",
    description=description,
    version="0.1.0",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "Platform Research and Development Team of Mobigen Co., Ltd.",
        "email": "irisdev@mobigen.com",
    },
    docs_url="/recommender/docs",
    redoc_url="/recommender/redoc",
    openapi_url="/recommender/openapi.json",
    openapi_tags=tags_metadata,
)
