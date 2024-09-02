from fastapi import FastAPI

description = """
Hello!
This is Recommender API with Fast API. 🚀
"""

tags_metadata = [
]

responses = {  # todo - status_code에 대한 수정 필요
    200: {"content": {
        "application/json": {
            "example": {"username": "return username place"}
        }
    }},
    404: {"description": "User not Found",
          "content": {
              "application/json": {
                  "example": {"message": "User not Found"}
              }
          }},
    302: {"description": "The user was moved"},
    403: {"description": "Not enough privileges"},
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
