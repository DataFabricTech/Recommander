from fastapi import FastAPI

description = """
Hello!
This is Recommender API with Fast API. 🚀
"""

tags_metadata = [
]

responses = { # todo - status_code에 대한 수정 필요
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

'''
FastAPI를 이용한 문서 작성
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/~",
    title="title",
    summary = "Summary",
    description = "Description",
    version="1.0.0",
    openapi_tags=[{'name':'items', 'description': 'description'}]
    response_model = Item(있으면),
)
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
    
def read_item(item_id: int, q: str = Query(None, min_length=3, max_length=50, description="Description"):

fastapi에 대한 비동기 예시

from fastapi import FastAPI

app = FastAPI()

@app.get("/async")
async def async_task():
    # 5초 동안 대기하는 비동기 작업 (예: 외부 API 호출)
    await asyncio.sleep(5)
    return {"message": "비동기 작업 완료"}
'''
