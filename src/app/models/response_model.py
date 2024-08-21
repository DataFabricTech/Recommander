from pydantic import BaseModel


class BaseCommonModel(BaseModel):
    status: int
    data: any
    message: str
    code: str


class ErrorModel(BaseCommonModel):
    detail: str


class RecommendationModel(BaseCommonModel):
    fqn: str

