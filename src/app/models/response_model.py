from pydantic import BaseModel
from typing import Union


class MessageModel(BaseModel):
    message: str


class RecommendationModel(BaseModel):
    fqns: list


class ErrorModel(BaseModel):
    detail: str


class BaseCommonModel(BaseModel):
    status: int
    code: str = None
    data: Union[RecommendationModel, MessageModel] = None
    error: ErrorModel = None
