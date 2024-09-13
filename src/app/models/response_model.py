from pydantic import BaseModel
from typing import Union, List


class MessageModel(BaseModel):
    message: str


class RecommendationModel(BaseModel):
    recommended: Union[List[str], str] = None


class ErrorModel(BaseModel):
    detail: str


class BaseCommonModel(BaseModel):
    status: int
    code: str = None
    data: Union[RecommendationModel, MessageModel] = None
    error: ErrorModel = None
