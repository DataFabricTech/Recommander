from queue import PriorityQueue
from sqlalchemy import String, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from dataclasses import dataclass, field

Base = declarative_base()


class OverrodePriorityQueue(PriorityQueue):
    def __init__(self, maxsize=5):
        super().__init__(maxsize)
        self.maxsize = maxsize

    def put(self, item, block=True, timeout=None):
        if self.qsize() >= self.maxsize:
            min_item = self.get_nowait()

            if item > min_item:
                super().put(item, block, timeout)
            else:
                super().put(min_item, block, timeout)
        else:
            super().put(item, block, timeout)


class RecommenderWithDB(Base):
    __tablename__ = 'recommender'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    recommend_id: Mapped[list] = mapped_column(JSON)

    def __init__(self, id: str, recommend_id: list):
        self.id = id
        self.recommend_id = recommend_id


@dataclass(order=True)
class RecommendEntity:
    """
    비교 모델의 추천을 위한 엔티티로써, 비교 모델의 id과 일치(포함)하는 컬럼의 개수, 컬럼들의 유사도중에 Top(N개)의 평균값을 이용한다.

    추천 순서 : 일치(포함)하는 컬럼의 개수 > 유사도
    """
    sort_index: tuple = field(init=False, repr=False)

    target_id: str
    inclusion_column_count: int
    top_similarity_average: float

    def __plus_inclusion_column_count(self):
        self.inclusion_column_count += 1

    def __set_top_similarity_average(self, top_similarity_average: float):
        self.top_similarity_average = top_similarity_average

    def __post_init__(self):
        self.sort_index = (self.inclusion_column_count, self.top_similarity_average)
