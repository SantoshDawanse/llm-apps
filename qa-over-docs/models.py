from pydantic.v1 import BaseModel, Field


class Response(BaseModel):
    answer: str = Field(description="answer conflicting values")
