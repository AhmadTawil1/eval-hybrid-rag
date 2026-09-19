from pydantic import BaseModel


class GroundedAnswer(BaseModel):
    answer: str
    referenced_chunks: list[str]
