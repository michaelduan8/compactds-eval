from pydantic import BaseModel

class BreakDownResponse(BaseModel):
    # problem_analysis: str
    # core_concepts: list[str]
    search_queries: list[str]
