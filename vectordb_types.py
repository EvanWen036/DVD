from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CreateCollection(BaseModel):
    name: str
    dim: int
    metric: str = Field("cosine", pattern="^(cosine|l2)$")

class Point(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None

class UpsertRequest(BaseModel):
    points: List[Point]

class DeleteRequest(BaseModel):
    ids: List[str]

class QueryRequest(BaseModel):
    vector: List[float]
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None  # placeholder (not applied yet)

class QueryHit(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    hits: List[QueryHit]
