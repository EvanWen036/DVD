from fastapi import FastAPI, HTTPException, Query
from store import Registry
from vectordb_types import (
    CreateCollection, UpsertRequest, QueryRequest,
    DeleteRequest, QueryResponse, QueryHit,
)

app = FastAPI(title="Mini Vector DB")
reg = Registry()

@app.post("/collections")
def create_collection(body: CreateCollection):
    try:
        reg.create(body.name, body.dim, body.metric, backend=body.backend)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/upsert")
def upsert(collection: str = Query(...), body: UpsertRequest = None):
    if body is None:
        raise HTTPException(400, "Missing request body")
    try:
        col = reg.get(collection)
        n = col.upsert([p.model_dump() for p in body.points])
        return {"upserted": n}
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))

@app.post("/delete")
def delete(collection: str = Query(...), body: DeleteRequest = None):
    if body is None:
        raise HTTPException(400, "Missing request body")
    try:
        col = reg.get(collection)
        n = col.delete(body.ids)
        return {"deleted": n}
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))

@app.post("/query", response_model=QueryResponse)
def query(collection: str = Query(...), body: QueryRequest = None):
    if body is None:
        raise HTTPException(400, "Missing request body")
    try:
        col = reg.get(collection)
        hits = col.query(body.vector, body.top_k)
        return QueryResponse(hits=[QueryHit(**h) for h in hits])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
