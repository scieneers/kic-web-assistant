from fastapi import FastAPI
from pydantic import BaseModel

from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder

app = FastAPI()
model = MultilingualE5LargeEmbedder()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class EmbedRequest(BaseModel):
    query: str


@app.post("/embed")
def read_root(query: EmbedRequest):
    vector = model.embed(query)
    return {"Embedding": vector}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("embedding_api:app", host="0.0.0.0", port=80, reload=True, log_level="debug")

    # Test request
    # curl -X POST 'http://localhost:80/embed' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{'query':'Hello test'}'
