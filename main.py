from fastapi import FastAPI, HTTPException
from api.chunking_embed_routes import router2
from api.routes import router
import logging

app = FastAPI(
    title="RAG using Azure",
    description="basic RAG Pipeline using Azure search service and Foundry",
)

app.include_router(router)
app.include_router(router2)
import subprocess

result = subprocess.run(["ipconfig", "/flushdns"], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)


@app.get("/health")
def get_health():
    status = "OK"
    return {"status": status}
