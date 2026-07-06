from fastapi import FastAPI

from app.database.init_db import init_db
from app.routes import api_router

app = FastAPI(
    title="Inventory Demand Forecasting Assistant API",
    version="0.1.0",
    description="REST API for dataset upload, forecasting, recommendations and reports.",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(api_router)
