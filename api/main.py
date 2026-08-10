# main.py — FastAPI application entrypoint
# creates the app, configures CORS, and mounts all routers

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import training, metrics, predict

# create the FastAPI app instance
app = FastAPI(
    title="FedCredit API",
    description="Federated learning credit scoring system — training metrics and loan predictions",
    version="1.0.0"
)

# CORS middleware, allows the React frontend to talk to this API
# without this the browser blocks requests from localhost:3000 to localhost:8000
app.add_middleware(
    CORSMiddleware,
    # in production replace * with your actual frontend URL
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount routers , each handles a group of related endpoints
app.include_router(training.router, prefix="/training", tags=["training"])
app.include_router(metrics.router,  prefix="/metrics",  tags=["metrics"])
app.include_router(predict.router,  prefix="/predict",  tags=["predict"])

# health check, confirms the API is running
@app.get("/")
def root():
    return {"status": "FedCredit API is running"}