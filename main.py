from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import salary
from app.services.salary_inference import salary_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Memuat model Salary Estimator...")
    salary_predictor.load()
    print("Model salary siap.")
    yield


app = FastAPI(title="MAPAN AI Service", lifespan=lifespan)
app.include_router(salary.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "salary_model_loaded": salary_predictor.is_loaded,
    }