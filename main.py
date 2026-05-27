from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import cv, salary
from app.services.cv_inference import cv_parser
from app.services.salary_inference import salary_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Memuat model Salary Estimator...")
    salary_predictor.load()
    print("Model salary siap.")

    print("Memuat model CV NER...")
    cv_parser.load()
    print("Model CV NER siap.")

    yield


app = FastAPI(title="MAPAN AI Service", lifespan=lifespan)
app.include_router(salary.router)
app.include_router(cv.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "salary_model_loaded": salary_predictor.is_loaded,
        "cv_model_loaded": cv_parser.is_loaded,
    }