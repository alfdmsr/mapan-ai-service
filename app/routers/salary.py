from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.salary_inference import salary_predictor

router = APIRouter(prefix="/predict", tags=["salary"])


class SalaryPredictRequest(BaseModel):
    job_title: str = Field(..., min_length=1, description="Judul pekerjaan")
    company: str = Field(..., min_length=1, description="Nama perusahaan")
    location: str = Field(..., min_length=1, description="Lokasi kerja")


class SalaryPredictResponse(BaseModel):
    predicted_salary_idr: float
    currency: str = "IDR"
    model_version: str = "salary_estimator_v1"


@router.post("/salary", response_model=SalaryPredictResponse)
def predict_salary(request: SalaryPredictRequest) -> SalaryPredictResponse:
    if not salary_predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model salary belum dimuat. Server sedang startup.",
        )
    try:
        salary_idr = salary_predictor.predict(
            job_title=request.job_title,
            company=request.company,
            location=request.location,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}") from exc

    return SalaryPredictResponse(predicted_salary_idr=salary_idr)