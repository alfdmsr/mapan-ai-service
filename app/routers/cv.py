from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.cv_inference import cv_parser

router = APIRouter(prefix="/parse", tags=["cv"])


class CVParseRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, description="Teks CV mentah")


class CVParseResponse(BaseModel):
    tokens: list[str]
    labels: list[str]
    skills: list[str]
    roles: list[str]
    model_version: str = "cv_ner_multilingual_v1"


@router.post("/cv", response_model=CVParseResponse)
def parse_cv(request: CVParseRequest) -> CVParseResponse:
    if not cv_parser.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model CV NER belum dimuat. Server sedang startup.",
        )

    try:
        result = cv_parser.parse(request.raw_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse CV gagal: {exc}") from exc

    return CVParseResponse(**result)
