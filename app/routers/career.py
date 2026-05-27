from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.career_inference import career_recommender

router = APIRouter(prefix="/recommend", tags=["career"])


class CareerRecommendRequest(BaseModel):
    raw_text: str | None = Field(
        None,
        description="Teks CV mentah (diparse via Model 2 NER)",
    )
    skills: list[str] | None = Field(
        None,
        description="Daftar skill langsung (lewati NER)",
    )
    top_k: int = Field(5, ge=1, le=50, description="Jumlah rekomendasi teratas")
    sector_filter: str | None = Field(
        None,
        description="Filter sektor MAPAN (opsional)",
    )
    min_match_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Skor sigmoid minimum",
    )


class RoleRecommendation(BaseModel):
    rank: int
    role_id: str
    role_name: str
    role_name_id: str | None = None
    sector: str
    match_score: float
    match_percent: float
    overlap_ratio: float
    matched_skills: list[str]
    missing_skills: list[str]


class CareerRecommendResponse(BaseModel):
    user_skills_canonical: list[str]
    recommendations: list[RoleRecommendation]
    model_version: str
    message: str | None = None


@router.post("/career", response_model=CareerRecommendResponse)
def recommend_career(request: CareerRecommendRequest) -> CareerRecommendResponse:
    if not career_recommender.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model career belum dimuat. Server sedang startup.",
        )

    if not request.raw_text and not request.skills:
        raise HTTPException(
            status_code=400,
            detail="Berikan salah satu: raw_text atau skills.",
        )

    try:
        result = career_recommender.recommend(
            raw_text=request.raw_text,
            skills=request.skills,
            top_k=request.top_k,
            sector_filter=request.sector_filter,
            min_match_score=request.min_match_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Rekomendasi karier gagal: {exc}",
        ) from exc

    return CareerRecommendResponse(**result)
