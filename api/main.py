"""
main.py
=======
FastAPI application for House Price Prediction.

Endpoints
---------
  GET  /          – health greeting
  GET  /health    – health check with model info
  GET  /model/info – detailed model metadata & metrics
  POST /predict   – single prediction
  POST /predict/batch – batch predictions (up to 100 rows)

Run locally:
    uvicorn api.main:app --reload
Swagger UI:
    http://127.0.0.1:8000/docs
ReDoc:
    http://127.0.0.1:8000/redoc
"""

import logging
import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── path fix so `src/` modules are importable ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from predict import load_model_artifact, predict_batch, predict_price  # noqa: E402

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("house_price_api")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🏠 House Price Prediction API",
    description=(
        "Predict residential property sale prices using a machine-learning model "
        "trained on Ames Housing-style data.\n\n"
        "**Model pipeline:** data cleaning → feature engineering → "
        "ordinal + one-hot encoding → StandardScaler → best of 4 regressors.\n\n"
        "**Supported models:** Linear Regression, Random Forest, "
        "Gradient Boosting, XGBoost."
    ),
    version="1.0.0",
    contact={"name": "House Price API", "email": "api@houseprice.dev"},
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f} ms)")
    return response


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    """Input features for a single house price prediction."""

    overall_qual: int = Field(
        ..., ge=1, le=10,
        description="Overall material and finish quality (1=Poor … 10=Excellent)",
        example=7,
    )
    gr_liv_area: int = Field(
        ..., ge=300, le=6000,
        description="Above-grade (ground) living area in square feet",
        example=1800,
    )
    garage_cars: int = Field(
        ..., ge=0, le=5,
        description="Garage capacity in car spaces",
        example=2,
    )
    total_bsmt_sf: int = Field(
        ..., ge=0, le=6000,
        description="Total basement area in square feet (0 if no basement)",
        example=1000,
    )
    full_bath: int = Field(
        ..., ge=0, le=5,
        description="Number of full bathrooms above grade",
        example=2,
    )
    year_built: int = Field(
        ..., ge=1800, le=2024,
        description="Year the house was originally constructed",
        example=2005,
    )

    @field_validator("year_built")
    @classmethod
    def year_not_future(cls, v):
        if v > 2024:
            raise ValueError("year_built cannot be in the future.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "overall_qual":  7,
                "gr_liv_area":   1800,
                "garage_cars":   2,
                "total_bsmt_sf": 1000,
                "full_bath":     2,
                "year_built":    2005,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_house_price: float = Field(..., description="Predicted sale price in USD", example=325000.45)
    model_name: str             = Field(..., description="Model used for prediction")
    model_version: str          = Field(..., description="Model artifact version")


class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures] = Field(..., min_length=1, max_length=100,
                                        description="List of houses to predict (max 100)")


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]  = None
    model_version: Optional[str] = None
    r2_score: Optional[float]  = None


# ── Startup: pre-load model ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-warm the model so the first request is fast."""
    try:
        art = load_model_artifact()
        logger.info(
            f"✅ Model ready: {art['model_name']} v{art['model_version']} "
            f"R²={art['metrics']['R2']:.4f}"
        )
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not yet trained: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"], summary="API root")
async def root():
    """Returns a greeting confirming the API is running."""
    return {"message": "House Price Prediction API is running"}


@app.get("/health", response_model=HealthResponse, tags=["Status"], summary="Health check")
async def health():
    """
    Returns the health status of the API and whether the model is loaded.
    Use this endpoint for liveness / readiness probes.
    """
    try:
        art = load_model_artifact()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=art["model_name"],
            model_version=art["model_version"],
            r2_score=round(art["metrics"]["R2"], 4),
        )
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model/info", tags=["Model"], summary="Model metadata and performance")
async def model_info():
    """
    Returns full model metadata including:
    - Model name & version
    - Selected features
    - Test-set metrics (MAE, RMSE, R²)
    - Comparison across all trained models
    """
    try:
        art = load_model_artifact()
        return {
            "model_name":        art["model_name"],
            "model_version":     art["model_version"],
            "selected_features": art["selected_features"],
            "metrics":           art["metrics"],
            "all_model_results": art.get("all_results", {}),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict a single house price",
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error — check input ranges"},
        503: {"description": "Model not loaded — run training first"},
    },
)
async def predict(features: HouseFeatures):
    """
    Predict the sale price of a single house.

    **Required fields:**
    - `overall_qual`  : 1–10 quality rating
    - `gr_liv_area`   : above-grade living area (sq ft)
    - `garage_cars`   : number of garage spaces
    - `total_bsmt_sf` : basement area (sq ft)
    - `full_bath`     : full bathrooms above grade
    - `year_built`    : year of construction

    **Returns** the predicted price in USD.
    """
    try:
        art   = load_model_artifact()
        price = predict_price(features.model_dump())
        return PredictionResponse(
            predicted_house_price=price,
            model_name=art["model_name"],
            model_version=art["model_version"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict prices for multiple houses (max 100)",
)
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """
    Predict sale prices for a list of houses in one call.
    Maximum 100 houses per request.
    """
    try:
        art    = load_model_artifact()
        inputs = [h.model_dump() for h in request.houses]
        prices = predict_batch(inputs)
        return BatchPredictionResponse(
            predictions=prices,
            count=len(prices),
            model_name=art["model_name"],
            model_version=art["model_version"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
