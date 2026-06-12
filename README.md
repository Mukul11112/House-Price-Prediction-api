# 🏠 House Price Prediction API

A production-grade machine learning project that predicts residential property sale prices and exposes predictions through a FastAPI REST endpoint.

---

## 📁 Project Structure

```
house-price-prediction-api/
├── data/
│   └── housing.csv                   # 2,000-row Ames-style housing dataset
├── notebooks/
│   └── training.ipynb                # Full EDA + training walkthrough
├── models/
│   ├── house_price_model.pkl         # Saved best model + preprocessor
│   ├── feature_importance.png        # Feature importance chart
│   ├── eda_dashboard.png             # EDA visualisation dashboard
│   └── model_performance_report.json # Metrics for all 4 models
├── api/
│   └── main.py                       # FastAPI application
├── src/
│   ├── preprocess.py                 # Data cleaning + feature engineering
│   ├── train.py                      # Model training pipeline
│   └── predict.py                    # Inference helpers
├── Dockerfile
├── docker-compose.yml
├── postman_collection.json
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1 · Clone & install dependencies

```bash
git clone <your-repo-url>
cd house-price-prediction-api
pip install -r requirements.txt
```

### 2 · Train the model

```bash
python src/train.py
```

Expected output (printed table):
```
========================================================================
Model                      MAE         RMSE       R²      CV R²
------------------------------------------------------------------------
LinearRegression ★   $   6,680 $   8,503   0.9258     0.9050
RandomForest         $   9,047 $  11,314   0.8687     0.8320
GradientBoosting     $   7,413 $   9,542   0.9066     0.8877
XGBoost              $   7,744 $   9,845   0.9006     0.8808
========================================================================
```

### 3 · Start the API server

```bash
uvicorn api.main:app --reload
```

The server starts at **http://127.0.0.1:8000**

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API greeting |
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/model/info` | Model metadata + all metrics |
| `POST` | `/predict` | Predict a single house price |
| `POST` | `/predict/batch` | Predict up to 100 houses |

### Interactive Docs

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/docs | Swagger UI (try it live) |
| http://127.0.0.1:8000/redoc | ReDoc documentation |

---

## 🔧 API Usage Examples

### Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "overall_qual": 7,
    "gr_liv_area": 1800,
    "garage_cars": 2,
    "total_bsmt_sf": 1000,
    "full_bath": 2,
    "year_built": 2005
  }'
```

**Response:**
```json
{
  "predicted_house_price": 236245.66,
  "model_name": "LinearRegression",
  "model_version": "1.0.0"
}
```

### Batch Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [
      {"overall_qual": 7, "gr_liv_area": 1800, "garage_cars": 2, "total_bsmt_sf": 1000, "full_bath": 2, "year_built": 2005},
      {"overall_qual": 5, "gr_liv_area": 1100, "garage_cars": 1, "total_bsmt_sf": 700,  "full_bath": 1, "year_built": 1985}
    ]
  }'
```

---

## 📥 Input Field Reference

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `overall_qual` | int | 1–10 | Overall material & finish quality |
| `gr_liv_area` | int | 300–6000 | Above-grade living area (sq ft) |
| `garage_cars` | int | 0–5 | Garage capacity (car spaces) |
| `total_bsmt_sf` | int | 0–6000 | Total basement area (sq ft) |
| `full_bath` | int | 0–5 | Full bathrooms above grade |
| `year_built` | int | 1800–2024 | Year of original construction |

---

## 🐳 Docker

### Build & run with Docker

```bash
# Build the image (trains model automatically inside container)
docker build -t house-price-api .

# Run
docker run -p 8000:8000 house-price-api
```

### Docker Compose

```bash
docker compose up --build
```

---

## 📊 ML Pipeline Details

### Data Preprocessing
- **Duplicate removal** – exact row deduplication
- **Missing value imputation** – median (numeric), mode (categorical)
- **Outlier capping** – Tukey IQR fences (winsorise, not drop)
- **Feature engineering** – HouseAge, TotalSF, TotalBathrooms, binary flags
- **Encoding** – ordinal for quality cols, one-hot for nominals
- **Scaling** – StandardScaler (fit on train only)
- **Feature selection** – SelectKBest F-regression, top 20

### Models Compared
| Model | Strengths |
|-------|-----------|
| Linear Regression | Fast, interpretable, strong when features are well-engineered |
| Random Forest | Handles non-linearity, robust to outliers |
| Gradient Boosting | High accuracy, sequential error correction |
| XGBoost | Regularised boosting, often best on tabular data |

### Evaluation Metrics
- **MAE** – Mean Absolute Error (dollar error on average)
- **RMSE** – Root Mean Squared Error (penalises large errors)
- **R²** – Proportion of variance explained (1.0 = perfect)
- **CV R²** – 5-fold cross-validated R² on training set

---

## 🚀 Deployment Guide

### Render

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Set **Build Command:** `pip install -r requirements.txt && python src/train.py`
4. Set **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Deploy!

### Railway

1. Push to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variable: `PORT=8000`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

---

## 🧪 Testing the API (Postman)

Import `postman_collection.json` into Postman. The collection includes 8 pre-built requests covering all endpoints, validation errors, luxury homes, and batch predictions.

---

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for package versions
