import traceback
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from gnn_services.model import JobGraphSAGE, CVJobLinkPredictor
from cv_services.extractCV import extract_cv_profile
import os
import uuid
import requests
from pydantic import BaseModel
# =========================
# CONFIG
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

INDUSTRY_JSON_PATH = BASE_DIR / "cv_services" / "skills_by_industry.json"

GRAPH_PATH = BASE_DIR / "gnn_services" / "graph_data" / "job_graphsage_graph.pt"
MODEL_PATH = BASE_DIR / "gnn_services" / "graph_data" / "best_cv_job_link_prediction_graphsage.pt"
MAPPING_PATH = BASE_DIR / "gnn_services" / "graph_data" / "job_mapping.pt"

TOP_K_DEFAULT = 10


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="GNN Recommendation Service")


class RecommendRequest(BaseModel):
    cv_text: str
    top_k: int = TOP_K_DEFAULT
class ExtractCvUrlRequest(BaseModel):
    file_url: str

# =========================
# HELPER
# =========================

def get_mapping_value(mapping_data, key, index):
    if not isinstance(mapping_data, dict):
        return None

    values = mapping_data.get(key)

    if values is None:
        return None

    if isinstance(values, dict):
        return values.get(index)

    if isinstance(values, list):
        if 0 <= index < len(values):
            return values[index]
        return None

    return None


def load_state_dict_safely(model, state_dict):
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)


# =========================
# LOAD MODEL ON STARTUP
# =========================

print("DEVICE:", DEVICE)

print("Loading graph...")

graph = torch.load(
    GRAPH_PATH,
    map_location=DEVICE,
    weights_only=False
)

graph = graph.to(DEVICE)

print("Loading mapping...")

mapping = torch.load(
    MAPPING_PATH,
    map_location="cpu",
    weights_only=False
)

print("Mapping keys:", list(mapping.keys()) if isinstance(mapping, dict) else type(mapping))

print("Building model...")

graph_model = JobGraphSAGE(
    metadata=graph.metadata(),
    hidden_dim=256,
    out_dim=256,
    dropout=0.3
)

model = CVJobLinkPredictor(
    graph_model=graph_model,
    cv_input_dim=768,
    hidden_dim=256
)

print("Loading model weights...")

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

load_state_dict_safely(model, state_dict)

model = model.to(DEVICE)
model.eval()

print("Loading text embedding model...")

text_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=str(DEVICE)
)

print("AI service ready!")


# =========================
# RECOMMEND FUNCTION
# =========================

@torch.no_grad()
def recommend_jobs_by_cv_text(cv_text: str, top_k: int = 10):
    model.eval()

    cv_emb = text_model.encode(
        [cv_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    cv_emb = torch.tensor(
        cv_emb,
        dtype=torch.float32,
        device=DEVICE
    )

    cv_z = model.encode_cv(cv_emb)
    job_z = model.encode_job(graph)

    scores = torch.matmul(cv_z, job_z.T).squeeze(0)

    k = min(top_k, scores.shape[0])

    top_scores, top_indices = torch.topk(scores, k=k)

    results = []

    for job_idx, score in zip(
        top_indices.cpu().tolist(),
        top_scores.cpu().tolist()
    ):
        job_idx = int(job_idx)

        item = {
            "job_id": job_idx,
            "score": float(score),
            "title": get_mapping_value(mapping, "job_titles", job_idx),
            "company": get_mapping_value(mapping, "job_companies", job_idx),
            "industry": get_mapping_value(mapping, "job_industries", job_idx),
            "skills": get_mapping_value(mapping, "job_skills", job_idx),
        }

        results.append(item)

    return results


# =========================
# ROUTES
# =========================

@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "GNN Recommendation Service is running"
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        if not req.cv_text or not req.cv_text.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "cv_text không được để trống",
                    "data": []
                }
            )

        results = recommend_jobs_by_cv_text(
            cv_text=req.cv_text,
            top_k=req.top_k
        )

        return {
            "success": True,
            "data": results
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e),
                "data": []
            }
        )
@app.post("/extract-cv-url")
def extract_cv_url(req: ExtractCvUrlRequest):
    temp_path = None

    try:
        if not req.file_url or not req.file_url.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "file_url không được để trống"
                }
            )

        response = requests.get(
            req.file_url,
            timeout=30
        )

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Không tải được file CV từ file_url",
                    "status_code": response.status_code
                }
            )

        temp_filename = f"{uuid.uuid4()}.pdf"
        temp_path = UPLOAD_DIR / temp_filename

        with open(temp_path, "wb") as f:
            f.write(response.content)

        result = extract_cv_profile(
            pdf_path=str(temp_path),
            industry_json_path=str(INDUSTRY_JSON_PATH)
        )

        return {
            "success": True,
            "data": {
                "cv_text": result["cv_text"],
                "industry": result["industry"],
                "industry_score": result["industry_score"],
                "matched_industry_skills": result["matched_industry_skills"],
                "top_industries": result["top_industries"],
                "skills": result["skills"],
                "degree": result["degree"],
                "location": result["location"],
                "exp_min": result["exp_min"],
                "exp_max": result["exp_max"]
            }
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Lỗi extract CV từ file_url",
                "error": str(e)
            }
        )

    finally:
        if temp_path and temp_path.exists():
            os.remove(temp_path)