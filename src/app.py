import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from gnn_services.model import JobGraphSAGE, CVJobLinkPredictor
from cv_services.extractCV import extract_cv_profile
from cv_services.skill_extractor import SkillExtractor
import torch.nn.functional as F
import requests
import copy
import threading
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

INDUSTRY_JSON_PATH = BASE_DIR / "cv_services" / "skills_by_industry.json"

GRAPH_PATH = BASE_DIR / "gnn_services" / "graph_data" / "job_graphsage_graph.pt"
MODEL_PATH = BASE_DIR / "gnn_services" / "graph_data" / "best_cv_job_link_prediction_graphsage.pt"
MAPPING_PATH = BASE_DIR / "gnn_services" / "graph_data" / "job_mapping.pt"

# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="GNN Recommendation Service")


class RecommendRequest(BaseModel):
    cv_text: str
class ExtractCvUrlRequest(BaseModel):
    file_url: str


class IndexJobRequest(BaseModel):
    id: Optional[int] = None
    job_id: Optional[int] = None
    name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    job_requirement: Optional[str] = None
    jobRequirement: Optional[str] = None
    job_benefit: Optional[str] = None
    jobBenefit: Optional[str] = None
    location: Optional[str] = None
    exp_min: Optional[float] = None
    expMin: Optional[float] = None
    exp_max: Optional[float] = None
    expMax: Optional[float] = None
    company: Optional[Any] = None
    industries: Optional[List[Any]] = None
    industry: Optional[Any] = None

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


def build_extract_cv_response(result):
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


def download_pdf_to_memory(file_url: str) -> BytesIO:
    parsed_url = urlparse(file_url)

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("file_url phải là URL HTTP/HTTPS hợp lệ")

    response = requests.get(file_url, timeout=30)
    response.raise_for_status()

    pdf_bytes = response.content

    if not pdf_bytes:
        raise ValueError("PDF rỗng hoặc không tải được nội dung")

    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        content_type = response.headers.get("content-type", "")
        raise ValueError(f"URL không trả về nội dung PDF hợp lệ. content-type={content_type}")

    return BytesIO(pdf_bytes)


GRAPH_LOCK = threading.Lock()


def clean_text(value):
    if value is None:
        return ""

    return str(value).lower().strip()


def normalize_string(value):
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def normalize_set(items):
    output = set()

    if not items:
        return output

    for item in items:
        text = clean_text(item)

        if text:
            output.add(text)

    return output


def remove_rule_skills(skills):
    output = set()

    for skill in skills:
        skill = clean_text(skill)

        if not skill:
            continue

        if skill.endswith("_years_exp"):
            continue

        if skill.startswith("degree_"):
            continue

        output.add(skill)

    return output


def dice_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0

    return 2 * len(a & b) / (len(a) + len(b))


def compute_skill_overlap_score(cv_skills, job_skills):
    cv_skills = remove_rule_skills(cv_skills)
    job_skills = remove_rule_skills(job_skills)

    if not cv_skills or not job_skills:
        return 0.0, 0

    shared_count = len(cv_skills & job_skills)

    if shared_count == 0:
        return 0.0, 0

    cv_coverage = shared_count / len(cv_skills)
    job_coverage = shared_count / len(job_skills)
    overlap_score = 0.7 * cv_coverage + 0.3 * job_coverage

    return max(0.0, min(1.0, overlap_score)), shared_count


def extract_years_experience(skills):
    exp_list = []

    for skill in skills:
        skill = clean_text(skill)

        if "_years_exp" not in skill:
            continue

        try:
            exp_list.append(int(skill.split("_")[0]))
        except Exception:
            pass

    return min(exp_list) if exp_list else 0


def exp_penalty_between_jobs(skills_1, skills_2):
    gap = abs(
        extract_years_experience(skills_1)
        - extract_years_experience(skills_2)
    )

    if gap >= 5:
        return 0.50

    if gap >= 3:
        return 0.30

    if gap >= 2:
        return 0.10

    if gap >= 1:
        return 0.05

    return 0.0


def extract_highest_degree(skills):
    degree_rank = {
        "degree_trung_cap": 1,
        "degree_cao_dang": 2,
        "degree_cu_nhan": 3,
        "degree_dai_hoc": 3,
        "degree_thac_si": 4,
        "degree_tien_si": 5,
    }

    highest = 0

    for skill in skills:
        highest = max(highest, degree_rank.get(clean_text(skill), 0))

    return highest


def degree_penalty_between_jobs(skills_1, skills_2):
    degree_1 = extract_highest_degree(skills_1)
    degree_2 = extract_highest_degree(skills_2)

    if degree_1 == 0 or degree_2 == 0:
        return 0.0

    gap = abs(degree_1 - degree_2)

    if gap >= 3:
        return 0.40

    if gap >= 2:
        return 0.25

    if gap >= 1:
        return 0.10

    return 0.0


def get_config_value(key, default):
    config = mapping.get("config") if isinstance(mapping, dict) else None

    if not isinstance(config, dict):
        return default

    return config.get(key, default)


def get_job_id_from_request(req: IndexJobRequest):
    return req.job_id if req.job_id is not None else req.id


def get_exp_min_from_request(req: IndexJobRequest):
    return req.exp_min if req.exp_min is not None else req.expMin


def get_company_name(company):
    if isinstance(company, dict):
        return normalize_string(company.get("name"))

    return normalize_string(company)


def extract_industry_name(item):
    if isinstance(item, dict):
        return normalize_string(item.get("name"))

    return normalize_string(item)


def get_industry_names(req: IndexJobRequest):
    industries = []

    for item in req.industries or []:
        name = extract_industry_name(item)

        if name:
            industries.append(name)

    industry = extract_industry_name(req.industry)

    if industry:
        industries.append(industry)

    seen = set()
    output = []

    for industry_name in industries:
        key = clean_text(industry_name)

        if key not in seen:
            seen.add(key)
            output.append(industry_name)

    return output
def build_job_source_text(req: IndexJobRequest):
    parts = [
        req.title,
        req.name,
        req.description,
        req.job_requirement,
        req.jobRequirement,
    ]

    return " ".join(
        str(part).strip()
        for part in parts
        if part is not None and str(part).strip()
    )

def add_exp_skill(skills, exp_min):
    if exp_min is None:
        return skills

    try:
        exp_year = int(float(exp_min))
    except Exception:
        return skills

    if exp_year > 0:
        skills.add(f"{exp_year}_years_exp")

    return skills


def clean_embedding_skills(skills):
    return [
        skill
        for skill in sorted(remove_rule_skills(skills))
        if len(skill) > 1
    ][:20]

def build_embedding_text(req: IndexJobRequest, skills, industries):

    title = normalize_string(req.title) or normalize_string(req.name)
    real_skills = clean_embedding_skills(skills)

    parts = []

    if title:
        parts.append(f"Job title: {title}.")

    if industries:
        parts.append("Industry: " + ", ".join(industries[:3]) + ".")

    if real_skills:
        parts.append("Skills: " + ", ".join(real_skills) + ".")

    return " ".join(parts).strip()

def ensure_mapping_list(key, default_builder):
    num_jobs = int(mapping.get("num_jobs", graph["job"].x.size(0)))
    value = mapping.get(key)

    if not isinstance(value, list):
        value = [default_builder(i) for i in range(num_jobs)]
        mapping[key] = value

    while len(value) < num_jobs:
        value.append(default_builder(len(value)))

    return value


def get_or_create_node_id(mapping_key, node_type, name):
    node_map = mapping.setdefault(mapping_key, {})
    key = clean_text(name)

    if key in node_map:
        return node_map[key], False

    node_id = len(node_map)
    node_map[key] = node_id

    dim = graph["job"].x.size(1)
    new_x = torch.zeros((1, dim), dtype=graph[node_type].x.dtype, device=DEVICE)
    graph[node_type].x = torch.cat([graph[node_type].x, new_x], dim=0)

    return node_id, True


def filter_edge_type(edge_type, keep_mask):
    edge_store = graph[edge_type]

    edge_store.edge_index = edge_store.edge_index[:, keep_mask]

    if hasattr(edge_store, "edge_weight"):
        edge_store.edge_weight = edge_store.edge_weight[keep_mask]


def remove_job_edges(node_idx):
    edge_type = ("job", "requires", "skill")
    filter_edge_type(edge_type, graph[edge_type].edge_index[0] != node_idx)

    edge_type = ("skill", "required_by", "job")
    filter_edge_type(edge_type, graph[edge_type].edge_index[1] != node_idx)

    edge_type = ("job", "belongs_to", "industry")
    filter_edge_type(edge_type, graph[edge_type].edge_index[0] != node_idx)

    edge_type = ("industry", "contains", "job")
    filter_edge_type(edge_type, graph[edge_type].edge_index[1] != node_idx)

    edge_type = ("job", "similar_to", "job")
    edge_index = graph[edge_type].edge_index
    keep_mask = (edge_index[0] != node_idx) & (edge_index[1] != node_idx)
    filter_edge_type(edge_type, keep_mask)


def append_edges(edge_type, src, dst, weights=None):
    if len(src) == 0:
        return

    new_edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long,
        device=DEVICE
    )

    graph[edge_type].edge_index = torch.cat(
        [graph[edge_type].edge_index, new_edge_index],
        dim=1
    )

    if weights is not None:
        new_weights = torch.tensor(
            weights,
            dtype=torch.float,
            device=DEVICE
        )

        graph[edge_type].edge_weight = torch.cat(
            [graph[edge_type].edge_weight, new_weights],
            dim=0
        )


def rebuild_mean_features(node_type, edge_type):
    num_nodes = graph[node_type].x.size(0)
    job_x = graph["job"].x
    dim = job_x.size(1)
    edge_index = graph[edge_type].edge_index

    feat = torch.zeros(
        (num_nodes, dim),
        dtype=job_x.dtype,
        device=DEVICE
    )
    count = torch.zeros(
        num_nodes,
        dtype=job_x.dtype,
        device=DEVICE
    )

    if edge_index.numel() > 0:
        src_jobs = edge_index[0]
        dst_nodes = edge_index[1]
        feat.index_add_(0, dst_nodes, job_x[src_jobs])
        count.index_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=job_x.dtype))

    count = torch.clamp(count, min=1.0)
    feat = feat / count.unsqueeze(1)
    feat = F.normalize(feat, p=2, dim=1)
    feat = torch.nan_to_num(feat)

    graph[node_type].x = feat


def build_similar_edges_for_job(node_idx):
    top_k = int(get_config_value("top_k_similar", 10))
    threshold = float(get_config_value("final_sim_threshold", 0.35))

    job_skills = ensure_mapping_list("job_skills", lambda _idx: set())
    job_real_skills = ensure_mapping_list("job_real_skills", lambda _idx: set())
    job_industries = ensure_mapping_list("job_industries", lambda _idx: set())

    target_industries = job_industries[node_idx]
    target_real_skills = job_real_skills[node_idx]
    target_full_skills = job_skills[node_idx]

    neighbors = []

    if not target_industries or not target_real_skills:
        return [], [], []

    for other_idx in range(len(job_skills)):
        if other_idx == node_idx:
            continue

        if not target_industries.intersection(job_industries[other_idx]):
            continue

        other_real_skills = job_real_skills[other_idx]

        if not other_real_skills:
            continue

        skill_sim = dice_similarity(target_real_skills, other_real_skills)
        final_score = max(
            0.0,
            skill_sim
            - exp_penalty_between_jobs(target_full_skills, job_skills[other_idx])
            - degree_penalty_between_jobs(target_full_skills, job_skills[other_idx])
        )

        if final_score >= threshold:
            neighbors.append((final_score, other_idx))

    neighbors.sort(key=lambda item: item[0], reverse=True)
    neighbors = neighbors[:top_k]

    src = []
    dst = []
    weights = []

    for score, other_idx in neighbors:
        src.extend([node_idx, other_idx])
        dst.extend([other_idx, node_idx])
        weights.extend([score, score])

    return src, dst, weights


def save_graph_state():
    graph_cpu = copy.deepcopy(graph).to("cpu")

    torch.save(graph_cpu, GRAPH_PATH)
    torch.save(mapping, MAPPING_PATH)


def index_job_in_graph(req: IndexJobRequest):
    job_id = get_job_id_from_request(req)

    if job_id is None:
        raise ValueError("job_id hoac id la bat buoc")

    job_id = int(job_id)
    title = normalize_string(req.title) or normalize_string(req.name)

    if not title:
        raise ValueError("Tên công việc không được để trống")

    source_text = build_job_source_text(req)

    if not source_text:
        raise ValueError("Text công việc không được để trống")

    extracted_skills = set(skill_extractor.extract_skills(source_text))
    extracted_skills = add_exp_skill(extracted_skills, get_exp_min_from_request(req))
    real_skills = remove_rule_skills(extracted_skills)
    industries = normalize_set(get_industry_names(req))
    embedding_text = build_embedding_text(req, extracted_skills, sorted(industries))

    if not embedding_text:
        raise ValueError("khong the tao text embedding cho job")

    job_emb = text_model.encode(
        [embedding_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    job_x = torch.tensor(
        job_emb,
        dtype=torch.float32,
        device=DEVICE
    )

    job_ids = ensure_mapping_list("job_ids", lambda idx: idx)
    job_titles = ensure_mapping_list("job_titles", lambda _idx: None)
    job_skills = ensure_mapping_list("job_skills", lambda _idx: set())
    job_real_skills = ensure_mapping_list("job_real_skills", lambda _idx: set())
    job_industries = ensure_mapping_list("job_industries", lambda _idx: set())
    job_companies = ensure_mapping_list("job_companies", lambda _idx: None)

    if job_id in job_ids:
        node_idx = job_ids.index(job_id)
        action = "updated"
        graph["job"].x[node_idx] = job_x[0]
        remove_job_edges(node_idx)
    else:
        node_idx = graph["job"].x.size(0)
        action = "created"
        graph["job"].x = torch.cat([graph["job"].x, job_x], dim=0)
        job_ids.append(job_id)
        job_titles.append(None)
        job_skills.append(set())
        job_real_skills.append(set())
        job_industries.append(set())
        job_companies.append(None)

    job_titles[node_idx] = title
    job_skills[node_idx] = extracted_skills
    job_real_skills[node_idx] = real_skills
    job_industries[node_idx] = industries
    job_companies[node_idx] = get_company_name(req.company)

    skill_ids = [
        get_or_create_node_id("skill2id", "skill", skill)[0]
        for skill in sorted(real_skills)
    ]
    industry_ids = [
        get_or_create_node_id("industry2id", "industry", industry)[0]
        for industry in sorted(industries)
    ]

    append_edges(
        ("job", "requires", "skill"),
        [node_idx] * len(skill_ids),
        skill_ids
    )
    append_edges(
        ("skill", "required_by", "job"),
        skill_ids,
        [node_idx] * len(skill_ids)
    )
    append_edges(
        ("job", "belongs_to", "industry"),
        [node_idx] * len(industry_ids),
        industry_ids
    )
    append_edges(
        ("industry", "contains", "job"),
        industry_ids,
        [node_idx] * len(industry_ids)
    )

    sim_src, sim_dst, sim_weights = build_similar_edges_for_job(node_idx)
    append_edges(("job", "similar_to", "job"), sim_src, sim_dst, sim_weights)

    rebuild_mean_features("skill", ("job", "requires", "skill"))
    rebuild_mean_features("industry", ("job", "belongs_to", "industry"))

    mapping["num_jobs"] = graph["job"].x.size(0)
    mapping["embedding_dim"] = graph["job"].x.size(1)

    save_graph_state()

    return {
        "action": action,
        "job_id": job_id,
        "node_idx": node_idx,
        "title": title,
        "skills": sorted(real_skills),
        "industries": sorted(industries),
        "similar_edges_added": len(sim_src),
        "total_jobs": int(mapping["num_jobs"]),
    }


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

print("Loading skill extractor...")

skill_extractor = SkillExtractor()

print("AI service ready!")


# =========================
# RECOMMEND FUNCTION
# =========================

@torch.no_grad()
def recommend_jobs_by_cv_text(cv_text: str):
    model.eval()
    cv_skills = remove_rule_skills(skill_extractor.extract_skills(cv_text))

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

    graphsage_raw_scores = torch.matmul(cv_z, job_z.T).squeeze(0)
    graphsage_scores = ((graphsage_raw_scores + 1) / 2).clamp(0, 1)

    # Lấy tất cả job, sắp xếp score giảm dần
    results = []

    for job_idx, graphsage_score in enumerate(graphsage_scores.cpu().tolist()):
        job_idx = int(job_idx)
        job_skills = get_mapping_value(mapping, "job_real_skills", job_idx)

        if job_skills is None:
            job_skills = get_mapping_value(mapping, "job_skills", job_idx)

        overlap_score, shared_skill_count = compute_skill_overlap_score(
            cv_skills=cv_skills,
            job_skills=job_skills or []
        )

        final_score = 0.6 * overlap_score + 0.4 * float(graphsage_score)

        results.append({
            "job_id": job_idx,
            "score": round(final_score, 4),
            "title": get_mapping_value(mapping, "job_titles", job_idx),
            "company": get_mapping_value(mapping, "job_companies", job_idx),
            "industry": get_mapping_value(mapping, "job_industries", job_idx),
            "skills": get_mapping_value(mapping, "job_skills", job_idx),
        })

    results = sorted(
        results,
        key=lambda item: (
            item["final_score"],
            item["overlap_skill_score"],
            item["graphsage_score"]
        ),
        reverse=True
    )

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
            cv_text=req.cv_text
        )

        return {
            "success": True,
            "total": len(results),
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

@app.post("/extract-cv")
@app.post("/extract-cv-url")
def extract_cv(req: ExtractCvUrlRequest):
    try:
        if not req.file_url or not req.file_url.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "file_url không được để trống"
                }
            )

        pdf_file = download_pdf_to_memory(req.file_url.strip())

        result = extract_cv_profile(
            pdf_path=pdf_file,
            industry_json_path=str(INDUSTRY_JSON_PATH)
        )

        return build_extract_cv_response(result)

    except (requests.RequestException, ValueError) as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Không thể tải hoặc đọc PDF từ file_url",
                "error": str(e)
            }
        )

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
@app.post("/jobs/upsert")
def upsert_job(req: IndexJobRequest):
    try:
        with GRAPH_LOCK:
            result = index_job_in_graph(req)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )
        
@app.get("/debug/job/{job_id}")
def debug_job(job_id: int):
    try:
        job_ids = ensure_mapping_list("job_ids", lambda idx: idx)

        if job_id not in job_ids:
            return {
                "success": False,
                "message": f"Job {job_id} không tồn tại trong graph"
            }

        node_idx = job_ids.index(job_id)

        similar_edge = graph["job", "similar_to", "job"]

        edge_index = similar_edge.edge_index.cpu()

        neighbors = []

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            if src == node_idx:
                weight = None

                if hasattr(similar_edge, "edge_weight"):
                    weight = float(
                        similar_edge.edge_weight[i].cpu()
                    )

                neighbors.append({
                    "node_idx": dst,
                    "job_id": job_ids[dst],
                    "title": mapping["job_titles"][dst],
                    "score": weight
                })

        return {
            "success": True,
            "job_id": job_id,
            "node_idx": node_idx,
            "title": mapping["job_titles"][node_idx],
            "skills": list(mapping["job_skills"][node_idx]),
            "industries": list(mapping["job_industries"][node_idx]),
            "neighbors": sorted(
                neighbors,
                key=lambda x: x["score"] or 0,
                reverse=True
            )
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )
