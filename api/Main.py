"""
FastAPI Backend — AI Resume Screening & Candidate Ranking API
Endpoints: upload resume, rank candidates, get analytics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json, os, io, re, time, hashlib
from datetime import datetime

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Resume Screening API",
    description="NLP-powered recruitment assistant that ranks resumes against job descriptions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory store (replace with PostgreSQL in production) ──────────────────

_store = {
    "job_descriptions": {},
    "candidates": {},
    "rankings": {},
    "analytics": {
        "total_resumes_processed": 0,
        "total_rankings": 0,
        "api_calls": 0
    }
}

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class JobDescription(BaseModel):
    title: str = Field(..., example="Senior Data Scientist")
    description: str = Field(..., example="We are looking for...")
    required_skills: List[str] = Field(default_factory=list, example=["Python", "Machine Learning"])
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience: int = Field(default=2, ge=0, le=30)
    location: Optional[str] = "Remote"

class ResumeInput(BaseModel):
    name: str
    email: str
    resume_text: str
    experience_years: Optional[int] = 0
    skills: Optional[List[str]] = []
    education: Optional[Dict] = {}

class RankingRequest(BaseModel):
    jd_id: str
    candidate_ids: Optional[List[str]] = None   # None = rank all
    top_k: int = Field(default=10, ge=1, le=100)
    model: str = Field(default="ensemble", pattern="^(tfidf|features|semantic|ensemble)$")

class RankingResult(BaseModel):
    candidate_id: str
    name: str
    rank: int
    ensemble_score: float
    tfidf_score: float
    feature_score: float
    semantic_score: float
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: int
    recommendation: str

# ─── Helper Functions ─────────────────────────────────────────────────────────

def generate_id(prefix: str) -> str:
    ts = str(time.time()).encode()
    return f"{prefix}_{hashlib.md5(ts).hexdigest()[:8].upper()}"

def extract_skills_simple(text: str, skill_list: List[str]) -> List[str]:
    """Simple skill extraction for demo without NLP dependencies."""
    found = []
    text_lower = text.lower()
    for skill in skill_list:
        if skill.lower() in text_lower:
            found.append(skill)
    return found

def compute_score(resume: dict, jd: dict) -> Dict[str, Any]:
    """Compute ranking scores for a resume-JD pair."""
    r_text = resume.get("resume_text", "").lower()
    r_skills = set(s.lower() for s in resume.get("skills", []))
    r_exp = resume.get("experience_years", 0)

    req_skills = set(s.lower() for s in jd.get("required_skills", []))
    pref_skills = set(s.lower() for s in jd.get("preferred_skills", []))
    jd_text = jd.get("description", "").lower()

    # Skill overlap
    req_match = len(r_skills & req_skills) / max(len(req_skills), 1)
    pref_match = len(r_skills & pref_skills) / max(len(pref_skills), 1)
    skill_score = 0.7 * req_match + 0.3 * pref_match

    # TF-IDF approximation via word overlap
    r_words = set(r_text.split())
    jd_words = set(jd_text.split())
    tfidf_approx = len(r_words & jd_words) / max(len(jd_words), 1)
    tfidf_score = min(tfidf_approx * 3, 1.0)

    # Experience score
    min_exp = jd.get("min_experience", 2)
    if r_exp >= min_exp:
        exp_score = min(1.0, 1.0 - max(0, r_exp - min_exp - 3) * 0.05)
    else:
        exp_score = max(0.0, 1.0 - (min_exp - r_exp) * 0.2)

    # Education
    edu_text = json.dumps(resume.get("education", {})).lower()
    edu_score = 0.6
    if "ph.d" in edu_text or "phd" in edu_text: edu_score = 1.0
    elif "m.tech" in edu_text or "m.sc" in edu_text or "mca" in edu_text: edu_score = 0.85
    elif "b.tech" in edu_text or "b.sc" in edu_text: edu_score = 0.70

    # Certification bonus
    cert_bonus = min(len(resume.get("certifications", [])) * 0.03, 0.10)

    feature_score = (0.40 * skill_score + 0.30 * exp_score +
                     0.15 * edu_score + 0.15 * cert_bonus)
    semantic_score = min((tfidf_score + skill_score) / 2 * 1.1, 1.0)
    ensemble = 0.25 * tfidf_score + 0.45 * feature_score + 0.30 * semantic_score

    # Recommendation
    if ensemble >= 0.75: rec = "🟢 Strongly Recommended"
    elif ensemble >= 0.55: rec = "🟡 Recommended"
    elif ensemble >= 0.35: rec = "🟠 Consider with Reservations"
    else: rec = "🔴 Not Recommended"

    matched = list(r_skills & req_skills) + list(r_skills & pref_skills)
    missing = list(req_skills - r_skills)

    return {
        "tfidf_score": round(tfidf_score, 4),
        "feature_score": round(feature_score, 4),
        "semantic_score": round(semantic_score, 4),
        "ensemble_score": round(ensemble, 4),
        "match_percentage": round(ensemble * 100, 1),
        "matched_skills": list(set(matched))[:10],
        "missing_skills": missing[:10],
        "recommendation": rec
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Resume Screening API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": ["/docs", "/jd", "/candidates", "/rank", "/analytics"]
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "candidates": len(_store["candidates"]),
            "jds": len(_store["job_descriptions"])}

# ── Job Descriptions ──

@app.post("/jd", tags=["Job Descriptions"], status_code=201)
async def create_jd(jd: JobDescription):
    """Create a new job description."""
    jd_id = generate_id("JD")
    _store["job_descriptions"][jd_id] = {**jd.dict(), "jd_id": jd_id,
                                          "created_at": datetime.utcnow().isoformat()}
    _store["analytics"]["api_calls"] += 1
    return {"jd_id": jd_id, "message": "Job description created successfully"}

@app.get("/jd", tags=["Job Descriptions"])
async def list_jds():
    """List all job descriptions."""
    return {"count": len(_store["job_descriptions"]),
            "job_descriptions": list(_store["job_descriptions"].values())}

@app.get("/jd/{jd_id}", tags=["Job Descriptions"])
async def get_jd(jd_id: str):
    if jd_id not in _store["job_descriptions"]:
        raise HTTPException(404, detail=f"JD '{jd_id}' not found")
    return _store["job_descriptions"][jd_id]

@app.delete("/jd/{jd_id}", tags=["Job Descriptions"])
async def delete_jd(jd_id: str):
    if jd_id not in _store["job_descriptions"]:
        raise HTTPException(404, detail=f"JD '{jd_id}' not found")
    del _store["job_descriptions"][jd_id]
    return {"message": f"JD '{jd_id}' deleted"}

# ── Candidates ──

@app.post("/candidates", tags=["Candidates"], status_code=201)
async def add_candidate(resume: ResumeInput):
    """Add a candidate resume."""
    cid = generate_id("CAND")
    _store["candidates"][cid] = {
        **resume.dict(), "candidate_id": cid,
        "created_at": datetime.utcnow().isoformat()
    }
    _store["analytics"]["total_resumes_processed"] += 1
    _store["analytics"]["api_calls"] += 1
    return {"candidate_id": cid, "message": "Candidate added successfully"}

@app.post("/candidates/bulk", tags=["Candidates"], status_code=201)
async def add_candidates_bulk(resumes: List[ResumeInput]):
    """Bulk upload candidates."""
    ids = []
    for resume in resumes:
        cid = generate_id("CAND")
        _store["candidates"][cid] = {
            **resume.dict(), "candidate_id": cid,
            "created_at": datetime.utcnow().isoformat()
        }
        ids.append(cid)
    _store["analytics"]["total_resumes_processed"] += len(resumes)
    return {"count": len(ids), "candidate_ids": ids}

@app.post("/candidates/upload", tags=["Candidates"])
async def upload_resume_file(file: UploadFile = File(...)):
    """Upload a PDF/DOCX resume file."""
    content = await file.read()
    # In production: use PyPDF2 / python-docx to extract text
    text = content.decode('utf-8', errors='ignore')
    cid = generate_id("CAND")
    _store["candidates"][cid] = {
        "candidate_id": cid, "name": file.filename.replace(".txt", ""),
        "email": "", "resume_text": text, "experience_years": 0,
        "skills": [], "education": {},
        "created_at": datetime.utcnow().isoformat()
    }
    _store["analytics"]["total_resumes_processed"] += 1
    return {"candidate_id": cid, "filename": file.filename, "chars": len(text)}

@app.get("/candidates", tags=["Candidates"])
async def list_candidates():
    """List all candidates."""
    candidates = [{"candidate_id": v["candidate_id"], "name": v["name"],
                   "email": v["email"], "experience_years": v.get("experience_years", 0),
                   "skills_count": len(v.get("skills", []))}
                  for v in _store["candidates"].values()]
    return {"count": len(candidates), "candidates": candidates}

@app.get("/candidates/{candidate_id}", tags=["Candidates"])
async def get_candidate(candidate_id: str):
    if candidate_id not in _store["candidates"]:
        raise HTTPException(404, detail=f"Candidate '{candidate_id}' not found")
    return _store["candidates"][candidate_id]

# ── Ranking ──

@app.post("/rank", tags=["Ranking"])
async def rank_candidates(request: RankingRequest):
    """Rank candidates against a job description."""
    if request.jd_id not in _store["job_descriptions"]:
        raise HTTPException(404, detail=f"JD '{request.jd_id}' not found")

    jd = _store["job_descriptions"][request.jd_id]

    # Select candidates to rank
    if request.candidate_ids:
        candidates = {cid: _store["candidates"][cid]
                      for cid in request.candidate_ids
                      if cid in _store["candidates"]}
    else:
        candidates = _store["candidates"]

    if not candidates:
        raise HTTPException(400, detail="No valid candidates found")

    # Score all candidates
    scored = []
    for cid, candidate in candidates.items():
        scores = compute_score(candidate, jd)
        scored.append({
            "candidate_id": cid,
            "name": candidate.get("name", "Unknown"),
            "experience_years": candidate.get("experience_years", 0),
            **scores
        })

    # Sort by ensemble score
    scored.sort(key=lambda x: x["ensemble_score"], reverse=True)
    for i, r in enumerate(scored[:request.top_k]):
        r["rank"] = i + 1

    ranking_id = generate_id("RANK")
    result = {
        "ranking_id": ranking_id,
        "jd_id": request.jd_id,
        "jd_title": jd["title"],
        "model_used": request.model,
        "total_candidates": len(candidates),
        "ranked_at": datetime.utcnow().isoformat(),
        "results": scored[:request.top_k]
    }
    _store["rankings"][ranking_id] = result
    _store["analytics"]["total_rankings"] += 1

    return result

@app.get("/rank/{ranking_id}", tags=["Ranking"])
async def get_ranking(ranking_id: str):
    if ranking_id not in _store["rankings"]:
        raise HTTPException(404, detail=f"Ranking '{ranking_id}' not found")
    return _store["rankings"][ranking_id]

@app.get("/rank", tags=["Ranking"])
async def list_rankings():
    rankings = [{"ranking_id": k, "jd_id": v["jd_id"], "jd_title": v["jd_title"],
                 "total_candidates": v["total_candidates"], "ranked_at": v["ranked_at"]}
                for k, v in _store["rankings"].items()]
    return {"count": len(rankings), "rankings": rankings}

# ── Analytics ──

@app.get("/analytics", tags=["Analytics"])
async def get_analytics():
    """Get system-wide analytics."""
    skill_counts: Dict[str, int] = {}
    exp_dist = []
    for c in _store["candidates"].values():
        exp_dist.append(c.get("experience_years", 0))
        for s in c.get("skills", []):
            skill_counts[s] = skill_counts.get(s, 0) + 1

    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    return {
        **_store["analytics"],
        "total_jds": len(_store["job_descriptions"]),
        "top_skills_in_pool": [{"skill": s, "count": c} for s, c in top_skills],
        "experience_distribution": {
            "0-2 years": sum(1 for e in exp_dist if e <= 2),
            "3-5 years": sum(1 for e in exp_dist if 3 <= e <= 5),
            "6-10 years": sum(1 for e in exp_dist if 6 <= e <= 10),
            "10+ years": sum(1 for e in exp_dist if e > 10),
        },
        "avg_experience": round(sum(exp_dist) / max(len(exp_dist), 1), 1)
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Resume Screening API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)