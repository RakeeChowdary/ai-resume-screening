"""
NLP Preprocessing Utilities
Handles text cleaning, tokenization, skill extraction, and feature engineering
"""

import re
import json
import string
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize resume/JD text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)           # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)                   # Remove emails
    text = re.sub(r'\+?\d[\d\s\-().]{8,}\d', '', text)   # Remove phone numbers
    text = re.sub(r'[^\w\s\+#\.]', ' ', text)             # Keep tech-relevant chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_sections(text: str) -> Dict[str, str]:
    """Extract common resume sections."""
    sections = {
        "summary": "", "skills": "", "experience": "",
        "education": "", "projects": "", "certifications": ""
    }
    section_patterns = {
        "summary": r'(?:summary|objective|profile|about)',
        "skills": r'(?:skills?|technologies|tech stack|competencies)',
        "experience": r'(?:experience|work history|employment|positions?)',
        "education": r'(?:education|academic|qualification)',
        "projects": r'(?:projects?|portfolio)',
        "certifications": r'(?:certifications?|certificates?|awards?|achievements?)'
    }
    lines = text.split('\n')
    current_section = "summary"
    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for section, pattern in section_patterns.items():
            if re.search(pattern, line_lower) and len(line_lower) < 50:
                current_section = section
                matched = True
                break
        if not matched and line.strip():
            sections[current_section] += line + "\n"
    return sections


# ─── Skill Extraction ──────────────────────────────────────────────────────────

SKILL_TAXONOMY = {
    "Python": ["python", "py"],
    "Java": ["java", "java8", "java11"],
    "JavaScript": ["javascript", "js", "node.js", "nodejs"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++", "cpp", "c plus plus"],
    "Go": ["golang", "go lang"],
    "R": [" r ", "r programming", "rstudio"],
    "Scala": ["scala"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Machine Learning": ["machine learning", "ml", "sklearn", "scikit-learn", "scikit learn"],
    "Deep Learning": ["deep learning", "dl", "neural network", "neural net"],
    "NLP": ["nlp", "natural language processing", "text mining"],
    "Computer Vision": ["computer vision", "cv", "image processing", "opencv"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "Keras": ["keras"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "BERT": ["bert", "roberta", "distilbert", "albert"],
    "GPT": ["gpt", "gpt-2", "gpt-3", "gpt-4", "chatgpt", "openai"],
    "XGBoost": ["xgboost", "xgb"],
    "LightGBM": ["lightgbm", "lgbm"],
    "Apache Spark": ["apache spark", "pyspark", "spark"],
    "Kafka": ["kafka", "apache kafka"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt", "data build tool"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],
    "Databricks": ["databricks"],
    "AWS": ["aws", "amazon web services", "s3", "ec2", "lambda", "sagemaker"],
    "GCP": ["gcp", "google cloud", "gcloud"],
    "Azure": ["azure", "microsoft azure"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform", "iac"],
    "React": ["react", "reactjs", "react.js"],
    "FastAPI": ["fastapi", "fast api"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Agile": ["agile", "scrum", "kanban", "sprint"],
    "Git": ["git", "github", "gitlab", "bitbucket"],
    "CI/CD": ["ci/cd", "ci cd", "jenkins", "github actions", "gitlab ci"],
    "MLflow": ["mlflow", "mlops", "model registry"],
    "Statistical Analysis": ["statistics", "statistical analysis", "hypothesis testing", "regression"],
}


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using the taxonomy."""
    text_lower = text.lower()
    found_skills = []
    for skill_name, aliases in SKILL_TAXONOMY.items():
        for alias in aliases:
            # Use word boundary for short terms
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill_name)
                break
    return list(set(found_skills))


def extract_experience_years(text: str) -> int:
    """Extract years of experience from text."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(m) for m in matches])
    return max(years) if years else 0


def extract_education_level(text: str) -> int:
    """Map education to numeric level."""
    text_lower = text.lower()
    if re.search(r'\bp\.?h\.?d\.?\b|doctor', text_lower):
        return 5
    elif re.search(r'\bm\.?tech\.?\b|\bm\.?sc\.?\b|\bmca\b|\bmba\b|\bm\.?eng', text_lower):
        return 4
    elif re.search(r'\bb\.?tech\.?\b|\bb\.?sc\.?\b|\bbca\b|\bb\.?eng', text_lower):
        return 3
    elif re.search(r'\bdiploma\b|\bassociate', text_lower):
        return 2
    return 1


def extract_cgpa(text: str) -> Optional[float]:
    """Extract CGPA/GPA from text."""
    patterns = [
        r'cgpa[:\s]+(\d+\.?\d*)',
        r'gpa[:\s]+(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*/\s*10',
        r'(\d+\.?\d*)\s*/\s*4\.0',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if val <= 4.0:
                val = val * 2.5   # Convert 4.0 scale to 10
            return min(val, 10.0)
    return None


# ─── Feature Engineering ───────────────────────────────────────────────────────

def compute_skill_match(resume_skills: List[str], required_skills: List[str],
                         preferred_skills: List[str]) -> Dict[str, float]:
    """Compute skill overlap metrics between resume and JD."""
    resume_set = set(s.lower() for s in resume_skills)
    req_set = set(s.lower() for s in required_skills)
    pref_set = set(s.lower() for s in preferred_skills)

    req_match = len(resume_set & req_set) / len(req_set) if req_set else 0
    pref_match = len(resume_set & pref_set) / len(pref_set) if pref_set else 0
    matched_req = list(resume_set & req_set)
    missing_req = list(req_set - resume_set)

    return {
        "required_skill_match": round(req_match, 4),
        "preferred_skill_match": round(pref_match, 4),
        "total_skills_count": len(resume_set),
        "matched_required_skills": matched_req,
        "missing_required_skills": missing_req,
        "combined_score": round(0.7 * req_match + 0.3 * pref_match, 4)
    }


def compute_experience_score(candidate_exp: int, required_exp: int) -> float:
    """Score experience relative to job requirement."""
    if candidate_exp >= required_exp:
        # Slight penalty for very overqualified
        excess = candidate_exp - required_exp
        return min(1.0, 1.0 - max(0, excess - 3) * 0.05)
    else:
        gap = required_exp - candidate_exp
        return max(0.0, 1.0 - gap * 0.2)


def engineer_features(resume: dict, jd: dict) -> Dict[str, float]:
    """Create feature vector for resume-JD pair."""
    r_text = resume.get("resume_text", "")
    r_skills = resume.get("skills", extract_skills(r_text))
    r_exp = resume.get("experience_years", extract_experience_years(r_text))
    r_edu_level = extract_education_level(r_text)
    r_cgpa = resume.get("education", {}).get("cgpa") or extract_cgpa(r_text) or 7.0

    skill_metrics = compute_skill_match(
        r_skills, jd.get("required_skills", []), jd.get("preferred_skills", [])
    )
    exp_score = compute_experience_score(r_exp, jd.get("min_experience", 2))

    return {
        "required_skill_match": skill_metrics["required_skill_match"],
        "preferred_skill_match": skill_metrics["preferred_skill_match"],
        "skill_combined_score": skill_metrics["combined_score"],
        "total_skills": skill_metrics["total_skills_count"],
        "experience_years": r_exp,
        "experience_score": exp_score,
        "education_level": r_edu_level,
        "cgpa_normalized": min(r_cgpa / 10.0, 1.0),
        "certifications_count": len(resume.get("certifications", [])),
        "projects_count": len(resume.get("projects", [])),
        "resume_length": len(r_text.split()),
        "matched_required": len(skill_metrics["matched_required_skills"]),
        "missing_required": len(skill_metrics["missing_required_skills"]),
    }


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_heuristic_score(features: Dict[str, float]) -> float:
    """Compute weighted heuristic ranking score."""
    weights = {
        "skill_combined_score": 0.40,
        "experience_score": 0.25,
        "education_level": 0.10,       # normalized below
        "cgpa_normalized": 0.10,
        "certifications_count": 0.05,  # normalized below
        "projects_count": 0.05,        # normalized below
        "resume_length": 0.05,         # normalized below
    }
    edu_norm = min(features["education_level"] / 5.0, 1.0)
    cert_norm = min(features["certifications_count"] / 5.0, 1.0)
    proj_norm = min(features["projects_count"] / 5.0, 1.0)
    len_norm = min(features["resume_length"] / 600, 1.0)

    score = (
        weights["skill_combined_score"] * features["skill_combined_score"] +
        weights["experience_score"] * features["experience_score"] +
        weights["education_level"] * edu_norm +
        weights["cgpa_normalized"] * features["cgpa_normalized"] +
        weights["certifications_count"] * cert_norm +
        weights["projects_count"] * proj_norm +
        weights["resume_length"] * len_norm
    )
    return round(score, 4)