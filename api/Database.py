"""
Database Models — SQLAlchemy ORM for PostgreSQL
Tables: candidates, job_descriptions, rankings, ranking_results
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text,
    DateTime, ForeignKey, JSON, Boolean, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import os

Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/resume_screening")

# ─── Models ───────────────────────────────────────────────────────────────────

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    email = Column(String(200), index=True)
    phone = Column(String(50))
    experience_years = Column(Integer, default=0)
    skills = Column(JSON, default=list)          # ["Python", "SQL", ...]
    education = Column(JSON, default=dict)        # {degree, university, cgpa}
    certifications = Column(JSON, default=list)
    projects = Column(JSON, default=list)
    resume_text = Column(Text)
    resume_filename = Column(String(300))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    ranking_results = relationship("RankingResult", back_populates="candidate")

    __table_args__ = (
        Index("idx_candidate_experience", "experience_years"),
        Index("idx_candidate_active", "is_active"),
    )

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "email": self.email,
            "experience_years": self.experience_years,
            "skills": self.skills or [],
            "education": self.education or {},
            "certifications": self.certifications or [],
        }


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    jd_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(JSON, default=list)
    preferred_skills = Column(JSON, default=list)
    min_experience = Column(Integer, default=0)
    location = Column(String(200))
    department = Column(String(200))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    rankings = relationship("Ranking", back_populates="job_description")

    def to_dict(self):
        return {
            "jd_id": self.jd_id,
            "title": self.title,
            "required_skills": self.required_skills or [],
            "preferred_skills": self.preferred_skills or [],
            "min_experience": self.min_experience,
        }


class Ranking(Base):
    __tablename__ = "rankings"

    id = Column(Integer, primary_key=True, index=True)
    ranking_id = Column(String(50), unique=True, index=True, nullable=False)
    jd_id = Column(String(50), ForeignKey("job_descriptions.jd_id"), nullable=False)
    model_used = Column(String(50), default="ensemble")
    total_candidates = Column(Integer, default=0)
    created_by = Column(String(200))
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    job_description = relationship("JobDescription", back_populates="rankings")
    results = relationship("RankingResult", back_populates="ranking",
                           order_by="RankingResult.rank")

    def to_dict(self):
        return {
            "ranking_id": self.ranking_id,
            "jd_id": self.jd_id,
            "model_used": self.model_used,
            "total_candidates": self.total_candidates,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RankingResult(Base):
    __tablename__ = "ranking_results"

    id = Column(Integer, primary_key=True, index=True)
    ranking_id = Column(String(50), ForeignKey("rankings.ranking_id"), nullable=False)
    candidate_id = Column(String(50), ForeignKey("candidates.candidate_id"), nullable=False)
    rank = Column(Integer, nullable=False)
    ensemble_score = Column(Float, default=0.0)
    tfidf_score = Column(Float, default=0.0)
    feature_score = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    match_percentage = Column(Float, default=0.0)
    matched_skills = Column(JSON, default=list)
    missing_skills = Column(JSON, default=list)
    recommendation = Column(String(100))
    shortlisted = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    ranking = relationship("Ranking", back_populates="results")
    candidate = relationship("Candidate", back_populates="ranking_results")

    __table_args__ = (
        Index("idx_result_ranking_rank", "ranking_id", "rank"),
        Index("idx_result_score", "ensemble_score"),
    )


# ─── Database Utilities ───────────────────────────────────────────────────────

def get_engine(db_url: str = None):
    url = db_url or DATABASE_URL
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)

def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")

def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# ─── SQL Queries (raw) ────────────────────────────────────────────────────────

SQL_QUERIES = {
    "top_candidates_per_jd": """
        SELECT
            rr.rank,
            c.name,
            c.email,
            c.experience_years,
            rr.ensemble_score,
            rr.match_percentage,
            rr.recommendation,
            rr.matched_skills
        FROM ranking_results rr
        JOIN candidates c ON c.candidate_id = rr.candidate_id
        WHERE rr.ranking_id = :ranking_id
        ORDER BY rr.rank ASC
        LIMIT :limit;
    """,

    "skill_demand_analysis": """
        SELECT
            skill_value,
            COUNT(*) AS demand_count
        FROM job_descriptions,
             json_array_elements_text(required_skills) AS skill_value
        WHERE is_active = TRUE
        GROUP BY skill_value
        ORDER BY demand_count DESC
        LIMIT 20;
    """,

    "candidate_pool_analysis": """
        SELECT
            CASE
                WHEN experience_years BETWEEN 0 AND 2 THEN '0-2 years'
                WHEN experience_years BETWEEN 3 AND 5 THEN '3-5 years'
                WHEN experience_years BETWEEN 6 AND 10 THEN '6-10 years'
                ELSE '10+ years'
            END AS experience_bucket,
            COUNT(*) AS candidate_count,
            ROUND(AVG((education->>'cgpa')::float), 2) AS avg_cgpa
        FROM candidates
        WHERE is_active = TRUE
        GROUP BY experience_bucket
        ORDER BY experience_bucket;
    """,

    "score_distribution": """
        SELECT
            jd.title,
            COUNT(rr.id) AS ranked_candidates,
            ROUND(AVG(rr.ensemble_score)::numeric, 3) AS avg_score,
            ROUND(MAX(rr.ensemble_score)::numeric, 3) AS max_score,
            ROUND(MIN(rr.ensemble_score)::numeric, 3) AS min_score
        FROM ranking_results rr
        JOIN rankings r ON r.ranking_id = rr.ranking_id
        JOIN job_descriptions jd ON jd.jd_id = r.jd_id
        GROUP BY jd.title;
    """
}

if __name__ == "__main__":
    print("Database URL:", DATABASE_URL)
    print("Models defined:", [Base.__tablename__ for Base in [Candidate, JobDescription, Ranking, RankingResult]])
    print("\nTo create tables:")
    print("  engine = get_engine('postgresql://user:pass@localhost/resume_db')")
    print("  create_tables(engine)")