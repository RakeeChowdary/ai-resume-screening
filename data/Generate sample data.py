"""
Sample Data Generator for AI Resume Screening System
Generates realistic synthetic resume and job description data
"""

import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── Skill pools ─────────────────────────────────────────────────────────────

TECH_SKILLS = {
    "programming": ["Python", "Java", "JavaScript", "C++", "Go", "Rust", "TypeScript", "R", "Scala", "Kotlin"],
    "ml_ai": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "scikit-learn",
              "Keras", "XGBoost", "LightGBM", "Hugging Face", "BERT", "GPT", "LLM Fine-tuning"],
    "data": ["SQL", "PostgreSQL", "MongoDB", "Redis", "Apache Spark", "Hadoop", "Kafka", "Airflow",
             "dbt", "Snowflake", "BigQuery", "Databricks", "Pandas", "NumPy"],
    "cloud": ["AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "CI/CD", "GitHub Actions"],
    "web": ["React", "Node.js", "FastAPI", "Django", "Flask", "REST APIs", "GraphQL", "Microservices"],
    "viz": ["Tableau", "Power BI", "Matplotlib", "Seaborn", "Plotly", "D3.js"],
    "soft": ["Leadership", "Communication", "Problem Solving", "Teamwork", "Agile", "Scrum", "Project Management"]
}

UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "NIT Trichy", "BITS Pilani",
    "Stanford University", "MIT", "Carnegie Mellon", "UC Berkeley", "University of Michigan",
    "VIT University", "Anna University", "Osmania University", "JNTU Hyderabad", "Delhi University"
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Uber", "Airbnb",
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Accenture", "Deloitte",
    "Goldman Sachs", "JPMorgan", "McKinsey", "Cognizant", "Capgemini"
]

DEGREES = ["B.Tech", "M.Tech", "B.Sc", "M.Sc", "MBA", "Ph.D", "BCA", "MCA"]

SPECIALIZATIONS = ["Computer Science", "Data Science", "Information Technology",
                   "Electronics & Communication", "Electrical Engineering", "Mathematics & Computing"]

JOB_ROLES = [
    {
        "title": "Senior Data Scientist",
        "required_skills": ["Python", "Machine Learning", "SQL", "TensorFlow", "Statistical Analysis"],
        "preferred_skills": ["PyTorch", "MLflow", "Spark", "AWS", "Deep Learning"],
        "min_experience": 4,
        "description": "Lead end-to-end ML projects, build predictive models, collaborate with product teams."
    },
    {
        "title": "ML Engineer",
        "required_skills": ["Python", "PyTorch", "Docker", "REST APIs", "Deep Learning"],
        "preferred_skills": ["Kubernetes", "MLOps", "C++", "CUDA", "Transformer Models"],
        "min_experience": 3,
        "description": "Deploy and scale ML models in production, optimize inference pipelines."
    },
    {
        "title": "Data Engineer",
        "required_skills": ["Python", "SQL", "Apache Spark", "Kafka", "ETL Pipelines"],
        "preferred_skills": ["Airflow", "dbt", "Snowflake", "Terraform", "Databricks"],
        "min_experience": 3,
        "description": "Build and maintain data pipelines, ensure data quality and availability."
    },
    {
        "title": "NLP Research Scientist",
        "required_skills": ["Python", "NLP", "BERT", "Transformers", "Deep Learning"],
        "preferred_skills": ["LLM Fine-tuning", "PyTorch", "Hugging Face", "GPT", "Research Publications"],
        "min_experience": 5,
        "description": "Research and develop state-of-the-art NLP models for text understanding."
    },
    {
        "title": "Full Stack Developer",
        "required_skills": ["JavaScript", "React", "Node.js", "SQL", "REST APIs"],
        "preferred_skills": ["TypeScript", "GraphQL", "Docker", "AWS", "Microservices"],
        "min_experience": 2,
        "description": "Build scalable web applications, collaborate with designers and product managers."
    },
    {
        "title": "Data Analyst",
        "required_skills": ["SQL", "Python", "Tableau", "Excel", "Data Visualization"],
        "preferred_skills": ["Power BI", "R", "Machine Learning", "Pandas", "Statistical Analysis"],
        "min_experience": 1,
        "description": "Analyze business data, create dashboards, provide actionable insights."
    }
]

# ─── Resume generator ─────────────────────────────────────────────────────────

def generate_resume(candidate_id: int, target_role: dict = None) -> dict:
    first_names = ["Rahul", "Priya", "Amit", "Sneha", "Arjun", "Kavya", "Vikram", "Anjali",
                   "Rohit", "Divya", "Kiran", "Meera", "Sanjay", "Pooja", "Arun", "Nisha",
                   "Alex", "Sarah", "Michael", "Emma", "David", "Lisa", "James", "Jennifer"]
    last_names = ["Sharma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Nair", "Joshi",
                  "Verma", "Mehta", "Iyer", "Rao", "Smith", "Johnson", "Williams", "Brown"]

    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    email = f"{name.lower().replace(' ', '.')}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
    phone = f"+91-{random.randint(7000000000, 9999999999)}"
    experience_years = random.randint(0, 12)

    # Skills
    all_skills = []
    for category in TECH_SKILLS.values():
        sample_size = random.randint(1, min(4, len(category)))
        all_skills.extend(random.sample(category, sample_size))

    # If targeting a role, add some of its required/preferred skills
    if target_role:
        overlap = random.randint(2, len(target_role["required_skills"]))
        all_skills.extend(random.sample(target_role["required_skills"],
                                         min(overlap, len(target_role["required_skills"]))))

    skills = list(set(all_skills))[:random.randint(8, 20)]

    # Education
    degree = random.choice(DEGREES)
    university = random.choice(UNIVERSITIES)
    specialization = random.choice(SPECIALIZATIONS)
    grad_year = datetime.now().year - experience_years - random.randint(0, 2)
    cgpa = round(random.uniform(6.0, 9.8), 2)

    # Work experience
    work_experience = []
    remaining_exp = experience_years
    current_year = datetime.now().year

    while remaining_exp > 0:
        duration = random.randint(1, min(3, remaining_exp))
        company = random.choice(COMPANIES)
        role_titles = ["Software Engineer", "Data Scientist", "ML Engineer", "Data Analyst",
                       "Backend Developer", "Research Engineer", "Junior Developer", "Senior Engineer"]
        responsibilities = [
            f"Developed {random.choice(['machine learning', 'data', 'software'])} solutions using {random.choice(skills[:3])}",
            f"Improved model accuracy by {random.randint(5, 35)}% through feature engineering",
            f"Collaborated with cross-functional teams of {random.randint(3, 15)} engineers",
            f"Reduced processing time by {random.randint(20, 60)}% via pipeline optimization",
            f"Deployed models serving {random.randint(10, 500)}K daily requests"
        ]
        work_experience.append({
            "company": company,
            "role": random.choice(role_titles),
            "duration_years": duration,
            "end_year": current_year,
            "responsibilities": random.sample(responsibilities, random.randint(2, 4))
        })
        current_year -= duration
        remaining_exp -= duration

    # Projects
    project_names = [
        "Real-time Fraud Detection System", "NLP Sentiment Analysis Engine",
        "Customer Churn Prediction Model", "Image Classification CNN",
        "Recommendation Engine", "Time Series Forecasting Dashboard",
        "Automated Resume Parser", "Chat Bot with BERT", "Data Pipeline on AWS"
    ]
    projects = []
    for _ in range(random.randint(1, 4)):
        proj_skills = random.sample(skills, min(3, len(skills)))
        projects.append({
            "name": random.choice(project_names),
            "tech_stack": proj_skills,
            "description": f"Built using {', '.join(proj_skills[:2])} achieving {random.randint(80, 99)}% accuracy"
        })

    # Certifications
    certs = ["AWS Certified ML Specialist", "Google Professional Data Engineer",
             "TensorFlow Developer Certificate", "Databricks Certified Associate",
             "Microsoft Azure Data Scientist", "Coursera Deep Learning Specialization"]
    certifications = random.sample(certs, random.randint(0, 3))

    # Build resume text
    responsibilities_text = ""
    for exp in work_experience:
        resp_items = "\n    • ".join(exp["responsibilities"])
        responsibilities_text += f"""
  {exp['role']} at {exp['company']} ({exp['duration_years']} year{'s' if exp['duration_years']>1 else ''})
    • {resp_items}"""

    projects_text = ""
    for proj in projects:
        projects_text += f"\n  • {proj['name']}: {proj['description']}"

    resume_text = f"""
{name}
Email: {email} | Phone: {phone} | LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}

SUMMARY
Experienced {random.choice(['data professional', 'software engineer', 'ML practitioner'])} with {experience_years} years of experience
in {random.choice(skills[:3])}. Passionate about building scalable solutions.

SKILLS
{', '.join(skills)}

EDUCATION
{degree} in {specialization}
{university} | Graduated {grad_year} | CGPA: {cgpa}/10

WORK EXPERIENCE{responsibilities_text}

PROJECTS{projects_text}

CERTIFICATIONS
{chr(10).join(['  • ' + c for c in certifications]) if certifications else '  • None listed'}
    """.strip()

    return {
        "candidate_id": f"CAND_{candidate_id:04d}",
        "name": name,
        "email": email,
        "phone": phone,
        "experience_years": experience_years,
        "skills": skills,
        "education": {
            "degree": degree,
            "specialization": specialization,
            "university": university,
            "grad_year": grad_year,
            "cgpa": cgpa
        },
        "work_experience": work_experience,
        "projects": projects,
        "certifications": certifications,
        "resume_text": resume_text
    }


def generate_job_description(job_role: dict, jd_id: int) -> dict:
    jd_text = f"""
JOB TITLE: {job_role['title']}
LOCATION: Hyderabad / Bangalore / Remote

JOB DESCRIPTION
{job_role['description']}

We are looking for a talented {job_role['title']} to join our growing team.
You will work on cutting-edge problems with a world-class engineering team.

REQUIRED QUALIFICATIONS
• {job_role['min_experience']}+ years of professional experience
• Strong proficiency in: {', '.join(job_role['required_skills'])}
• Bachelor's or Master's degree in Computer Science, Engineering, or related field
• Excellent problem-solving and communication skills

PREFERRED QUALIFICATIONS
• Experience with: {', '.join(job_role['preferred_skills'])}
• Published research papers or open-source contributions
• Experience with Agile/Scrum methodologies

RESPONSIBILITIES
• Design and implement scalable {job_role['title'].lower()} solutions
• Collaborate with cross-functional teams to define requirements
• Mentor junior team members and conduct code reviews
• Stay current with latest research and industry trends
• Document and present findings to stakeholders

BENEFITS
• Competitive salary and equity package
• Health insurance and wellness benefits
• Flexible work arrangements
• Learning and development budget
    """.strip()

    return {
        "jd_id": f"JD_{jd_id:03d}",
        "title": job_role["title"],
        "required_skills": job_role["required_skills"],
        "preferred_skills": job_role["preferred_skills"],
        "min_experience": job_role["min_experience"],
        "description": jd_text
    }


def generate_dataset(n_resumes: int = 200) -> tuple:
    print(f"Generating {n_resumes} resumes and {len(JOB_ROLES)} job descriptions...")

    # Generate job descriptions
    job_descriptions = [generate_job_description(role, i) for i, role in enumerate(JOB_ROLES)]

    # Generate resumes (mix of targeted and random)
    resumes = []
    for i in range(n_resumes):
        target = random.choice(JOB_ROLES) if random.random() > 0.3 else None
        resumes.append(generate_resume(i + 1, target))

    print(f"✓ Generated {len(resumes)} resumes")
    print(f"✓ Generated {len(job_descriptions)} job descriptions")
    return resumes, job_descriptions


if __name__ == "__main__":
    resumes, job_descriptions = generate_dataset(200)

    # Save as JSON
    with open("raw/resumes.json", "w") as f:
        json.dump(resumes, f, indent=2)
    with open("raw/job_descriptions.json", "w") as f:
        json.dump(job_descriptions, f, indent=2)

    # Save as CSV (flat version)
    df_resumes = pd.DataFrame([{
        "candidate_id": r["candidate_id"],
        "name": r["name"],
        "email": r["email"],
        "experience_years": r["experience_years"],
        "skills_count": len(r["skills"]),
        "skills": ", ".join(r["skills"]),
        "degree": r["education"]["degree"],
        "university": r["education"]["university"],
        "cgpa": r["education"]["cgpa"],
        "certifications_count": len(r["certifications"]),
        "projects_count": len(r["projects"]),
        "resume_text": r["resume_text"]
    } for r in resumes])

    df_jd = pd.DataFrame([{
        "jd_id": jd["jd_id"],
        "title": jd["title"],
        "required_skills": ", ".join(jd["required_skills"]),
        "preferred_skills": ", ".join(jd["preferred_skills"]),
        "min_experience": jd["min_experience"],
        "description": jd["description"]
    } for jd in job_descriptions])

    df_resumes.to_csv("processed/resumes.csv", index=False)
    df_jd.to_csv("processed/job_descriptions.csv", index=False)

    print("\n✓ Data saved to raw/ and processed/")
    print(f"Resume stats:\n{df_resumes[['experience_years','skills_count','cgpa']].describe()}")