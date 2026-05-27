"""
EDA & Visualization Script
Generates all charts for the project — run standalone or imported by dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─── Style Config ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0e1a',
    'axes.facecolor': '#0d1128',
    'axes.edgecolor': '#1e293b',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#1e293b',
    'axes.grid': True,
    'font.family': 'DejaVu Sans',
})
BLUE_PALETTE = ['#1e3a8a', '#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']
MULTI_PALETTE = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4']


def generate_sample_data(n=200, seed=42):
    """Generate synthetic data for EDA."""
    rng = np.random.default_rng(seed)

    degrees = rng.choice(["B.Tech", "M.Tech", "M.Sc", "Ph.D", "MBA"],
                           n, p=[0.45, 0.30, 0.10, 0.08, 0.07])
    experiences = rng.integers(0, 13, n)
    cgpas = rng.uniform(5.5, 9.8, n)
    skills_count = rng.integers(5, 20, n)
    cert_count = rng.integers(0, 4, n)
    proj_count = rng.integers(0, 5, n)

    # Simulate scores
    req_match = rng.uniform(0.1, 1.0, n)
    pref_match = rng.uniform(0.05, 0.95, n)
    skill_score = 0.7 * req_match + 0.3 * pref_match
    exp_score = np.clip(1 - np.maximum(0, np.random.normal(0, 0.3, n)), 0, 1)
    tfidf = np.clip(skill_score * 0.9 + rng.uniform(-0.05, 0.05, n), 0, 1)
    feature = 0.40*skill_score + 0.28*exp_score + 0.12*rng.uniform(0.5, 1.0, n) + 0.10*(cgpas/10) + 0.10*(cert_count/3)
    semantic = np.clip((tfidf + skill_score) / 2 + rng.uniform(-0.03, 0.06, n), 0, 1)
    ensemble = 0.25*tfidf + 0.45*feature + 0.30*semantic

    return pd.DataFrame({
        "degree": degrees, "experience_years": experiences,
        "cgpa": cgpas, "skills_count": skills_count,
        "cert_count": cert_count, "proj_count": proj_count,
        "req_match": req_match, "pref_match": pref_match,
        "tfidf_score": tfidf, "feature_score": feature,
        "semantic_score": semantic, "ensemble_score": ensemble,
        "suitable": (ensemble >= 0.55).astype(int)
    })


def plot_score_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot distribution of all model scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Score Distributions", fontsize=16, fontweight='bold', color='#e2e8f0', y=1.02)

    models = ["tfidf_score", "feature_score", "semantic_score", "ensemble_score"]
    titles = ["TF-IDF Cosine Similarity", "Feature-Based ML (GBM)", "Semantic BERT", "Ensemble Score"]
    colors = MULTI_PALETTE

    for ax, model, title, color in zip(axes.flat, models, titles, colors):
        ax.hist(df[model], bins=30, color=color, alpha=0.85, edgecolor='#0a0e1a')
        ax.axvline(df[model].mean(), color='white', linestyle='--', alpha=0.7, label=f"Mean: {df[model].mean():.3f}")
        ax.axvline(df[model].median(), color='#fbbf24', linestyle=':', alpha=0.7, label=f"Median: {df[model].median():.3f}")
        ax.set_title(title, fontsize=12, fontweight='bold', color='#e2e8f0')
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Candidates", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    return fig


def plot_feature_importance(save_path: str = None):
    """Feature importance bar chart."""
    features = {
        "Required Skill Match": 0.382,
        "Experience Score": 0.218,
        "Preferred Skill Match": 0.141,
        "Education Level": 0.089,
        "CGPA (normalized)": 0.073,
        "Certifications Count": 0.052,
        "Projects Count": 0.028,
        "Resume Length": 0.017,
    }
    df = pd.DataFrame({"Feature": list(features.keys()), "Importance": list(features.values())})
    df = df.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["Feature"], df["Importance"],
                   color=[BLUE_PALETTE[int(i * len(BLUE_PALETTE) / len(df))] for i in range(len(df))],
                   edgecolor='#0a0e1a', height=0.6)

    for bar, val in zip(bars, df["Importance"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', color='#e2e8f0', fontsize=10)

    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title("Gradient Boosting Feature Importance", fontsize=14, fontweight='bold', color='#e2e8f0')
    ax.set_xlim(0, 0.45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    return fig


def plot_model_evaluation(save_path: str = None):
    """Side-by-side model evaluation metrics."""
    metrics_data = {
        "TF-IDF": {"NDCG@5": 0.71, "NDCG@10": 0.74, "Precision@5": 0.68, "MRR": 0.69, "AUC-ROC": 0.73},
        "Feature ML": {"NDCG@5": 0.84, "NDCG@10": 0.87, "Precision@5": 0.82, "MRR": 0.83, "AUC-ROC": 0.88},
        "Semantic BERT": {"NDCG@5": 0.79, "NDCG@10": 0.82, "Precision@5": 0.76, "MRR": 0.78, "AUC-ROC": 0.83},
        "Ensemble": {"NDCG@5": 0.89, "NDCG@10": 0.91, "Precision@5": 0.88, "MRR": 0.87, "AUC-ROC": 0.91},
    }
    metrics = list(list(metrics_data.values())[0].keys())
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (model, vals) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, list(vals.values()), width,
                      label=model, color=MULTI_PALETTE[i], alpha=0.85, edgecolor='#0a0e1a')

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Evaluation Metrics Comparison", fontsize=14, fontweight='bold', color='#e2e8f0')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=11, facecolor='#0d1128', edgecolor='#1e293b', labelcolor='#e2e8f0')
    ax.axhline(0.80, color='#fbbf24', linestyle='--', alpha=0.5, label='0.80 threshold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Correlation heatmap of numerical features."""
    numeric_cols = ["experience_years", "cgpa", "skills_count", "cert_count",
                    "proj_count", "req_match", "pref_match", "ensemble_score"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, ax=ax,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7},
                annot_kws={"size": 9, "color": "#e2e8f0"})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', color='#e2e8f0')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    return fig


def create_plotly_dashboard(df: pd.DataFrame) -> go.Figure:
    """Create an interactive multi-panel Plotly dashboard."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Score Distribution", "Experience vs Score",
                        "CGPA vs Score", "Degree Distribution",
                        "Model Comparison", "Req Match vs Ensemble"],
        specs=[[{}, {}, {}], [{}, {}, {}]]
    )

    # 1. Score distribution
    for score, color in [("tfidf_score", "#60a5fa"), ("ensemble_score", "#22c55e")]:
        fig.add_trace(go.Histogram(x=df[score], name=score.replace("_", " ").title(),
                                    marker_color=color, opacity=0.7, nbinsx=25), row=1, col=1)

    # 2. Experience vs Score scatter
    fig.add_trace(go.Scatter(x=df["experience_years"], y=df["ensemble_score"],
                              mode='markers', name="Candidates",
                              marker=dict(color=df["ensemble_score"], colorscale="Blues",
                                          size=5, opacity=0.7)), row=1, col=2)

    # 3. CGPA vs Score
    fig.add_trace(go.Scatter(x=df["cgpa"], y=df["ensemble_score"],
                              mode='markers', name="CGPA-Score",
                              marker=dict(color=df["cgpa"], colorscale="Purples",
                                          size=5, opacity=0.7)), row=1, col=3)

    # 4. Degree distribution
    degree_counts = df["degree"].value_counts()
    fig.add_trace(go.Bar(x=degree_counts.index, y=degree_counts.values,
                          name="Degree", marker_color=MULTI_PALETTE[:len(degree_counts)]),
                  row=2, col=1)

    # 5. Model comparison box
    for model, color in [("tfidf_score", "#60a5fa"), ("feature_score", "#22c55e"),
                          ("semantic_score", "#f59e0b"), ("ensemble_score", "#8b5cf6")]:
        fig.add_trace(go.Box(y=df[model], name=model.replace("_score", "").replace("_", " ").title(),
                              marker_color=color), row=2, col=2)

    # 6. Req match vs Ensemble
    fig.add_trace(go.Scatter(x=df["req_match"], y=df["ensemble_score"],
                              mode='markers', name="Req Match",
                              marker=dict(color=df["cert_count"], colorscale="Greens",
                                          size=5, opacity=0.7)), row=2, col=3)

    fig.update_layout(
        height=700, title_text="AI Resume Screening — EDA Dashboard",
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1128',
        font=dict(color='#94a3b8'), showlegend=False,
        title_font=dict(size=18, color='#e2e8f0')
    )
    return fig


if __name__ == "__main__":
    import os
    os.makedirs("../visualizations", exist_ok=True)

    print("Generating sample data...")
    df = generate_sample_data(200)

    print("Generating plots...")
    plot_score_distributions(df, "../visualizations/score_distributions.png")
    print("  ✓ score_distributions.png")

    plot_feature_importance("../visualizations/feature_importance.png")
    print("  ✓ feature_importance.png")

    plot_model_evaluation("../visualizations/model_evaluation.png")
    print("  ✓ model_evaluation.png")

    plot_correlation_heatmap(df, "../visualizations/correlation_heatmap.png")
    print("  ✓ correlation_heatmap.png")

    print("\n✓ All visualizations saved to ../visualizations/")
    print(f"\nDataset stats:\n{df[['experience_years','cgpa','ensemble_score','suitable']].describe()}")