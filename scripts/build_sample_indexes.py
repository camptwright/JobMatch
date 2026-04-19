"""
build_sample_indexes.py

Builds small, commit-able BM25F + semantic indexes from a stratified sample
of the processed data.  Full indexes for 107K jobs are too large for GitHub
(~200-500 MB); a 5K-job sample keeps each file under GitHub's 100 MB limit
while still giving a real demo.

Usage (from project root):
    python scripts/build_sample_indexes.py
    python scripts/build_sample_indexes.py --n-jobs 5000 --seed 42

After running, commit the generated files:
    git add data/indexes/
    git commit -m "add sample indexes for deployment"
"""

import os
import sys
import argparse
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
INDEX_DIR = os.path.join(BASE_DIR, "data", "indexes")


def _sample_jobs(jobs_df, n, seed):
    """Stratified sample by job_category, falling back to random if needed."""
    import pandas as pd

    if "job_category" not in jobs_df.columns or jobs_df["job_category"].isna().all():
        return jobs_df.sample(n=min(n, len(jobs_df)), random_state=seed)

    categories = jobs_df["job_category"].dropna().unique()
    per_cat = max(1, n // len(categories))
    parts = []
    for cat in categories:
        cat_df = jobs_df[jobs_df["job_category"] == cat]
        parts.append(cat_df.sample(n=min(per_cat, len(cat_df)), random_state=seed))

    sampled = pd.concat(parts).drop_duplicates()

    # Top up to n if we're short (due to small categories)
    if len(sampled) < n:
        remaining = jobs_df[~jobs_df.index.isin(sampled.index)]
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=seed).head(n)


def build(n_jobs: int = 5000, seed: int = 42, device: str = None):
    import pandas as pd
    from engine.bm25f import BM25FIndex, build_resume_index

    os.makedirs(INDEX_DIR, exist_ok=True)

    jobs_csv = os.path.join(PROC_DIR, "jobs_clean.csv")
    resumes_csv = os.path.join(PROC_DIR, "resumes_clean.csv")

    for path, label in [(jobs_csv, "jobs_clean.csv"), (resumes_csv, "resumes_clean.csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            print("Run: python build.py --step preprocess   first.")
            sys.exit(1)

    # ── Jobs (sampled) ──────────────────────────────────────────────────────
    jobs_df = pd.read_csv(jobs_csv)
    print(f"Full job corpus: {len(jobs_df):,} postings")

    sampled = _sample_jobs(jobs_df, n_jobs, seed)
    print(f"Sampled {len(sampled):,} jobs (stratified by category, seed={seed})")

    # Write temp CSV for index builders
    tmp_jobs = os.path.join(INDEX_DIR, "_sample_jobs_tmp.csv")
    sampled.to_csv(tmp_jobs, index=False)

    field_configs = {
        "title":       {"weight": 3.0, "b": 0.3},
        "description": {"weight": 1.0, "b": 0.75},
    }

    print("\n--- Building job BM25F index ---")
    from engine.bm25f import build_job_index
    job_bm25 = build_job_index(tmp_jobs, field_configs=field_configs)
    job_bm25.save(os.path.join(INDEX_DIR, "jobs_bm25f.pkl"))

    try:
        from engine.semantic import build_job_semantic_index
        print("\n--- Building job semantic index ---")
        job_sem = build_job_semantic_index(tmp_jobs, device=device)
        job_sem.save(os.path.join(INDEX_DIR, "jobs_semantic"))
    except ImportError:
        print("\nWARNING: sentence-transformers not installed. Skipping semantic job index.")

    os.remove(tmp_jobs)

    # ── Resumes (full — only 2,484 rows, always fits) ──────────────────────
    print("\n--- Building resume BM25F index ---")
    resume_bm25 = build_resume_index(resumes_csv)
    resume_bm25.save(os.path.join(INDEX_DIR, "resumes_bm25f.pkl"))

    try:
        from engine.semantic import build_resume_semantic_index
        print("\n--- Building resume semantic index ---")
        resume_sem = build_resume_semantic_index(resumes_csv, device=device)
        resume_sem.save(os.path.join(INDEX_DIR, "resumes_semantic"))
    except ImportError:
        print("\nWARNING: sentence-transformers not installed. Skipping semantic resume index.")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Sample indexes built successfully.")
    total_mb = 0.0
    for root, _, files in os.walk(INDEX_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            mb = os.path.getsize(fpath) / 1e6
            total_mb += mb
            rel = os.path.relpath(fpath, BASE_DIR)
            print(f"  {rel:<45} {mb:6.1f} MB")
    print(f"  {'Total':<45} {total_mb:6.1f} MB")
    print("=" * 55)
    print("\nNext step — commit the indexes:")
    print("  git add data/indexes/")
    print("  git commit -m 'add sample indexes for deployment'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build small sample indexes for deployment")
    parser.add_argument("--n-jobs", type=int, default=5000,
                        help="Number of job postings to sample (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Device for sentence-transformers encoding")
    args = parser.parse_args()

    build(n_jobs=args.n_jobs, seed=args.seed, device=args.device)
