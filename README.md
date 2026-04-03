# JobMatch — Resume & Job Retrieval System

CSCE 470 (Information Storage & Retrieval), Spring 2026, Texas A&M University.
Team: Brayden Bailey, Campbell Wright.

Bidirectional retrieval system: upload a resume to find matching jobs, or paste a job description to find matching resumes. Built on BM25F + semantic embeddings (all-MiniLM-L6-v2), with a hybrid fusion mode combining both.

---

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` and `torch` are required for semantic and hybrid modes. BM25F works without them.

---

## Quick Test (no data download required)

Each engine includes a self-contained demo using synthetic data. Run these immediately after install to verify the algorithms work:

```bash
python engine/bm25f.py          # BM25F index + ranked retrieval on 5 synthetic jobs
python engine/semantic.py       # Semantic embedding search (requires sentence-transformers)
python engine/hybrid.py         # BM25F-only fallback, or full hybrid if sentence-transformers installed
python evaluation/evaluate.py   # Sanity check: P@K, NDCG@K, AP on synthetic rankings
```

Expected output from `python engine/bm25f.py`:

```
BM25F index built: 5 docs, 64 unique terms

Query: 'python machine learning engineer'
  1. [1.981] Machine Learning Engineer
  2. [0.914] Senior Python Developer
  3. [0.710] Data Scientist
...
```

---

## Full Pipeline (with real data)

### 1. Download datasets

Place the following in `data/raw/`:

| Dataset | Source | Records | Path |
|---------|--------|---------|------|
| Job Postings (arshkon) | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | 123,849 | `data/raw/postings.csv/postings.csv` |
| Job Posts + Skills (asaniczka) | [Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | 1.3M | `data/raw/linkedin_job_postings.csv/linkedin_job_postings.csv` |
| Resumes (snehaanbhawal) | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 2,484 | `data/raw/Resume/Resume.csv` |
| Resumes (florex) | [GitHub](https://github.com/florex/resume_corpus) | 29,783 | `data/raw/resume_corpus-master/` |

Raw data is not tracked in git (too large).

### 2. Run the pipeline

```bash
python build.py --step preprocess   # Clean raw CSVs -> data/processed/
python build.py --step index        # Build BM25F + semantic indexes -> data/indexes/
python build.py --step evaluate     # Generate ground truth + compute metrics
python build.py --step demo         # Interactive CLI search demo
python build.py --step all          # Run all steps in order
```

**Note on semantic indexing:** Encoding 124K job descriptions with `all-MiniLM-L6-v2` takes ~30-60 min on CPU. Use `--device cuda` if a GPU is available, or subsample by editing `build_job_semantic_index()` in `engine/semantic.py`.

---

## Retrieval Modes

**BM25F (lexical)** — Field-weighted BM25 with separate inverted index channels for title (weight=3.0, b=0.3) and description (weight=1.0, b=0.75). Handles keyword matching with IDF weighting and per-field length normalization.

**Semantic** — Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384-dim). Encodes documents and queries into dense vectors; retrieval by cosine similarity. Handles vocabulary mismatch (e.g. "ML" vs "machine learning").

**Hybrid** — Min-max normalized weighted sum: `score = alpha * BM25F + (1 - alpha) * semantic`. Default alpha=0.5, tunable at runtime.

---

## Evaluation

Ground truth is generated two ways:

- **Category-based (free, instant):** Resume categories (24 from snehaanbhawal) are mapped to job categories via a hand-built mapping. A retrieved job is graded 3 (same category), 1 (related), or 0 (unrelated).
- **LLM-scored:** Resume-job pairs sent to an LLM with a 0-3 graded relevance prompt.

```bash
# Category-based (no API key needed)
python evaluation/generate_ground_truth.py --api category --num-queries 100

# LLM-scored
python evaluation/generate_ground_truth.py --api anthropic --num-queries 50 --top-k 20
```

Metrics computed: Precision@K, NDCG@K, MAP — compared across BM25F, semantic, and hybrid modes.

---

## Project Structure

```
JobMatch/
├── build.py                        # Master pipeline (preprocess/index/evaluate/demo)
├── requirements.txt
│
├── engine/
│   ├── bm25f.py                    # BM25F inverted index with multi-field weighting
│   ├── semantic.py                 # Sentence-transformer embedding retrieval
│   └── hybrid.py                   # Hybrid fusion (BM25F + semantic)
│
├── evaluation/
│   ├── evaluate.py                 # P@K, NDCG@K, MAP metrics + comparison table
│   └── generate_ground_truth.py    # Category-based and LLM-scored relevance judgments
│
├── scripts/
│   ├── preprocess.py               # HTML stripping, deduplication, category extraction
│   ├── generate_figures.py         # EDA figures for checkpoints
│   └── benchmark_encoding.py       # Semantic encoding speed benchmark
│
├── data/
│   ├── raw/                        # NOT in git — download from sources above
│   ├── processed/                  # Generated by preprocess.py
│   └── indexes/                    # Generated by build.py --step index
│
├── docs/
│   ├── project-requirements (1).pdf
│   ├── Proposal JobMatch.pdf
│   └── CSCE 470 Project Checkpoint - Data.pdf
│
└── app/                            # Web app (Checkpoint 3+)
```
