# JobMatch — Resume & Job Retrieval System

CSCE 470 (Information Storage & Retrieval), Spring 2026, Texas A&M University.  
Team: Brayden Bailey, Campbell Wright.

Bidirectional retrieval: upload a resume to find matching jobs, or paste a job description to find matching resumes. Built on BM25F (field-weighted inverted index), sentence-transformer embeddings, and a Dirichlet language model — fused in a hybrid mode with optional Rocchio query expansion and Learning-to-Rank reranking.

---

## Prerequisites

- Python 3.10+
- A Kaggle account (to download datasets)
- An OpenAI API key (optional — only needed for GPT-scored ground truth)

---

## Setup

```bash
git clone <repo-url>
cd jobmatch
pip install -r requirements.txt
```

Copy the environment file and add your OpenAI key if you plan to use GPT-scored ground truth:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### Verify algorithms without any data

All retrieval engines have self-contained demos using synthetic data — no download required:

```bash
python -m engine.bm25f      # BM25F field-weighted ranking
python -m engine.semantic   # Sentence-transformer embedding search
python -m engine.hybrid     # Hybrid fusion + pseudo-relevance feedback
python -m engine.lm         # Dirichlet language model retrieval
python -m engine.cluster    # k-means corpus clustering
python -m engine.ltr        # LTR feature extraction demo
python evaluation/evaluate.py  # NDCG / MAP / P@K metric sanity check
```

---

## Datasets

Download each dataset and place it at the exact path shown — Kaggle wraps downloads in a folder of the same name, so the nested paths below are intentional.

| Dataset | Source | Records | Place at |
|---------|--------|---------|----------|
| arshkon LinkedIn job postings | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | 123,849 | `data/raw/postings.csv/postings.csv` |
| asaniczka 1.3M jobs & skills | [Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | 1,348,454 | `data/raw/linkedin_job_postings.csv/` |
| snehaanbhawal resumes | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 2,484 | `data/raw/Resume/Resume.csv` |
| florex resume corpus | [GitHub](https://github.com/florex/resume_corpus) | 29,783 | `data/raw/resume_corpus-master/` |

Raw data is gitignored and never committed.

---

## Building the Pipeline

Each step below builds on the previous one. Run them in order.

### Step 1 — Preprocess

Cleans raw CSVs: strips HTML from resumes, removes short/empty documents, normalizes category labels. Outputs to `data/processed/`.

```bash
python build.py --step preprocess
```

### Step 2 — Build indexes

Builds all retrieval indexes and writes them to `data/indexes/`:

- **BM25F** index (jobs + resumes)
- **Semantic** embedding index using `all-MiniLM-L6-v2` (jobs + resumes)
- **k-means cluster** index, 50 clusters over the semantic embedding space (jobs + resumes)
- **Language model** index with Dirichlet smoothing µ=2000 (jobs + resumes)

```bash
python build.py --step index

# Accelerate semantic encoding on a GPU (recommended — CPU takes ~30-60 min for 107K jobs)
python build.py --step index --device cuda
```

### Step 3 — Generate ground truth

Pools candidates from both BM25F and semantic retrieval, grades them by category match (3 = same, 1 = related, 0 = different), and writes `evaluation/ground_truth.csv`.

```bash
# Category-based grades — free, instant, no API key needed
python evaluation/generate_ground_truth.py --api category --num-queries 100

# GPT-scored grades — requires OPENAI_API_KEY; ~$0.01-0.02 for 50 queries × 20 candidates
python evaluation/generate_ground_truth.py --api openai --num-queries 50 --top-k 20
```

### Step 4 — Evaluate retrieval

Runs P@K, NDCG@K, and MAP across BM25F, Semantic, and Hybrid modes against `ground_truth.csv`.

```bash
python build.py --step evaluate
```

### Step 5 — Train the LTR reranker (optional)

Trains a logistic Learning-to-Rank model on the ground truth. Requires ground truth from Step 3 and indexes from Step 2.

```bash
python build.py --step ltr
```

### Run everything at once

```bash
python build.py --step all
```

---

## Running the Web App

### Option A — Use pre-built sample indexes (fastest, no raw data needed)

The repo includes sample indexes built from a stratified 5,000-job subset. If they are present in `data/indexes/`, start the app directly:

```bash
python -m flask --app app/server.py run --debug
```

If the indexes are missing, build a fresh sample set from your processed data:

```bash
python scripts/build_sample_indexes.py          # default: 5,000 jobs
python scripts/build_sample_indexes.py --n-jobs 3000 --device cuda
```

### Option B — Use full indexes (107K jobs)

Run the full pipeline through Step 2, then start the app:

```bash
python build.py --step preprocess
python build.py --step index --device cuda
python -m flask --app app/server.py run --debug
```

Open **http://127.0.0.1:5000** in a browser.  
The first request takes 5–30 seconds while indexes load into memory. Subsequent requests are fast.

---

## Retrieval Modes

Four modes are available, selectable per query in the UI or via the `mode=` parameter:

| Mode | Algorithm | Best for |
|------|-----------|----------|
| `hybrid` | Min-max normalized BM25F + semantic fusion (default, α=0.5) | General-purpose — balances keyword precision and semantic recall |
| `semantic` | Cosine similarity over `all-MiniLM-L6-v2` embeddings (384-dim) | Vocabulary mismatch — "ML" ↔ "machine learning", "Postgres" ↔ "PostgreSQL" |
| `bm25f` | Field-weighted BM25 (title ×3.0, description ×1.0) | Exact terminology, skill keywords, job titles |
| `lm` | Dirichlet-smoothed unigram language model (µ=2000) | Probabilistic scoring; handles rare and unseen terms via collection smoothing |

### Query Expansion (PRF)

Checking **Query expansion (PRF)** in the UI activates Rocchio-style pseudo-relevance feedback. The system retrieves an initial set of results, averages the top-5 semantic result embeddings into a feedback centroid, blends it with the original query (weight 0.7 original, 0.3 centroid), then re-scores all candidates with the expanded query. This improves recall on vocabulary-mismatch queries without requiring any user labeling.

### Learning-to-Rank

When `data/indexes/ltr_model.pkl` is present (built with `python build.py --step ltr`), the app automatically loads the LTR reranker. It uses 7 features per candidate: BM25F score, semantic cosine similarity, LM log-probability, query–title Jaccard overlap, query–description Jaccard overlap, query token count, and title token count. The logistic model is trained on labeled job–resume pairs from `evaluation/ground_truth.csv`.

### Corpus Clustering

Jobs and resumes are partitioned into 50 topic clusters using k-means over the semantic embedding space. Each result card shows its cluster ID, and the results page displays a topic distribution summary across all returned results. Clusters are built automatically during `--step index`.

---

## Evaluation Results

Evaluated on 48 query resumes across 24 job categories. Ground truth pooled from BM25F and semantic candidates with category-based relevance grades.

| Metric | BM25F | Semantic | Hybrid |
|--------|-------|----------|--------|
| P@5 | 0.296 | **0.421** | 0.367 |
| P@10 | 0.288 | **0.442** | 0.358 |
| NDCG@10 | 0.313 | **0.458** | 0.369 |
| NDCG@20 | 0.358 | **0.487** | 0.429 |
| MAP | 0.185 | **0.321** | 0.225 |

Semantic outperforms BM25F by ~15 NDCG points across the board, consistent with vocabulary mismatch being the dominant challenge in resume-job matching. The live **Evaluate** page in the web app runs all four modes (including Language Model) against ground truth in real time.

---

## Project Structure

```
jobmatch/
├── build.py                         # Pipeline: preprocess / index / evaluate / ltr / demo
├── requirements.txt
├── .env.example                     # Copy to .env — set OPENAI_API_KEY for GPT scoring
│
├── engine/
│   ├── bm25f.py                     # BM25F inverted index, multi-field weighting
│   ├── semantic.py                  # Sentence-transformer embeddings + FAISS search
│   ├── hybrid.py                    # Hybrid fusion + Rocchio pseudo-relevance feedback
│   ├── lm.py                        # Dirichlet-smoothed unigram language model
│   ├── cluster.py                   # k-means corpus clustering over embedding space
│   └── ltr.py                       # Logistic Learning-to-Rank reranker (7 features)
│
├── evaluation/
│   ├── generate_ground_truth.py     # Category-based and GPT-scored relevance judgments
│   └── evaluate.py                  # P@K, NDCG@K, MAP across all retrieval modes
│
├── scripts/
│   ├── preprocess.py                # HTML stripping, dedup, category normalization
│   ├── build_sample_indexes.py      # Stratified 5K-job sample index build for quick demo
│   └── generate_figures.py          # EDA figures
│
├── data/
│   ├── raw/                         # NOT in git — download from Kaggle/GitHub
│   ├── processed/                   # NOT in git — generated by preprocess.py
│   └── indexes/                     # Pre-built sample indexes committed for demo
│
└── app/
    ├── server.py                    # Flask app — lazy index loading, all retrieval modes
    └── templates/
        ├── base.html
        ├── index.html               # Search form with mode selector and PRF toggle
        ├── results.html             # Ranked results with mode badge and cluster summary
        ├── evaluate.html            # Four-mode metric comparison table
        └── partials/
            ├── job_card.html        # Job result card with cluster badge
            └── resume_card.html     # Resume result card with text preview
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Verify algorithms (no data) | `python -m engine.hybrid` |
| Preprocess raw data | `python build.py --step preprocess` |
| Build all indexes | `python build.py --step index` |
| Build sample indexes | `python scripts/build_sample_indexes.py` |
| Generate ground truth | `python evaluation/generate_ground_truth.py --api category` |
| Run evaluation | `python build.py --step evaluate` |
| Train LTR reranker | `python build.py --step ltr` |
| Run full pipeline | `python build.py --step all` |
| Start web app | `python -m flask --app app/server.py run --debug` |
| Clean generated files | `python build.py --step clean-all` |

---

## References

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
- Borisyuk et al., "Semantic Search at LinkedIn," arXiv:2602.07309
- Jiechieu & Tsopze, "Skills prediction based on multi-label resume classification using CNN," Neural Comput & Applic (2020)
