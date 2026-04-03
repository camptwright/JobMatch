# JobMatch — Resume & Job Retrieval System

CSCE 470 (Information Storage & Retrieval), Spring 2026, Texas A&M University.
Team: Brayden Bailey, Campbell Wright.

Bidirectional retrieval system: upload a resume to find matching jobs, or paste a job description to find matching resumes. Built on BM25F + semantic embeddings.

---

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` and `torch` are required for semantic/hybrid modes. BM25F-only works without them.

---

## Data

Raw datasets are **not tracked in git**. Download and place in `data/raw/`:

| Dataset | Source | Records | Path |
|---------|--------|---------|------|
| Job Postings (arshkon) | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | 123,849 | `data/raw/postings.csv/postings.csv` |
| Job Posts + Skills (asaniczka) | [Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | 1.3M | `data/raw/linkedin_job_postings.csv/linkedin_job_postings.csv` |
| Resumes (snehaanbhawal) | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 2,484 | `data/raw/Resume/Resume.csv` |
| Resumes (florex) | [GitHub](https://github.com/florex/resume_corpus) | 29,783 | `data/raw/resume_corpus-master/resume_samples/resume_samples.txt` |

---

## Running the Pipeline

```bash
python build.py --step preprocess   # Clean raw data -> data/processed/
python build.py --step index        # Build BM25F + semantic indexes
python build.py --step evaluate     # Generate ground truth + run metrics
python build.py --step demo         # Interactive search demo
python build.py --step all          # Run everything
```

Individual components:

```bash
python scripts/generate_figures.py                                          # EDA figures
python engine/bm25f.py                                                      # BM25F self-test
python engine/semantic.py                                                    # Semantic self-test
python evaluation/evaluate.py                                               # Metrics sanity test
python evaluation/generate_ground_truth.py --api category --num-queries 100
python evaluation/generate_ground_truth.py --api anthropic --num-queries 50 --top-k 20
```

---

## Project Structure

```
data/raw/          # Raw datasets — NOT in git
data/processed/    # Cleaned CSVs — NOT in git
data/indexes/      # Built indexes — NOT in git
engine/            # Core retrieval algorithms
evaluation/        # Ground truth generation + metrics
scripts/           # Preprocessing and EDA figure generation
figures/           # Generated EDA PNGs
app/               # Web app (Checkpoint 3+)
build.py           # Master pipeline
```

## File Descriptions

```
engine/
  bm25f.py        BM25F inverted index with multi-field weighting — title (w=3.0, b=0.3)
                  + description (w=1.0, b=0.75). BM25FIndex class with add_document(),
                  build(), search(), save()/load(). Convenience builders for jobs and resumes.

  semantic.py     SemanticIndex wrapping sentence-transformers (all-MiniLM-L6-v2, 384-dim).
                  Lazy model loading, cosine similarity retrieval via pre-normalized dot product.
                  Convenience builders for jobs and resumes.

  hybrid.py       HybridRetriever combining BM25F and semantic scores via min-max normalized
                  weighted sum: final = α*BM25F_norm + (1-α)*semantic_norm (default α=0.5).
                  Supports modes: bm25f | semantic | hybrid. compare_modes() for side-by-side.

evaluation/
  generate_ground_truth.py   Three modes: category-based (free), LLM-scored via OpenAI or
                              Anthropic (0-3 graded relevance scale), and pooled BM25F
                              retrieval with random negatives. Stratified resume sampling.

  evaluate.py                precision_at_k(), ndcg_at_k(), average_precision(), MAP.
                              evaluate_retrieval() runs full eval given ground truth + retriever.
                              compare_retrieval_modes() produces a comparison table.

scripts/
  preprocess.py         Loads raw CSVs, strips HTML, removes short docs, normalizes categories,
                        saves category_map.json. Outputs cleaned CSVs to data/processed/.

  generate_figures.py   Reads all 4 raw datasets and produces 6 EDA figures to figures/.
                        Skills figure splits comma-separated skill lists before counting.

  benchmark_encoding.py Benchmarks sentence-transformer encoding speed across corpus sizes
                        on CPU and GPU, with full-corpus time projections.

build.py           Master pipeline with --step flag: preprocess | index | evaluate | demo |
                   rebuild | clean-all. The demo step launches an interactive CLI search.
```
