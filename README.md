# CSCE 463 Final Project

## Project Structure
```
JOBMATCH/
├── .gitignore
├── README.md
├── requirements.txt
├── build.py                                    # Master pipeline: preprocess → index → evaluate → demo
│
├── data/
│   └── raw/                                    # ← NOT tracked in git (see Data Setup below)
│       ├── postings.csv/
│       │   └── postings.csv                    # arshkon — 123,849 LinkedIn job postings
│       ├── job_skills.csv/
│       │   └── job_skills.csv                  # asaniczka — 1.3M job skill entries
│       ├── linkedin_job_postings.csv/
│       │   └── linkedin_job_postings.csv       # asaniczka — 1,348,454 LinkedIn job postings
│       ├── Resume/
│       │   └── Resume.csv                      # snehaanbhawal — 2,484 labeled resumes
│       └── resume_corpus-master/
│           ├── resume_samples/
│           │   └── resume_samples.txt          # florex — 29,783 multi-labeled resumes
│           ├── resumes_corpus/                  # Individual .txt + .lab resume files
│           ├── normalized_classes.txt
│           ├── skills_it.txt
│           ├── resume_samples.zip
│           ├── resumes_corpus.zip
│           └── README.md
│
├── scripts/
│   ├── generate_figures.py                     # EDA figure generation for checkpoints
│   └── preprocess.py                           # Data cleaning pipeline
│
├── engine/                                     # Core retrieval algorithms
│   ├── __init__.py
│   ├── bm25f.py                                # BM25F with multi-field weighting
│   ├── semantic.py                             # Sentence-transformer embedding retrieval
│   └── hybrid.py                               # Hybrid fusion (BM25F + semantic)
│
├── evaluation/                                 # Evaluation framework
│   ├── generate_ground_truth.py                # LLM-based + category-based ground truth
│   └── evaluate.py                             # NDCG@K, P@K, MAP metrics
│
├── figures/                                    # Generated EDA figures
│   ├── corpus_overview.png
│   ├── job_description_lengths.png
│   ├── job_levels.png
│   ├── resume_categories.png
│   ├── resume_lengths.png
│   └── top_skills.png
│
├── docs/                                       # Checkpoint deliverables
│   └── Checkpoint1_Data.pdf
│
└── app/                                        # Web app (Checkpoint 3)
```
 
## Data Setup
 
The raw datasets are too large for GitHub. Download them and place in `data/raw/`:
 
| Dataset | Source | Records | Download |
|---------|--------|---------|----------|
| Job Postings (arshkon) | LinkedIn 2023–2024 | 123,849 | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| Job Postings + Skills (asaniczka) | LinkedIn 2024 | 1,348,454 | [Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) |
| Resumes (snehaanbhawal) | LiveCareer, 24 categories | 2,484 | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| Resumes (florex) | Multi-labeled occupations | 29,783 | [GitHub](https://github.com/florex/resume_corpus) |