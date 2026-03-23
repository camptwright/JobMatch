# CSCE 463 Final Project

## Project Structure
```
JOBMATCH/
├── .gitignore
├── README.md
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
│   └── generate_figures.py                     # EDA figure generation for checkpoints
└── figures/                                    # Generated EDA figures
   ├── corpus_overview.png
   ├── job_description_lengths.png
   ├── job_levels.png
   ├── resume_categories.png
   ├── resume_lengths.png
   └── top_skills.png
```
 
## Data Setup
 
The raw datasets are too large for GitHub. Download them and place in `data/raw/`:
 
| Dataset | Source | Records | Download |
|---------|--------|---------|----------|
| Job Postings (arshkon) | LinkedIn 2023–2024 | 123,849 | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| Job Postings + Skills (asaniczka) | LinkedIn 2024 | 1,348,454 | [Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) |
| Resumes (snehaanbhawal) | LiveCareer, 24 categories | 2,484 | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| Resumes (florex) | Multi-labeled occupations | 29,783 | [GitHub](https://github.com/florex/resume_corpus) |