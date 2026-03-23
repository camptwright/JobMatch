import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# CONFIG
ARSHKON_POSTINGS = "data/raw/postings.csv/postings.csv"

ASANICZKA_POSTINGS = "data/raw/linkedin_job_postings.csv/linkedin_job_postings.csv"
ASANICZKA_SKILLS   = "data/raw/job_skills.csv/job_skills.csv"

SNEHAANBHAWAL_RESUMES = "data/raw/Resume/Resume.csv"

FLOREX_RESUMES = "data/raw/resume_corpus-master/resume_samples/resume_samples.txt"

# Output directory for figures
FIG_DIR = "figures"

# STYLING

plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
})

C_BLUE   = '#2563EB'
C_GREEN  = '#059669'
C_ORANGE = '#D97706'
C_RED    = '#DC2626'
C_PURPLE = '#7C3AED'


def word_count(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len(text.split())


def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<[^>]+>', ' ', text).strip()


# LOADERS

def load_arshkon():
    """Load arshkon/linkedin-job-postings."""
    print(f"Loading arshkon postings from {ARSHKON_POSTINGS}...")
    df = pd.read_csv(ARSHKON_POSTINGS)
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns[:8])}...")

    # Compute word counts on description
    if 'description' in df.columns:
        df['word_count'] = df['description'].apply(word_count)
    else:
        print("  WARNING: 'description' column not found. Check column names.")
        df['word_count'] = 0

    return df


def load_asaniczka():
    """Load asaniczka/1.3M-linkedin-jobs-and-skills."""
    print(f"Loading asaniczka postings from {ASANICZKA_POSTINGS}...")
    cols_to_try = ['job_link', 'job_title', 'company', 'job_location', 'job_level', 'job_type']

    header = pd.read_csv(ASANICZKA_POSTINGS, nrows=0)
    available = [c for c in cols_to_try if c in header.columns]
    print(f"  Using columns: {available}")

    df = pd.read_csv(ASANICZKA_POSTINGS, usecols=available)
    print(f"  Loaded {len(df):,} rows")

    skills_df = None
    if os.path.exists(ASANICZKA_SKILLS):
        print(f"Loading asaniczka skills from {ASANICZKA_SKILLS}...")
        skills_df = pd.read_csv(ASANICZKA_SKILLS)
        print(f"  Loaded {len(skills_df):,} skill entries")

    return df, skills_df


def load_snehaanbhawal():
    """Load snehaanbhawal/resume-dataset."""
    print(f"Loading snehaanbhawal resumes from {SNEHAANBHAWAL_RESUMES}...")
    df = pd.read_csv(SNEHAANBHAWAL_RESUMES)
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")

    if 'Resume_str' in df.columns:
        df['word_count'] = df['Resume_str'].apply(word_count)
    elif 'Resume_html' in df.columns:
        df['word_count'] = df['Resume_html'].apply(strip_html).apply(word_count)

    return df


def load_florex():
    """
    Load florex/resume_corpus from resume_samples.txt.
    Format per line: id:::occupation1;occupation2:::resume_text
    """
    print(f"Loading florex resumes from {FLOREX_RESUMES}...")
    records = []
    with open(FLOREX_RESUMES, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if ':::' not in line:
                continue
            parts = line.split(':::')
            if len(parts) >= 3:
                rid = parts[0].strip()
                occupations = [o.strip() for o in parts[1].split(';') if o.strip()]
                text = parts[2].strip()
                records.append({
                    'id': rid,
                    'occupations': occupations,
                    'occupation_str': parts[1].strip(),
                    'text': text,
                    'word_count': word_count(text),
                    'num_labels': len(occupations),
                })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} resumes")
    return df


# FIGURE GENERATORS

def fig1_corpus_overview(arshkon_n, asaniczka_n, sneha_n, florex_n):
    """Bar chart showing total corpus sizes across all 4 datasets."""
    labels = [
        f'arshkon\nJob Posts\n({arshkon_n:,})',
        f'asaniczka\nJob Posts\n({asaniczka_n:,})',
        f'snehaanbhawal\nResumes\n({sneha_n:,})',
        f'florex\nResumes\n({florex_n:,})',
    ]
    counts = [arshkon_n, asaniczka_n, sneha_n, florex_n]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, width=0.6, edgecolor='white', linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Number of Documents')
    ax.set_title('Total Corpus Size by Dataset')
    ax.set_yscale('log')
    ax.set_ylim(bottom=100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'corpus_overview.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig2_job_description_lengths(arshkon_df):
    """Histogram of job description word counts (arshkon dataset)."""
    wc = arshkon_df['word_count'].dropna()
    wc = wc[wc > 0]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(wc, bins=60, color=C_BLUE, alpha=0.8, edgecolor='white', linewidth=0.3)
    median_val = int(wc.median())
    mean_val = int(wc.mean())
    ax.axvline(median_val, color=C_RED, linestyle='--', linewidth=1.2,
               label=f'Median: {median_val} words')
    ax.axvline(mean_val, color=C_ORANGE, linestyle=':', linewidth=1.2,
               label=f'Mean: {mean_val} words')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Job Description Length Distribution (arshkon, n={len(wc):,})')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'job_description_lengths.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig3_resume_lengths(sneha_df, florex_df):
    """Side-by-side histograms of resume word counts from both resume datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # snehaanbhawal
    wc1 = sneha_df['word_count'].dropna()
    wc1 = wc1[wc1 > 0]
    ax1.hist(wc1, bins=40, color=C_ORANGE, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax1.axvline(int(wc1.median()), color=C_RED, linestyle='--', linewidth=1,
                label=f'Median: {int(wc1.median())}')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'snehaanbhawal (n={len(wc1):,})')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # florex
    wc2 = florex_df['word_count'].dropna()
    wc2 = wc2[wc2 > 0]
    ax2.hist(wc2, bins=40, color=C_RED, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax2.axvline(int(wc2.median()), color=C_BLUE, linestyle='--', linewidth=1,
                label=f'Median: {int(wc2.median())}')
    ax2.set_xlabel('Word Count')
    ax2.set_title(f'florex (n={len(wc2):,})')
    ax2.legend(fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Resume Length Distributions', fontsize=12, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'resume_lengths.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig4_resume_categories(sneha_df):
    """Bar chart of snehaanbhawal resume categories (24 labels)."""
    cats = sneha_df['Category'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(cats)), cats.values, color=C_ORANGE, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(f'Resume Categories — snehaanbhawal (n={len(sneha_df):,})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'resume_categories.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig5_top_skills(skills_df):
    """Top 20 most common skills from asaniczka job_skills.csv."""
    # Column is likely 'job_skills' or 'skill'
    skill_col = None
    for candidate in ['job_skills', 'skill', 'skills']:
        if candidate in skills_df.columns:
            skill_col = candidate
            break

    if skill_col is None:
        print(f"  WARNING: Could not find skill column. Columns are: {list(skills_df.columns)}")
        print("  Skipping skills figure.")
        return None

    top = skills_df[skill_col].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(top)), top.values, color=C_GREEN, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top 20 Skills in Job Listings (asaniczka, n={len(skills_df):,} entries)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'top_skills.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig6_job_level_distribution(asaniczka_df):
    """Pie/bar chart of job level (seniority) distribution from asaniczka."""
    if 'job_level' not in asaniczka_df.columns:
        print("  Skipping job level figure — column not found.")
        return None

    levels = asaniczka_df['job_level'].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(levels)), levels.values, color=[C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE][:len(levels)],
                  alpha=0.85, width=0.6, edgecolor='white', linewidth=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels.index, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title('Job Seniority Level Distribution (asaniczka)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, count in zip(bars, levels.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + levels.max()*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'job_levels.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# MAIN

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Track what loaded successfully
    arshkon_df = None
    asaniczka_df = None
    skills_df = None
    sneha_df = None
    florex_df = None

    if os.path.exists(ARSHKON_POSTINGS):
        arshkon_df = load_arshkon()
    else:
        print(f"SKIP: {ARSHKON_POSTINGS} not found")

    if os.path.exists(ASANICZKA_POSTINGS):
        asaniczka_df, skills_df = load_asaniczka()
    else:
        print(f"SKIP: {ASANICZKA_POSTINGS} not found")

    if os.path.exists(SNEHAANBHAWAL_RESUMES):
        sneha_df = load_snehaanbhawal()
    else:
        print(f"SKIP: {SNEHAANBHAWAL_RESUMES} not found")

    if os.path.exists(FLOREX_RESUMES):
        florex_df = load_florex()
    else:
        print(f"SKIP: {FLOREX_RESUMES} not found")

    print("\n--- Generating Figures ---\n")
    generated = []

    # Fig 1: Corpus overview
    arshkon_n  = len(arshkon_df)  if arshkon_df is not None else 0
    asaniczka_n = len(asaniczka_df) if asaniczka_df is not None else 0
    sneha_n    = len(sneha_df)    if sneha_df is not None else 0
    florex_n   = len(florex_df)   if florex_df is not None else 0

    if any([arshkon_n, asaniczka_n, sneha_n, florex_n]):
        generated.append(fig1_corpus_overview(arshkon_n, asaniczka_n, sneha_n, florex_n))

    # Fig 2: Job description lengths (arshkon has full descriptions)
    if arshkon_df is not None:
        generated.append(fig2_job_description_lengths(arshkon_df))

    # Fig 3: Resume lengths (both resume datasets)
    if sneha_df is not None and florex_df is not None:
        generated.append(fig3_resume_lengths(sneha_df, florex_df))
    elif sneha_df is not None:
        # Fallback: just snehaanbhawal
        fig, ax = plt.subplots(figsize=(6, 3))
        wc = sneha_df['word_count'].dropna()
        wc = wc[wc > 0]
        ax.hist(wc, bins=40, color=C_ORANGE, alpha=0.8, edgecolor='white')
        ax.axvline(int(wc.median()), color=C_RED, linestyle='--', label=f'Median: {int(wc.median())}')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Resume Length Distribution (n={len(wc):,})')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        p = os.path.join(FIG_DIR, 'resume_lengths.png')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        generated.append(p)
        print(f"  Saved: {p}")

    # Fig 4: Resume categories (snehaanbhawal has 24 labels)
    if sneha_df is not None and 'Category' in sneha_df.columns:
        generated.append(fig4_resume_categories(sneha_df))

    # Fig 5: Top skills (asaniczka)
    if skills_df is not None:
        result = fig5_top_skills(skills_df)
        if result:
            generated.append(result)

    # Fig 6: Job levels (asaniczka)
    if asaniczka_df is not None:
        result = fig6_job_level_distribution(asaniczka_df)
        if result:
            generated.append(result)

    print(f"\n{'='*55}")
    print(f"  Done! Generated {len(generated)} figures in {FIG_DIR}/")
    print(f"{'='*55}")

    total_jobs = arshkon_n + asaniczka_n
    total_resumes = sneha_n + florex_n
    print(f"\n  Total job listings:  {total_jobs:,}")
    print(f"  Total resumes:       {total_resumes:,}")
    print(f"  Combined corpus:     {total_jobs + total_resumes:,} documents\n")

    print("  Figures generated:")
    for p in generated:
        print(f"    - {p}")

    print(f"\n  Paste these into your 1-page Checkpoint PDF.")
    print(f"  Pick 2-3 that best show the data characteristics.")


if __name__ == '__main__':
    main()
