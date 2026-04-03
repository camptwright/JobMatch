import pandas as pd
import numpy as np
import re
import os
import json
from html import unescape

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)


def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\-/.,;:()&+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_count(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len(text.split())


def process_jobs(filepath):
    print("Loading job postings...")
    df = pd.read_csv(filepath)
    print(f"  Raw records: {len(df)}")

    keep_cols = [
        'job_id', 'title', 'description', 'company_name', 'location',
        'max_salary', 'min_salary', 'formatted_experience_level',
        'formatted_work_type', 'applies', 'views'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.dropna(subset=['description'])
    print(f"  After dropping null descriptions: {len(df)}")

    df['description_clean'] = df['description'].apply(lambda t: clean_text(strip_html(t)))

    df['desc_word_count'] = df['description_clean'].apply(word_count)
    df = df[df['desc_word_count'] >= 20].copy()
    print(f"  After dropping short descriptions (<20 words): {len(df)}")

    df['title_clean'] = df['title'].fillna('').apply(clean_text)

    if 'formatted_experience_level' in df.columns:
        df['experience_level'] = df['formatted_experience_level'].fillna('Unknown').str.strip()

    for col in ['company_name', 'location']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    for col in ['max_salary', 'min_salary', 'applies', 'views']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['job_category'] = df['title_clean'].apply(extract_job_category)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=['description_clean'], keep='first').copy()
    print(f"  After deduplicating identical descriptions: {len(df)} (removed {before_dedup - len(df)})")

    print(f"  Final job postings: {len(df)}")
    return df


def extract_job_category(title):
    title_lower = title.lower()
    categories = {
        'Engineering': ['engineer', 'developer', 'software', 'devops', 'sre', 'backend', 'frontend', 'fullstack'],
        'Information-Technology': ['it ', 'system admin', 'network', 'helpdesk', 'cybersecurity', 'security analyst'],
        'Data Science': ['data scientist', 'data analyst', 'machine learning', 'ml engineer', 'ai '],
        'Design': ['designer', 'ux', 'ui ', 'graphic', 'creative director'],
        'Sales': ['sales', 'account executive', 'business development', 'bdr', 'sdr'],
        'Marketing': ['marketing', 'seo', 'content strategist', 'brand', 'growth'],
        'Finance': ['finance', 'financial', 'accountant', 'accounting', 'cfo', 'controller', 'auditor'],
        'HR': ['human resources', 'recruiter', 'recruiting', 'talent', 'hr '],
        'Healthcare': ['nurse', 'physician', 'medical', 'clinical', 'health', 'pharma'],
        'Management': ['manager', 'director', 'vp ', 'vice president', 'head of', 'lead'],
        'Consulting': ['consultant', 'consulting', 'advisory'],
        'Education': ['teacher', 'professor', 'instructor', 'educator', 'tutor'],
        'Legal': ['lawyer', 'attorney', 'legal', 'paralegal', 'advocate'],
        'Construction': ['construction', 'civil engineer', 'project engineer', 'architect'],
    }
    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in title_lower:
                return cat
    return 'Other'


def process_resumes(filepath):
    print("\nLoading resumes...")
    df = pd.read_csv(filepath)
    print(f"  Raw records: {len(df)}")

    df['resume_clean'] = df['Resume_str'].fillna('').apply(clean_text)

    mask_empty = df['resume_clean'].apply(word_count) < 10
    if 'Resume_html' in df.columns:
        df.loc[mask_empty, 'resume_clean'] = (
            df.loc[mask_empty, 'Resume_html']
            .fillna('')
            .apply(strip_html)
            .apply(clean_text)
        )

    df['resume_word_count'] = df['resume_clean'].apply(word_count)

    df = df[df['resume_word_count'] >= 30].copy()
    print(f"  After dropping short resumes (<30 words): {len(df)}")

    df['Category'] = df['Category'].str.strip()

    print(f"  Final resumes: {len(df)}")
    print(f"  Categories: {sorted(df['Category'].unique())}")
    return df


CATEGORY_MAP = {
    'ENGINEERING':            'Engineering',
    'INFORMATION-TECHNOLOGY': 'Information-Technology',
    'HR':                     'HR',
    'SALES':                  'Sales',
    'FINANCE':                'Finance',
    'ACCOUNTANT':             'Finance',
    'BANKING':                'Finance',
    'HEALTHCARE':             'Healthcare',
    'FITNESS':                'Healthcare',
    'DESIGNER':               'Design',
    'DIGITAL-MEDIA':          'Design',
    'ARTS':                   'Design',
    'ADVOCATE':               'Legal',
    'CONSTRUCTION':           'Construction',
    'TEACHER':                'Education',
    'BUSINESS-DEVELOPMENT':   'Sales',
    'CONSULTANT':             'Consulting',
    'CHEF':                   'Other',
    'AVIATION':               'Other',
    'APPAREL':                'Other',
    'AGRICULTURE':            'Other',
    'AUTOMOBILE':             'Engineering',
    'BPO':                    'Other',
    'PUBLIC-RELATIONS':       'Marketing',
}


def save_category_map():
    outpath = os.path.join(PROC_DIR, 'category_map.json')
    with open(outpath, 'w') as f:
        json.dump(CATEGORY_MAP, f, indent=2)
    print(f"\nCategory map saved to {outpath}")


def main():
    jobs_path = os.path.join(RAW_DIR, 'postings.csv', 'postings.csv')
    if not os.path.exists(jobs_path):
        jobs_path = os.path.join(RAW_DIR, 'postings.csv')

    resumes_path = os.path.join(RAW_DIR, 'Resume', 'Resume.csv')
    if not os.path.exists(resumes_path):
        resumes_path = os.path.join(RAW_DIR, 'Resume.csv')

    for path, name in [(jobs_path, 'Job postings'), (resumes_path, 'Resumes')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found at {path}")
            print(f"  Download from Kaggle and place in {RAW_DIR}/")
            return

    jobs_df = process_jobs(jobs_path)
    resumes_df = process_resumes(resumes_path)

    jobs_out = os.path.join(PROC_DIR, 'jobs_clean.csv')
    resumes_out = os.path.join(PROC_DIR, 'resumes_clean.csv')

    jobs_df.to_csv(jobs_out, index=False)
    resumes_df.to_csv(resumes_out, index=False)
    save_category_map()

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Jobs:    {len(jobs_df):,} records -> {jobs_out}")
    print(f"  Resumes: {len(resumes_df):,} records -> {resumes_out}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
