import os
import sys
import shutil
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'indexes')


def step_preprocess():
    print("\n" + "="*60)
    print("  STEP 1: Preprocessing")
    print("="*60)
    from scripts.preprocess import main as preprocess_main
    preprocess_main()


def step_index(device: str = None):
    print("\n" + "="*60)
    print("  STEP 2: Building Indexes")
    print("="*60)
    os.makedirs(INDEX_DIR, exist_ok=True)

    jobs_csv = os.path.join(PROC_DIR, 'jobs_clean.csv')
    resumes_csv = os.path.join(PROC_DIR, 'resumes_clean.csv')

    for path, name in [(jobs_csv, 'Jobs'), (resumes_csv, 'Resumes')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} data not found at {path}. Run preprocess first.")
            return

    from engine.bm25f import build_job_index, build_resume_index

    print("\n--- Job Posting Index (BM25F) ---")
    job_idx = build_job_index(jobs_csv)
    job_idx.save(os.path.join(INDEX_DIR, 'jobs_bm25f.pkl'))

    print("\n--- Resume Index (BM25F) ---")
    resume_idx = build_resume_index(resumes_csv)
    resume_idx.save(os.path.join(INDEX_DIR, 'resumes_bm25f.pkl'))

    try:
        from engine.semantic import build_job_semantic_index, build_resume_semantic_index

        print("\n--- Job Posting Index (Semantic) ---")
        job_sem = build_job_semantic_index(jobs_csv, device=device)
        job_sem.save(os.path.join(INDEX_DIR, 'jobs_semantic'))

        print("\n--- Resume Index (Semantic) ---")
        resume_sem = build_resume_semantic_index(resumes_csv, device=device)
        resume_sem.save(os.path.join(INDEX_DIR, 'resumes_semantic'))

    except ImportError:
        print("\nWARNING: sentence-transformers not installed. Skipping semantic index.")
        print("Install with: pip install sentence-transformers")

    print("\nIndexing complete!")


def step_evaluate():
    print("\n" + "="*60)
    print("  STEP 3: Evaluation")
    print("="*60)

    gt_path = os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv')
    idx_path = os.path.join(INDEX_DIR, 'jobs_bm25f.pkl')
    if not os.path.exists(gt_path):
        print("Generating pooled BM25F ground truth...")
        from evaluation.generate_ground_truth import generate_pooled_ground_truth
        generate_pooled_ground_truth(
            os.path.join(PROC_DIR, 'resumes_clean.csv'),
            os.path.join(PROC_DIR, 'jobs_clean.csv'),
            gt_path,
            bm25f_index_path=idx_path,
            num_queries=50,
            top_k=20,
            n_random_negatives=5,
        )

    from evaluation.evaluate import (
        load_ground_truth, evaluate_retrieval, print_evaluation_report
    )
    from engine.bm25f import BM25FIndex

    if not os.path.exists(idx_path):
        print("ERROR: BM25F index not found. Run --step index first.")
        return

    import pandas as pd
    job_idx = BM25FIndex.load(idx_path)

    resumes_df = pd.read_csv(os.path.join(PROC_DIR, 'resumes_clean.csv'))
    id_col = 'ID' if 'ID' in resumes_df.columns else None
    text_col = 'resume_clean' if 'resume_clean' in resumes_df.columns else 'Resume_str'
    queries = dict(zip(
        resumes_df[id_col].astype(int) if id_col else range(len(resumes_df)),
        resumes_df[text_col].fillna('').astype(str),
    ))

    def bm25f_retrieve(query_text):
        return job_idx.search(query_text, top_k=50)

    results = evaluate_retrieval(gt_path, bm25f_retrieve, queries)
    print_evaluation_report(results, "BM25F")


def step_rebuild(device: str = None):
    print("\n" + "="*60)
    print("  REBUILD: Wipe & Regenerate")
    print("="*60)

    targets = [
        os.path.join(PROC_DIR),
        os.path.join(INDEX_DIR),
        os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv'),
    ]

    for target in targets:
        if os.path.isdir(target):
            shutil.rmtree(target)
            print(f"  Removed dir:  {os.path.relpath(target, BASE_DIR)}/")
        elif os.path.isfile(target):
            os.remove(target)
            print(f"  Removed file: {os.path.relpath(target, BASE_DIR)}")

    print()
    step_preprocess()
    step_index(device=device)
    step_evaluate()


def step_clean_all():
    print("\n" + "="*60)
    print("  CLEAN: Removing Generated Files")
    print("="*60)

    targets = [
        (os.path.join(PROC_DIR),                                    'dir'),
        (os.path.join(INDEX_DIR),                                   'dir'),
        (os.path.join(BASE_DIR, 'figures'),                         'dir'),
        (os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv'),  'file'),
    ]

    removed = 0
    for target, kind in targets:
        if kind == 'dir' and os.path.isdir(target):
            shutil.rmtree(target)
            print(f"  Removed dir:  {os.path.relpath(target, BASE_DIR)}/")
            removed += 1
        elif kind == 'file' and os.path.isfile(target):
            os.remove(target)
            print(f"  Removed file: {os.path.relpath(target, BASE_DIR)}")
            removed += 1

    if removed == 0:
        print("  Nothing to remove — already clean.")
    else:
        print(f"\n  Done. {removed} item(s) removed.")
        print("  Raw data in data/raw/ was not touched.")
        print("  Run: python build.py --step all   to regenerate everything.")


def step_demo():
    print("\n" + "="*60)
    print("  JobMatch Interactive Demo")
    print("="*60)

    idx_path = os.path.join(INDEX_DIR, 'jobs_bm25f.pkl')
    if not os.path.exists(idx_path):
        print("ERROR: Index not built. Run: python build.py --step index")
        return

    from engine.bm25f import BM25FIndex
    job_idx = BM25FIndex.load(idx_path)

    print(f"\nLoaded index with {job_idx.N:,} job postings.")
    print("Type a resume summary or skills to find matching jobs.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query > ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        results = job_idx.search(query, top_k=10)
        if not results:
            print("  No results found.\n")
            continue

        print(f"\n  Top results for: '{query}'")
        for rank, (doc_id, score) in enumerate(results, 1):
            meta = job_idx.get_doc(doc_id)
            title = meta.get('title', 'Unknown')
            company = meta.get('company', '')
            category = meta.get('category', '')
            print(f"  {rank:2d}. [{score:.3f}] {title}")
            if company:
                print(f"       {company} | {category}")
        print()


def main():
    parser = argparse.ArgumentParser(description="JobMatch Build Pipeline")
    parser.add_argument(
        '--step',
        choices=['all', 'preprocess', 'index', 'evaluate', 'demo', 'rebuild', 'clean-all'],
        default='all',
        help="Which pipeline step to run"
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default=None,
        help="Compute device for semantic encoding (default: auto-detect GPU, fall back to CPU)"
    )
    args = parser.parse_args()

    if args.step == 'clean-all':
        step_clean_all()
    elif args.step == 'rebuild':
        step_rebuild(device=args.device)
    else:
        if args.step in ('all', 'preprocess'):
            step_preprocess()
        if args.step in ('all', 'index'):
            step_index(device=args.device)
        if args.step in ('all', 'evaluate'):
            step_evaluate()
        if args.step == 'demo':
            step_demo()


if __name__ == '__main__':
    main()
