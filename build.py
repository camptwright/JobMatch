import os
import sys
import shutil
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'indexes')


def step_preprocess():
	print('\n' + '='*60)
	print('  STEP 1: Preprocessing')
	print('='*60)
	from scripts.preprocess import main as preprocess_main
	preprocess_main()


def step_ltr():
	print('\n' + '='*60)
	print('  STEP: Train LTR Reranker')
	print('='*60)

	gt_path = os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv')
	bm25f_path = os.path.join(INDEX_DIR, 'jobs_bm25f.pkl')
	ltr_path = os.path.join(INDEX_DIR, 'ltr_model.pkl')

	if not os.path.exists(gt_path):
		print('ERROR: ground_truth.csv not found. Run --step evaluate first.')
		return
	if not os.path.exists(bm25f_path):
		print('ERROR: BM25F index not found. Run --step index first.')
		return

	from engine.bm25f import BM25FIndex
	from engine.ltr import LTRReranker

	bm25f_idx = BM25FIndex.load(bm25f_path)

	sem_dir = os.path.join(INDEX_DIR, 'jobs_semantic')
	try:
		from engine.semantic import SemanticIndex
		from engine.hybrid import HybridRetriever
		sem_idx = SemanticIndex.load(sem_dir) if os.path.isdir(sem_dir) else None
	except Exception:
		sem_idx = None

	retriever = HybridRetriever(bm25f_idx, sem_idx) if sem_idx else None

	if retriever is None:
		print('WARNING: Semantic index not available. LTR will use only BM25F/LM features.')

	import pandas as pd
	resumes_df = pd.read_csv(os.path.join(PROC_DIR, 'resumes_clean.csv'))
	id_col = 'ID' if 'ID' in resumes_df.columns else None
	text_col = 'resume_clean' if 'resume_clean' in resumes_df.columns else 'Resume_str'
	queries = dict(zip(
		resumes_df[id_col].astype(int) if id_col else range(len(resumes_df)),
		resumes_df[text_col].fillna('').astype(str),
	))

	# Use a minimal retriever wrapper if sem_idx not available
	class _MinimalRetriever:
		def __init__(self, bm25f):
			self.bm25f = bm25f

	ltr = LTRReranker()
	ltr.train(gt_path, retriever if retriever else _MinimalRetriever(bm25f_idx), queries_dict=queries)
	ltr.save(ltr_path)
	print('\nLTR model saved to', ltr_path)


def step_index(device=None, resume_only=False):
	print('\n' + '='*60)
	if resume_only:
		print('  STEP 2: Building Resume Indexes Only')
	else:
		print('  STEP 2: Building Indexes')
	print('='*60)
	os.makedirs(INDEX_DIR, exist_ok=True)

	jobs_csv = os.path.join(PROC_DIR, 'jobs_clean.csv')
	resumes_csv = os.path.join(PROC_DIR, 'resumes_clean.csv')

	if not resume_only:
		for path, name in [(jobs_csv, 'Jobs'), (resumes_csv, 'Resumes')]:
			if not os.path.exists(path):
				print('ERROR: {} data not found at {}. Run preprocess first.'.format(name, path))
				return
	else:
		if not os.path.exists(resumes_csv):
			print('ERROR: Resume data not found at {}. Run preprocess first.'.format(resumes_csv))
			return

	from engine.bm25f import build_resume_index

	if not resume_only:
		from engine.bm25f import build_job_index
		print('\n--- Job Posting Index (BM25F) ---')
		job_idx = build_job_index(jobs_csv)
		job_idx.save(os.path.join(INDEX_DIR, 'jobs_bm25f.pkl'))

	print('\n--- Resume Index (BM25F) ---')
	resume_idx = build_resume_index(resumes_csv)
	resume_idx.save(os.path.join(INDEX_DIR, 'resumes_bm25f.pkl'))

	job_sem = None
	try:
		from engine.semantic import build_resume_semantic_index
		from engine.cluster import build_cluster_index

		if not resume_only:
			from engine.semantic import build_job_semantic_index
			print('\n--- Job Posting Index (Semantic) ---')
			job_sem = build_job_semantic_index(jobs_csv, device=device)
			job_sem.save(os.path.join(INDEX_DIR, 'jobs_semantic'))

			print('\n--- Job Cluster Index (k-means, n=50) ---')
			build_cluster_index(
				job_sem,
				n_clusters=50,
				save_path=os.path.join(INDEX_DIR, 'jobs_clusters'),
			)

		print('\n--- Resume Index (Semantic) ---')
		resume_sem = build_resume_semantic_index(resumes_csv, device=device)
		resume_sem.save(os.path.join(INDEX_DIR, 'resumes_semantic'))

		print('\n--- Resume Cluster Index (k-means, n=50) ---')
		build_cluster_index(
			resume_sem,
			n_clusters=50,
			save_path=os.path.join(INDEX_DIR, 'resumes_clusters'),
		)

	except ImportError:
		print('\nWARNING: sentence-transformers not installed. Skipping semantic + cluster indexes.')
		print('Install with: pip install sentence-transformers')

	if not resume_only:
		print('\n--- Job Posting Index (Language Model) ---')
		from engine.lm import build_job_lm_index
		job_lm = build_job_lm_index(jobs_csv)
		job_lm.save(os.path.join(INDEX_DIR, 'jobs_lm.pkl'))

	print('\n--- Resume Index (Language Model) ---')
	from engine.lm import build_resume_lm_index
	resume_lm = build_resume_lm_index(resumes_csv)
	resume_lm.save(os.path.join(INDEX_DIR, 'resumes_lm.pkl'))

	print('\nIndexing complete!')


def step_evaluate():
	print('\n' + '='*60)
	print('  STEP 3: Evaluation')
	print('='*60)

	gt_path = os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv')
	idx_path = os.path.join(INDEX_DIR, 'jobs_bm25f.pkl')
	if not os.path.exists(gt_path):
		print('Generating pooled ground truth (BM25F + semantic)...')
		from evaluation.generate_ground_truth import generate_pooled_ground_truth
		sem_dir = os.path.join(INDEX_DIR, 'jobs_semantic')
		generate_pooled_ground_truth(
			os.path.join(PROC_DIR, 'resumes_clean.csv'),
			os.path.join(PROC_DIR, 'jobs_clean.csv'),
			gt_path,
			bm25f_index_path=idx_path,
			semantic_index_dir=sem_dir if os.path.isdir(sem_dir) else None,
			num_queries=50,
			top_k=20,
			n_random_negatives=5,
		)

	from evaluation.evaluate import load_ground_truth, evaluate_retrieval, print_evaluation_report
	from engine.bm25f import BM25FIndex

	if not os.path.exists(idx_path):
		print('ERROR: BM25F index not found. Run --step index first.')
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
	print_evaluation_report(results, 'BM25F')


def step_rebuild(device=None):
	print('\n' + '='*60)
	print('  REBUILD: Wipe & Regenerate')
	print('='*60)

	targets = [
		PROC_DIR,
		INDEX_DIR,
		os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv'),
	]

	for target in targets:
		if os.path.isdir(target):
			shutil.rmtree(target)
			print('  Removed dir:  {}/'.format(os.path.relpath(target, BASE_DIR)))
		elif os.path.isfile(target):
			os.remove(target)
			print('  Removed file: {}'.format(os.path.relpath(target, BASE_DIR)))

	print()
	step_preprocess()
	step_index(device=device)
	step_evaluate()


def step_clean_all():
	print('\n' + '='*60)
	print('  CLEAN: Removing Generated Files')
	print('='*60)

	targets = [
		(PROC_DIR,                                                    'dir'),
		(INDEX_DIR,                                                   'dir'),
		(os.path.join(BASE_DIR, 'figures'),                          'dir'),
		(os.path.join(BASE_DIR, 'evaluation', 'ground_truth.csv'),   'file'),
	]

	removed = 0
	for target, kind in targets:
		if kind == 'dir' and os.path.isdir(target):
			shutil.rmtree(target)
			print('  Removed dir:  {}/'.format(os.path.relpath(target, BASE_DIR)))
			removed += 1
		elif kind == 'file' and os.path.isfile(target):
			os.remove(target)
			print('  Removed file: {}'.format(os.path.relpath(target, BASE_DIR)))
			removed += 1

	if removed == 0:
		print('  Nothing to remove -- already clean.')
	else:
		print('\n  Done. {} item(s) removed.'.format(removed))
		print('  Raw data in data/raw/ was not touched.')
		print('  Run: python build.py --step all   to regenerate everything.')


def step_demo():
	print('\n' + '='*60)
	print('  JobMatch Interactive Demo')
	print('='*60)

	idx_path = os.path.join(INDEX_DIR, 'jobs_bm25f.pkl')
	if not os.path.exists(idx_path):
		print('ERROR: Index not built. Run: python build.py --step index')
		return

	from engine.bm25f import BM25FIndex
	job_idx = BM25FIndex.load(idx_path)

	print('\nLoaded index with {:,} job postings.'.format(job_idx.N))
	print('Type a resume summary or skills to find matching jobs.')
	print("Type 'quit' to exit.\n")

	while True:
		query = input('Query > ').strip()
		if query.lower() in ('quit', 'exit', 'q'):
			break
		if not query:
			continue

		results = job_idx.search(query, top_k=10)
		if not results:
			print('  No results found.\n')
			continue

		print("\n  Top results for: '{}'".format(query))
		for rank, (doc_id, score) in enumerate(results, 1):
			meta = job_idx.get_doc(doc_id)
			title = meta.get('title', 'Unknown')
			company = meta.get('company', '')
			category = meta.get('category', '')
			print('  {:2d}. [{:.3f}] {}'.format(rank, score, title))
			if company:
				print('       {} | {}'.format(company, category))
		print()


def main():
	parser = argparse.ArgumentParser(description='JobMatch Build Pipeline')
	parser.add_argument(
		'--step',
		choices=['all', 'preprocess', 'index', 'evaluate', 'ltr', 'demo', 'rebuild', 'clean-all'],
		default='all',
		help='Which pipeline step to run'
	)
	parser.add_argument(
		'--device',
		choices=['cpu', 'cuda'],
		default=None,
		help='Compute device for semantic encoding (default: auto-detect GPU, fall back to CPU)'
	)
	parser.add_argument(
		'--resume-only',
		action='store_true',
		help='When used with --step index, rebuild only the resume indexes (leaves job indexes untouched)'
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
			step_index(device=args.device, resume_only=args.resume_only)
		if args.step in ('all', 'evaluate'):
			step_evaluate()
		if args.step in ('all', 'ltr'):
			step_ltr()
		if args.step == 'demo':
			step_demo()


if __name__ == '__main__':
	main()
