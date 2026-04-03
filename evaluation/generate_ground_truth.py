import os
import sys
import json
import csv
import time
import argparse
import random

import pandas as pd


_RELATED_CATEGORY_PAIRS = {
	frozenset({'Engineering', 'Information-Technology'}),
	frozenset({'Finance', 'Consulting'}),
	frozenset({'Sales', 'Marketing'}),
	frozenset({'Healthcare', 'Education'}),
	frozenset({'Design', 'Marketing'}),
}


RELEVANCE_PROMPT = """You are an expert recruiter evaluating how well a job posting matches a candidate's resume.

Rate the relevance of this job posting for this resume on a 0-3 scale:
  0 = Not relevant (completely different field or skills)
  1 = Marginally relevant (same broad industry but different role/level)
  2 = Relevant (related skills and field, reasonable match)
  3 = Highly relevant (strong skill overlap, ideal match for the candidate)

RESUME:
{resume_text}

JOB POSTING:
Title: {job_title}
Description: {job_description}

Respond with ONLY a single integer (0, 1, 2, or 3) and nothing else."""


def score_with_openai(resume_text, job_title, job_description, model='gpt-4o-mini'):
	import openai

	client = openai.OpenAI()
	prompt = RELEVANCE_PROMPT.format(
		resume_text=resume_text[:2000],
		job_title=job_title,
		job_description=job_description[:1000],
	)

	response = client.chat.completions.create(
		model=model,
		messages=[{'role': 'user', 'content': prompt}],
		max_tokens=5,
		temperature=0.0,
	)

	text = response.choices[0].message.content.strip()
	try:
		grade = int(text[0])
		return min(max(grade, 0), 3)
	except (ValueError, IndexError):
		return 0


def score_with_anthropic(resume_text, job_title, job_description, model='claude-sonnet-4-6'):
	import anthropic

	client = anthropic.Anthropic()
	prompt = RELEVANCE_PROMPT.format(
		resume_text=resume_text[:2000],
		job_title=job_title,
		job_description=job_description[:1000],
	)

	response = client.messages.create(
		model=model,
		max_tokens=5,
		messages=[{'role': 'user', 'content': prompt}],
	)

	text = response.content[0].text.strip()
	try:
		grade = int(text[0])
		return min(max(grade, 0), 3)
	except (ValueError, IndexError):
		return 0


def generate_ground_truth(resumes_csv, jobs_csv, output_path, num_queries=50, top_k=20, api='openai', seed=42):
	random.seed(seed)

	resumes_df = pd.read_csv(resumes_csv)
	jobs_df = pd.read_csv(jobs_csv)

	cat_map_path = os.path.join(os.path.dirname(resumes_csv), 'category_map.json')
	if os.path.exists(cat_map_path):
		with open(cat_map_path) as f:
			category_map = json.load(f)
	else:
		category_map = {}

	categories = resumes_df['Category'].unique()
	sampled_resumes = []
	per_cat = max(2, num_queries // len(categories))

	for cat in categories:
		cat_resumes = resumes_df[resumes_df['Category'] == cat]
		n = min(per_cat, len(cat_resumes))
		sampled_resumes.extend(cat_resumes.sample(n=n, random_state=seed).to_dict('records'))

	random.shuffle(sampled_resumes)
	sampled_resumes = sampled_resumes[:num_queries]

	print('Selected {} query resumes across {} categories'.format(len(sampled_resumes), len(categories)))

	score_fn = score_with_openai if api == 'openai' else score_with_anthropic

	judgments = []
	total_pairs = 0

	for qi, resume in enumerate(sampled_resumes):
		resume_id = resume.get('ID', qi)
		resume_text = resume.get('resume_clean', resume.get('Resume_str', ''))
		resume_cat = resume.get('Category', '')
		mapped_cat = category_map.get(resume_cat, resume_cat)

		same_cat_jobs = jobs_df[jobs_df['job_category'] == mapped_cat]
		other_jobs = jobs_df[jobs_df['job_category'] != mapped_cat]

		n_same = min(top_k // 2, len(same_cat_jobs))
		n_other = top_k - n_same

		candidates = pd.concat([
			same_cat_jobs.sample(n=n_same, random_state=seed + qi) if n_same > 0 else pd.DataFrame(),
			other_jobs.sample(n=min(n_other, len(other_jobs)), random_state=seed + qi),
		])

		print('\n[{}/{}] Resume {} ({}) -> {} candidates'.format(
			qi + 1, len(sampled_resumes), resume_id, resume_cat, len(candidates)))

		for ji, (_, job) in enumerate(candidates.iterrows()):
			job_id = job.get('job_id', ji)
			job_title = str(job.get('title', ''))
			job_desc = str(job.get('description_clean', job.get('description', '')))

			try:
				grade = score_fn(resume_text, job_title, job_desc)
			except Exception as e:
				print('  Error scoring pair: {}'.format(e))
				grade = 0

			judgments.append({
				'query_id': resume_id,
				'query_category': resume_cat,
				'doc_id': job_id,
				'doc_category': str(job.get('job_category', '')),
				'doc_title': job_title,
				'relevance': grade,
			})
			total_pairs += 1

			if (ji + 1) % 5 == 0:
				print('    Scored {}/{} pairs'.format(ji + 1, len(candidates)))

			time.sleep(0.2)

	with open(output_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=judgments[0].keys())
		writer.writeheader()
		writer.writerows(judgments)

	print('\n' + '='*50)
	print('Ground truth saved: {}'.format(output_path))
	print('  Total pairs scored: {}'.format(total_pairs))
	print('  Relevance distribution:')
	for grade in range(4):
		count = sum(1 for j in judgments if j['relevance'] == grade)
		print('    Grade {}: {} ({:.1f}%)'.format(grade, count, count / total_pairs * 100))
	print('='*50)


def generate_category_ground_truth(resumes_csv, jobs_csv, output_path, num_queries=100, top_k=20, seed=42):
	random.seed(seed)

	resumes_df = pd.read_csv(resumes_csv)
	jobs_df = pd.read_csv(jobs_csv)

	cat_map_path = os.path.join(os.path.dirname(resumes_csv), 'category_map.json')
	if os.path.exists(cat_map_path):
		with open(cat_map_path) as f:
			category_map = json.load(f)
	else:
		category_map = {}

	sampled = resumes_df.sample(n=min(num_queries, len(resumes_df)), random_state=seed)

	judgments = []
	for qi, (_, resume) in enumerate(sampled.iterrows()):
		resume_id = resume.get('ID', qi)
		resume_cat = resume.get('Category', '')
		mapped_cat = category_map.get(resume_cat, resume_cat)

		candidate_jobs = jobs_df.sample(n=min(top_k, len(jobs_df)), random_state=seed + qi)

		for _, job in candidate_jobs.iterrows():
			job_cat = str(job.get('job_category', ''))

			if job_cat == mapped_cat:
				grade = 3
			elif frozenset({job_cat, mapped_cat}) in _RELATED_CATEGORY_PAIRS:
				grade = 1
			else:
				grade = 0

			judgments.append({
				'query_id': resume_id,
				'query_category': resume_cat,
				'doc_id': job.get('job_id', ''),
				'doc_category': job_cat,
				'doc_title': str(job.get('title', '')),
				'relevance': grade,
			})

	with open(output_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=judgments[0].keys())
		writer.writeheader()
		writer.writerows(judgments)

	total = len(judgments)
	print('Category-based ground truth saved: {}'.format(output_path))
	print('  Total pairs: {}'.format(total))
	for grade in [0, 1, 3]:
		count = sum(1 for j in judgments if j['relevance'] == grade)
		print('  Grade {}: {} ({:.1f}%)'.format(grade, count, count / total * 100))


def generate_pooled_ground_truth(resumes_csv, jobs_csv, output_path, bm25f_index_path,
                                  num_queries=50, top_k=20, n_random_negatives=5, seed=42):
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
	from engine.bm25f import BM25FIndex

	random.seed(seed)

	resumes_df = pd.read_csv(resumes_csv)
	jobs_df = pd.read_csv(jobs_csv)

	cat_map_path = os.path.join(os.path.dirname(resumes_csv), 'category_map.json')
	if os.path.exists(cat_map_path):
		with open(cat_map_path) as f:
			category_map = json.load(f)
	else:
		category_map = {}

	print('Loading BM25F index from {}...'.format(bm25f_index_path))
	idx = BM25FIndex.load(bm25f_index_path)
	print('Loaded {:,} docs'.format(idx.N))

	categories = resumes_df['Category'].unique()
	sampled = []
	per_cat = max(2, num_queries // len(categories))
	for cat in categories:
		cat_resumes = resumes_df[resumes_df['Category'] == cat]
		n = min(per_cat, len(cat_resumes))
		sampled.extend(cat_resumes.sample(n=n, random_state=seed).to_dict('records'))
	random.shuffle(sampled)
	sampled = sampled[:num_queries]
	print('Selected {} query resumes'.format(len(sampled)))

	jobs_by_id = jobs_df.set_index('job_id').to_dict('index')
	all_job_ids = list(jobs_by_id.keys())

	judgments = []
	for qi, resume in enumerate(sampled):
		resume_id = resume.get('ID', qi)
		resume_cat = resume.get('Category', '')
		mapped_cat = category_map.get(resume_cat, resume_cat)
		query_text = resume.get('resume_clean', resume.get('Resume_str', ''))

		retrieved = idx.search(query_text, top_k=top_k)
		candidate_ids = [doc_id for doc_id, _ in retrieved]

		candidate_set = set(candidate_ids)
		neg_pool = [jid for jid in all_job_ids if jid not in candidate_set]
		neg_sample = random.sample(neg_pool, min(n_random_negatives, len(neg_pool)))
		candidate_ids.extend(neg_sample)

		if (qi + 1) % 10 == 0 or qi == 0:
			print('  [{}/{}] Resume {} ({}): {} candidates'.format(
				qi + 1, len(sampled), resume_id, resume_cat, len(candidate_ids)))

		for doc_id in candidate_ids:
			job = jobs_by_id.get(doc_id, {})
			job_cat = str(job.get('job_category', ''))
			grade = 3 if job_cat == mapped_cat else 0

			judgments.append({
				'query_id': resume_id,
				'query_category': resume_cat,
				'doc_id': doc_id,
				'doc_category': job_cat,
				'doc_title': str(job.get('title', '')),
				'relevance': grade,
			})

	with open(output_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=judgments[0].keys())
		writer.writeheader()
		writer.writerows(judgments)

	total = len(judgments)
	relevant = sum(1 for j in judgments if j['relevance'] >= 2)
	print('\nPooled ground truth saved: {}'.format(output_path))
	print('  Total pairs: {}, Relevant: {} ({:.1f}%)'.format(total, relevant, relevant / total * 100))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate ground truth relevance judgments')
	parser.add_argument('--resumes', default='data/processed/resumes_clean.csv')
	parser.add_argument('--jobs', default='data/processed/jobs_clean.csv')
	parser.add_argument('--output', default='evaluation/ground_truth.csv')
	parser.add_argument('--api', choices=['openai', 'anthropic', 'category'], default='category')
	parser.add_argument('--num-queries', type=int, default=50)
	parser.add_argument('--top-k', type=int, default=20)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

	if args.api == 'category':
		generate_category_ground_truth(
			args.resumes, args.jobs, args.output,
			num_queries=args.num_queries, top_k=args.top_k, seed=args.seed,
		)
	else:
		generate_ground_truth(
			args.resumes, args.jobs, args.output,
			num_queries=args.num_queries, top_k=args.top_k,
			api=args.api, seed=args.seed,
		)
