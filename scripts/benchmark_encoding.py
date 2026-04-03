import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

JOBS_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'jobs_clean.csv')
RESUMES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'resumes_clean.csv')

MODEL_NAME   = 'all-MiniLM-L6-v2'
BATCH_SIZE   = 128
SAMPLE_SIZES = [500, 2000, 5000, 10000, 20000]

FULL_JOBS    = 107278
FULL_RESUMES = 2483


def load_job_texts(n):
	df = pd.read_csv(JOBS_CSV, nrows=n)
	title_col = 'title_clean' if 'title_clean' in df.columns else 'title'
	desc_col  = 'description_clean' if 'description_clean' in df.columns else 'description'
	titles = df[title_col].fillna('').astype(str)
	descs  = df[desc_col].fillna('').astype(str).str[:500]
	return (titles + '. ' + titles + '. ' + descs).tolist()


def load_resume_texts():
	df = pd.read_csv(RESUMES_CSV)
	text_col = 'resume_clean' if 'resume_clean' in df.columns else 'Resume_str'
	return df[text_col].fillna('').astype(str).str[:1000].tolist()


def main():
	cuda_ok  = torch.cuda.is_available()
	gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'

	print('=' * 65)
	print('  JobMatch -- Semantic Encoding Benchmark')
	print('=' * 65)
	print('  Model     : {}'.format(MODEL_NAME))
	print('  Batch size: {}'.format(BATCH_SIZE))
	if cuda_ok:
		print('  CUDA      : YES -- ' + gpu_name)
	else:
		print('  CUDA      : NO (CPU only)')
	print()

	devices = ['cpu']
	if cuda_ok:
		devices.append('cuda')

	print('JOB CORPUS BENCHMARK')
	print('-' * 65)
	print('{:>10}  {:>6}  {:>10}  {:>10}  {:>8}'.format('N docs', 'Device', 'Time (s)', 'docs/s', 'Speedup'))
	print('-' * 65)

	results = {}

	for n in SAMPLE_SIZES:
		texts = load_job_texts(n)
		for dev in devices:
			model = SentenceTransformer(MODEL_NAME, device=dev)
			t0 = time.perf_counter()
			model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
			elapsed = time.perf_counter() - t0
			rate = n / elapsed
			results[(n, dev)] = (elapsed, rate)

		cpu_rate = results[(n, 'cpu')][1]
		for dev in devices:
			elapsed, rate = results[(n, dev)]
			if dev != 'cpu':
				speedup_str = '{:.1f}x'.format(rate / cpu_rate)
			else:
				speedup_str = 'baseline'
			print('{:>10,}  {:>6}  {:>10.2f}  {:>10.0f}  {:>8}'.format(n, dev, elapsed, rate, speedup_str))
		if len(devices) > 1:
			print()

	print()

	print('FULL CORPUS PROJECTIONS  (extrapolated from 20k sample)')
	print('-' * 65)

	for dev in devices:
		_, rate_20k = results[(20000, dev)]
		job_est    = FULL_JOBS    / rate_20k
		resume_est = FULL_RESUMES / rate_20k
		total_est  = job_est + resume_est
		print('  {:4s}  jobs ({:,}): {:.1f} min  |  resumes ({:,}): {:.1f}s  |  total: {:.1f} min'.format(
			dev.upper(), FULL_JOBS, job_est / 60, FULL_RESUMES, resume_est, total_est / 60))

	if cuda_ok:
		cpu_rate = results[(20000, 'cpu')][1]
		gpu_rate = results[(20000, 'cuda')][1]
		speedup  = gpu_rate / cpu_rate
		print('\n  GPU is {:.1f}x faster than CPU for this model/batch size.'.format(speedup))
	else:
		print()
		print('  NOTE: No GPU detected on this machine.')
		print('  Expected GPU speedup for all-MiniLM-L6-v2 on a modern NVIDIA GPU')
		print('  (e.g. T4/A100): ~15-20x over CPU, based on published benchmarks.')
		cpu_rate = results[(20000, 'cpu')][1]
		for mult in [15, 20]:
			gpu_rate_est = cpu_rate * mult
			job_est_gpu  = FULL_JOBS    / gpu_rate_est
			res_est_gpu  = FULL_RESUMES / gpu_rate_est
			print('    At {}x speedup: jobs ~{:.1f} min, resumes ~{:.0f}s'.format(
				mult, job_est_gpu / 60, res_est_gpu))

	print()
	print('RESUME CORPUS (full {:,} docs)'.format(FULL_RESUMES))
	print('-' * 65)
	resume_texts = load_resume_texts()
	for dev in devices:
		model = SentenceTransformer(MODEL_NAME, device=dev)
		t0 = time.perf_counter()
		model.encode(resume_texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
		elapsed = time.perf_counter() - t0
		rate = len(resume_texts) / elapsed
		print('  {:4s}  {:.2f}s  ({:.0f} docs/s)'.format(dev.upper(), elapsed, rate))

	print()
	print('Done.')


if __name__ == '__main__':
	main()
