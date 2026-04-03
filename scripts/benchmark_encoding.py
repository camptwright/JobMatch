import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

JOBS_CSV   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'jobs_clean.csv')
RESUMES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'resumes_clean.csv')

MODEL_NAME  = 'all-MiniLM-L6-v2'
BATCH_SIZE  = 128
SAMPLE_SIZES = [500, 2_000, 5_000, 10_000, 20_000]

FULL_JOBS    = 107_278
FULL_RESUMES = 2_483


def load_job_texts(n: int) -> list[str]:
    df = pd.read_csv(JOBS_CSV, nrows=n)
    title_col = 'title_clean' if 'title_clean' in df.columns else 'title'
    desc_col  = 'description_clean' if 'description_clean' in df.columns else 'description'
    titles = df[title_col].fillna('').astype(str)
    descs  = df[desc_col].fillna('').astype(str).str[:500]
    return (titles + '. ' + titles + '. ' + descs).tolist()


def load_resume_texts() -> list[str]:
    df = pd.read_csv(RESUMES_CSV)
    text_col = 'resume_clean' if 'resume_clean' in df.columns else 'Resume_str'
    return df[text_col].fillna('').astype(str).str[:1000].tolist()


def bench_device(device: str, texts: list[str], label: str) -> float:
    model = SentenceTransformer(MODEL_NAME, device=device)
    t0 = time.perf_counter()
    model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{device.upper():4s}] {label}: {elapsed:.2f}s  ({len(texts)/elapsed:.0f} docs/s)")
    return elapsed


def main():
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"

    print("=" * 65)
    print("  JobMatch — Semantic Encoding Benchmark")
    print("=" * 65)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  CUDA      : {'YES — ' + gpu_name if cuda_ok else 'NO (CPU only)'}")
    print()

    devices = ['cpu']
    if cuda_ok:
        devices.append('cuda')

    print("JOB CORPUS BENCHMARK")
    print("-" * 65)
    print(f"{'N docs':>10}  {'Device':>6}  {'Time (s)':>10}  {'docs/s':>10}  {'Speedup':>8}")
    print("-" * 65)

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
            speedup = rate / cpu_rate if dev != 'cpu' else 1.0
            speedup_str = f"{speedup:.1f}x" if dev != 'cpu' else "baseline"
            print(f"{n:>10,}  {dev:>6}  {elapsed:>10.2f}  {rate:>10.0f}  {speedup_str:>8}")
        if len(devices) > 1:
            print()

    print()

    print("FULL CORPUS PROJECTIONS  (extrapolated from 20k sample)")
    print("-" * 65)

    for dev in devices:
        _, rate_20k = results[(20_000, dev)]
        job_est    = FULL_JOBS    / rate_20k
        resume_est = FULL_RESUMES / rate_20k
        total_est  = job_est + resume_est
        print(f"  {dev.upper():4s}  jobs ({FULL_JOBS:,}): {job_est/60:.1f} min  |  "
              f"resumes ({FULL_RESUMES:,}): {resume_est:.1f}s  |  total: {total_est/60:.1f} min")

    if cuda_ok:
        cpu_rate = results[(20_000, 'cpu')][1]
        gpu_rate = results[(20_000, 'cuda')][1]
        speedup  = gpu_rate / cpu_rate
        print(f"\n  GPU is {speedup:.1f}x faster than CPU for this model/batch size.")
    else:
        print()
        print("  NOTE: No GPU detected on this machine.")
        print("  Expected GPU speedup for all-MiniLM-L6-v2 on a modern NVIDIA GPU")
        print("  (e.g. T4/A100): ~15–20x over CPU, based on published benchmarks.")
        cpu_rate = results[(20_000, 'cpu')][1]
        for mult in [15, 20]:
            gpu_rate_est = cpu_rate * mult
            job_est_gpu  = FULL_JOBS    / gpu_rate_est
            res_est_gpu  = FULL_RESUMES / gpu_rate_est
            print(f"    At {mult}x speedup: jobs ~{job_est_gpu/60:.1f} min, resumes ~{res_est_gpu:.0f}s")

    print()
    print("RESUME CORPUS (full 2,483 docs)")
    print("-" * 65)
    resume_texts = load_resume_texts()
    for dev in devices:
        model = SentenceTransformer(MODEL_NAME, device=dev)
        t0 = time.perf_counter()
        model.encode(resume_texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
        elapsed = time.perf_counter() - t0
        rate = len(resume_texts) / elapsed
        print(f"  {dev.upper():4s}  {elapsed:.2f}s  ({rate:.0f} docs/s)")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
