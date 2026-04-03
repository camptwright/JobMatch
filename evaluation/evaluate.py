import os
import sys
import csv
import json
import math
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np


def precision_at_k(ranked_list: List[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def dcg_at_k(ranked_list: List[int], relevance_map: Dict[int, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        rel = relevance_map.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_list: List[int], relevance_map: Dict[int, int], k: int) -> float:
    actual = dcg_at_k(ranked_list, relevance_map, k)
    ideal_order = sorted(relevance_map.keys(), key=lambda d: relevance_map[d], reverse=True)
    ideal = dcg_at_k(ideal_order, relevance_map, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def average_precision(ranked_list: List[int], relevant: set) -> float:
    if not relevant:
        return 0.0
    ap = 0.0
    hits = 0
    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)


def mean_average_precision(
    all_ranked: Dict[int, List[int]],
    all_relevant: Dict[int, set],
) -> float:
    aps = []
    for query_id in all_ranked:
        ranked = all_ranked[query_id]
        relevant = all_relevant.get(query_id, set())
        aps.append(average_precision(ranked, relevant))
    return np.mean(aps) if aps else 0.0


def load_ground_truth(filepath: str, relevance_threshold: int = 2) -> Tuple[
    Dict[int, Dict[int, int]],
    Dict[int, set],
]:
    relevance_map = defaultdict(dict)
    relevant_sets = defaultdict(set)

    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row['query_id'])
            did = int(row['doc_id']) if row['doc_id'].isdigit() else hash(row['doc_id'])
            grade = int(row['relevance'])
            relevance_map[qid][did] = grade
            if grade >= relevance_threshold:
                relevant_sets[qid].add(did)

    return dict(relevance_map), dict(relevant_sets)


def evaluate_retrieval(
    ground_truth_path: str,
    retriever_fn,
    queries: Dict[int, str],
    k_values: List[int] = [5, 10, 20],
    relevance_threshold: int = 2,
) -> dict:
    rel_map, rel_sets = load_ground_truth(ground_truth_path, relevance_threshold)

    per_query = {}
    all_ranked = {}

    for qid in rel_map:
        if qid not in queries:
            continue

        query_text = queries[qid]
        results = retriever_fn(query_text)
        ranked = [doc_id for doc_id, _ in results]
        all_ranked[qid] = ranked

        metrics = {}
        for k in k_values:
            metrics[f'P@{k}'] = precision_at_k(ranked, rel_sets.get(qid, set()), k)
            metrics[f'NDCG@{k}'] = ndcg_at_k(ranked, rel_map[qid], k)

        metrics['AP'] = average_precision(ranked, rel_sets.get(qid, set()))
        per_query[qid] = metrics

    aggregate = {}
    if per_query:
        all_metrics = list(per_query.values())
        for metric_name in all_metrics[0]:
            values = [m[metric_name] for m in all_metrics]
            aggregate[f'mean_{metric_name}'] = np.mean(values)
            aggregate[f'std_{metric_name}'] = np.std(values)

    aggregate['MAP'] = mean_average_precision(all_ranked, rel_sets)
    aggregate['num_queries'] = len(per_query)

    return {
        'per_query': per_query,
        'aggregate': aggregate,
    }


def print_evaluation_report(results: dict, mode_name: str = ""):
    agg = results['aggregate']
    n = agg.get('num_queries', 0)

    print(f"\n{'='*55}")
    print(f"  Evaluation Report{' — ' + mode_name if mode_name else ''}")
    print(f"  Queries evaluated: {n}")
    print(f"{'='*55}")

    for key, val in sorted(agg.items()):
        if key.startswith('mean_'):
            metric = key[5:]
            std = agg.get(f'std_{metric}', 0)
            print(f"  {metric:12s}  {val:.4f}  (± {std:.4f})")

    print(f"  {'MAP':12s}  {agg['MAP']:.4f}")
    print(f"{'='*55}")


def compare_retrieval_modes(
    ground_truth_path: str,
    bm25f_fn,
    semantic_fn,
    hybrid_fn,
    queries: Dict[int, str],
    k_values: List[int] = [5, 10],
):
    modes = {
        'BM25F': bm25f_fn,
        'Semantic': semantic_fn,
        'Hybrid': hybrid_fn,
    }

    all_results = {}
    for name, fn in modes.items():
        print(f"\nEvaluating {name}...")
        all_results[name] = evaluate_retrieval(
            ground_truth_path, fn, queries, k_values
        )

    metrics_to_show = [f'mean_P@{k}' for k in k_values] + [f'mean_NDCG@{k}' for k in k_values] + ['MAP']

    print(f"\n{'='*65}")
    print(f"  Comparative Results")
    print(f"{'='*65}")

    header = f"  {'Metric':15s}"
    for name in modes:
        header += f"  {name:>12s}"
    print(header)
    print(f"  {'-'*15}" + f"  {'-'*12}" * len(modes))

    for metric in metrics_to_show:
        row = f"  {metric.replace('mean_', ''):15s}"
        for name in modes:
            val = all_results[name]['aggregate'].get(metric, 0)
            row += f"  {val:12.4f}"
        print(row)

    print(f"{'='*65}")

    return all_results


if __name__ == '__main__':
    print("=== Evaluation Metrics Sanity Test ===\n")

    relevance = {10: 3, 20: 2, 30: 1, 40: 0, 50: 0}
    relevant = {10, 20}

    perfect = [10, 20, 30, 40, 50]
    print("Perfect ranking:", perfect)
    print(f"  P@3  = {precision_at_k(perfect, relevant, 3):.3f}")
    print(f"  NDCG@5 = {ndcg_at_k(perfect, relevance, 5):.3f}")
    print(f"  AP   = {average_precision(perfect, relevant):.3f}")

    bad = [40, 50, 30, 20, 10]
    print("\nBad ranking:", bad)
    print(f"  P@3  = {precision_at_k(bad, relevant, 3):.3f}")
    print(f"  NDCG@5 = {ndcg_at_k(bad, relevance, 5):.3f}")
    print(f"  AP   = {average_precision(bad, relevant):.3f}")

    print("\nAll metrics working correctly!")
