import os
import sys
import csv
import json
import math
import argparse


def precision_at_k(ranked_list, relevant, k):
	if k == 0:
		return 0.0
	top_k = ranked_list[:k]
	hits = sum(1 for doc_id in top_k if doc_id in relevant)
	return hits / k


def dcg_at_k(ranked_list, relevance_map, k):
	dcg = 0.0
	for i, doc_id in enumerate(ranked_list[:k]):
		rel = relevance_map.get(doc_id, 0)
		dcg += (2**rel - 1) / math.log2(i + 2)
	return dcg


def ndcg_at_k(ranked_list, relevance_map, k):
	actual = dcg_at_k(ranked_list, relevance_map, k)
	ideal_order = sorted(relevance_map.keys(), key=lambda d: relevance_map[d], reverse=True)
	ideal = dcg_at_k(ideal_order, relevance_map, k)
	if ideal == 0:
		return 0.0
	return actual / ideal


def average_precision(ranked_list, relevant):
	if not relevant:
		return 0.0
	ap = 0.0
	hits = 0
	for i, doc_id in enumerate(ranked_list):
		if doc_id in relevant:
			hits += 1
			ap += hits / (i + 1)
	return ap / len(relevant)


def mean_average_precision(all_ranked, all_relevant):
	aps = []
	for query_id in all_ranked:
		ranked = all_ranked[query_id]
		relevant = all_relevant.get(query_id, set())
		aps.append(average_precision(ranked, relevant))
	if not aps:
		return 0.0
	return sum(aps) / len(aps)


def load_ground_truth(filepath, relevance_threshold=2):
	relevance_map = {}
	relevant_sets = {}

	with open(filepath, encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			qid = int(row['query_id'])
			did = int(row['doc_id']) if row['doc_id'].isdigit() else hash(row['doc_id'])
			grade = int(row['relevance'])

			if qid not in relevance_map:
				relevance_map[qid] = {}
			relevance_map[qid][did] = grade

			if grade >= relevance_threshold:
				if qid not in relevant_sets:
					relevant_sets[qid] = set()
				relevant_sets[qid].add(did)

	return relevance_map, relevant_sets


def evaluate_retrieval(ground_truth_path, retriever_fn, queries, k_values=None, relevance_threshold=2):
	if k_values is None:
		k_values = [5, 10, 20]

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
			metrics['P@{}'.format(k)] = precision_at_k(ranked, rel_sets.get(qid, set()), k)
			metrics['NDCG@{}'.format(k)] = ndcg_at_k(ranked, rel_map[qid], k)

		metrics['AP'] = average_precision(ranked, rel_sets.get(qid, set()))
		per_query[qid] = metrics

	aggregate = {}
	if per_query:
		all_metrics = list(per_query.values())
		for metric_name in all_metrics[0]:
			values = [m[metric_name] for m in all_metrics]
			mean_val = sum(values) / len(values)
			variance = sum((v - mean_val) ** 2 for v in values) / len(values)
			aggregate['mean_' + metric_name] = mean_val
			aggregate['std_' + metric_name] = variance ** 0.5

	aggregate['MAP'] = mean_average_precision(all_ranked, rel_sets)
	aggregate['num_queries'] = len(per_query)

	return {
		'per_query': per_query,
		'aggregate': aggregate,
	}


def print_evaluation_report(results, mode_name=''):
	agg = results['aggregate']
	n = agg.get('num_queries', 0)

	print('\n' + '='*55)
	if mode_name:
		print('  Evaluation Report -- ' + mode_name)
	else:
		print('  Evaluation Report')
	print('  Queries evaluated: {}'.format(n))
	print('='*55)

	for key in sorted(agg.keys()):
		if key.startswith('mean_'):
			metric = key[5:]
			std = agg.get('std_' + metric, 0)
			print('  {:12s}  {:.4f}  (+-{:.4f})'.format(metric, agg[key], std))

	print('  {:12s}  {:.4f}'.format('MAP', agg['MAP']))
	print('='*55)


def compare_retrieval_modes(ground_truth_path, bm25f_fn, semantic_fn, hybrid_fn, queries, k_values=None):
	if k_values is None:
		k_values = [5, 10]

	modes = {
		'BM25F':    bm25f_fn,
		'Semantic': semantic_fn,
		'Hybrid':   hybrid_fn,
	}

	all_results = {}
	for name, fn in modes.items():
		print('\nEvaluating {}...'.format(name))
		all_results[name] = evaluate_retrieval(ground_truth_path, fn, queries, k_values)

	metrics_to_show = ['mean_P@{}'.format(k) for k in k_values]
	metrics_to_show += ['mean_NDCG@{}'.format(k) for k in k_values]
	metrics_to_show += ['MAP']

	print('\n' + '='*65)
	print('  Comparative Results')
	print('='*65)

	header = '  {:15s}'.format('Metric')
	for name in modes:
		header += '  {:>12s}'.format(name)
	print(header)
	print('  ' + '-'*15 + ('  ' + '-'*12) * len(modes))

	for metric in metrics_to_show:
		row = '  {:15s}'.format(metric.replace('mean_', ''))
		for name in modes:
			val = all_results[name]['aggregate'].get(metric, 0)
			row += '  {:12.4f}'.format(val)
		print(row)

	print('='*65)

	return all_results


if __name__ == '__main__':
	print('=== Evaluation Metrics Sanity Test ===\n')

	relevance = {10: 3, 20: 2, 30: 1, 40: 0, 50: 0}
	relevant = {10, 20}

	perfect = [10, 20, 30, 40, 50]
	print('Perfect ranking:', perfect)
	print('  P@3    = {:.3f}'.format(precision_at_k(perfect, relevant, 3)))
	print('  NDCG@5 = {:.3f}'.format(ndcg_at_k(perfect, relevance, 5)))
	print('  AP     = {:.3f}'.format(average_precision(perfect, relevant)))

	bad = [40, 50, 30, 20, 10]
	print('\nBad ranking:', bad)
	print('  P@3    = {:.3f}'.format(precision_at_k(bad, relevant, 3)))
	print('  NDCG@5 = {:.3f}'.format(ndcg_at_k(bad, relevance, 5)))
	print('  AP     = {:.3f}'.format(average_precision(bad, relevant)))

	print('\nAll metrics working correctly!')
