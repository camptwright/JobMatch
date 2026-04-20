import csv
import pickle
import re

import numpy as np


def _token_set(text):
	if not isinstance(text, str):
		return set()
	return set(re.findall(r'[a-z0-9]+', text.lower()))


def extract_features(query, doc_meta, bm25f_score=0.0, semantic_score=0.0, lm_score=0.0):
	q_tokens = _token_set(query)
	title = doc_meta.get('title', '') or ''
	desc = str(doc_meta.get('description', '') or '')[:200]
	t_tokens = _token_set(title)
	d_tokens = _token_set(desc)

	def jaccard(a, b):
		if not a or not b:
			return 0.0
		return len(a & b) / len(a | b)

	lm_feat = max(lm_score, -500.0) + 500.0

	return np.array([
		min(bm25f_score, 50.0),
		float(semantic_score),
		lm_feat,
		jaccard(q_tokens, t_tokens),
		jaccard(q_tokens, d_tokens),
		float(len(q_tokens)),
		float(len(t_tokens)),
	], dtype=np.float32)


class LTRReranker:
	def __init__(self):
		self.model = None
		self.scaler = None
		self._trained = False

	def train(self, ground_truth_csv, retriever, queries_dict=None, top_k=50, verbose=True):
		try:
			from sklearn.linear_model import LogisticRegression
			from sklearn.preprocessing import StandardScaler
		except ImportError:
			raise ImportError('scikit-learn required: pip install scikit-learn')

		rows = []
		with open(ground_truth_csv, newline='', encoding='utf-8') as f:
			reader = csv.DictReader(f)
			for row in reader:
				rows.append(row)

		if not rows:
			raise ValueError('ground_truth.csv is empty')

		X, y = [], []

		for row in rows:
			qid = row.get('query_id') or row.get('resume_id') or ''
			doc_id = row.get('doc_id') or row.get('job_id') or ''
			relevance = int(float(row.get('relevance', 0)))

			if queries_dict is None:
				query_text = row.get('query_text', '')
			else:
				query_text = queries_dict.get(int(qid), '') if qid else ''

			if not query_text:
				continue

			bm25f_score = float(row.get('bm25f_score', 0.0) or 0.0)
			semantic_score = float(row.get('semantic_score', 0.0) or 0.0)
			lm_score = float(row.get('lm_score', 0.0) or 0.0)

			doc_id_int = int(doc_id) if doc_id else None
			meta = retriever.bm25f.get_doc(doc_id_int) or {} if doc_id_int is not None else {}

			feat = extract_features(query_text, meta, bm25f_score, semantic_score, lm_score)
			X.append(feat)
			y.append(min(relevance, 1))

		if len(X) < 10:
			raise ValueError(
				'Too few labeled examples ({}) to train LTR. '
				'Run evaluation/generate_ground_truth.py first.'.format(len(X))
			)

		X = np.array(X)
		y = np.array(y)

		self.scaler = StandardScaler()
		X_scaled = self.scaler.fit_transform(X)

		self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
		self.model.fit(X_scaled, y)
		self._trained = True

		if verbose:
			pos = int(y.sum())
			print('LTR trained on {} pairs ({} relevant, {} non-relevant)'.format(
				len(y), pos, len(y) - pos))

	def rerank(self, query, results):
		if not self._trained or not results:
			return results

		feats = []
		for doc_id, score, meta in results:
			bm25f_score = meta.get('bm25f_score', score)
			semantic_score = meta.get('semantic_score', score)
			lm_score = meta.get('lm_score', 0.0)
			feats.append(extract_features(query, meta, bm25f_score, semantic_score, lm_score))

		X = np.array(feats)
		X_scaled = self.scaler.transform(X)
		probs = self.model.predict_proba(X_scaled)[:, 1]

		return sorted(
			[(doc_id, float(p), meta) for (doc_id, _, meta), p in zip(results, probs)],
			key=lambda x: x[1],
			reverse=True,
		)

	def save(self, path):
		with open(path, 'wb') as f:
			pickle.dump(
				{'model': self.model, 'scaler': self.scaler, '_trained': self._trained},
				f, protocol=4,
			)
		print('LTRReranker saved to {}'.format(path))

	@classmethod
	def load(cls, path):
		ltr = cls()
		with open(path, 'rb') as f:
			state = pickle.load(f)
		ltr.model = state['model']
		ltr.scaler = state['scaler']
		ltr._trained = state['_trained']
		return ltr


if __name__ == '__main__':
	print('=== LTR Feature Extraction Demo ===\n')

	meta = {'title': 'Senior Python Developer', 'description': 'Flask REST API Django backend Python'}
	feat = extract_features('python backend developer', meta, bm25f_score=12.5, semantic_score=0.82)
	labels = ['bm25f', 'semantic', 'lm', 'title_overlap', 'desc_overlap', 'query_len', 'title_len']
	for label, val in zip(labels, feat):
		print('  {:20s} {:.4f}'.format(label, val))

	print('\nTo train: python build.py --step ltr')
	print('Requires ground_truth.csv — run: python evaluation/generate_ground_truth.py --api category')
