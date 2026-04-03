from engine.bm25f import BM25FIndex
from engine.semantic import SemanticIndex


class HybridRetriever:
	def __init__(self, bm25f_index, semantic_index, alpha=0.5, candidate_pool=100):
		self.bm25f = bm25f_index
		self.semantic = semantic_index
		self.alpha = alpha
		self.candidate_pool = candidate_pool

	def search(self, query, top_k=10, alpha=None, mode='hybrid', return_metadata=True):
		a = alpha if alpha is not None else self.alpha

		if mode == 'bm25f':
			results = self.bm25f.search(query, top_k=top_k)
			if not return_metadata:
				return results
			return [(doc_id, score, self.bm25f.get_doc(doc_id)) for doc_id, score in results]

		if mode == 'semantic':
			results = self.semantic.search(query, top_k=top_k)
			if not return_metadata:
				return results
			return [(doc_id, score, self.semantic.get_doc(doc_id)) for doc_id, score in results]

		bm25_results = self.bm25f.search(query, top_k=self.candidate_pool)
		sem_results = self.semantic.search(query, top_k=self.candidate_pool)

		all_candidates = set()
		bm25_scores = {}
		sem_scores = {}

		for doc_id, score in bm25_results:
			all_candidates.add(doc_id)
			bm25_scores[doc_id] = score

		for doc_id, score in sem_results:
			all_candidates.add(doc_id)
			sem_scores[doc_id] = score

		bm25_norm = _min_max_normalize(bm25_scores)
		sem_norm = _min_max_normalize(sem_scores)

		combined = []
		for doc_id in all_candidates:
			b_score = bm25_norm.get(doc_id, 0.0)
			s_score = sem_norm.get(doc_id, 0.0)
			final = a * b_score + (1.0 - a) * s_score
			combined.append((doc_id, final))

		combined.sort(key=lambda x: x[1], reverse=True)

		if not return_metadata:
			return combined[:top_k]

		results = []
		for doc_id, score in combined[:top_k]:
			meta = self.bm25f.get_doc(doc_id)
			if not meta:
				meta = self.semantic.get_doc(doc_id)
			meta = dict(meta)
			meta['bm25f_score'] = bm25_scores.get(doc_id, 0.0)
			meta['semantic_score'] = sem_scores.get(doc_id, 0.0)
			results.append((doc_id, score, meta))

		return results

	def compare_modes(self, query, top_k=5):
		return {
			'bm25f':    self.search(query, top_k=top_k, mode='bm25f'),
			'semantic': self.search(query, top_k=top_k, mode='semantic'),
			'hybrid':   self.search(query, top_k=top_k, mode='hybrid'),
		}


def _min_max_normalize(scores):
	if not scores:
		return {}
	values = list(scores.values())
	min_s = min(values)
	max_s = max(values)
	spread = max_s - min_s
	if spread == 0:
		return {k: 1.0 for k in scores}
	return {k: (v - min_s) / spread for k, v in scores.items()}


if __name__ == '__main__':
	print('=== Hybrid Retriever Demo ===\n')
	print('Run from the project root: python -m engine.hybrid\n')

	from engine.bm25f import BM25FIndex

	config = {
		'title':       {'weight': 3.0, 'b': 0.3},
		'description': {'weight': 1.0, 'b': 0.75},
	}

	bm25 = BM25FIndex(field_configs=config)
	jobs = [
		{'title': 'Senior Python Developer', 'description': 'Experienced Python developer. Django Flask REST APIs. Machine learning is a plus.'},
		{'title': 'Data Scientist', 'description': 'Build ML models. Python TensorFlow scikit-learn. Statistics background.'},
		{'title': 'Frontend React Developer', 'description': 'Modern web apps. React TypeScript CSS. Responsive design accessibility.'},
		{'title': 'ML Engineer', 'description': 'Deploy ML pipelines. Python PyTorch cloud. NLP computer vision.'},
		{'title': 'IT Project Manager', 'description': 'Lead agile teams. Software delivery Scrum. Technical background.'},
	]

	for i, job in enumerate(jobs):
		bm25.add_document(i, job, {'title': job['title']})
	bm25.build()

	try:
		sem = SemanticIndex()
		texts = ['{} {}'.format(j['title'], j['description']) for j in jobs]
		meta = [{'title': j['title']} for j in jobs]
		sem.encode_documents(list(range(5)), texts, meta)

		hybrid = HybridRetriever(bm25, sem, alpha=0.5)

		query = 'python machine learning'
		print("Query: '{}'\n".format(query))

		comparison = hybrid.compare_modes(query, top_k=3)
		for mode, results in comparison.items():
			print('  [{}]'.format(mode.upper()))
			for rank, (doc_id, score, m) in enumerate(results, 1):
				print('    {}. [{:.3f}] {}'.format(rank, score, m['title']))
			print()

	except ImportError:
		print('sentence-transformers not installed. Testing BM25F only.')
		results = bm25.search('python machine learning', top_k=3)
		for rank, (doc_id, score) in enumerate(results, 1):
			print('  {}. [{:.3f}] {}'.format(rank, score, bm25.get_doc(doc_id)['title']))
