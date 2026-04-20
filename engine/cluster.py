import numpy as np


class ClusterIndex:
	def __init__(self, n_clusters=50, random_state=42):
		self.n_clusters = n_clusters
		self.random_state = random_state
		self.labels_ = None
		self.centroids_ = None
		self.doc_ids = None
		self._label_map = {}

	def fit(self, semantic_index):
		from sklearn.cluster import KMeans

		emb = semantic_index.embeddings.astype(np.float32)
		self.doc_ids = list(semantic_index.doc_ids)

		k = min(self.n_clusters, len(self.doc_ids))
		print('Clustering {} documents into {} clusters...'.format(len(self.doc_ids), k))
		km = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
		km.fit(emb)

		self.labels_ = km.labels_
		self.centroids_ = km.cluster_centers_
		self._label_map = {did: int(self.labels_[i]) for i, did in enumerate(self.doc_ids)}
		print('  Clustering complete.')
		return self

	def get_cluster(self, doc_id):
		return self._label_map.get(doc_id)

	def cluster_counts(self, doc_ids):
		counts = {}
		for did in doc_ids:
			c = self._label_map.get(did)
			if c is not None:
				counts[c] = counts.get(c, 0) + 1
		return counts

	def nearest_cluster(self, query_embedding):
		if self.centroids_ is None:
			return None
		q = np.array(query_embedding, dtype=np.float32)
		dists = np.linalg.norm(self.centroids_ - q, axis=1)
		return int(np.argmin(dists))

	def docs_in_cluster(self, cluster_id):
		return [did for did, c in self._label_map.items() if c == cluster_id]

	def save(self, path):
		np.savez(
			path,
			labels=self.labels_,
			centroids=self.centroids_,
			doc_ids=np.array(self.doc_ids),
		)
		print('ClusterIndex saved to {}'.format(path))

	@classmethod
	def load(cls, path, n_clusters=50):
		data = np.load(path + '.npz' if not path.endswith('.npz') else path)
		idx = cls(n_clusters=n_clusters)
		idx.labels_ = data['labels']
		idx.centroids_ = data['centroids']
		idx.doc_ids = data['doc_ids'].tolist()
		idx._label_map = {did: int(idx.labels_[i]) for i, did in enumerate(idx.doc_ids)}
		return idx


def build_cluster_index(semantic_index, n_clusters=50, save_path=None):
	ci = ClusterIndex(n_clusters=n_clusters)
	ci.fit(semantic_index)
	if save_path:
		ci.save(save_path)
	return ci


if __name__ == '__main__':
	print('=== ClusterIndex Demo ===\n')
	try:
		from engine.semantic import SemanticIndex

		sem = SemanticIndex()
		doc_ids = list(range(10))
		texts = [
			'Python developer Flask REST API backend',
			'Django web developer Python SQL',
			'Data scientist machine learning TensorFlow Python',
			'ML engineer PyTorch deep learning NLP',
			'Frontend React TypeScript CSS JavaScript',
			'Angular developer web UI components',
			'Finance analyst Excel modeling forecasting',
			'Accountant CPA audit financial statements',
			'Nurse RN patient care hospital clinical',
			'Physician MD surgery internal medicine',
		]
		meta = [{'title': t.split()[0]} for t in texts]
		sem.encode_documents(doc_ids, texts, meta)

		ci = ClusterIndex(n_clusters=3)
		ci.fit(sem)

		print('\nCluster assignments:')
		for did, text in zip(doc_ids, texts):
			print('  [cluster {}] {}'.format(ci.get_cluster(did), text[:50]))
	except ImportError as e:
		print('Missing dependency:', e)
