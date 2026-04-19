import os
import json
import numpy as np


def _detect_device():
	try:
		import torch
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	except ImportError:
		return 'cpu'


class SemanticIndex:
	def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
		self.model_name = model_name
		self.device = device if device is not None else _detect_device()
		self.model = None
		self.embeddings = None
		self.doc_ids = []
		self.doc_store = {}
		self._dim = None

	def _load_model(self):
		if self.model is None:
			from sentence_transformers import SentenceTransformer
			print('Loading model: {} (device={})...'.format(self.model_name, self.device))
			self.model = SentenceTransformer(self.model_name, device=self.device)
			self._dim = self.model.get_sentence_embedding_dimension()
			print('  Embedding dimension: {}'.format(self._dim))

	def encode_documents(self, doc_ids, texts, metadata=None, batch_size=128, show_progress=True):
		self._load_model()

		self.doc_ids = doc_ids
		if metadata:
			for did, meta in zip(doc_ids, metadata):
				self.doc_store[did] = meta

		print('Encoding {} documents...'.format(len(texts)))
		self.embeddings = self.model.encode(
			texts,
			batch_size=batch_size,
			show_progress_bar=show_progress,
			normalize_embeddings=True,
		)
		print('  Embeddings shape: {}'.format(self.embeddings.shape))

	def search(self, query, top_k=10):
		self._load_model()

		if self.embeddings is None or len(self.doc_ids) == 0:
			return []

		q_emb = self.model.encode([query], normalize_embeddings=True)[0]

		scores = self.embeddings @ q_emb

		top_indices = np.argsort(scores)[::-1][:top_k]
		results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]
		return results

	def get_doc(self, doc_id):
		return self.doc_store.get(doc_id, {})

	def save(self, directory):
		os.makedirs(directory, exist_ok=True)
		np.save(os.path.join(directory, 'embeddings.npy'), self.embeddings)
		meta = {
			'model_name': self.model_name,
			'doc_ids': self.doc_ids,
			'doc_store': {str(k): v for k, v in self.doc_store.items()},
		}
		with open(os.path.join(directory, 'meta.json'), 'w') as f:
			json.dump(meta, f)
		print('Semantic index saved to {}/'.format(directory))

	@classmethod
	def load(cls, directory):
		with open(os.path.join(directory, 'meta.json')) as f:
			meta = json.load(f)

		idx = cls(model_name=meta['model_name'])
		idx.doc_ids = meta['doc_ids']
		idx.doc_store = {int(k): v for k, v in meta['doc_store'].items()}
		idx.embeddings = np.load(os.path.join(directory, 'embeddings.npy'))
		return idx


def build_job_semantic_index(jobs_csv, model_name='all-MiniLM-L6-v2', sample_size=None, device=None):
	import pandas as pd

	df = pd.read_csv(jobs_csv)
	if sample_size and sample_size < len(df):
		df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
		print('Subsampled to {} jobs for semantic index'.format(sample_size))

	for col in ['company_name', 'job_category']:
		if col in df.columns:
			df[col] = df[col].fillna('')

	title_col = 'title_clean' if 'title_clean' in df.columns else 'title'
	desc_col = 'description_clean' if 'description_clean' in df.columns else 'description'
	id_col = 'job_id' if 'job_id' in df.columns else None

	doc_ids = df[id_col].astype(int).tolist() if id_col else list(range(len(df)))
	titles = df[title_col].fillna('').astype(str)
	descs = df[desc_col].fillna('').astype(str).str[:500]
	texts = (titles + '. ' + titles + '. ' + descs).tolist()
	metadata = df[['title', 'company_name', 'job_category']].rename(
		columns={'company_name': 'company', 'job_category': 'category'}
	).fillna('').astype(str).to_dict('records')

	idx = SemanticIndex(model_name=model_name, device=device)
	idx.encode_documents(doc_ids, texts, metadata)
	return idx


def build_resume_semantic_index(resumes_csv, model_name='all-MiniLM-L6-v2', device=None):
	import pandas as pd

	df = pd.read_csv(resumes_csv)

	id_col = 'ID' if 'ID' in df.columns else None
	text_col = 'resume_clean' if 'resume_clean' in df.columns else 'Resume_str'

	doc_ids = df[id_col].astype(int).tolist() if id_col else list(range(len(df)))
	texts = df[text_col].fillna('').astype(str).str[:1000].tolist()
	snippets = df[text_col].fillna('').astype(str).str[:600].tolist()
	categories = df['Category'].fillna('').astype(str).tolist() if 'Category' in df.columns else [''] * len(df)
	metadata = [{'category': c, 'text': t} for c, t in zip(categories, snippets)]

	idx = SemanticIndex(model_name=model_name, device=device)
	idx.encode_documents(doc_ids, texts, metadata)
	return idx


if __name__ == '__main__':
	print('=== Semantic Retrieval Demo ===\n')
	print('NOTE: This demo requires sentence-transformers.')
	print('Install with: pip install sentence-transformers\n')

	try:
		idx = SemanticIndex()

		doc_ids = [0, 1, 2, 3, 4]
		texts = [
			'Senior Python Developer. Experienced with Django, Flask, and REST APIs.',
			'Data Scientist. Machine learning models using Python, TensorFlow, scikit-learn.',
			'Frontend React Developer. Modern web apps with React and TypeScript.',
			'ML Engineer. Deploy ML pipelines with PyTorch. NLP and computer vision.',
			'Project Manager IT. Lead agile teams in software delivery.',
		]
		meta = [{'title': t.split('.')[0]} for t in texts]

		idx.encode_documents(doc_ids, texts, meta)

		queries = [
			'python machine learning engineer',
			'javascript web development frontend',
			'PostgreSQL database management',
		]

		for q in queries:
			print("\nQuery: '{}'".format(q))
			results = idx.search(q, top_k=3)
			for rank, (doc_id, score) in enumerate(results, 1):
				m = idx.get_doc(doc_id)
				print('  {}. [{:.3f}] {}'.format(rank, score, m.get('title', doc_id)))

	except ImportError:
		print('sentence-transformers not installed. Skipping demo.')
		print('Run: pip install sentence-transformers')
