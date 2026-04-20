import math
import pickle

from engine.bm25f import tokenize


class LanguageModelIndex:
	def __init__(self, mu=2000):
		self.mu = mu
		self.doc_tfs = {}
		self.doc_lengths = {}
		self.cf = {}
		self.total_tokens = 0
		self.doc_store = {}
		self.doc_ids = []

	def add_document(self, doc_id, text, metadata=None):
		tokens = tokenize(text)
		tf = {}
		for t in tokens:
			tf[t] = tf.get(t, 0) + 1
			self.cf[t] = self.cf.get(t, 0) + 1
		self.doc_tfs[doc_id] = tf
		self.doc_lengths[doc_id] = len(tokens)
		self.total_tokens += len(tokens)
		if metadata:
			self.doc_store[doc_id] = metadata
		self.doc_ids.append(doc_id)

	def build(self):
		pass

	def score(self, query_tokens, doc_id):
		tf = self.doc_tfs.get(doc_id, {})
		dl = self.doc_lengths.get(doc_id, 0)
		mu = self.mu
		total = self.total_tokens or 1
		log_score = 0.0
		for t in query_tokens:
			tf_t = tf.get(t, 0)
			cf_t = self.cf.get(t, 0)
			p_collection = cf_t / total
			p_smooth = (tf_t + mu * p_collection) / (dl + mu)
			if p_smooth > 0:
				log_score += math.log(p_smooth)
		return log_score

	def search(self, query, top_k=10):
		tokens = tokenize(query)
		if not tokens:
			return []
		scored = [(doc_id, self.score(tokens, doc_id)) for doc_id in self.doc_ids]
		scored.sort(key=lambda x: x[1], reverse=True)
		return scored[:top_k]

	def get_doc(self, doc_id):
		return self.doc_store.get(doc_id, {})

	def save(self, path):
		with open(path, 'wb') as f:
			pickle.dump(self.__dict__, f, protocol=4)
		print('LanguageModelIndex saved to {}'.format(path))

	@classmethod
	def load(cls, path):
		idx = cls.__new__(cls)
		with open(path, 'rb') as f:
			idx.__dict__.update(pickle.load(f))
		return idx

	@property
	def N(self):
		return len(self.doc_ids)


def build_job_lm_index(jobs_csv, mu=2000):
	import pandas as pd

	df = pd.read_csv(jobs_csv)
	title_col = 'title_clean' if 'title_clean' in df.columns else 'title'
	desc_col = 'description_clean' if 'description_clean' in df.columns else 'description'
	id_col = 'job_id' if 'job_id' in df.columns else None

	idx = LanguageModelIndex(mu=mu)
	for i, row in df.iterrows():
		doc_id = int(row[id_col]) if id_col else i
		text = '{} {} {}'.format(
			row.get(title_col, ''), row.get(title_col, ''), row.get(desc_col, '')
		)
		meta = {
			'title': row.get('title', ''),
			'company': row.get('company_name', ''),
			'category': row.get('job_category', ''),
		}
		idx.add_document(doc_id, text, meta)

	idx.build()
	print('LM index built: {:,} documents, {:,} unique terms'.format(idx.N, len(idx.cf)))
	return idx


def build_resume_lm_index(resumes_csv, mu=2000):
	import pandas as pd

	df = pd.read_csv(resumes_csv)
	id_col = 'ID' if 'ID' in df.columns else None
	text_col = 'resume_clean' if 'resume_clean' in df.columns else 'Resume_str'

	idx = LanguageModelIndex(mu=mu)
	for i, row in df.iterrows():
		doc_id = int(row[id_col]) if id_col else i
		text = str(row.get(text_col, ''))[:2000]
		meta = {'category': row.get('Category', ''), 'text': text[:600]}
		idx.add_document(doc_id, text, meta)

	idx.build()
	print('LM resume index built: {:,} documents'.format(idx.N))
	return idx


if __name__ == '__main__':
	print('=== Language Model Retrieval Demo ===\n')

	idx = LanguageModelIndex(mu=2000)
	docs = [
		(0, 'Senior Python Developer Django Flask REST API backend engineering',
		 {'title': 'Senior Python Developer'}),
		(1, 'Data Scientist machine learning Python TensorFlow scikit-learn statistics',
		 {'title': 'Data Scientist'}),
		(2, 'Frontend React developer TypeScript CSS responsive design',
		 {'title': 'Frontend React Developer'}),
		(3, 'ML Engineer deploy machine learning pipelines PyTorch NLP computer vision',
		 {'title': 'ML Engineer'}),
		(4, 'IT Project Manager agile scrum software delivery team lead',
		 {'title': 'IT Project Manager'}),
	]
	for doc_id, text, meta in docs:
		idx.add_document(doc_id, text, meta)
	idx.build()

	queries = ['python machine learning', 'react frontend javascript', 'postgresql database']
	for q in queries:
		print("Query: '{}'".format(q))
		for rank, (doc_id, score) in enumerate(idx.search(q, top_k=3), 1):
			print('  {}. [{:.3f}] {}'.format(rank, score, idx.get_doc(doc_id).get('title', '')))
		print()
