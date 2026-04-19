import math
import re
import json
import os
import pickle


STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())


def tokenize(text):
	if not isinstance(text, str):
		return []
	tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text.lower())
	return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_simple(token):
	for suffix in ['ing', 'tion', 'ment', 'ness', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ly', 'er', 'ed', 'es', 's']:
		if token.endswith(suffix) and len(token) - len(suffix) >= 3:
			return token[:-len(suffix)]
	return token


class BM25FIndex:
	def __init__(self, field_configs, k1=1.2, use_stemming=False):
		self.field_configs = field_configs
		self.fields = list(field_configs.keys())
		self.k1 = k1
		self.use_stemming = use_stemming

		# term -> field -> doc_id -> tf
		self.index = {}
		# doc_id -> metadata dict
		self.doc_store = {}
		# doc_id -> field -> token length
		self.field_lengths = {}
		self.N = 0
		# field -> average document length
		self.avgdl = {}
		# term -> document frequency
		self.df = {}
		self._built = False

	def _process_tokens(self, text):
		tokens = tokenize(text)
		if self.use_stemming:
			tokens = [stem_simple(t) for t in tokens]
		return tokens

	def add_document(self, doc_id, fields, metadata=None):
		self.doc_store[doc_id] = metadata or {}
		self.field_lengths[doc_id] = {}
		terms_seen = set()

		for field in self.fields:
			text = fields.get(field, '')
			tokens = self._process_tokens(text)
			self.field_lengths[doc_id][field] = len(tokens)

			tf = {}
			for t in tokens:
				tf[t] = tf.get(t, 0) + 1

			for term, count in tf.items():
				if term not in self.index:
					self.index[term] = {}
				if field not in self.index[term]:
					self.index[term][field] = {}
				self.index[term][field][doc_id] = count
				terms_seen.add(term)

		for term in terms_seen:
			self.df[term] = self.df.get(term, 0) + 1

		self.N += 1
		self._built = False

	def build(self):
		for field in self.fields:
			lengths = [self.field_lengths[doc_id].get(field, 0) for doc_id in self.doc_store]
			if lengths:
				self.avgdl[field] = sum(lengths) / len(lengths)
			else:
				self.avgdl[field] = 1.0
		self._built = True
		print('BM25F index built: {} docs, {} unique terms'.format(self.N, len(self.index)))

	def _idf(self, term):
		df = self.df.get(term, 0)
		if df == 0:
			return 0.0
		return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

	def _tf_hat(self, term, doc_id):
		tf_hat = 0.0
		for field in self.fields:
			cfg = self.field_configs[field]
			w_f = cfg['weight']
			b_f = cfg['b']
			tf_raw = self.index.get(term, {}).get(field, {}).get(doc_id, 0)
			if tf_raw == 0:
				continue
			dl = self.field_lengths[doc_id].get(field, 0)
			avgdl = self.avgdl.get(field, 1.0)
			norm = 1.0 + b_f * (dl / avgdl - 1.0)
			tf_hat += (w_f * tf_raw) / norm
		return tf_hat

	def score(self, query, doc_id):
		if not self._built:
			raise RuntimeError('Call build() before scoring')
		query_tokens = self._process_tokens(query)
		total = 0.0
		for term in query_tokens:
			idf = self._idf(term)
			tf_h = self._tf_hat(term, doc_id)
			total += idf * (tf_h / (self.k1 + tf_h))
		return total

	def search(self, query, top_k=10, max_query_terms=64):
		if not self._built:
			raise RuntimeError('Call build() before searching')

		unique_terms = list(set(self._process_tokens(query)))
		if not unique_terms:
			return []

		if len(unique_terms) > max_query_terms:
			unique_terms.sort(key=lambda t: self._idf(t), reverse=True)
			unique_terms = unique_terms[:max_query_terms]

		k1 = self.k1
		scores = {}

		for term in unique_terms:
			if term not in self.index:
				continue
			idf = self._idf(term)
			if idf == 0:
				continue

			tf_hat_map = {}
			for field in self.fields:
				field_postings = self.index[term].get(field, {})
				if not field_postings:
					continue
				cfg = self.field_configs[field]
				w_f = cfg['weight']
				b_f = cfg['b']
				avgdl = self.avgdl.get(field, 1.0)
				for doc_id, tf_raw in field_postings.items():
					dl = self.field_lengths[doc_id].get(field, 0)
					norm = 1.0 + b_f * (dl / avgdl - 1.0)
					tf_hat_map[doc_id] = tf_hat_map.get(doc_id, 0.0) + (w_f * tf_raw) / norm

			for doc_id, tf_hat in tf_hat_map.items():
				scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf_hat / (k1 + tf_hat))

		if not scores:
			return []

		ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
		return ranked[:top_k]

	def get_doc(self, doc_id):
		return self.doc_store.get(doc_id, {})

	def save(self, path):
		pkl_path = path if path.endswith('.pkl') else path.replace('.json', '') + '.pkl'
		with open(pkl_path, 'wb') as f:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		print('Index saved to {} ({:.1f} MB)'.format(pkl_path, os.path.getsize(pkl_path) / 1e6))

	@classmethod
	def load(cls, path):
		if path.endswith('.pkl'):
			pkl_path = path
		else:
			pkl_path = path.replace('.json', '') + '.pkl'
		if os.path.exists(pkl_path):
			with open(pkl_path, 'rb') as f:
				idx = pickle.load(f)
			return idx

		with open(path) as f:
			data = json.load(f)

		idx = cls(
			field_configs=data['field_configs'],
			k1=data['k1'],
			use_stemming=data['use_stemming'],
		)
		idx.N = data['N']
		idx.avgdl = data['avgdl']
		idx.df = data['df']
		idx.doc_store = {int(k): v for k, v in data['doc_store'].items()}
		idx.field_lengths = {int(k): v for k, v in data['field_lengths'].items()}

		for term, fields in data['index'].items():
			for field, docs in fields.items():
				if term not in idx.index:
					idx.index[term] = {}
				if field not in idx.index[term]:
					idx.index[term][field] = {}
				for doc_id_str, tf in docs.items():
					idx.index[term][field][int(doc_id_str)] = tf

		idx._built = True
		return idx


def build_job_index(jobs_csv, field_configs=None):
	import pandas as pd

	if field_configs is None:
		field_configs = {
			'title':       {'weight': 3.0, 'b': 0.3},
			'description': {'weight': 1.0, 'b': 0.75},
		}

	idx = BM25FIndex(field_configs=field_configs, k1=1.2)
	df = pd.read_csv(jobs_csv)
	for col in ['company_name', 'location', 'job_category']:
		if col in df.columns:
			df[col] = df[col].fillna('')
	print('Building job index from {} postings...'.format(len(df)))

	has_title_clean = 'title_clean' in df.columns
	has_desc_clean = 'description_clean' in df.columns
	has_job_id = 'job_id' in df.columns
	has_company = 'company_name' in df.columns
	has_location = 'location' in df.columns
	has_category = 'job_category' in df.columns

	for i, row in enumerate(df.itertuples(index=False)):
		doc_id = int(getattr(row, 'job_id') if has_job_id else i)
		fields = {
			'title': str(getattr(row, 'title_clean') if has_title_clean else getattr(row, 'title', '')),
			'description': str(getattr(row, 'description_clean') if has_desc_clean else getattr(row, 'description', '')),
		}
		metadata = {
			'title': str(getattr(row, 'title', '')),
			'company': str(getattr(row, 'company_name') if has_company else ''),
			'location': str(getattr(row, 'location') if has_location else ''),
			'category': str(getattr(row, 'job_category') if has_category else ''),
		}
		idx.add_document(doc_id, fields, metadata)
		if (i + 1) % 25000 == 0:
			print('  Indexed {:,} / {:,}'.format(i + 1, len(df)))

	idx.build()
	return idx


def build_resume_index(resumes_csv, field_configs=None):
	import pandas as pd

	if field_configs is None:
		field_configs = {
			'resume_text': {'weight': 1.0, 'b': 0.75},
		}

	idx = BM25FIndex(field_configs=field_configs, k1=1.2)
	df = pd.read_csv(resumes_csv)
	print('Building resume index from {} resumes...'.format(len(df)))

	has_id = 'ID' in df.columns
	has_resume_clean = 'resume_clean' in df.columns

	for i, row in enumerate(df.itertuples(index=False)):
		doc_id = int(getattr(row, 'ID') if has_id else i)
		full_text = str(getattr(row, 'resume_clean') if has_resume_clean else getattr(row, 'Resume_str', ''))
		fields = {
			'resume_text': full_text,
		}
		metadata = {
			'category': str(getattr(row, 'Category', '')),
			'text': full_text[:600],
		}
		idx.add_document(doc_id, fields, metadata)

	idx.build()
	return idx


if __name__ == '__main__':
	print('=== BM25F Engine Demo ===\n')

	config = {
		'title':       {'weight': 3.0, 'b': 0.3},
		'description': {'weight': 1.0, 'b': 0.75},
	}
	idx = BM25FIndex(field_configs=config, k1=1.2)

	jobs = [
		{
			'title': 'Senior Python Developer',
			'description': 'We are looking for an experienced Python developer with expertise in Django, Flask, and REST APIs. Machine learning experience is a plus.',
		},
		{
			'title': 'Data Scientist',
			'description': 'Join our analytics team to build machine learning models using Python, TensorFlow, and scikit-learn. Strong statistics background required.',
		},
		{
			'title': 'Frontend React Developer',
			'description': 'Build modern web applications with React, TypeScript, and CSS. Experience with responsive design and accessibility standards.',
		},
		{
			'title': 'Machine Learning Engineer',
			'description': 'Design and deploy ML pipelines using Python, PyTorch, and cloud services. NLP and computer vision experience preferred.',
		},
		{
			'title': 'Project Manager - IT',
			'description': 'Lead cross-functional teams in delivering software projects on time. Agile and Scrum certification preferred. Technical background in software development.',
		},
	]

	for i, job in enumerate(jobs):
		idx.add_document(i, job, {'title': job['title']})
	idx.build()

	queries = [
		'python machine learning engineer',
		'react frontend web developer',
		'project management agile',
	]

	for q in queries:
		print("\nQuery: '{}'".format(q))
		results = idx.search(q, top_k=3)
		for rank, (doc_id, score) in enumerate(results, 1):
			meta = idx.get_doc(doc_id)
			print('  {}. [{:.3f}] {}'.format(rank, score, meta['title']))
