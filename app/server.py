import os
import re
import sys
import urllib.parse

# Ensure the project root is importable so engine/ and evaluation/ can be found
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_INDEX_DIR = os.path.join(_ROOT, "data", "indexes")

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Index state — loaded once on first request, reused forever.
# ---------------------------------------------------------------------------
_job_retriever    = None
_resume_retriever = None
_semantic_available = False
_retrievers_loaded  = False
_load_error         = None

# Evaluation cache — populated on first /evaluate/run, then served instantly.
_eval_cache = None


def _load_retrievers():
    global _job_retriever, _resume_retriever, _semantic_available, \
           _retrievers_loaded, _load_error

    if _retrievers_loaded:
        return

    bm25_jobs_path    = os.path.join(_INDEX_DIR, "jobs_bm25f.pkl")
    bm25_resumes_path = os.path.join(_INDEX_DIR, "resumes_bm25f.pkl")

    if not os.path.exists(bm25_jobs_path) or not os.path.exists(bm25_resumes_path):
        _load_error = (
            "Indexes not found. Run: python build.py --step index "
            "(or python scripts/build_sample_indexes.py for a quick demo build)"
        )
        _retrievers_loaded = True
        return

    from engine.bm25f import BM25FIndex
    job_bm25    = BM25FIndex.load(bm25_jobs_path)
    resume_bm25 = BM25FIndex.load(bm25_resumes_path)

    # Optional: language model indexes
    job_lm = resume_lm = None
    lm_jobs_path    = os.path.join(_INDEX_DIR, "jobs_lm.pkl")
    lm_resumes_path = os.path.join(_INDEX_DIR, "resumes_lm.pkl")
    if os.path.exists(lm_jobs_path) and os.path.exists(lm_resumes_path):
        try:
            from engine.lm import LanguageModelIndex
            job_lm    = LanguageModelIndex.load(lm_jobs_path)
            resume_lm = LanguageModelIndex.load(lm_resumes_path)
        except Exception as e:
            print(f"[jobmatch] LM index unavailable: {e}", file=sys.stderr)

    # Optional: cluster indexes
    job_clusters = resume_clusters = None
    jobs_cluster_path    = os.path.join(_INDEX_DIR, "jobs_clusters.npz")
    resumes_cluster_path = os.path.join(_INDEX_DIR, "resumes_clusters.npz")
    if os.path.exists(jobs_cluster_path) and os.path.exists(resumes_cluster_path):
        try:
            from engine.cluster import ClusterIndex
            job_clusters    = ClusterIndex.load(jobs_cluster_path)
            resume_clusters = ClusterIndex.load(resumes_cluster_path)
        except Exception as e:
            print(f"[jobmatch] Cluster index unavailable: {e}", file=sys.stderr)

    # Optional: LTR reranker
    ltr_model = None
    ltr_path = os.path.join(_INDEX_DIR, "ltr_model.pkl")
    if os.path.exists(ltr_path):
        try:
            from engine.ltr import LTRReranker
            ltr_model = LTRReranker.load(ltr_path)
        except Exception as e:
            print(f"[jobmatch] LTR model unavailable: {e}", file=sys.stderr)

    try:
        from engine.semantic import SemanticIndex
        from engine.hybrid import HybridRetriever
        job_sem    = SemanticIndex.load(os.path.join(_INDEX_DIR, "jobs_semantic"))
        resume_sem = SemanticIndex.load(os.path.join(_INDEX_DIR, "resumes_semantic"))
        _job_retriever    = HybridRetriever(job_bm25, job_sem,
                                            alpha=0.25,
                                            cluster_index=job_clusters,
                                            lm_index=job_lm)
        _resume_retriever = HybridRetriever(resume_bm25, resume_sem,
                                            alpha=0.25,
                                            cluster_index=resume_clusters,
                                            lm_index=resume_lm)
        _semantic_available = True
    except Exception as e:
        print(f"[jobmatch] Semantic index unavailable, falling back to BM25F: {e}",
              file=sys.stderr)
        _job_retriever    = job_bm25
        _resume_retriever = resume_bm25

    # Attach LTR to retrievers if available
    if ltr_model is not None and _semantic_available:
        _job_retriever._ltr = ltr_model

    _retrievers_loaded = True


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _linkedin_search_url(title: str, company: str = "") -> str:
    query = f"{title} {company}".strip() if company else title
    return "https://www.linkedin.com/jobs/search/?" + urllib.parse.urlencode({"keywords": query})


def _search_jobs(query: str, top_k: int = 10, mode: str = "hybrid", use_prf: bool = False,
                 use_ltr: bool = False) -> list:
    _load_retrievers()
    if _semantic_available:
        if use_prf:
            raw = _job_retriever.search_with_prf(query, top_k=top_k)
        else:
            raw = _job_retriever.search(query, top_k=top_k, mode=mode)

        if use_ltr and getattr(_job_retriever, "_ltr", None) is not None:
            raw = _job_retriever._ltr.rerank(query, raw)

        results = []
        for _, score, m in raw:
            title   = m.get("title", "")
            company = m.get("company", "")
            results.append({
                "title":    title,
                "company":  company,
                "location": m.get("location", ""),
                "category": m.get("category", ""),
                "score":    round(score, 4),
                "cluster":  m.get("cluster"),
                "url":      _linkedin_search_url(title, company),
            })
        return results

    raw = _job_retriever.search(query, top_k=top_k)
    results = []
    for doc_id, score in raw:
        m       = _job_retriever.get_doc(doc_id)
        title   = m.get("title", "")
        company = m.get("company", "")
        results.append({**m, "score": round(score, 4), "url": _linkedin_search_url(title, company)})
    return results


def _search_resumes(query: str, top_k: int = 10, mode: str = "hybrid",
                    use_prf: bool = False) -> list:
    _load_retrievers()
    if _semantic_available:
        if use_prf:
            raw = _resume_retriever.search_with_prf(query, top_k=top_k)
        else:
            raw = _resume_retriever.search(query, top_k=top_k, mode=mode)
        results = []
        for _, score, m in raw:
            text = m.get("text", "")
            title = _extract_resume_title(text)
            results.append({
                "title":    title,
                "category": _infer_category(title, text, m.get("category", "")),
                "text":     text,
                "score":    round(score, 4),
                "cluster":  m.get("cluster"),
            })
        return results

    raw = _resume_retriever.search(query, top_k=top_k)
    results = []
    for doc_id, score in raw:
        m = _resume_retriever.get_doc(doc_id)
        text = m.get("text", "")
        title = _extract_resume_title(text)
        results.append({**m, "title": title, "category": _infer_category(title, text, m.get("category", "")), "score": round(score, 4)})
    return results


def _cluster_counts(results: list) -> dict:
    counts = {}
    for r in results:
        c = r.get("cluster")
        if c is not None:
            counts[c] = counts.get(c, 0) + 1
    return counts


def _cluster_dist_jobs(results: list) -> dict:
    return _cluster_counts(results)


def _cluster_dist_resumes(results: list) -> dict:
    return _cluster_counts(results)


_MAX_QUERY_CHARS = 4000  # ~512 tokens; model hard-caps at 256 anyway

# ---------------------------------------------------------------------------
# Resume title extraction
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r'\b(professional\s+summary|summary|objective|skills|experience|education|'
    r'work\s+history|certifications|references|achievements|awards)\b',
    re.IGNORECASE,
)

# Matches 2–4 Title-Case words (non-greedy: prefers shorter matches so "John Smith
# Software Engineer" yields "John Smith" not the full four-word phrase).
_NAME_RE = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){1,3}?)\b')

# Common job-title and section words that should not be mistaken for a person name
_NAME_SKIP_WORDS = frozenset({
    'Senior', 'Junior', 'Lead', 'Staff', 'Principal', 'Chief', 'Head',
    'Engineer', 'Developer', 'Manager', 'Director', 'Analyst', 'Designer',
    'Consultant', 'Specialist', 'Associate', 'Officer', 'Executive',
    'Administrator', 'Technician', 'Intern', 'Coordinator', 'Scientist',
    'Architect', 'President', 'Vice', 'General', 'Regional', 'Global',
    'Technical', 'Software', 'Systems', 'Network', 'Data', 'Information',
    'Business', 'Sales', 'Marketing', 'Finance', 'Human', 'Resources',
    'Medical', 'Clinical', 'Research', 'Project', 'Product', 'Program',
    'Service', 'Support', 'Quality', 'Operations', 'Security', 'Digital',
    'Infrastructure', 'Strategic', 'Professional', 'Certified', 'Licensed',
    'Machine', 'Learning', 'Intelligence', 'Artificial', 'Cloud', 'Platform',
})


def _cap_at_60(s: str) -> str:
    if len(s) <= 60:
        return s
    cut = s[:60]
    last_space = cut.rfind(' ')
    return cut[:last_space] if last_space > 0 else cut


def _extract_resume_title(text: str) -> str:
    if not text:
        return ""

    # Identify where body text begins (Summary, Experience, Skills, …)
    m = _SECTION_RE.search(text)
    header_end = m.start() if (m and m.start() < 250) else min(len(text), 200)
    header = text[:header_end].strip()

    if not header:
        return ""

    # Look for a person name: 2–4 Title-Case words, no common job-title terms
    for match in _NAME_RE.finditer(header):
        candidate = match.group(1)
        words = candidate.split()
        if not any(w in _NAME_SKIP_WORDS for w in words):
            return _cap_at_60(candidate)

    # Fallback: title-case the header segment (usually an ALL-CAPS job title)
    return _cap_at_60(header.title())


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

# Normalise raw ALL-CAPS category strings from the snehaanbhawal dataset.
_RAW_CATEGORY_NORM = {
    'ENGINEERING':            'Engineering',
    'INFORMATION-TECHNOLOGY': 'Information Technology',
    'HR':                     'Human Resources',
    'SALES':                  'Sales',
    'FINANCE':                'Finance',
    'ACCOUNTANT':             'Finance',
    'BANKING':                'Finance',
    'HEALTHCARE':             'Healthcare',
    'FITNESS':                'Healthcare',
    'DESIGNER':               'Design',
    'DIGITAL-MEDIA':          'Design',
    'ARTS':                   'Design',
    'ADVOCATE':               'Legal',
    'CONSTRUCTION':           'Construction',
    'TEACHER':                'Education',
    'BUSINESS-DEVELOPMENT':   'Sales',
    'CONSULTANT':             'Consulting',
    'CHEF':                   'Other',
    'AVIATION':               'Other',
    'APPAREL':                'Other',
    'AGRICULTURE':            'Other',
    'AUTOMOBILE':             'Engineering',
    'BPO':                    'Other',
    'PUBLIC-RELATIONS':       'Marketing',
}


def _normalize_raw_category(raw: str) -> str:
    """Return a display-friendly category label.

    If raw contains any lowercase char it is already a display label (synthetic
    or pre-normalised) and is returned as-is.  Otherwise it is treated as an
    ALL-CAPS snehaanbhawal label and looked up in _RAW_CATEGORY_NORM.
    """
    if not raw:
        return ""
    stripped = raw.strip()
    if any(c.islower() for c in stripped):
        return stripped
    return _RAW_CATEGORY_NORM.get(stripped, stripped.title().replace('-', ' '))


# More specific rules ordered by specificity; IR/Search and ML/NLP are first
# to avoid being absorbed by the broader Engineering/Data buckets.
_CATEGORY_RULES = [
    ('Information Retrieval',    ['information retrieval', 'search engineer', 'search scientist',
                                   'relevance engineer', 'elasticsearch', 'bm25', 'inverted index',
                                   'faiss', 'ann search', 'learning-to-rank', 'learning to rank',
                                   'whoosh', 'query expansion', 'ndcg', 'lucene', 'solr',
                                   'dense retrieval', 'sparse retrieval', 'semantic search engineer',
                                   'passage retrieval', 'document retrieval engineer']),
    ('ML/NLP Engineering',       ['natural language processing', 'nlp engineer', 'nlp scientist',
                                   'machine learning engineer', 'ml engineer', 'ai engineer',
                                   'deep learning', 'pytorch', 'hugging face', 'transformers',
                                   'bert', 'llm', 'rag pipeline', 'sentence-transformers',
                                   'spacy', 'mlflow', 'vector database', 'fine-tuning',
                                   'language model', 'text classification', 'named entity']),
    ('DevOps/Cloud Engineering', ['devops', 'site reliability', 'sre engineer', 'kubernetes',
                                   'k8s', 'terraform', 'ansible', 'argocd', 'github actions',
                                   'ci/cd', 'prometheus', 'grafana', 'pagerduty', 'helm',
                                   'infrastructure as code', 'platform engineer',
                                   'cloud infrastructure engineer', 'containerization']),
    ('Software Engineering',     ['software developer', 'software engineer', 'full stack',
                                   'fullstack', 'backend developer', 'frontend developer',
                                   'web developer', 'mobile developer', 'api developer']),
    ('Information Technology',   ['information technology', 'it specialist', 'it manager',
                                   'system admin', 'network admin', 'network engineer',
                                   'cybersecurity', 'helpdesk', 'database admin',
                                   'database engineer', 'it support', 'cloud admin']),
    ('Data & Analytics',         ['data scientist', 'data analyst', 'data engineer',
                                   'business intelligence', 'bi analyst', 'analytics engineer']),
    ('Engineering',              ['engineering manager', 'mechanical engineer',
                                   'electrical engineer', 'civil engineer', 'aerospace engineer',
                                   'chemical engineer', 'manufacturing engineer',
                                   'systems engineer', 'engineering tech']),
    ('Management',               ['director', 'vp ', 'vice president', 'chief ', 'cto', 'cio', 'ceo']),
    ('Consulting',               ['consultant', 'senior consultant', 'associate consultant', 'advisory']),
    ('Marketing',                ['marketing manager', 'seo specialist', 'content strategist',
                                   'brand manager', 'digital marketing', 'public relations',
                                   'growth hacker', 'marketing analyst']),
    ('Design',                   ['ux designer', 'ui designer', 'graphic designer',
                                   'creative director', 'visual designer', 'product designer',
                                   'interaction designer']),
    ('Finance',                  ['financial analyst', 'accountant', 'investment banker',
                                   'finance manager', 'accounting manager', 'auditor',
                                   'controller', 'cpa', 'cfa']),
    ('Sales',                    ['sales representative', 'account executive', 'account manager',
                                   'business development', 'sales manager', 'sales engineer']),
    ('Healthcare',               ['registered nurse', 'physician', 'medical doctor',
                                   'clinical coordinator', 'healthcare administrator',
                                   'pharmacist', 'physical therapist', 'occupational therapist']),
    ('Legal',                    ['lawyer', 'attorney', 'legal counsel', 'paralegal', 'legal analyst']),
    ('Human Resources',          ['human resources', 'recruiter', 'talent acquisition',
                                   'hr manager', 'people operations', 'hr business partner']),
    ('Education',                ['teacher', 'professor', 'instructor', 'educator', 'tutor',
                                   'curriculum developer', 'academic advisor']),
]


def _infer_category(title: str, text: str, meta_category: str = "") -> str:
    # Prefer metadata category (normalised) over keyword inference.
    if meta_category:
        normalized = _normalize_raw_category(meta_category)
        if normalized:
            return normalized

    # Keyword inference on title + first 400 chars of resume text.
    haystack = (title + " " + text[:400]).lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in haystack for kw in keywords):
            return category
    return ""


def _get_query_text() -> str:
    text = (request.form.get("text") or "").strip()
    if not text:
        pdf_file = request.files.get("pdf")
        if pdf_file and getattr(pdf_file, "filename", ""):
            text = _extract_pdf_text(pdf_file)
    return text[:_MAX_QUERY_CHARS]


def _extract_pdf_text(pdf_file) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_RELATED_CATEGORY_PAIRS = {
    frozenset({"Engineering", "Information-Technology"}),
    frozenset({"Finance", "Consulting"}),
    frozenset({"Sales", "Marketing"}),
    frozenset({"Healthcare", "Education"}),
    frozenset({"Design", "Marketing"}),
}


def _category_grade(resume_cat: str, job_cat: str) -> int:
    if not job_cat or not resume_cat:
        return 0
    if job_cat == resume_cat:
        return 3
    if frozenset({job_cat, resume_cat}) in _RELATED_CATEGORY_PAIRS:
        return 1
    return 0


def _inline_evaluate(retriever_fn, queries, job_categories, category_map, k_values):
    """Evaluate retriever_fn using inline category-based relevance against loaded job index."""
    import math

    def _p_at_k(ranked, rel_set, k):
        return sum(1 for d in ranked[:k] if d in rel_set) / k if k else 0.0

    def _dcg(ranked, rel_map, k):
        return sum(
            (2 ** rel_map.get(d, 0) - 1) / math.log2(i + 2)
            for i, d in enumerate(ranked[:k])
        )

    def _ndcg(ranked, rel_map, k):
        ideal = sorted(rel_map.values(), reverse=True)
        ideal_dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
        return _dcg(ranked, rel_map, k) / ideal_dcg if ideal_dcg else 0.0

    def _ap(ranked, rel_set):
        if not rel_set:
            return 0.0
        hits, ap = 0, 0.0
        for i, d in enumerate(ranked):
            if d in rel_set:
                hits += 1
                ap += hits / (i + 1)
        return ap / len(rel_set)

    per_query_ap = []
    agg = {f"mean_P@{k}": 0.0 for k in k_values}
    agg.update({f"mean_NDCG@{k}": 0.0 for k in k_values})
    n = 0

    for qid, query_text in queries.items():
        resume_cat = category_map.get(qid, "")
        results = retriever_fn(query_text)
        ranked = [doc_id for doc_id, _ in results]

        rel_map_q = {job_id: _category_grade(resume_cat, job_cat)
                     for job_id, job_cat in job_categories.items()}
        rel_set_q = {job_id for job_id, g in rel_map_q.items() if g >= 2}

        for k in k_values:
            agg[f"mean_P@{k}"] += _p_at_k(ranked, rel_set_q, k)
            agg[f"mean_NDCG@{k}"] += _ndcg(ranked, rel_map_q, k)

        per_query_ap.append(_ap(ranked, rel_set_q))
        n += 1

    if n:
        for key in agg:
            agg[key] /= n

    agg["MAP"] = sum(per_query_ap) / len(per_query_ap) if per_query_ap else 0.0
    agg["num_queries"] = n
    return agg


def _run_evaluation():
    """Evaluate all retrieval modes inline against the loaded job index.

    Ground truth is built on-the-fly from category matching so it is always
    consistent with whatever job set is indexed (sample or full).
    Results are cached in _eval_cache after the first run.
    """
    global _eval_cache
    if _eval_cache is not None:
        return _eval_cache

    _load_retrievers()
    if _load_error:
        raise RuntimeError(_load_error)

    import random
    import json

    resume_idx = _resume_retriever.bm25f if _semantic_available else _resume_retriever
    job_idx    = _job_retriever.bm25f    if _semantic_available else _job_retriever

    # Load category map (resume_category → normalized category)
    cat_map_path = os.path.join(_ROOT, "data", "processed", "category_map.json")
    raw_cat_map = {}
    if os.path.exists(cat_map_path):
        with open(cat_map_path, encoding="utf-8") as f:
            raw_cat_map = json.load(f)

    # Build {qid: normalized_category} for all resumes
    resume_cat_map = {}
    all_resume_ids = list(resume_idx.doc_store.keys())
    random.seed(42)
    sampled_ids = random.sample(all_resume_ids, min(20, len(all_resume_ids)))
    queries = {}
    for qid in sampled_ids:
        meta = resume_idx.get_doc(qid)
        text = meta.get("text", "")
        if not text:
            continue
        raw_cat = meta.get("category", "")
        resume_cat_map[qid] = raw_cat_map.get(raw_cat, raw_cat)
        queries[qid] = text

    if not queries:
        raise RuntimeError(
            "No resume texts found in resume index. "
            "Rebuild indexes: python build.py --step index"
        )

    # Build {job_id: normalized_category} for all jobs in the loaded index
    job_categories = {
        doc_id: job_idx.get_doc(doc_id).get("category", "")
        for doc_id in job_idx.doc_store
    }

    k_values = [5, 10, 20]

    def _bm25f_fn(q):
        return job_idx.search(q, top_k=50)

    def _semantic_fn(q):
        return _job_retriever.semantic.search(q, top_k=50)

    def _hybrid_fn(q):
        raw = _job_retriever.search(q, top_k=50, mode="hybrid")
        return [(doc_id, score) for doc_id, score, _ in raw]

    def _lm_fn(q):
        return _job_retriever.lm_index.search(q, top_k=50)

    modes = {"BM25F": _bm25f_fn}
    if _semantic_available:
        modes["Semantic"] = _semantic_fn
        modes["Hybrid"]   = _hybrid_fn
    if _semantic_available and getattr(_job_retriever, "lm_index", None) is not None:
        modes["LM"] = _lm_fn

    out_modes = {}
    for name, fn in modes.items():
        out_modes[name] = _inline_evaluate(fn, queries, job_categories, resume_cat_map, k_values)

    _eval_cache = {
        "modes":       out_modes,
        "num_queries": len(queries),
        "k_values":    k_values,
    }
    return _eval_cache


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=None,
    )

    # ── Health check ────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {"status": "ok"}, 200

    # ── Home ────────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        return render_template("index.html")

    # ── Match: resume → jobs ─────────────────────────────────────────────────
    @app.post("/match/jobs")
    def match_jobs():
        _load_retrievers()
        if _load_error:
            return render_template("error.html", message=_load_error), 503

        query   = _get_query_text()
        mode    = request.form.get("mode", "hybrid")
        use_prf = bool(request.form.get("use_prf"))

        if not query:
            return render_template("results.html", title="Jobs matched to resume",
                                   results=[], result_type="jobs",
                                   mode_used=mode, prf_used=False, cluster_dist={})

        results = _search_jobs(query, mode=mode, use_prf=use_prf)
        cluster_dist = _cluster_dist_jobs(results)
        return render_template("results.html", title="Jobs matched to resume",
                               results=results, result_type="jobs",
                               mode_used=mode, prf_used=use_prf,
                               cluster_dist=cluster_dist)

    # ── Match: job → resumes ─────────────────────────────────────────────────
    @app.post("/match/resumes")
    def match_resumes():
        _load_retrievers()
        if _load_error:
            return render_template("error.html", message=_load_error), 503

        query   = _get_query_text()
        mode    = request.form.get("mode", "hybrid")
        use_prf = bool(request.form.get("use_prf"))

        if not query:
            return render_template("results.html", title="Resumes matched to job",
                                   results=[], result_type="resumes",
                                   mode_used=mode, prf_used=False, cluster_dist={})

        results = _search_resumes(query, mode=mode, use_prf=use_prf)
        cluster_dist = _cluster_dist_resumes(results)
        return render_template("results.html", title="Resumes matched to job",
                               results=results, result_type="resumes",
                               mode_used=mode, prf_used=use_prf,
                               cluster_dist=cluster_dist)

    # ── Evaluate: page ───────────────────────────────────────────────────────
    @app.get("/evaluate")
    def evaluate_page():
        return render_template("evaluate.html", results=_eval_cache)

    # ── Evaluate: run (POST, returns JSON) ───────────────────────────────────
    @app.post("/evaluate/run")
    def evaluate_run():
        try:
            results = _run_evaluation()
            return jsonify({"ok": True, "results": results})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


app = create_app()
_load_retrievers()
