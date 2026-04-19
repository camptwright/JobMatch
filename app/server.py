import os
import sys
import urllib.parse

# Ensure the project root is importable so engine/ and evaluation/ can be found
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_INDEX_DIR = os.path.join(_ROOT, "data", "indexes")
_GT_PATH   = os.path.join(_ROOT, "evaluation", "ground_truth.csv")

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

    try:
        from engine.semantic import SemanticIndex
        from engine.hybrid import HybridRetriever
        job_sem    = SemanticIndex.load(os.path.join(_INDEX_DIR, "jobs_semantic"))
        resume_sem = SemanticIndex.load(os.path.join(_INDEX_DIR, "resumes_semantic"))
        _job_retriever    = HybridRetriever(job_bm25, job_sem)
        _resume_retriever = HybridRetriever(resume_bm25, resume_sem)
        _semantic_available = True
    except Exception as e:
        print(f"[jobmatch] Semantic index unavailable, falling back to BM25F: {e}",
              file=sys.stderr)
        _job_retriever    = job_bm25
        _resume_retriever = resume_bm25

    _retrievers_loaded = True


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _linkedin_search_url(title: str, company: str = "") -> str:
    query = f"{title} {company}".strip() if company else title
    return "https://www.linkedin.com/jobs/search/?" + urllib.parse.urlencode({"keywords": query})


def _search_jobs(query: str, top_k: int = 10) -> list:
    _load_retrievers()
    if _semantic_available:
        raw = _job_retriever.search(query, top_k=top_k, mode="hybrid")
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


def _search_resumes(query: str, top_k: int = 10) -> list:
    _load_retrievers()
    if _semantic_available:
        raw = _resume_retriever.search(query, top_k=top_k, mode="hybrid")
        return [
            {"category": m.get("category", ""), "text": m.get("text", ""), "score": round(score, 4)}
            for _, score, m in raw
        ]

    raw = _resume_retriever.search(query, top_k=top_k)
    return [{**_resume_retriever.get_doc(doc_id), "score": round(score, 4)} for doc_id, score in raw]


def _get_query_text() -> str:
    text = (request.form.get("text") or "").strip()
    if not text:
        pdf_file = request.files.get("pdf")
        if pdf_file and getattr(pdf_file, "filename", ""):
            text = _extract_pdf_text(pdf_file)
    return text


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

def _run_evaluation():
    """Run three-way BM25F / Semantic / Hybrid evaluation against ground_truth.csv.

    Query texts are pulled from the resume BM25F index metadata (stored there at
    index-build time), so no processed CSVs are needed at runtime.

    Results are cached in _eval_cache after the first run.
    """
    global _eval_cache
    if _eval_cache is not None:
        return _eval_cache

    _load_retrievers()
    if _load_error:
        raise RuntimeError(_load_error)

    if not os.path.exists(_GT_PATH):
        raise RuntimeError(
            "Ground truth file not found. "
            "Run: python evaluation/generate_ground_truth.py --api category"
        )

    from evaluation.evaluate import load_ground_truth, evaluate_retrieval

    rel_map, _ = load_ground_truth(_GT_PATH)

    # Pull resume text from the BM25F resume index (stored as 'text' metadata).
    resume_idx = _resume_retriever.bm25f if _semantic_available else _resume_retriever
    queries = {qid: resume_idx.get_doc(qid).get("text", "") for qid in rel_map}
    queries = {qid: text for qid, text in queries.items() if text}

    if not queries:
        raise RuntimeError(
            "No query texts found in resume index. "
            "Rebuild indexes: python build.py --step index"
        )

    k_values = [5, 10, 20]

    def _bm25f_fn(q):
        idx = _job_retriever.bm25f if _semantic_available else _job_retriever
        return idx.search(q, top_k=50)

    def _semantic_fn(q):
        return _job_retriever.semantic.search(q, top_k=50)

    def _hybrid_fn(q):
        return _job_retriever.search(q, top_k=50, mode="hybrid", return_metadata=False)

    modes = {"BM25F": _bm25f_fn}
    if _semantic_available:
        modes["Semantic"] = _semantic_fn
        modes["Hybrid"]   = _hybrid_fn

    out_modes = {}
    for name, fn in modes.items():
        res = evaluate_retrieval(_GT_PATH, fn, queries, k_values=k_values)
        out_modes[name] = res["aggregate"]

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

        query = _get_query_text()
        if not query:
            return render_template("results.html", title="Jobs matched to resume",
                                   results=[], result_type="jobs")

        return render_template("results.html", title="Jobs matched to resume",
                               results=_search_jobs(query), result_type="jobs")

    # ── Match: job → resumes ─────────────────────────────────────────────────
    @app.post("/match/resumes")
    def match_resumes():
        _load_retrievers()
        if _load_error:
            return render_template("error.html", message=_load_error), 503

        query = _get_query_text()
        if not query:
            return render_template("results.html", title="Resumes matched to job",
                                   results=[], result_type="resumes")

        return render_template("results.html", title="Resumes matched to job",
                               results=_search_resumes(query), result_type="resumes")

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
