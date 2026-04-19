import os
import sys
import urllib.parse

# Ensure the project root is importable so engine/ can be found
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_INDEX_DIR = os.path.join(_ROOT, "data", "indexes")

from flask import Flask, render_template, request

# Indexes are loaded once on first request and reused for all subsequent ones.
_job_retriever = None
_resume_retriever = None
_semantic_available = False
_retrievers_loaded = False
_load_error = None


def _load_retrievers():
    global _job_retriever, _resume_retriever, _semantic_available, _retrievers_loaded, _load_error
    if _retrievers_loaded:
        return

    bm25_jobs_path = os.path.join(_INDEX_DIR, "jobs_bm25f.pkl")
    bm25_resumes_path = os.path.join(_INDEX_DIR, "resumes_bm25f.pkl")

    if not os.path.exists(bm25_jobs_path) or not os.path.exists(bm25_resumes_path):
        _load_error = (
            "Indexes not found. Run: python build.py --step index  "
            "(or python scripts/build_sample_indexes.py for a quick demo build)"
        )
        _retrievers_loaded = True
        return

    from engine.bm25f import BM25FIndex
    job_bm25 = BM25FIndex.load(bm25_jobs_path)
    resume_bm25 = BM25FIndex.load(bm25_resumes_path)

    try:
        from engine.semantic import SemanticIndex
        from engine.hybrid import HybridRetriever
        job_sem = SemanticIndex.load(os.path.join(_INDEX_DIR, "jobs_semantic"))
        resume_sem = SemanticIndex.load(os.path.join(_INDEX_DIR, "resumes_semantic"))
        _job_retriever = HybridRetriever(job_bm25, job_sem)
        _resume_retriever = HybridRetriever(resume_bm25, resume_sem)
        _semantic_available = True
    except Exception:
        # sentence-transformers not installed or semantic indexes missing — BM25F only.
        _job_retriever = job_bm25
        _resume_retriever = resume_bm25

    _retrievers_loaded = True


def _linkedin_search_url(title: str, company: str = "") -> str:
    """Build a LinkedIn job search URL from title and company."""
    query = title
    if company:
        query = f"{title} {company}"
    return "https://www.linkedin.com/jobs/search/?" + urllib.parse.urlencode({"keywords": query})


def _search_jobs(query: str, top_k: int = 10) -> list:
    _load_retrievers()
    if _semantic_available:
        raw = _job_retriever.search(query, top_k=top_k, mode="hybrid")
        results = []
        for _, score, m in raw:
            title = m.get("title", "")
            company = m.get("company", "")
            results.append({
                "title": title,
                "company": company,
                "location": m.get("location", ""),
                "category": m.get("category", ""),
                "score": round(score, 4),
                "url": _linkedin_search_url(title, company),
            })
        return results

    raw = _job_retriever.search(query, top_k=top_k)
    results = []
    for doc_id, score in raw:
        m = _job_retriever.get_doc(doc_id)
        title = m.get("title", "")
        company = m.get("company", "")
        results.append({
            **m,
            "score": round(score, 4),
            "url": _linkedin_search_url(title, company),
        })
    return results


def _search_resumes(query: str, top_k: int = 10) -> list:
    _load_retrievers()
    if _semantic_available:
        raw = _resume_retriever.search(query, top_k=top_k, mode="hybrid")
        return [
            {
                "category": m.get("category", ""),
                "text": m.get("text", ""),
                "score": round(score, 4),
            }
            for _, score, m in raw
        ]

    raw = _resume_retriever.search(query, top_k=top_k)
    return [
        {**_resume_retriever.get_doc(doc_id), "score": round(score, 4)}
        for doc_id, score in raw
    ]


def _get_query_text() -> str:
    """Return query text from the textarea or an uploaded PDF."""
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


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "templates"),
        static_folder=os.path.join(_HERE, "static"),
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/match/jobs")
    def match_jobs():
        _load_retrievers()
        if _load_error:
            return render_template("error.html", message=_load_error), 503

        query = _get_query_text()
        if not query:
            return render_template(
                "results.html",
                title="Jobs matched to resume",
                results=[],
                result_type="jobs",
            )
        results = _search_jobs(query)
        return render_template(
            "results.html",
            title="Jobs matched to resume",
            results=results,
            result_type="jobs",
        )

    @app.post("/match/resumes")
    def match_resumes():
        _load_retrievers()
        if _load_error:
            return render_template("error.html", message=_load_error), 503

        query = _get_query_text()
        if not query:
            return render_template(
                "results.html",
                title="Resumes matched to job",
                results=[],
                result_type="resumes",
            )
        results = _search_resumes(query)
        return render_template(
            "results.html",
            title="Resumes matched to job",
            results=results,
            result_type="resumes",
        )

    return app


app = create_app()
