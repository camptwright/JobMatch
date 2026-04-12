import os
from flask import Flask, render_template, request


def create_app() -> Flask:
	app = Flask(
		__name__,
		template_folder=os.path.join(os.path.dirname(__file__), "templates"),
		static_folder=os.path.join(os.path.dirname(__file__), "static"),
	)

	@app.get("/")
	def index():
		return render_template("index.html")

	@app.post("/match/jobs")
	def match_jobs_placeholder():
		return render_template(
			"results.html",
			title="Jobs matched to resume",
			details="Results shown here",
			payload=_summarize_submission(),
		)

	@app.post("/match/resumes")
	def match_resumes_placeholder():
		return render_template(
			"results.html",
			title="Resumes matched to job",
			details="Results shown here",
			payload=_summarize_submission(),
		)

	return app


def _summarize_submission() -> dict:
	text = (request.form.get("text") or "").strip()
	file = request.files.get("pdf")
	return {
		"text_chars": len(text),
		"has_pdf": bool(file and getattr(file, "filename", ""))
	}


app = create_app()
