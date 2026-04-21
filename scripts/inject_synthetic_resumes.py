"""
Inject 30 synthetic resumes into data/processed/resumes_clean.csv covering
three underrepresented categories:
  - ML/NLP Engineering       (10 resumes)
  - DevOps/Cloud Engineering (10 resumes)
  - Information Retrieval    (10 resumes)

Then rebuilds resume indexes (BM25F, Semantic, LM, Clusters) without
touching job indexes.

Usage:
    python scripts/inject_synthetic_resumes.py [--dry-run]
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'indexes')
RESUMES_CSV = os.path.join(PROC_DIR, 'resumes_clean.csv')

# ---------------------------------------------------------------------------
# Synthetic resume corpus
# Each entry: {"category": str, "text": str}
# Varied format (bullets / prose), seniority, industry, and skill emphasis
# so the semantic encoder does not collapse all 10-per-category into one cluster.
# ---------------------------------------------------------------------------

_SYNTHETIC_RESUMES = [

    # ── ML / NLP Engineering ────────────────────────────────────────────────

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Alex Chen  NLP Engineer  San Francisco CA\n"
            "Four years of experience building production natural language processing systems. "
            "At Veritas AI designed and shipped a BERT-based document classification pipeline "
            "processing 20 million daily requests. Implemented named entity recognition using "
            "spaCy and fine-tuned transformer models on domain-specific corpora with Hugging "
            "Face Transformers. Previously at DataStream built text preprocessing workflows "
            "and feature engineering for gradient-boosted classifiers. Strong background in "
            "PyTorch, MLflow experiment tracking, and Docker containerisation. Comfortable "
            "building RAG pipelines that combine FAISS dense retrieval with generative models. "
            "Deployed production models on AWS using Lambda and ECS. "
            "B.S. Computer Science UC San Diego 2019."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Priya Sharma  Senior Machine Learning Engineer\n"
            "Skills: PyTorch, Hugging Face Transformers, RLHF, LLM fine-tuning, sentence-"
            "transformers, FAISS, vector databases, MLflow, Kubernetes, Python, AWS SageMaker\n"
            "Experience:\n"
            "Anthropic-adjacent startup (2021-present): Led fine-tuning of large language "
            "models using RLHF and direct preference optimisation. Built retrieval-augmented "
            "generation (RAG) system over a 10M-document knowledge base using FAISS and "
            "sentence-transformers. Reduced hallucination rate by 38 pct on internal benchmarks.\n"
            "MindBridge Analytics (2018-2021): Trained multi-task NLP models for intent "
            "classification and slot filling in a conversational AI product. "
            "Managed model versioning with MLflow and deployment on SageMaker.\n"
            "M.S. Machine Learning Carnegie Mellon 2018. B.Tech IIT Delhi 2016."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Marcus Johnson  Research Scientist Machine Learning\n"
            "I study efficient transformer architectures and multi-lingual representation "
            "learning. My doctoral work at Stanford focused on parameter-efficient fine-tuning "
            "techniques (LoRA, adapter layers) applied to low-resource NLP tasks. Published "
            "four papers at ACL and EMNLP on cross-lingual transfer and few-shot text "
            "classification. Expert in PyTorch, Hugging Face Transformers, and the full "
            "evaluation pipeline from annotation through benchmark comparison. Industrial "
            "internship at Google Brain working on multilingual BERT pre-training. "
            "Comfortable with distributed training using DeepSpeed and FSDP across GPU "
            "clusters. Seeking a research engineering role focused on language model pre-"
            "training or fine-tuning at scale.\n"
            "Ph.D. Computer Science Stanford University 2023. B.S. Mathematics MIT 2017."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Sofia Rodriguez  Computational Linguist / Biomedical NLP Scientist\n"
            "Specialised in clinical and biomedical text mining using transformer-based models. "
            "At MedScript AI extracted structured information (diagnoses, medications, procedures) "
            "from unstructured EHR notes using a fine-tuned BioBERT pipeline built with spaCy "
            "and Hugging Face. Developed custom tokenisers for medical abbreviations and "
            "implemented coreference resolution for patient records. Previously at Linguasoft "
            "built multilingual text classification systems serving 12 languages. Comfortable "
            "with scikit-learn, pandas, SQL, and Python data pipelines. Familiar with HIPAA "
            "compliance requirements for healthcare AI systems.\n"
            "Skills: spaCy, BioBERT, PyTorch, Transformers, text mining, NER, relation "
            "extraction, clinical NLP, AWS, Docker\n"
            "M.S. Computational Linguistics University of Washington 2020."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "James Park  Staff ML Platform Engineer\n"
            "I build the infrastructure that lets data scientists ship models to production "
            "faster. At Stripe I designed the internal ML platform: feature store (Feast), "
            "experiment tracking (MLflow), model registry, and automated deployment to "
            "Kubernetes via Argo Workflows. Reduced median time-to-production from 6 weeks "
            "to 4 days. Previously at Lyft maintained the Kubeflow-based training pipeline "
            "for 200+ active models including NLP-based support ticket classifiers and "
            "ride-demand forecasting. Deep expertise in Python, Kubernetes, Docker, Spark, "
            "and Airflow. Familiar with sentence-transformers for embedding-based retrieval "
            "and Triton Inference Server for high-throughput model serving.\n"
            "B.S. Electrical Engineering and Computer Science UC Berkeley 2015."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Yuki Tanaka  AI Engineer\n"
            "Startup generalist working on LLM-powered products from prototype to production.\n"
            "Recent work:\n"
            "- Fine-tuned Llama 3 on proprietary customer support corpus using QLoRA; "
            "achieved 91 pct accuracy on held-out tickets, beating GPT-4 baseline.\n"
            "- Built end-to-end RAG pipeline: PDF ingestion, chunking, embedding with "
            "all-MiniLM-L6-v2 via sentence-transformers, storage in Qdrant vector database, "
            "and retrieval-augmented response generation.\n"
            "- Implemented RLHF preference data collection pipeline (human annotators "
            "comparing model outputs) and trained reward model with PyTorch.\n"
            "- Deployed conversational AI with FastAPI + LangChain orchestration on AWS.\n"
            "Stack: Python, PyTorch, Hugging Face Transformers, LangChain, sentence-"
            "transformers, Qdrant, FastAPI, AWS, Docker\n"
            "B.S. Computer Science Keio University 2020."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Nadia Williams  Senior Data Scientist  Natural Language Processing\n"
            "Seven years building recommendation and personalisation systems at scale. "
            "At Zalando led the semantic product-search team: trained two-tower embedding "
            "models (query and product encoders) using PyTorch, indexed 40M product "
            "embeddings in FAISS, and A/B tested against BM25 baseline with a 15 pct "
            "click-through improvement. Previously at Spotify built playlist continuation "
            "models using word2vec-style track embeddings and trained collaborative "
            "filtering with implicit feedback. Strong background in statistics, A/B testing "
            "methodology, and SQL-based data analysis alongside deep learning work.\n"
            "Tools: Python, PyTorch, FAISS, sentence-transformers, scikit-learn, Spark, "
            "Airflow, Looker, Tableau\n"
            "M.S. Statistics Columbia University 2017."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Ethan Brooks  Machine Learning Engineer  Semantic Search\n"
            "Speciality: dense retrieval, bi-encoder models, and vector similarity search.\n"
            "At Pinterest built the visual-semantic search backend: fine-tuned CLIP and "
            "sentence-transformers on paired (image, query) data, hosted 2B-vector FAISS "
            "index on GPU cluster with sub-20 ms p99 latency. Implemented hybrid re-ranking "
            "pipeline blending sparse BM25 scores with dense cosine similarities. At "
            "Amazon Search trained bi-encoder recall model using hard-negative mining and "
            "in-batch negatives; improved Recall@100 by 22 pct over BM25 baseline.\n"
            "Skills: Python, PyTorch, sentence-transformers, FAISS, Elasticsearch, "
            "transformers, AWS, Kubernetes, MLflow, Triton\n"
            "B.S. Computer Science Cornell University 2018."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "Leila Hassan  NLP Software Engineer\n"
            "I build robust, production-grade NLP APIs that handle millions of requests daily. "
            "At Typeform designed a real-time text-analysis microservice using spaCy and "
            "custom transformer components, serving sentiment, intent, and language detection "
            "through a FastAPI REST interface with Redis caching. Maintained 99.95 pct "
            "uptime and P95 latency under 50 ms. Previously at Rosetta Stone built language-"
            "identification and grammar-checking pipelines using rule-based and neural "
            "hybrid approaches. Comfortable with async Python, containerisation with Docker, "
            "CI/CD with GitHub Actions, and model monitoring with Prometheus and Grafana.\n"
            "Stack: Python, spaCy, Transformers, FastAPI, Redis, Docker, Kubernetes, "
            "GitHub Actions, Prometheus\n"
            "M.S. Computer Science University of Edinburgh 2019."
        ),
    },

    {
        "category": "ML/NLP Engineering",
        "text": (
            "David Kim  Machine Learning Engineer  Financial Services\n"
            "Fraud Detection  Anomaly Detection  Ensemble Models\n"
            "At Capital One built real-time transaction fraud classifier combining gradient "
            "boosting (XGBoost) with a feed-forward neural network; reduced false-positive "
            "rate by 31 pct while maintaining 99.2 pct recall on fraudulent transactions. "
            "Implemented SHAP-based model interpretability dashboard for compliance review. "
            "Text-based fraud signals extracted using TF-IDF and fine-tuned DistilBERT on "
            "merchant description fields. At JPMorgan developed credit-risk scoring model "
            "integrating tabular features with document embeddings from loan application text. "
            "Experience with MLflow, SageMaker, and Airflow-based batch scoring pipelines.\n"
            "B.S. Statistics and Computer Science University of Michigan 2017."
        ),
    },

    # ── DevOps / Cloud Engineering ──────────────────────────────────────────

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Chris Murphy  DevOps Engineer  AWS  Terraform  CI/CD\n"
            "Two years of experience automating cloud infrastructure at a SaaS startup. "
            "Provisioned and maintained AWS environment (EC2, RDS, S3, Lambda, ECS) using "
            "Terraform for all infrastructure as code. Built GitHub Actions CI/CD pipelines "
            "reducing deployment time from 45 minutes to 8 minutes. Containerised legacy "
            "monolith into Docker microservices and deployed on ECS Fargate. Configured "
            "CloudWatch alarms and PagerDuty escalation policies for on-call coverage. "
            "Wrote Ansible playbooks for AMI baking and configuration drift remediation. "
            "Comfortable with Bash scripting, Python automation scripts, and Linux "
            "administration on Ubuntu and Amazon Linux.\n"
            "Certifications: AWS Solutions Architect Associate, HashiCorp Terraform Associate\n"
            "B.S. Information Systems Northeastern University 2022."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Rachel Torres  Senior Site Reliability Engineer\n"
            "Seven years building reliable distributed systems. At Shopify I own the "
            "SLO program for the checkout critical path: define SLIs, set error budgets, "
            "and lead monthly reliability reviews with engineering leadership. Migrated "
            "core services from bare-metal to Kubernetes (GKE); reduced infrastructure "
            "cost 40 pct and improved deployment frequency from weekly to 30 deployments "
            "per day. Built the observability stack: Prometheus metrics collection, Grafana "
            "dashboards, Loki for log aggregation, and Jaeger for distributed tracing. "
            "Led incident command for P0 events; authored runbooks that cut mean time to "
            "resolve by 55 pct. Strong background in Go, Python, Kubernetes, Helm, and "
            "capacity planning.\n"
            "B.S. Computer Engineering University of Texas Austin 2016."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Benjamin Wallace  Cloud Infrastructure Engineer  GCP  Kubernetes  Ansible\n"
            "I design and operate large-scale infrastructure on Google Cloud Platform. "
            "At Twilio migrated 300 microservices from AWS to GCP using Terraform modules; "
            "implemented Kubernetes clusters on GKE with Helm charts for service packaging "
            "and Ansible for node configuration management. Established GitOps workflow using "
            "ArgoCD: all cluster state declarative in git, pull-based continuous delivery "
            "with automatic drift detection. Configured Prometheus and Grafana for service "
            "monitoring with alerting to PagerDuty. Managed Cloud Armor WAF rules and VPC "
            "peering for multi-region deployment.\n"
            "Skills: GCP, Kubernetes, Terraform, Helm, ArgoCD, Ansible, Prometheus, "
            "Grafana, PagerDuty, Docker, Python, Bash\n"
            "B.S. Computer Science Georgia Tech 2017."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Ananya Patel  Platform Engineer  Developer Experience  GitOps\n"
            "I build the internal developer platform that lets 400 engineers ship code "
            "without thinking about infrastructure. At Figma designed and maintain the "
            "Kubernetes-based deployment platform: custom Kubernetes operators written in "
            "Go, ArgoCD for GitOps-based continuous delivery, and a self-service Backstage "
            "portal for service provisioning. Reduced environment setup time from 3 days to "
            "2 hours. Implemented progressive delivery with Argo Rollouts (canary and "
            "blue-green strategies) and automatic rollback on error-rate SLO breach. "
            "Previously at Dropbox maintained Airflow-based data platform on Kubernetes.\n"
            "Technologies: Kubernetes, ArgoCD, Argo Rollouts, Helm, Go, Python, Terraform, "
            "Backstage, Prometheus, Grafana, AWS EKS\n"
            "M.S. Computer Science Purdue University 2019."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Tyler Harris  Senior DevOps Consultant\n"
            "Eight years of consulting across 30+ clients in finance, healthcare, and media. "
            "Typical engagement: assess existing CI/CD maturity, design target-state "
            "architecture, implement infrastructure-as-code migration (Terraform), and "
            "train engineering teams. Delivered Docker containerisation and Kubernetes "
            "migration projects for clients on AWS, GCP, and Azure. Expert in Linux "
            "administration (RHEL, Ubuntu), Bash scripting, and Python automation. "
            "Designed multi-cloud disaster recovery strategies with RTO under 15 minutes. "
            "Comfortable with Vault for secrets management, Trivy for container scanning, "
            "and integrating security gates into GitHub Actions and Jenkins pipelines.\n"
            "Certifications: CKA (Certified Kubernetes Administrator), AWS DevOps Pro, "
            "GCP Professional Cloud DevOps Engineer\n"
            "B.S. Information Technology Penn State 2015."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Lauren Mitchell  Senior Cloud Engineer  Microsoft Azure\n"
            "Specialist in enterprise Azure deployments for regulated industries. "
            "At Cigna Healthcare engineered landing-zone architecture using Terraform "
            "and Azure Blueprints covering 12 subscriptions and 800 VMs. Implemented "
            "hub-spoke network topology with Azure Firewall, ExpressRoute, and Private "
            "Endpoints. Managed Azure Active Directory, Privileged Identity Management, "
            "and Conditional Access policies for SOC 2 Type II compliance. Automated "
            "compliance reporting with Azure Policy and built custom dashboards in "
            "Azure Monitor and Grafana. Designed Kubernetes clusters on AKS with Helm "
            "and GitOps via Flux.\n"
            "Skills: Azure, Terraform, AKS, Helm, Flux, Azure DevOps, GitHub Actions, "
            "Ansible, Python, Bash, Azure Monitor, Grafana\n"
            "B.S. Network Engineering Purdue University 2016."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Omar Al-Rashid  Site Reliability Engineering Manager\n"
            "I lead a 12-person SRE team at Booking.com responsible for the core booking "
            "funnel serving 1.5M transactions per hour. Established the company-wide SLO "
            "framework: worked with 40 service teams to define SLIs, set error budgets, "
            "and implement alerting in Prometheus with PagerDuty escalation. Drove MTTR "
            "from 47 minutes to 11 minutes by overhauling incident command process and "
            "runbook culture. Led migration of stateful services to Kubernetes using Helm "
            "and Argo CD. Implemented chaos engineering program using Chaos Monkey and "
            "LitmusChaos, discovering 14 latent failure modes before they caused outages. "
            "Mentor and hiring manager: grew team from 4 to 12 engineers in 2 years.\n"
            "B.S. Computer Science Delft University of Technology 2014."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Samantha Lee  Infrastructure Automation Engineer  AWS  Ansible  Python\n"
            "I automate everything that can be automated. At Robinhood:\n"
            "- Wrote Python/Ansible automation that eliminated 3000 hours/year of manual "
            "server provisioning work.\n"
            "- Built event-driven infrastructure using AWS Lambda, EventBridge, and SNS "
            "to auto-remediate common operational issues (disk full, high CPU, stale AMIs).\n"
            "- Implemented cost-optimisation tooling using AWS Cost Explorer API and "
            "automated rightsizing recommendations; saved 600K/year.\n"
            "- Managed Terraform state for 200+ modules across 6 AWS accounts using "
            "Terraform Cloud with remote state locking.\n"
            "- Maintained GitHub Actions CI/CD pipelines for infrastructure changes with "
            "Checkov static analysis and Terratest integration tests.\n"
            "B.S. Computer Science and Mathematics University of Illinois 2019."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Kevin Zhang  Kubernetes Platform Engineer\n"
            "Deep specialisation in Kubernetes internals, networking, and multi-tenancy. "
            "At Cloudflare I own the production Kubernetes platform running 15 000 pods "
            "across 400 nodes. Implemented multi-tenancy using Hierarchical Namespaces, "
            "OPA Gatekeeper policy enforcement, and pod security standards. Designed and "
            "deployed Istio service mesh for mutual TLS, traffic management, and "
            "distributed tracing. Built custom Kubernetes operators in Go for automated "
            "certificate rotation and database credential injection using HashiCorp Vault. "
            "Packaged all internal applications as Helm charts with semantic versioning "
            "and chart testing via ct (chart-testing).\n"
            "Stack: Kubernetes, Istio, Helm, Terraform, Go, Python, Vault, Prometheus, "
            "Grafana, ArgoCD, AWS EKS, Cilium CNI\n"
            "B.S. Computer Engineering University of Waterloo 2018."
        ),
    },

    {
        "category": "DevOps/Cloud Engineering",
        "text": (
            "Jessica Freeman  DevSecOps Engineer  Security Automation  Compliance\n"
            "I embed security into CI/CD pipelines so vulnerabilities are caught before "
            "production. At Palantir built the DevSecOps programme from scratch: integrated "
            "Trivy container scanning, Semgrep SAST, and Snyk SCA into GitHub Actions; "
            "automated HashiCorp Vault secret rotation for 300+ services; implemented "
            "OPA policy-as-code for Kubernetes admission control blocking non-compliant "
            "workloads. Led SOC 2 Type II readiness project covering infrastructure access, "
            "audit logging, and automated compliance evidence collection. Comfortable with "
            "Terraform, Ansible, Kubernetes, and AWS security services (GuardDuty, "
            "Security Hub, Inspector).\n"
            "Certifications: CISSP, CKS (Certified Kubernetes Security Specialist)\n"
            "B.S. Cybersecurity and Computer Science George Mason University 2018."
        ),
    },

    # ── Information Retrieval ────────────────────────────────────────────────

    {
        "category": "Information Retrieval",
        "text": (
            "Nathan Clark  Search Engineer  Elasticsearch  Relevance Tuning\n"
            "Five years building and optimising search systems for e-commerce. "
            "At Wayfair I own the product-search relevance stack: Elasticsearch cluster "
            "with custom BM25 field weights, synonym expansion, and query-time boosting "
            "by inventory and revenue signals. Tuned relevance through interleaving "
            "experiments and offline NDCG evaluation against human-labelled judgements. "
            "Implemented autocomplete using edge-ngram tokenisation and query-intent "
            "classification with a fine-tuned DistilBERT model to route queries to "
            "category-specific sub-indexes. Familiar with Lucene internals: scoring "
            "functions, custom similarities, and index segment management.\n"
            "Skills: Elasticsearch, Lucene, BM25, Python, relevance evaluation, "
            "query analysis, NDCG, A/B testing, NLP\n"
            "B.S. Computer Science University of Massachusetts Amherst 2018."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Elena Petrov  Senior Search Scientist  Learning-to-Rank\n"
            "I bridge the gap between information retrieval theory and production search systems. "
            "At Etsy led the learning-to-rank initiative: collected implicit relevance signals "
            "(clicks, purchases, dwell time), built LambdaMART reranker using XGBoost that "
            "improved NDCG@10 by 19 pct over BM25 baseline. Designed offline evaluation "
            "framework using pooled human judgements with five-point relevance scale (0-4). "
            "Ran online A/B experiments with statistical significance testing; translated "
            "offline NDCG gains into 3.2 pct purchase conversion uplift. Deep knowledge "
            "of MAP, NDCG, MRR, and precision-recall trade-offs in ranking problems. "
            "Published two papers on counterfactual LTR at SIGIR.\n"
            "Ph.D. Information Retrieval University of Amsterdam 2018. B.S. CS Moscow State 2013."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Raj Iyer  Information Retrieval Engineer  Dense Retrieval  Vector Search\n"
            "Specialist in neural information retrieval: dense passage retrieval, approximate "
            "nearest-neighbour search, and hybrid sparse-dense systems.\n"
            "At Microsoft Bing worked on dense retrieval recall: trained bi-encoder models "
            "using sentence-transformers, hard-negative mining from BM25 top-k, and in-batch "
            "negatives. Indexed 100B passages in a FAISS IVF-HNSW index achieving sub-5ms "
            "query latency. Implemented hybrid reranking pipeline combining Elasticsearch "
            "BM25 scores with dense cosine similarity via Reciprocal Rank Fusion. "
            "Evaluated on BEIR benchmark (12 datasets); model outperformed BM25 on 9 of 12.\n"
            "Tools: Python, PyTorch, sentence-transformers, FAISS, Elasticsearch, "
            "Hugging Face, BEIR, NDCG, MAP\n"
            "M.S. Computer Science IIT Bombay 2019."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Megan Carter  Search Platform Engineer  Apache Solr  Lucene  Distributed Search\n"
            "I build and operate the search infrastructure that powers large enterprise "
            "document repositories. At LexisNexis designed Solr Cloud cluster serving "
            "4B legal documents: custom schema with field-level boosting, per-field "
            "similarity configurations (BM25 for full text, TF-IDF for citation fields), "
            "and Shard routing by jurisdiction. Implemented query expansion using "
            "thesaurus-based synonym injection and WordNet-derived term expansion. Built "
            "Lucene custom query parsers for boolean and proximity operators needed by "
            "legal researchers. Familiar with index optimisation: merge policies, segment "
            "compaction, warm-up queries, and NRT (near-real-time) replication.\n"
            "Skills: Apache Solr, Lucene, Zookeeper, Elasticsearch, Java, Python, "
            "BM25, inverted index, query parsing, distributed search\n"
            "B.S. Computer Science University of Michigan 2016."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Carlos Morales  Search Relevance Engineer  Query Understanding  NLP\n"
            "At Airbnb I own the query understanding layer for accommodation search. "
            "Work includes: query segmentation and entity tagging (location, dates, "
            "guest count) using a fine-tuned BERT model; query expansion via embedding-"
            "based synonym discovery; spell correction using noisy-channel model trained "
            "on search logs; and intent classification routing navigational queries to "
            "listing detail pages. Built offline evaluation framework combining human "
            "relevance judgements (NDCG) with click-through rate metrics. Ran 30+ A/B "
            "experiments on query transformations; improvements shipped to 150M users.\n"
            "Stack: Python, PyTorch, Elasticsearch, BERT, spaCy, Spark, Airflow, "
            "query expansion, relevance evaluation, information retrieval\n"
            "M.S. Computer Science Stanford University 2020."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Stephanie Wu  Search Software Engineer  E-Commerce  Faceted Search\n"
            "Six years making product catalogs searchable and discoverable. "
            "At ASOS built faceted search over 80 000 SKUs using Elasticsearch: "
            "dynamic facet generation, filter aggregations, and nested document model "
            "for variant products. Implemented BM25-based keyword search with "
            "field-weighted scoring (title weight 3.0, description weight 1.0, "
            "brand weight 2.0) aligned to query term analysis experiments. Added "
            "semantic search re-ranking using a sentence-transformers bi-encoder for "
            "zero-shot queries with no exact keyword match. Reduced zero-result rate "
            "from 12 pct to 2 pct. Familiar with inverted index internals, tokeniser "
            "design, and Elasticsearch index lifecycle management.\n"
            "B.S. Computer Science Carnegie Mellon University 2017."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Aaron Goldstein  ML Relevance Engineer  Semantic Search  Neural Reranking\n"
            "I build hybrid search systems that combine lexical recall with neural precision.\n"
            "At LinkedIn Search:\n"
            "- Trained bi-encoder recall model (query and document towers) using contrastive "
            "learning; improved candidate recall@1000 by 28 pct over BM25 alone.\n"
            "- Trained cross-encoder reranker (DistilBERT) that scores (query, document) pairs "
            "for top-k reranking; NDCG@10 improvement of 14 pct vs BM25 baseline.\n"
            "- Implemented Reciprocal Rank Fusion to blend sparse and dense retrieval lists.\n"
            "- Evaluated models on custom benchmark with pooled human relevance judgements.\n"
            "- Familiar with BEIR, MS MARCO, and other standard IR evaluation benchmarks.\n"
            "Tools: Python, PyTorch, sentence-transformers, FAISS, Elasticsearch, transformers\n"
            "M.S. Information Science University of Washington 2019."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Fatima Malik  Research Scientist  Information Retrieval\n"
            "My research sits at the intersection of dense retrieval, evaluation methodology, "
            "and efficient approximate nearest-neighbour search. Postdoctoral researcher at "
            "Carnegie Mellon Language Technologies Institute. Published six papers at SIGIR, "
            "ECIR, and CIKM covering: learned sparse retrieval (SPLADE variants), ANN "
            "index evaluation on BEIR benchmark, and training-efficient bi-encoder models. "
            "Designed reproducible evaluation framework comparing BM25, DPR, ColBERT, and "
            "hybrid retrieval across 18 BEIR datasets using MAP, NDCG@10, and Recall@100. "
            "Comfortable with Pyserini, Tevatron, and custom PyTorch retrieval training "
            "pipelines. Reviewer for SIGIR, ACL, and EMNLP.\n"
            "Ph.D. Computer Science University of Glasgow 2022. M.S. CS AUB 2018."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Liam O'Brien  Document Retrieval and QA Engineer\n"
            "At IBM Watson I built open-domain question-answering pipelines combining "
            "retrieval with generative reading comprehension. Retrieval layer uses "
            "BM25 (Whoosh) for fast candidate selection followed by dense reranking "
            "with a fine-tuned cross-encoder. Indexed 5M enterprise documents; "
            "implemented incremental index updates using a change-feed from the document "
            "management system. Added query expansion via pseudo-relevance feedback: "
            "top-k retrieved documents mined for expansion terms using RM3 and "
            "incorporated into the inverted index query. Familiar with PyTerrier "
            "experimental framework and standard TREC evaluation metrics (MAP, NDCG, "
            "bpref). Also experienced with Elasticsearch for production deployment.\n"
            "Skills: Whoosh, BM25, PyTerrier, Elasticsearch, Python, PyTorch, "
            "cross-encoder, NDCG, MAP, query expansion, knowledge base retrieval\n"
            "B.S. Computer Science Trinity College Dublin 2018."
        ),
    },

    {
        "category": "Information Retrieval",
        "text": (
            "Hannah Svensson  Search Infrastructure Engineer  Index Optimisation  Monitoring\n"
            "I keep search clusters healthy and fast at KTH spin-off Findwise. Responsibilities "
            "span Elasticsearch cluster administration (rolling upgrades, shard rebalancing, "
            "ILM policies), query performance profiling, and relevance monitoring pipelines. "
            "Built automated relevance regression testing: daily batch evaluation run against "
            "held-out query set with NDCG@10 metric; alerts fire when score drops more than "
            "two percent from baseline. Implemented A/B testing harness for relevance "
            "configuration changes with interleaved experiment design. Familiar with Lucene "
            "segment internals, merge policies, and fielddata/docValues trade-offs. "
            "Experience with Solr, Elasticsearch, Whoosh, and inverted index construction "
            "from scratch in Python.\n"
            "M.S. Information Systems KTH Royal Institute of Technology 2017."
        ),
    },

]


def main(dry_run: bool = False) -> None:
    import pandas as pd

    if not os.path.exists(RESUMES_CSV):
        print(f"ERROR: {RESUMES_CSV} not found. Run preprocess first.")
        sys.exit(1)

    df = pd.read_csv(RESUMES_CSV)
    print(f"Existing resume count: {len(df)}")

    max_id = int(df['ID'].max()) if 'ID' in df.columns else 90_000_000

    rows = []
    for i, r in enumerate(_SYNTHETIC_RESUMES):
        text = r['text'].strip()
        word_count = len(text.split())
        rows.append({
            'ID':               max_id + 1 + i,
            'Resume_str':       text,
            'Resume_html':      '',
            'Category':         r['category'],
            'resume_clean':     text,
            'resume_word_count': word_count,
        })

    by_cat: dict[str, int] = {}
    for r in _SYNTHETIC_RESUMES:
        by_cat[r['category']] = by_cat.get(r['category'], 0) + 1

    print("\nSynthetic resumes to inject:")
    for cat, n in by_cat.items():
        print(f"  {cat}: {n}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(RESUMES_CSV, index=False)
    print(f"\nNew total resume corpus: {len(combined)} ({len(rows)} added)")

    # ── Rebuild resume indexes ───────────────────────────────────────────────
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("\n--- Resume BM25F Index ---")
    from engine.bm25f import build_resume_index
    resume_bm25 = build_resume_index(RESUMES_CSV)
    resume_bm25.save(os.path.join(INDEX_DIR, 'resumes_bm25f.pkl'))
    print(f"Saved resumes_bm25f.pkl  ({resume_bm25.N} docs)")

    try:
        from engine.semantic import build_resume_semantic_index
        from engine.cluster import build_cluster_index

        print("\n--- Resume Semantic Index ---")
        resume_sem = build_resume_semantic_index(RESUMES_CSV)
        resume_sem.save(os.path.join(INDEX_DIR, 'resumes_semantic'))
        print("Saved resumes_semantic/")

        print("\n--- Resume Cluster Index (k-means n=50) ---")
        build_cluster_index(
            resume_sem,
            n_clusters=50,
            save_path=os.path.join(INDEX_DIR, 'resumes_clusters'),
        )
        print("Saved resumes_clusters.npz")
    except ImportError:
        print("sentence-transformers not installed - skipping semantic + cluster indexes.")

    try:
        from engine.lm import build_resume_lm_index
        print("\n--- Resume LM Index ---")
        resume_lm = build_resume_lm_index(RESUMES_CSV)
        resume_lm.save(os.path.join(INDEX_DIR, 'resumes_lm.pkl'))
        print("Saved resumes_lm.pkl")
    except Exception as e:
        print(f"LM index skipped: {e}")

    print("\n=== Done - resume indexes rebuilt ===")
    print(f"Added per category:")
    for cat, n in by_cat.items():
        print(f"  {cat}: {n}")
    print(f"Total resumes now: {len(combined)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inject synthetic resumes and rebuild resume indexes')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without writing')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
