# syntax=docker/dockerfile:1
#
# Production image for the Streamlit GUI. Deliberately lean: it installs ONLY the
# core + GUI dependencies, not the file-parsing/ingest libraries (pdfplumber,
# python-docx, ...). Those are used only by the offline `ingest` command, which
# runs on a workstation before deploy -- never inside the served container. The
# app reads its corpora from Postgres and calls the model API; it does not read
# corpus files at runtime.
#
# 3.12-slim is chosen for full, stable wheel coverage (psycopg[binary], streamlit)
# rather than the newest interpreter.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer stays cached until the requirements change.
COPY requirements.txt requirements-gui.txt ./
RUN pip install -r requirements.txt -r requirements-gui.txt

# Application code. Sensitive/large paths (data/, .env, .git) are kept out of the
# build context by .dockerignore.
COPY app.py ./
COPY pubmed_rag/ ./pubmed_rag/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Streamlit's listen port. Azure App Service maps WEBSITES_PORT to this.
EXPOSE 8501

# Liveness via Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
