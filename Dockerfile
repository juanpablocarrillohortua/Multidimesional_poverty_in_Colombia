# Same Python as the venv where the model was saved and tested
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first: this layer stays cached while only the code changes
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Same layout as the repo: app_config looks for the model in ../models
COPY models/multidimensional_poverty_model_v1.joblib models/
COPY app/ app/

# Unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

# Streamlit's own health endpoint, no curl needed in the image
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# 0.0.0.0 so the host can reach it through the published port
CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
