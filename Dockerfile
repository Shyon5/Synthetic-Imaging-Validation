FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIV_DATA_DIR=/data \
    SIV_OUTPUT_DIR=/output \
    SIV_CONTAINER=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY docker/constraints.txt ./docker/constraints.txt
RUN python -m pip install --constraint docker/constraints.txt ".[app]" \
    && useradd --uid 1000 --create-home appuser \
    && mkdir /data /output \
    && chown appuser:appuser /output
COPY apps/ ./apps/
COPY .streamlit/config.toml ./.streamlit/config.toml

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "apps/local_validation/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
