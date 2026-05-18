FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY README.md ./README.md

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "src.cli", "--help"]
