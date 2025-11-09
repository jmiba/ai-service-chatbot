FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        curl \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

COPY . .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=20s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
