
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
