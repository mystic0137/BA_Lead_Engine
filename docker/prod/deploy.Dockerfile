FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt


FROM base AS hf-cache

WORKDIR /app

COPY hf_models/ /app/hf_models/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/hf_models')"

FROM base AS fastapi

WORKDIR /app

COPY --from=hf-cache /app/hf_models /app/hf_models

COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/

RUN make train

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS streamlit
                                                                                                            
WORKDIR /app

COPY frontend/ /app/frontend/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV API_URL=http://fastapi:8000

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
