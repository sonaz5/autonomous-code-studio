FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY requests-main/ ./requests-main/
COPY main.py .
COPY app.py .

# Pre-cache models during build
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Expose Streamlit default port
EXPOSE 8501

# Launch Streamlit web app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]