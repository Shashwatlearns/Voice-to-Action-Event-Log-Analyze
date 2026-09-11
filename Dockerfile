FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VTA_DB_PATH=/data/tasks.db

# ffmpeg lets pydub convert mp3/m4a/ogg uploads to WAV for SpeechRecognition
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui,voice]" \
    && python -m spacy download en_core_web_sm

COPY app.py ./
COPY examples ./examples

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
