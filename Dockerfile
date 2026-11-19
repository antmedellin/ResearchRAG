FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd --gid 1000 rag && useradd --uid 1000 --gid rag --create-home rag
ENV HOME=/home/rag XDG_CACHE_HOME=/home/rag/.cache

USER 1000:1000

