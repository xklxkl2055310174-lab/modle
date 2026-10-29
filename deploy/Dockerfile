FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 \
       --index-url https://download.pytorch.org/whl/cpu
COPY app app
COPY client client
COPY scripts scripts
RUN useradd --create-home --uid 10001 inference && mkdir -p weights reports \
    && chown -R inference:inference /service
USER inference
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"
CMD ["python", "-m", "app"]

