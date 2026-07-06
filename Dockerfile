# Tremor dashboard — container image (works on Google Cloud Run and anywhere).
FROM python:3.11-slim

WORKDIR /app

# system libs some wheels expect (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080). Streamlit must bind 0.0.0.0:$PORT.
ENV PORT=8080
EXPOSE 8080

CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
