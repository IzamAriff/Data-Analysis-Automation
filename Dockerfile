# DataPilot — containerised deployment (internal servers / cloud VMs)
FROM python:3.11-slim

WORKDIR /app

# Copy dependency list first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + bundled demo datasets.
COPY app.py ./
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Kaleido (chart PNG export) needs Chrome; install it on headless images.
RUN pip install --no-cache-dir kaleido && \
    kaleido_get_chrome || echo "Chrome download skipped — HTML chart export still works"

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
