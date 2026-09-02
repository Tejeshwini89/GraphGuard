FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRAPHGUARD_MODEL=/app/artifacts/baseline/xgboost.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config.yaml ./config.yaml
COPY artifacts/baseline ./artifacts/baseline

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "graphguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
