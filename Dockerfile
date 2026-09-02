FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRAPHGUARD_MODEL=/app/artifacts/baseline/xgboost.json

WORKDIR /app

# Keep the production API image independent of the heavyweight training stack.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY config.yaml ./config.yaml

ENV PYTHONPATH=/app/src
EXPOSE 8000

# The trained model is intentionally kept out of Git and mounted at runtime.
CMD ["uvicorn", "graphguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
