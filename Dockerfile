FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TRADING_MODE=PAPER
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m trader && chown -R trader /app
USER trader

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
