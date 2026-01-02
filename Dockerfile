# ---- Base image ----
FROM python:3.13-slim

# ---- System config ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Working directory ----
WORKDIR /app

# ---- Install dependencies first (better caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# ---- Fly.io expects internal port 8080 ----
ENV PORT=8080

# ---- Start FastAPI via CLI ----
CMD ["python", "main.py", "serve"]
