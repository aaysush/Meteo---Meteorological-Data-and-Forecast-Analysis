
FROM python:3.10-slim

WORKDIR /app

# ---------- System Dependencies ----------
# Add curl + build dependencies for pip packages that might need compiling
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# ---------- Expose Internal Port as the 
EXPOSE 8000

# ---------- Start the App with Uvicorn ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
