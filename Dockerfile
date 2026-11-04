FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Make sure the image is self-contained: generate demo data if none is shipped,
# then train the production models and run the whole test-suite at build time.
RUN python scripts/generate_data.py > /dev/null \
 && python ingestion.py data/cs-train \
 && python model.py \
 && python run-tests.py

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "app:app"]
