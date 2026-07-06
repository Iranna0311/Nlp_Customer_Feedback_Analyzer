nlp_customer_feedback_analyzer
==============================

FastAPI-based NLP customer feedback analyzer with text preprocessing, TF-IDF feature extraction, and sentiment models.

Local run
---------

Train the models:

```bash
python -m src.main --train
```

Start the API:

```bash
uvicorn src.main:app --reload
```

Docker
------

Build the image:

```bash
docker build -f docker/Dockerfile -t nlp-customer-feedback-analyzer .
```

Run the container:

```bash
docker run --rm -p 8000:8000 nlp-customer-feedback-analyzer
```

Or use Compose:

```bash
docker compose up --build
```

Docs
----

See [docs/docker.md](docs/docker.md) for the Docker workflow and notes.