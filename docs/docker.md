# Docker

This project ships with a Docker image that runs the FastAPI app.

## Build

```bash
docker build -f docker/Dockerfile -t nlp-customer-feedback-analyzer .
```

## Run

```bash
docker run --rm -p 8000:8000 nlp-customer-feedback-analyzer
```

## Compose

```bash
docker compose up --build
```

## Notes

- The container starts the API with `uvicorn src.main:app`.
- If you want the model artifacts available inside the image, run training before building or mount a volume with `data/models`.
- The compose file mounts `data/models` so trained artifacts persist between container runs.