FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY marg ./marg
COPY eval ./eval
COPY models ./models
RUN pip install --no-cache-dir '.[api]'

EXPOSE 8080
CMD ["python", "-m", "marg.api", "--data", "/app/outputs", "--host", "0.0.0.0", "--port", "8080"]
