FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY marg ./marg
COPY eval ./eval
COPY models ./models
RUN pip install --no-cache-dir '.[api]'

EXPOSE 8080
CMD ["python", "-m", "marg.entrypoint"]
