FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY marg ./marg
COPY eval ./eval
COPY models ./models
RUN pip install --no-cache-dir '.[api]' \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir opencv-python-headless==5.0.0.93

EXPOSE 8080
CMD ["python", "-m", "marg.entrypoint"]
