FROM python:3.11-slim
WORKDIR /app
# App code only (tests/evals/traces are not needed in the image).
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY app.py .env.example ./
# -e . pulls every runtime dependency from pyproject; gradio is pinned to match
# the version the app was tested against (drift-safe image).
RUN pip install --no-cache-dir -e . "gradio==6.29.1"
# Render/Docker hosts route traffic to $PORT (app.py reads it); 7860 is local-friendly.
EXPOSE 7860
CMD ["python", "app.py"]
