FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY app.py .env.example ./
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir gradio httpx trafilatura duckduckgo-search tavily-python openai pydantic-settings python-dotenv
EXPOSE 7860
CMD ["python", "app.py"]
