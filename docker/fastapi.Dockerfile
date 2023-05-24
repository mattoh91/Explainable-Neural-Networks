FROM python:3.10.1-slim

WORKDIR /app

COPY api_requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r api_requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8080"]
