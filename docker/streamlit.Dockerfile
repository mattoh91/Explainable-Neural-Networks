FROM python:3.10.1-slim

WORKDIR /app

COPY dashboard_requirements.txt .
RUN pip install -r dashboard_requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
