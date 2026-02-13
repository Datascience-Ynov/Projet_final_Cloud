FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Heroku provides PORT as environment variable
ENV BACKEND_URL=http://localhost:8000
ENV MLFLOW_URL=http://localhost:5000

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
