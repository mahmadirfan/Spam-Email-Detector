FROM python:3.12.6-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas scikit-learn

CMD ["python3", "spamdetector.py"]
