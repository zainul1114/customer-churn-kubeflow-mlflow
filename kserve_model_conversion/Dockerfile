FROM python:3.11-slim

RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    skops \
    boto3

WORKDIR /app

COPY convert_model.py /app/convert_model.py

ENTRYPOINT ["python", "/app/convert_model.py"]
