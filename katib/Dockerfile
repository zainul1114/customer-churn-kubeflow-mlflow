FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2

COPY train.py /app/train.py

ENTRYPOINT ["python", "/app/train.py"]
