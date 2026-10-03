import os
import boto3
import skops.io as sio
import joblib

S3_ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

BUCKET = "mlflow-artifacts"

SOURCE_KEY = (
    "3/models/"
    "m-a45cc18f6e494d8885ce64f02df0fadf/"
    "artifacts/model.skops"
)

OUTPUT_KEY = (
    "3/models/"
    "m-a45cc18f6e494d8885ce64f02df0fadf/"
    "artifacts/kserve-model/model.joblib"
)

INPUT_FILE = "/tmp/model.skops"
OUTPUT_FILE = "/tmp/model.joblib"


print("==============================================")
print("Customer Churn Model Conversion")
print("==============================================")

print("S3 endpoint:", S3_ENDPOINT)
print("Source:", f"s3://{BUCKET}/{SOURCE_KEY}")
print("Output:", f"s3://{BUCKET}/{OUTPUT_KEY}")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)

print("\nDownloading model.skops...")

s3.download_file(
    BUCKET,
    SOURCE_KEY,
    INPUT_FILE,
)

print("Downloaded:", INPUT_FILE)
print("Size:", os.path.getsize(INPUT_FILE), "bytes")

print("\nLoading skops model...")

model = sio.load(
    INPUT_FILE,
    trusted=[
        "sklearn.tree._tree.Tree"
    ],
)

print("Model type:", type(model))
print("Model:", model)

print("\nConverting to joblib...")

joblib.dump(
    model,
    OUTPUT_FILE,
)

print("Created:", OUTPUT_FILE)
print("Size:", os.path.getsize(OUTPUT_FILE), "bytes")

print("\nUploading model.joblib...")

s3.upload_file(
    OUTPUT_FILE,
    BUCKET,
    OUTPUT_KEY,
)

print("\n==============================================")
print("CONVERSION SUCCESSFUL")
print("==============================================")
print("s3://" + BUCKET + "/" + OUTPUT_KEY)
