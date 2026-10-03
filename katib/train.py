import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_dataset():
    np.random.seed(42)

    n_samples = 1000

    age = np.random.randint(18, 70, n_samples)

    tenure = np.random.randint(1, 72, n_samples)

    monthly_charges = np.round(
        np.random.uniform(20, 150, n_samples),
        2
    )

    total_charges = np.round(
        monthly_charges * tenure,
        2
    )

    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples
    )

    support_calls = np.random.randint(
        0,
        10,
        n_samples
    )

    churn_probability = (
        0.35
        + 0.20 * (contract_type == "Month-to-month")
        + 0.02 * support_calls
        - 0.003 * tenure
        + 0.001 * monthly_charges
    )

    churn_probability = np.clip(
        churn_probability,
        0,
        1
    )

    churn = (
        np.random.random(n_samples)
        < churn_probability
    ).astype(int)

    df = pd.DataFrame({
        "age": age,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "support_calls": support_calls,
        "churn": churn
    })

    return df


def preprocess(df):

    df = pd.get_dummies(
        df,
        columns=["contract_type"],
        dtype=int
    )

    X = df.drop(columns=["churn"])
    y = df["churn"]

    return X, y


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        required=True
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        required=True
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        required=True
    )

    args = parser.parse_args()

    print("========================================")
    print("Katib Customer Churn Trial")
    print("========================================")

    print(f"n_estimators       = {args.n_estimators}")
    print(f"max_depth          = {args.max_depth}")
    print(f"min_samples_split  = {args.min_samples_split}")

    # Create dataset
    df = create_dataset()

    # Preprocess
    X, y = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("========================================")
    print(f"accuracy={accuracy}")
    print("========================================")

    # Important:
    # Katib's StdOut metrics collector can use
    # the metric printed above.
    print(f"accuracy={accuracy}")


if __name__ == "__main__":
    main()
