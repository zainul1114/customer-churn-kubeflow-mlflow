# What is Customer Churn Prediction?

## 1. What is Customer Churn?

**Customer churn** means that a customer stops using a company's product or service.

For example:

* A telecom customer cancels their mobile subscription.
* A streaming customer cancels their subscription.
* A SaaS customer does not renew their subscription.
* A customer stops using an online service.

In simple terms:

> **Customer churn means a customer is leaving the business.**

---

## 2. What is Customer Churn Prediction?

**Customer Churn Prediction** is a machine learning technique used to predict whether a customer is likely to leave a company or continue using its service.

The machine learning model learns patterns from historical customer data.

The basic workflow is:

```text
Customer Data
      |
      v
Machine Learning Model
      |
      v
Churn Prediction
```

The prediction can be:

```text
Churn = Yes
```

or:

```text
Churn = No
```

A model can also produce a probability:

```text
Churn Probability = 82%
```

---

## 3. Example

Suppose a telecom company has the following customer information:

| Feature         |          Value |
| --------------- | -------------: |
| Age             |             35 |
| Tenure          |       4 months |
| Monthly Charges |            $85 |
| Total Charges   |           $340 |
| Contract Type   | Month-to-month |
| Support Calls   |              5 |

The machine learning model uses these features to make a prediction.

```text
Customer Data
      |
      v
+----------------------+
| Machine Learning     |
| Model                |
+----------+-----------+
           |
           v
+----------------------+
| Churn Prediction     |
+----------+-----------+
           |
           v
      Churn = Yes
```

---

## 4. Why is Churn Prediction Important?

Customer churn can affect business revenue and customer relationships.

By identifying customers who may be likely to churn, a company can investigate their experience and take appropriate customer-retention actions.

For example:

```text
Customer
   |
   v
Customer Data
   |
   v
Churn Prediction
   |
   +----> Lower Risk
   |
   +----> Higher Risk
              |
              v
       Customer Retention
           Actions
```

The objective is not simply to predict churn, but to provide information that can support customer-retention decisions.

---

## 5. Machine Learning Problem

Customer churn prediction is commonly treated as a **classification problem**.

The model learns from historical data where the churn outcome is already known.

Example:

| Customer | Tenure | Monthly Charges | Support Calls | Contract       | Churn |
| -------- | -----: | --------------: | ------------: | -------------- | ----- |
| C001     |     36 |              45 |             1 | Long-term      | No    |
| C002     |      3 |              85 |             5 | Month-to-month | Yes   |
| C003     |     24 |              50 |             1 | Long-term      | No    |
| C004     |      5 |              90 |             6 | Month-to-month | Yes   |

### Input Features

The model uses customer attributes such as:

```text
age
tenure
monthly_charges
total_charges
contract_type
support_calls
```

### Target Variable

The value that the model learns to predict is:

```text
churn
```

The target can represent:

```text
Yes
No
```

---

## 6. Customer Churn Prediction in This Project

This project uses a synthetic customer dataset containing:

```text
customer_id
age
tenure
monthly_charges
total_charges
contract_type
support_calls
churn
```

The machine learning workflow is:

```text
Dataset Creation
       |
       v
Data Validation
       |
       v
Data Preprocessing
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
MLflow Tracking
```

---

## 7. Machine Learning Model

The current project uses:

```text
RandomForestClassifier
```

Random Forest is an ensemble classification algorithm that combines multiple decision trees to produce a prediction.

The project uses customer features as inputs:

```text
Customer Features
       |
       v
Random Forest
       |
       v
Churn Prediction
```

The model is trained using historical customer data and evaluated using previously unseen test data.

---

## 8. Model Evaluation

After training, the model is evaluated using classification metrics.

The current project evaluates:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Accuracy

Measures the proportion of predictions that are correct.

### Precision

Measures how many customers predicted as churn actually belong to the churn class.

### Recall

Measures how many actual churn cases were correctly identified.

### F1 Score

Provides a combined measure based on precision and recall.

### Confusion Matrix

Shows the relationship between actual and predicted classes.

---

## 9. From Machine Learning to MLOps

A machine learning model is only one part of a production ML system.

This project extends the churn prediction problem into an MLOps workflow.

```text
                    Customer Data
                         |
                         v
                 +---------------+
                 |   Kubeflow    |
                 |   Pipelines   |
                 +-------+-------+
                         |
                         v
                Data Validation
                         |
                         v
                Data Preprocessing
                         |
                         v
                  Model Training
                         |
                         v
                  Model Evaluation
                         |
                         v
                     MLflow
                         |
              +----------+----------+
              |                     |
              v                     v
          PostgreSQL              MinIO
          Metadata              Artifacts
```

Kubeflow Pipelines provides the workflow orchestration layer, while MLflow provides experiment tracking and model/artifact tracking.

---

## 10. Technologies Used

The project uses:

* Kubernetes
* Kubeflow Community Distribution
* Kubeflow Pipelines
* MLflow
* PostgreSQL
* MinIO
* Python
* Scikit-learn
* Random Forest
* Docker

---

## 11. Project Roadmap

The project is being developed as an end-to-end MLOps platform.

```text
Dataset
   |
   v
Validation
   |
   v
Preprocessing
   |
   v
Training
   |
   v
Evaluation
   |
   v
MLflow Tracking
   |
   v
Katib Hyperparameter Optimization
   |
   v
Model Registry
   |
   v
KServe
   |
   v
Prediction API
   |
   v
Monitoring
   |
   v
CI/CD + GitOps
```

The first stages are implemented in the current project. The later stages are planned extensions.

---

## 12. Summary

Customer Churn Prediction uses machine learning to estimate whether a customer is likely to leave a product or service.

In this project, the machine learning problem is implemented as an end-to-end MLOps workflow using **Kubeflow and MLflow**.

The overall lifecycle is:

```text
Data
  |
  v
Validation
  |
  v
Preprocessing
  |
  v
Training
  |
  v
Evaluation
  |
  v
Experiment Tracking
  |
  v
Artifact Management
  |
  v
Model Registry
  |
  v
Model Serving
  |
  v
Monitoring
```

The current implementation covers the data, training, evaluation, MLflow tracking, and artifact-management stages, while model registry, serving, monitoring, and GitOps are planned future phases.

