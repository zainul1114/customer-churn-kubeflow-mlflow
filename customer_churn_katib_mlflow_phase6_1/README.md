# Phase 6.1 — Katib → MLflow Optimized Training

Katib has completed 8 Random Search trials and selected:

- Best Trial: `customer-churn-randomforest-n4zqb8cq`
- Katib accuracy: `0.635`
- `n_estimators=117`
- `max_depth=7`
- `min_samples_split=9`

## Flow

Customer Churn → Katib HPO → Best Hyperparameters → Final RandomForest → MLflow → PostgreSQL + MinIO

## Files

- `pipelines/customer_churn_katib_optimized_mlflow_pipeline.py` — KFP pipeline.
- `scripts/get_katib_best_params.sh` — reads the current Katib optimum.
- `docs/workflow.md` — phase workflow.

## Compile

```bash
python3 pipelines/customer_churn_katib_optimized_mlflow_pipeline.py
```

This creates:

```text
customer_churn_katib_optimized_mlflow_pipeline.yaml
```

## Verify Katib result

```bash
bash scripts/get_katib_best_params.sh
```

The final pipeline accepts the Katib values as parameters, avoiding additional Kubernetes RBAC requirements inside KFP pods.

MLflow tracking URI:

```text
http://mlflow.ml-registry.svc.cluster.local:5000
```

MLflow run:

```text
katib-optimized-randomforest
```

The final MLflow run independently calculates `training_accuracy` and `test_accuracy`. The Katib value `0.635` is the HPO objective result, not automatically the final MLflow test accuracy.
