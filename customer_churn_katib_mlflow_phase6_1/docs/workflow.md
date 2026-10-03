# Katib → MLflow

```text
Katib HPO
   ↓
8 Random Search Trials
   ↓
Best Trial
   ↓
n_estimators=117
max_depth=7
min_samples_split=9
accuracy=0.635
   ↓
Final RandomForest Training
   ↓
MLflow
   ├── PostgreSQL → run metadata
   └── MinIO → model/artifacts
```

The next phase after successful optimized training is Model Registry.
