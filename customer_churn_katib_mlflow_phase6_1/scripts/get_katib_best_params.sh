#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="kubeflow-user-example-com"
EXPERIMENT="customer-churn-randomforest"

OPTIMAL_JSON="$(kubectl get experiment "$EXPERIMENT" -n "$NAMESPACE"   -o jsonpath='{.status.currentOptimalTrial}')"

if [[ -z "$OPTIMAL_JSON" ]]; then
  echo "ERROR: Katib has no current optimal trial."
  exit 1
fi

python3 - "$OPTIMAL_JSON" <<'PY'
import json, sys
x = json.loads(sys.argv[1])
print("Best Trial          :", x["bestTrialName"])
print("Katib Accuracy      :", x["observation"]["metrics"][0]["max"])
for p in x["parameterAssignments"]:
    print(f'{p["name"]:20}: {p["value"]}')
PY
