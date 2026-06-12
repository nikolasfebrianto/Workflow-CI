# Workflow-CI

Repository ini berisi MLflow Project dan GitHub Actions untuk retraining otomatis model Breast Cancer Wisconsin Diagnostic.

## Struktur

```text
Workflow-CI
├── .github/workflows/mlflow-project-ci.yml
├── .workflow/README.md
└── MLProject
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── breast_cancer_preprocessing/
    ├── Tautan_ke_Docker_Hub.txt
    └── requirements.txt
```

Workflow akan:

1. Menjalankan `mlflow run MLProject --env-manager=local`.
2. Mengunggah artefak output training.
3. Membuat Docker image memakai `mlflow models build-docker`.
4. Push image ke Docker Hub.
