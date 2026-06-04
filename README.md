# Reproducible ML Pipeline — Session 5

## What this is

A logistic-regression classifier on the Iris dataset, built to be fully reproducible:
versioned code, data, experiments and environment.

## Data

data/iris.csv is tracked with DVC.
Pointer file: data/iris.csv.dvc

## Reproduce the result

1. pip install -r requirements.txt
2. dvc pull
3. python src/train.py --seed 42

## Expected output

seed=42 accuracy=0.9211

## Multi-seed evaluation

mean=0.9474
std=0.0333

## Environment

Python 3.11
Exact packages in requirements.txt
See Dockerfile
