# Voices Without Harm

Python NLP project for **fair toxicity classification** on the **Jigsaw Unintended Bias in Toxicity Classification** dataset.

## What this project does

- Trains a **TF-IDF + Logistic Regression** baseline
- Optionally fine-tunes **DistilBERT**
- Applies **counterfactual data augmentation** by swapping identity terms in non-toxic comments
- Evaluates both standard performance and **fairness metrics**
  - Accuracy
  - F1
  - ROC-AUC
  - Subgroup AUC
  - BPSN AUC
  - BNSP AUC

## Dataset

Competition page:
- Kaggle Jigsaw Unintended Bias in Toxicity Classification

Expected input file:
- `train.csv`

Important columns:
- `comment_text`
- `target`
- identity columns such as:
  - `male`
  - `female`
  - `homosexual_gay_or_lesbian`
  - `christian`
  - `jewish`
  - `muslim`
  - `black`
  - `white`
  - `psychiatric_or_mental_illness`

## Setup

Create an environment and install packages:

```bash
pip install pandas numpy scikit-learn
```

For DistilBERT mode, also install:

```bash
pip install torch transformers datasets accelerate
```

## How to run

### 1. Baseline
```bash
python voices_without_harm.py --data_path train.csv --model baseline
```

### 2. Baseline with counterfactual augmentation
```bash
python voices_without_harm.py --data_path train.csv --model baseline --augment
```

### 3. DistilBERT
```bash
python voices_without_harm.py --data_path train.csv --model distilbert --sample_size 50000 --epochs 2
```

## Output files

The script writes to `outputs/` by default:

- `overall_metrics.json`
- `bias_metrics.csv`
- `validation_predictions.csv`

## Notes

- The original dataset uses floating toxicity and identity annotations. This code binarizes them at `0.5`.
- The counterfactual augmentation is intentionally simple and easy to explain in a class project.
- If you want a stronger final project, you can extend this with:
  - weighted loss
  - balanced subgroup sampling
  - threshold tuning by subgroup
  - explainability with SHAP or attention analysis
  - comparison plots for fairness before and after augmentation
