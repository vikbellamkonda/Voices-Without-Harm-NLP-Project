
"""
Voices Without Harm
A toxicity classification project for the Kaggle Jigsaw Unintended Bias dataset.

Features
- TF-IDF and Logistic Regression baseline
- Optional DistilBERT fine-tuning with Hugging Face Transformers
- Counterfactual data augmentation for identity-term swaps
- Fairness metrics: Subgroup AUC, BPSN AUC, BNSP AUC
- Overall metrics: F1, ROC-AUC, Accuracy

Example usage
-------------
1) Baseline only:
python voices_without_harm.py --data_path train.csv --model baseline

2) Baseline with augmentation:
python voices_without_harm.py --data_path train.csv --model baseline --augment

3) DistilBERT:
python voices_without_harm.py --data_path train.csv --model distilbert --sample_size 50000 --epochs 2

Dataset notes
-------------
Expected columns include:
- comment_text
- target
- identity columns such as male, female, homosexual_gay_or_lesbian, christian, jewish, muslim, black, white, psychiatric_or_mental_illness
Plus many others in the full dataset.

The Kaggle competition uses floating labels. This script binarizes target and identity annotations at 0.5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DEFAULT_IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

IDENTITY_SWAP_MAP = {
    "christian": ["muslim", "jewish", "hindu", "buddhist"],
    "muslim": ["christian", "jewish", "hindu", "buddhist"],
    "jewish": ["christian", "muslim", "hindu", "buddhist"],
    "hindu": ["christian", "muslim", "jewish", "buddhist"],
    "black": ["white", "asian", "latino", "african"],
    "white": ["black", "asian", "latino", "african"],
    "male": ["female", "transgender", "nonbinary"],
    "female": ["male", "transgender", "nonbinary"],
    "gay": ["straight", "bisexual", "lesbian"],
    "lesbian": ["gay", "straight", "bisexual"],
    "homosexual": ["heterosexual", "bisexual"],
    "heterosexual": ["homosexual", "bisexual"],
    "disabled": ["neurotypical", "able-bodied"],
    "psychiatric": ["physical", "neurological"],
}

# Sets random seeds for reproducibility across Python, NumPy, and environment
# Ensures consistent experiment results across multiple runs
# Important for debugging and fair model comparison
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Safely computes ROC-AUC score for binary classification tasks
# Handles edge case where only one class is present in true labels
# Prevents runtime errors during evaluation
def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

# Filters and returns identity columns that exist in the dataset
# Prevents errors caused by referencing missing columns
# Ensures only valid identity features are used
def available_identity_columns(df: pd.DataFrame, requested: List[str]) -> List[str]:
    return [c for c in requested if c in df.columns]

# Converts a numeric or float series into binary labels
# Applies thresholding to transform values into 0 or 1
# Handles missing values by filling them with zero
def binarize_series(series: pd.Series, threshold: float = 0.5) -> pd.Series:
    return series.fillna(0).astype(float).ge(threshold).astype(int)

# Preprocesses the dataset by validating and cleaning columns
# Converts text column to string and creates binary labels
# Binarizes identity columns for fairness evaluation
def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    label_col: str = "target",
    identity_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df["label"] = binarize_series(df[label_col])

    if identity_cols is None:
        identity_cols = DEFAULT_IDENTITY_COLUMNS
    present = available_identity_columns(df, identity_cols)

    for col in present:
        df[col] = binarize_series(df[col])

    return df

# Builds regex patterns for identity terms used in augmentation
# Enables efficient matching and replacement of identity words
# Supports case-insensitive matching for robustness
def build_identity_term_patterns(identity_swap_map: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    patterns = {}
    for term in identity_swap_map:
        patterns[term] = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    return patterns

# Performs counterfactual data augmentation on non toxic samples
# Replaces identity terms with alternative identities
# Helps reduce bias by diversifying identity representations
def counterfactual_augment(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    label_col: str = "label",
    max_aug_per_example: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment only non-toxic examples by swapping identity terms with alternatives.
    """
    rng = random.Random(random_state)
    patterns = build_identity_term_patterns(IDENTITY_SWAP_MAP)
    augmented_rows = []

    non_toxic_df = df[df[label_col] == 0].copy()

    for _, row in non_toxic_df.iterrows():
        text = row[text_col]
        lowered = text.lower()

        matched_terms = [term for term in IDENTITY_SWAP_MAP if re.search(rf"\b{re.escape(term)}\b", lowered)]
        if not matched_terms:
            continue

        rng.shuffle(matched_terms)
        used = 0

        for term in matched_terms:
            replacements = IDENTITY_SWAP_MAP.get(term, [])
            if not replacements:
                continue

            replacement = rng.choice(replacements)
            new_text = patterns[term].sub(replacement, text)
            if new_text != text:
                new_row = row.copy()
                new_row[text_col] = new_text
                augmented_rows.append(new_row)
                used += 1

            if used >= max_aug_per_example:
                break

    if not augmented_rows:
        return df

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined

# Computes fairness metrics for a specific identity subgroup
# Calculates subgroup, BPSN, and BNSP AUC scores
# Evaluates model bias across different identity groups
def compute_bias_metrics_for_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Jigsaw-style subgroup metrics:
    - subgroup_auc: examples within subgroup only
    - bpsn_auc: background positive, subgroup negative
    - bnsp_auc: background negative, subgroup positive
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    subgroup_mask = np.asarray(subgroup_mask).astype(bool)

    subgroup_examples = subgroup_mask
    background_examples = ~subgroup_mask

    subgroup_auc = safe_roc_auc(y_true[subgroup_examples], y_pred[subgroup_examples])

    bpsn_mask = (background_examples & (y_true == 1)) | (subgroup_examples & (y_true == 0))
    bnsp_mask = (background_examples & (y_true == 0)) | (subgroup_examples & (y_true == 1))

    bpsn_auc = safe_roc_auc(y_true[bpsn_mask], y_pred[bpsn_mask])
    bnsp_auc = safe_roc_auc(y_true[bnsp_mask], y_pred[bnsp_mask])

    return {
        "subgroup_auc": subgroup_auc,
        "bpsn_auc": bpsn_auc,
        "bnsp_auc": bnsp_auc,
        "subgroup_size": int(subgroup_mask.sum()),
    }

# Computes bias metrics for all identity groups in the dataset
# Iterates through each identity column and aggregates results
# Returns a sorted DataFrame of fairness metrics
def compute_all_bias_metrics(
    df_eval: pd.DataFrame,
    prob_col: str,
    identity_cols: List[str],
    label_col: str = "label",
) -> pd.DataFrame:
    rows = []
    for col in identity_cols:
        if col not in df_eval.columns:
            continue
        subgroup_mask = df_eval[col].fillna(0).astype(int).values == 1
        metrics = compute_bias_metrics_for_subgroup(
            y_true=df_eval[label_col].values,
            y_pred=df_eval[prob_col].values,
            subgroup_mask=subgroup_mask,
        )
        metrics["identity"] = col
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["identity", "subgroup_auc", "bpsn_auc", "bnsp_auc", "subgroup_size"])

    result = pd.DataFrame(rows)[["identity", "subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]]
    return result.sort_values(by="subgroup_auc", ascending=True).reset_index(drop=True)

# Computes overall classification metrics for model performance
# Includes accuracy, F1 score, and ROC-AUC
# Uses thresholding to convert probabilities to predictions
def overall_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
    }


@dataclass
class BaselineArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression

# Trains a baseline model using TF-IDF and Logistic Regression
# Converts text into numerical features and fits the model
# Returns trained artifacts and validation probabilities
def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "comment_text",
    label_col: str = "label",
    max_features: int = 50000,
) -> Tuple[BaselineArtifacts, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=max_features,
    )
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_val = vectorizer.transform(val_df[text_col])

    model = LogisticRegression(
        max_iter=1000,
        C=4.0,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(X_train, train_df[label_col])
    val_probs = model.predict_proba(X_val)[:, 1]

    return BaselineArtifacts(vectorizer=vectorizer, model=model), val_probs

# Trains a baseline model using TF-IDF and Logistic Regression
# Converts text into numerical features and fits the model
# Returns trained artifacts and validation probabilities
def try_import_transformers():
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
        return {
            "torch": torch,
            "Dataset": Dataset,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorWithPadding": DataCollatorWithPadding,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
        }
    except Exception as e:
        raise ImportError(
            "DistilBERT mode requires: torch, datasets, transformers, accelerate"
        ) from e

# Trains a DistilBERT model for text classification
# Tokenizes input text and fine tunes transformer model
# Returns predicted probabilities on validation set
def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "comment_text",
    label_col: str = "label",
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "distilbert_outputs",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
) -> np.ndarray:
    pkg = try_import_transformers()
    torch = pkg["torch"]
    Dataset = pkg["Dataset"]
    AutoModelForSequenceClassification = pkg["AutoModelForSequenceClassification"]
    AutoTokenizer = pkg["AutoTokenizer"]
    DataCollatorWithPadding = pkg["DataCollatorWithPadding"]
    Trainer = pkg["Trainer"]
    TrainingArguments = pkg["TrainingArguments"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]].rename(columns={label_col: "labels"}))
    val_ds = Dataset.from_pandas(val_df[[text_col, label_col]].rename(columns={label_col: "labels"}))

    def tokenize_batch(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    cols_to_remove = [c for c in train_ds.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    train_ds = train_ds.remove_columns(cols_to_remove)
    cols_to_remove_val = [c for c in val_ds.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    val_ds = val_ds.remove_columns(cols_to_remove_val)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    preds = trainer.predict(val_ds)
    logits = preds.predictions
    probs = softmax(logits)[:, 1]
    return probs

# Computes softmax probabilities from model logits
# Converts raw outputs into normalized probability distribution
# Ensures numerical stability during computation
def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

# Formats overall and bias metrics into a readable string
# Prepares output for console display
# Handles missing or NaN values gracefully
def format_results(overall: Dict[str, float], bias_df: pd.DataFrame) -> str:
    lines = []
    lines.append("\n=== Overall Metrics ===")
    for k, v in overall.items():
        if isinstance(v, float) and not math.isnan(v):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")

    lines.append("\n=== Bias Metrics by Identity ===")
    if bias_df.empty:
        lines.append("No identity metrics available.")
    else:
        lines.append(bias_df.to_string(index=False))
    return "\n".join(lines)

# Optionally downsamples dataset for faster experimentation
# Limits dataset size while preserving randomness
# Useful for quick prototyping or debugging
def maybe_downsample(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

# Saves model results and predictions to output directory
# Writes overall metrics and bias metrics to files
# Stores validation predictions for further analysis
def save_results(
    out_dir: str,
    overall: Dict[str, float],
    bias_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)

    bias_df.to_csv(os.path.join(out_dir, "bias_metrics.csv"), index=False)
    eval_df.to_csv(os.path.join(out_dir, "validation_predictions.csv"), index=False)

# Parses command line arguments for script execution
# Defines configurable parameters for training and evaluation
# Returns structured argument namespace
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voices Without Harm NLP project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Kaggle Jigsaw train.csv")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "distilbert"])
    parser.add_argument("--text_col", type=str, default="comment_text")
    parser.add_argument("--label_col", type=str, default="target")
    parser.add_argument("--augment", action="store_true", help="Apply counterfactual augmentation on training data")
    parser.add_argument("--max_aug_per_example", type=int, default=2)
    parser.add_argument("--sample_size", type=int, default=0, help="Optional row cap for faster experiments")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_features", type=int, default=50000)
    return parser.parse_args()

# Main execution function for the entire pipeline
# Loads data, preprocesses, trains model, and evaluates results
# Handles both baseline and DistilBERT workflows
def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)

    identity_cols = available_identity_columns(df, DEFAULT_IDENTITY_COLUMNS)
    df = preprocess_dataframe(
        df,
        text_col=args.text_col,
        label_col=args.label_col,
        identity_cols=DEFAULT_IDENTITY_COLUMNS,
    )
    df = maybe_downsample(df, args.sample_size if args.sample_size > 0 else None, args.random_state)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    if args.augment:
        print("Applying counterfactual augmentation...")
        train_df = counterfactual_augment(
            train_df,
            text_col=args.text_col,
            label_col="label",
            max_aug_per_example=args.max_aug_per_example,
            random_state=args.random_state,
        )
        print(f"Training rows after augmentation: {len(train_df)}")

    print(f"Training model: {args.model}")
    if args.model == "baseline":
        _, val_probs = train_baseline(
            train_df,
            val_df,
            text_col=args.text_col,
            label_col="label",
            max_features=args.max_features,
        )
    else:
        val_probs = train_distilbert(
            train_df,
            val_df,
            text_col=args.text_col,
            label_col="label",
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=os.path.join(args.out_dir, "distilbert_ckpt"),
        )

    val_df = val_df.copy()
    val_df["pred_prob"] = val_probs
    overall = overall_metrics(val_df["label"].values, val_df["pred_prob"].values)
    bias_df = compute_all_bias_metrics(val_df, prob_col="pred_prob", identity_cols=identity_cols)

    print(format_results(overall, bias_df))
    save_results(args.out_dir, overall, bias_df, val_df)
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
