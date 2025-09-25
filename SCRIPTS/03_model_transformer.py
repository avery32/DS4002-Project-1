#!/usr/bin/env python3
"""
03_model_transformer.py

Fine-tune a pretrained Transformer (RoBERTa/BERT) for 3-class hate/offensive/neutral classification,
with proper train/val/test split, early stopping on validation only, and class imbalance options.

Changes vs original:
- Stratified 80/10/10 split (configurable).
- Trainer evaluates/saves on VAL set only (no test leakage).
- Final TEST evaluation happens once, after training.
- Optional oversampling of minority classes in TRAIN.
- Optional Focal Loss instead of plain CrossEntropy.
- Safer 'evaluation_strategy' arg for Transformers.

Usage (example)
--------------
python SCRIPTS/03_model_transformer.py \
  --input DATA/processed/labeled_data_clean.csv \
  --out_dir OUTPUT \
  --model_name roberta-base \
  --epochs 4 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 128 \
  --seed 42 \
  --oversample \
  --use_focal_loss \
  --focal_gamma 2.0
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.utils import resample

import matplotlib.pyplot as plt

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers.trainer import TrainerCallback
from datasets import Dataset

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(path: Path):
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'text' and 'label' columns (from 01_preprocess.py).")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    allowed = {0, 1, 2}
    found = set(np.unique(df["label"]))
    if not found.issubset(allowed):
        raise ValueError(f"Unexpected labels {found - allowed}; expected only {allowed}.")
    return df

def plot_confusion_matrix(y_true, y_pred, labels: List[int], path: Path, title="Confusion Matrix — Transformer"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

class EpochPrinter(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"[INFO] Starting epoch {int(state.epoch)+1 if state.epoch is not None else '?'}")
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[INFO] Finished epoch {int(state.epoch) if state.epoch is not None else '?'}")

# -----------------------
# Optional Focal Loss
# -----------------------
class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = self.ce(logits, targets)  # [B]
        # pt = exp(-ce) but compute from logits for stability:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, targets.view(-1,1)).clamp_(1e-8, 1.0).squeeze(1)  # [B]
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------
# Custom Trainer with weighted / focal loss
# -----------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal=False, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            self.class_weights = class_weights.to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.use_focal:
            loss_fct = FocalLoss(gamma=self.focal_gamma, weight=self.class_weights, reduction="mean")
            loss = loss_fct(logits, labels)
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# -----------------------
# Metrics for Trainer (VAL metrics during training)
# -----------------------
def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "micro_f1": float(f1_score(labels, preds, average="micro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }
    return metrics

# -----------------------
# Oversampling helper (TRAIN only)
# -----------------------
def oversample_to_max(df: pd.DataFrame, label_col="label", seed=42) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    max_n = counts.max()
    dfs = []
    for cls, n in counts.items():
        df_cls = df[df[label_col] == cls]
        if n < max_n:
            df_up = resample(df_cls, replace=True, n_samples=max_n, random_state=seed)
            dfs.append(df_up)
        else:
            dfs.append(df_cls)
    df_bal = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_bal

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer (RoBERTa/BERT) for hate/offensive detection with proper val/test handling.")
    parser.add_argument("--input", type=str, required=True, help="Path to DATA/processed/labeled_data_clean.csv")
    parser.add_argument("--out_dir", type=str, default="OUTPUT", help="Directory to write outputs.")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="HF model name (e.g., roberta-base, bert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_size", type=float, default=0.10, help="Validation fraction (default 0.10).")
    parser.add_argument("--test_size", type=float, default=0.10, help="Test fraction (default 0.10).")
    parser.add_argument("--oversample", action="store_true", help="Oversample minority classes in TRAIN.")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use Focal Loss instead of CrossEntropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # 1) Load data
    df_all = load_data(Path(args.input))
    X_all = df_all["text"].astype(str).tolist()
    y_all = df_all["label"].astype(int).tolist()
    labels_sorted = sorted(list(set(y_all)))  # expected [0,1,2]

    # 2) Split: first create temp/test, then split temp into train/val (all stratified)
    assert 0.0 < args.test_size < 0.5, "test_size should be in (0, 0.5)"
    assert 0.0 < args.val_size < 0.5 and args.val_size + args.test_size < 0.9, "val_size should be reasonable."

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed, stratify=y_all
    )

    # Compute relative val fraction on the remaining temp set
    val_frac_of_temp = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, random_state=args.seed, stratify=y_temp
    )

    # (Optional) Oversample training set to balance classes
    if args.oversample:
        df_tr = pd.DataFrame({"text": X_train, "label": y_train})
        df_tr_bal = oversample_to_max(df_tr, label_col="label", seed=args.seed)
        X_train = df_tr_bal["text"].tolist()
        y_train = df_tr_bal["label"].tolist()
        print("[INFO] Applied oversampling to balance classes in TRAIN.")

    print(f"[INFO] Split sizes -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3) Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length"
        )

    train_ds = Dataset.from_dict({"text": X_train, "labels": y_train}).map(tokenize_batch, batched=True)
    val_ds   = Dataset.from_dict({"text": X_val,   "labels": y_val  }).map(tokenize_batch, batched=True)
    test_ds  = Dataset.from_dict({"text": X_test,  "labels": y_test }).map(tokenize_batch, batched=True)

    # set format for PyTorch
    cols = ["input_ids", "attention_mask", "labels"]
    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=cols)

    # 4) Config & Model
    num_labels = len(labels_sorted)
    # Keep these names consistent with your labels 0,1,2
    id2label = {0: "hate", 1: "offensive", 2: "neither"}
    label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # 5) Class weights from TRAIN ONLY
    class_counts = pd.Series(y_train).value_counts().reindex(labels_sorted, fill_value=0).astype(float)
    inv_freq = (class_counts.sum() / (len(labels_sorted) * class_counts.clip(lower=1.0)))
    class_weights = inv_freq / inv_freq.min()  # normalize so min weight ~1
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)
    print(f"[INFO] Class weights (train-based): {class_weights.to_dict()}")

    # 6) Training args (EVAL on VAL only; no touching TEST)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],               # no wandb by default
        fp16=torch.cuda.is_available(),
    )

    # 7) Trainer (VAL = eval_dataset)
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        use_focal=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), EpochPrinter()]
    )

    # 8) Train (select best by VAL macro-F1)
    print("[INFO] Starting fine-tuning (selection on VAL macro-F1)...")
    trainer.train()

    # 9) Final evaluation on TEST (once)
    print("[INFO] Final evaluation on held-out TEST set...")
    test_eval = trainer.evaluate(eval_dataset=test_ds)
    # Predict on TEST for detailed report
    preds = trainer.predict(test_ds)
    y_probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    y_pred = y_probs.argmax(axis=1)

    # 10) Save reports & artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_sorted = sorted(labels_sorted)
    target_names = [id2label[i] for i in labels_sorted]

    report = classification_report(
        y_test, y_pred, labels=labels_sorted, digits=4, target_names=target_names
    )
    (out_dir / "transformer_classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_micro_f1": float(f1_score(y_test, y_pred, average="micro")),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "test_macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "eval_metrics_from_trainer_on_test": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_eval.items()},
        "class_weights_train": {int(k): float(v) for k, v in class_weights.to_dict().items()},
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "oversample": bool(args.oversample),
        "use_focal_loss": bool(args.use_focal_loss),
        "focal_gamma": float(args.focal_gamma)
    }
    (out_dir / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels_sorted, out_dir / "confusion_matrix_transformer.png")

    # Save best model & tokenizer
    save_dir = out_dir / "model"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    print("[INFO] Finished Transformer fine-tuning.")
    print(f"[INFO] TEST (held-out) metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("test_")}, indent=2))
    print(f"[INFO] Model saved to: {save_dir.resolve()}")
    print(f"[INFO] Outputs written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
