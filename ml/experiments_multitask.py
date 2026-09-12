"""Shared-backbone multi-task neural network for all 16 ADMET targets (1
regression + 15 classification), trained on the 7 shared descriptors.

Unlike ml/experiments_per_task_models.py (one independent XGBoost model per
task), this trains ONE small PyTorch MLP with a shared backbone and 16
task-specific heads, per this project's planning doc's description of the
multitask approach: "shared backbone + task-specific heads, each with its
own loss function." It is trained and evaluated here purely as a comparison
point against the per-task models - a negative result (multitask doesn't
help, or only helps some tasks) is an expected, reportable outcome, not a
failure. It does not feed into ml/model_registry.py.

Missing labels (most rows have a non-missing label for only 1-2 of the 16
tasks, since the 4 source datasets barely overlap by molecule): every batch
carries a per-task mask of which rows actually have a label for that task.
Per-task loss is computed elementwise with `reduction="none"`, multiplied by
that mask, then averaged over only the non-missing entries
(`(loss * mask).sum() / mask.sum()`). Missing entries are never imputed
with zero or any other value - the label tensor's missing slots are only
ever read where the mask is 0, and multiplying loss (not labels) by the
mask keeps a bad/undefined label value from ever influencing gradients.

Task weighting: Tox21 alone contributes ~7000+ non-missing rows across its
12 tasks per batch, vs. ClinTox's 2 tasks at ~1400 rows and BBBP's 1 task at
~1949 rows and solubility's ~1115 rows. An unweighted sum of the 16 masked
per-task losses would let Tox21's 12 tasks dominate the gradient simply by
having 12x as many heads with signal in a typical batch. Fixed, precomputed
inverse-frequency weights (1 / non-missing-train-count for that task,
renormalized to sum to 1 across the 16 tasks) are applied per task before
summing into the total loss, so every task gets a comparable say in the
combined gradient regardless of how many labeled rows it has.

Usage:
    python ml/experiments_multitask.py

Writes ml/results/multitask_metrics.json (schema matching
ml/results/baseline_metrics.json's family: one record per task/split) and
ml/results/multitask_comparison_report.md (per-task-best-of-baseline/xgboost
vs. this model, on val, with an overall honest verdict).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    REGRESSION_TASKS,
    compute_descriptors,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
BASELINE_METRICS_PATH = ROOT / "ml" / "results" / "baseline_metrics.json"
PER_TASK_METRICS_PATH = ROOT / "ml" / "results" / "per_task_xgboost_metrics.json"
RESULTS_PATH = ROOT / "ml" / "results" / "multitask_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "multitask_comparison_report.md"

SEED = 42
MODEL_NAME = "multitask_nn"
HIDDEN_DIM = 64
MAX_EPOCHS = 300
PATIENCE = 20
LR = 1e-3
BATCH_SIZE = 256


class MultiTaskNet(nn.Module):
    """Shared 7 -> 64 -> 64 backbone, one linear head per task. Each of the
    15 classification heads produces a single logit (sigmoid applied only at
    eval time / inside BCEWithLogitsLoss); the 1 regression head (solubility)
    produces a raw linear output for MSE loss."""

    def __init__(self, n_descriptors: int, tasks: list[str], hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.tasks = tasks
        self.backbone = nn.Sequential(
            nn.Linear(n_descriptors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, 1) for task in tasks})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.backbone(x)
        return {task: self.heads[task](shared).squeeze(-1) for task in self.tasks}


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits loss averaged over only the non-missing (mask==1)
    entries. Targets at masked-out positions are never imputed - they may
    hold any placeholder value since the mask zeroes their loss contribution
    before any reduction happens, so a bad placeholder can never leak into
    the gradient."""
    per_row = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_row * mask).sum() / denom


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_row = (preds - targets) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (per_row * mask).sum() / denom


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_proba) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n_positive": int(np.sum(y_true)), "n_total": len(y_true)}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "n_positive": int(np.sum(y_true)),
        "n_total": len(y_true),
    }


def build_label_and_mask_matrices(df: pd.DataFrame, tasks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Missing labels stay untouched under the mask (0.0 placeholder is
    written only so the tensor is numeric - the mask guarantees it is never
    read by the loss)."""
    mask = df[tasks].notna().to_numpy().astype(np.float32)
    labels = df[tasks].fillna(0.0).to_numpy().astype(np.float32)
    return labels, mask


def compute_task_weights(train_mask: np.ndarray, tasks: list[str]) -> np.ndarray:
    """Fixed inverse-frequency weights: 1 / (# non-missing train rows for
    that task), renormalized to sum to 1. See module docstring for why this
    is needed (Tox21's 12 tasks otherwise dominate the shared gradient)."""
    counts = train_mask.sum(axis=0)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum()
    return weights.astype(np.float32)


def train_model(
    X_train, labels_train, mask_train, X_val, labels_val, mask_val, tasks, task_weights, seed=SEED
):
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    model = MultiTaskNet(n_features, tasks)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    reg_idx = [i for i, t in enumerate(tasks) if t in REGRESSION_TASKS]

    X_train_t = torch.from_numpy(X_train)
    labels_train_t = torch.from_numpy(labels_train)
    mask_train_t = torch.from_numpy(mask_train)
    X_val_t = torch.from_numpy(X_val)
    labels_val_t = torch.from_numpy(labels_val)
    mask_val_t = torch.from_numpy(mask_val)
    weights_t = torch.from_numpy(task_weights)

    n_train = X_train_t.shape[0]
    rng = np.random.default_rng(seed)

    def compute_loss(logits_by_task, labels_batch, mask_batch):
        total = torch.zeros(())
        for i, task in enumerate(tasks):
            target = labels_batch[:, i]
            mask_col = mask_batch[:, i]
            pred = logits_by_task[task]
            if i in reg_idx:
                task_loss = masked_mse_loss(pred, target, mask_col)
            else:
                task_loss = masked_bce_loss(pred, target, mask_col)
            total = total + weights_t[i] * task_loss
        return total

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start : start + BATCH_SIZE]
            batch_idx_t = torch.from_numpy(batch_idx)
            xb = X_train_t[batch_idx_t]
            yb = labels_train_t[batch_idx_t]
            mb = mask_train_t[batch_idx_t]

            optimizer.zero_grad()
            preds = model(xb)
            loss = compute_loss(preds, yb, mb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = compute_loss(val_preds, labels_val_t, mask_val_t).item()

        if np.isnan(val_loss):
            raise RuntimeError(f"NaN val loss at epoch {epoch} - masking is likely broken.")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss, epoch + 1


def evaluate(model, X, labels, mask, tasks) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        preds = model(torch.from_numpy(X))
    return preds


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    trained_at = datetime.now(timezone.utc).isoformat()

    train_df = df[df["split"] == "train"]

    scaler = StandardScaler().fit(descriptors.loc[train_df.index, DESCRIPTOR_COLUMNS])
    X_all = scaler.transform(descriptors[DESCRIPTOR_COLUMNS]).astype(np.float32)
    X_train = X_all[df["split"] == "train"]
    X_val = X_all[df["split"] == "val"]
    X_test = X_all[df["split"] == "test"]

    labels_all, mask_all = build_label_and_mask_matrices(df, ALL_TASKS)
    labels_train, mask_train = labels_all[df["split"] == "train"], mask_all[df["split"] == "train"]
    labels_val, mask_val = labels_all[df["split"] == "val"], mask_all[df["split"] == "val"]
    labels_test, mask_test = labels_all[df["split"] == "test"], mask_all[df["split"] == "test"]

    task_weights = compute_task_weights(mask_train, ALL_TASKS)
    print("Task weights (inverse-frequency, renormalized):")
    for task, w in zip(ALL_TASKS, task_weights):
        print(f"  {task}: {w:.5f} (train n={int(mask_train[:, ALL_TASKS.index(task)].sum())})")

    model, best_val_loss, n_epochs = train_model(
        X_train, labels_train, mask_train, X_val, labels_val, mask_val, ALL_TASKS, task_weights
    )
    print(f"Trained for {n_epochs} epochs, best val combined loss = {best_val_loss:.5f}")

    all_records = []
    for split_name, X_split, labels_split, mask_split in [
        ("val", X_val, labels_val, mask_val),
        ("test", X_test, labels_test, mask_test),
    ]:
        preds_by_task = evaluate(model, X_split, labels_split, mask_split, ALL_TASKS)
        for i, task in enumerate(ALL_TASKS):
            m = mask_split[:, i].astype(bool)
            if m.sum() == 0:
                continue
            y_true = labels_split[m, i]
            raw_pred = preds_by_task[task].numpy()[m]

            if task in REGRESSION_TASKS:
                metrics = regression_metrics(y_true, raw_pred)
                record = {
                    "task": task, "model_name": MODEL_NAME, "split": split_name,
                    "trained_at": trained_at, "n_samples": int(m.sum()), **metrics,
                }
            else:
                proba = 1.0 / (1.0 + np.exp(-raw_pred))
                metrics = classification_metrics(y_true.astype(int), proba)
                record = {
                    "task": task, "model_name": MODEL_NAME, "split": split_name,
                    "trained_at": trained_at, **metrics,
                }

            if any(np.isnan(v) for v in metrics.values() if isinstance(v, float)):
                raise RuntimeError(f"NaN metric for task={task} split={split_name}: {metrics}")

            all_records.append(record)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_records)} records to {RESULTS_PATH}")

    baseline_records = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    per_task_records = json.loads(PER_TASK_METRICS_PATH.read_text(encoding="utf-8"))
    write_report(all_records, baseline_records, per_task_records)
    print(f"Wrote {REPORT_PATH}")


def _best_per_task_baseline(baseline_records: list[dict], per_task_records: list[dict], tasks: list[str], key: str, higher_is_better: bool) -> dict:
    """For each task, pick the best val score across ALL candidate models
    (baseline's majority/logreg/rf AND per-task XGBoost) using `key` (roc_auc
    for classification tasks judged in the report table, rmse for
    regression). Returns {task: {"model_name":..., "val": {...}, "test": {...}}}."""
    by_task_model = {}
    for r in baseline_records + per_task_records:
        if r["task"] not in tasks:
            continue
        by_task_model.setdefault(r["task"], {}).setdefault(r["model_name"], {})[r["split"]] = r

    best = {}
    for task, models in by_task_model.items():
        best_name, best_score = None, (-np.inf if higher_is_better else np.inf)
        for model_name, splits in models.items():
            val = splits.get("val")
            if val is None or val.get(key) is None:
                continue
            score = val[key]
            if (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                best_score = score
                best_name = model_name
        if best_name is None:
            continue
        best[task] = {"model_name": best_name, **models[best_name]}
    return best


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_report(all_records: list[dict], baseline_records: list[dict], per_task_records: list[dict]) -> None:
    mt_by_task = {}
    for r in all_records:
        mt_by_task.setdefault(r["task"], {})[r["split"]] = r

    best_reg = _best_per_task_baseline(baseline_records, per_task_records, REGRESSION_TASKS, "rmse", higher_is_better=False)
    best_cls = _best_per_task_baseline(baseline_records, per_task_records, CLASSIFICATION_TASKS, "roc_auc", higher_is_better=True)

    lines = ["# Multitask NN vs. per-task-best comparison\n\n"]
    lines.append(
        "One shared-backbone (7 -> 64 -> 64, ReLU) multi-task MLP with 16 "
        "task-specific linear heads (15 classification logits for BCE loss, "
        "1 regression output for MSE loss), trained jointly on all 16 tasks. "
        "Missing labels are masked per-task per-row in the loss "
        "(`(loss * mask).sum() / mask.sum()`), never imputed. Per-task loss "
        "contributions are weighted by fixed inverse-frequency weights "
        "(1 / train-split non-missing count, renormalized to sum to 1) so "
        "Tox21's 12 tasks (~5700-7100 rows each) don't drown out ClinTox's "
        "2 tasks (~1400 rows) or BBBP (~1949) or solubility (~1115).\n\n"
        "\"Per-task-best\" below is whichever model - baseline "
        "(majority/logreg/random-forest) or per-task-tuned XGBoost - had the "
        "best val score for that task, from `ml/results/baseline_metrics.json` "
        "and `ml/results/per_task_xgboost_metrics.json`.\n\n"
    )

    lines.append("## Regression (solubility)\n\n")
    lines.append("| Split | Per-task-best | Best RMSE | Multitask RMSE | Multitask R2 | Multitask wins? |\n|---|---|---|---|---|---|\n")
    reg_beats = 0
    for task in REGRESSION_TASKS:
        best = best_reg.get(task)
        if best is None:
            continue
        for split in ["val", "test"]:
            base_r = best.get(split)
            mt_r = mt_by_task.get(task, {}).get(split)
            if base_r is None or mt_r is None:
                continue
            wins = mt_r["rmse"] < base_r["rmse"]
            if split == "val" and wins:
                reg_beats += 1
            lines.append(
                f"| {split} | {best['model_name']} | {_fmt(base_r['rmse'])} | {_fmt(mt_r['rmse'])} | "
                f"{_fmt(mt_r['r2'])} | {'Yes' if wins else 'No'} |\n"
            )

    lines.append("\n## Classification (15 tasks)\n\n")
    lines.append(
        "| Task | Split | Per-task-best | Best ROC-AUC | Best PR-AUC | "
        "Multitask ROC-AUC | Multitask PR-AUC | Multitask wins (val ROC-AUC)? |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    cls_beats, cls_losses = 0, 0
    per_task_wins_examples, per_task_loss_examples = [], []
    for task in CLASSIFICATION_TASKS:
        best = best_cls.get(task)
        mt_splits = mt_by_task.get(task, {})
        if best is None or not mt_splits:
            continue
        val_wins = None
        for split in ["val", "test"]:
            base_r = best.get(split)
            mt_r = mt_splits.get(split)
            if base_r is None or mt_r is None:
                continue
            if base_r.get("roc_auc") is None or mt_r.get("roc_auc") is None:
                wins_str = "n/a"
            else:
                wins = mt_r["roc_auc"] > base_r["roc_auc"]
                wins_str = "Yes" if wins else "No"
                if split == "val":
                    val_wins = wins
            lines.append(
                f"| {task} | {split} | {best['model_name']} | {_fmt(base_r.get('roc_auc'))} | {_fmt(base_r.get('pr_auc'))} | "
                f"{_fmt(mt_r.get('roc_auc'))} | {_fmt(mt_r.get('pr_auc'))} | {wins_str} |\n"
            )
        if val_wins is True:
            cls_beats += 1
            per_task_wins_examples.append(task)
        elif val_wins is False:
            cls_losses += 1
            per_task_loss_examples.append(task)

    total_beats = reg_beats + cls_beats
    total_compared = 1 + cls_beats + cls_losses  # solubility always has a defined comparison
    if total_beats > total_compared / 2:
        verdict = "multitask learning helps more often than not on this dataset"
    elif total_beats == 0:
        verdict = "multitask learning did not help on a single task here"
    else:
        verdict = "the result is mixed - multitask helps a minority of tasks, not a clear overall win"
    lines.append(
        f"\n**Summary**: on val, the multitask NN beat the per-task-best model on "
        f"{total_beats} of {total_compared} tasks (regression: {reg_beats}/1, "
        f"classification: {cls_beats}/{cls_beats + cls_losses} with a defined ROC-AUC comparison). "
        f"Examples where multitask helped: {', '.join(per_task_wins_examples[:5]) or 'none'}. "
        f"Examples where it did not: {', '.join(per_task_loss_examples[:5]) or 'none'}. "
        f"Honest verdict: {verdict}. This is directionally consistent with the "
        "per-task XGBoost finding (only 3/15 classification tasks improved on "
        "baseline there) that, at this dataset size (~10k molecules total, most "
        "tasks missing labels for most rows), model choice matters less than "
        "the fundamental data scarcity per task.\n"
    )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
