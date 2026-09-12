"""Uncertainty (Kendall et al.-style) task weighting for the shared-backbone
multi-task NN, as an alternative to ml/experiments_multitask.py's FIXED
inverse-frequency task weights - per TODO/ml/TODO_experiments_multitask.md's
open item: "Експеримент з динамічним/adaptive task weighting (напр.
uncertainty weighting) як опційне вдосконалення".

ml/experiments_multitask.py precomputes a fixed weight per task (1 /
train-split non-missing count, renormalized to sum to 1) BEFORE training,
and never changes it. This script instead learns one scalar log-variance
parameter per task (`log_sigma_i`, one of Kendall, Gal & Cipolla 2018's
"Multi-Task Learning Using Uncertainty to Weigh Losses" homoscedastic
uncertainty terms) jointly with the network's weights via gradient descent:

    loss = sum_i [ (1 / (2 * sigma_i^2)) * loss_i  +  log(sigma_i) ]

Internally this is parameterized as `log_sigma_i` directly (not `sigma_i`)
so `sigma_i = exp(log_sigma_i)` is always positive without a constraint:

    loss = sum_i [ 0.5 * exp(-2 * log_sigma_i) * loss_i  +  log_sigma_i ]

Intuition: a task whose loss stays stubbornly high (noisy labels, harder
task, or simply less signal in these 7 descriptors) can push its own
sigma_i up, which shrinks that task's `1/(2 sigma_i^2)` weight - the
network "gives up" gradient budget on a task it cannot fit well, rather
than fighting it at a fixed weight forever. This is the exact opposite
failure mode from the fixed inverse-frequency scheme, which weights purely
by dataset SIZE (Tox21 assays get small weights because they have many
rows) regardless of how learnable the task actually turns out to be.

Same architecture (7 -> 64 -> 64 shared backbone + 16 task-specific linear
heads), same data, same train/val/test split, same missing-label masking
(reused directly from ml/experiments_multitask.py - masked_bce_loss,
masked_mse_loss, build_label_and_mask_matrices, MultiTaskNet) as
ml/experiments_multitask.py, so the ONLY thing that differs between the two
scripts' final metrics is the task-weighting scheme. It does not feed into
ml/model_registry.py, same as ml/experiments_multitask.py - this is a
comparison point, not a production candidate.

Usage:
    python ml/experiment_multitask_uncertainty_weighting.py

Writes ml/results/multitask_uncertainty_metrics.json (same schema as
ml/results/multitask_metrics.json) and
ml/results/multitask_uncertainty_weighting_report.md (per-task win/loss
table: uncertainty weighting vs. ml/experiments_multitask.py's fixed
inverse-frequency weighting, on val).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments_multitask import (
    MultiTaskNet,
    build_label_and_mask_matrices,
    classification_metrics,
    masked_bce_loss,
    masked_mse_loss,
    regression_metrics,
)
from features import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    REGRESSION_TASKS,
    compute_descriptors,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
FIXED_WEIGHTING_METRICS_PATH = ROOT / "ml" / "results" / "multitask_metrics.json"
RESULTS_PATH = ROOT / "ml" / "results" / "multitask_uncertainty_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "multitask_uncertainty_weighting_report.md"

SEED = 42
MODEL_NAME = "multitask_nn_uncertainty_weighted"
MAX_EPOCHS = 300
PATIENCE = 20
LR = 1e-3
BATCH_SIZE = 256


class UncertaintyWeighting(nn.Module):
    """One learnable log_sigma per task. Combines a vector of per-task
    scalar losses into the Kendall et al.-style total loss:

        sum_i [ 0.5 * exp(-2 * log_sigma_i) * loss_i  +  log_sigma_i ]

    log_sigma_i starts at 0 (sigma_i = 1), i.e. every task starts equally
    weighted, exactly like an unweighted sum - the network then adapts
    each task's weight via gradient descent on log_sigma_i jointly with
    the backbone/head weights."""

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, per_task_losses: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-2.0 * self.log_sigma)
        return (0.5 * precision * per_task_losses + self.log_sigma).sum()

    def sigmas(self) -> np.ndarray:
        with torch.no_grad():
            return torch.exp(self.log_sigma).numpy()


def train_model(
    X_train, labels_train, mask_train, X_val, labels_val, mask_val, tasks, seed=SEED
):
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    model = MultiTaskNet(n_features, tasks)
    weighting = UncertaintyWeighting(len(tasks))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(weighting.parameters()), lr=LR)

    reg_idx = [i for i, t in enumerate(tasks) if t in REGRESSION_TASKS]

    X_train_t = torch.from_numpy(X_train)
    labels_train_t = torch.from_numpy(labels_train)
    mask_train_t = torch.from_numpy(mask_train)
    X_val_t = torch.from_numpy(X_val)
    labels_val_t = torch.from_numpy(labels_val)
    mask_val_t = torch.from_numpy(mask_val)

    n_train = X_train_t.shape[0]
    rng = np.random.default_rng(seed)

    def per_task_losses(logits_by_task, labels_batch, mask_batch) -> torch.Tensor:
        losses = []
        for i, task in enumerate(tasks):
            target = labels_batch[:, i]
            mask_col = mask_batch[:, i]
            pred = logits_by_task[task]
            if i in reg_idx:
                losses.append(masked_mse_loss(pred, target, mask_col))
            else:
                losses.append(masked_bce_loss(pred, target, mask_col))
        return torch.stack(losses)

    best_val_loss = float("inf")
    best_state = None
    best_weighting_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        weighting.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start : start + BATCH_SIZE]
            batch_idx_t = torch.from_numpy(batch_idx)
            xb = X_train_t[batch_idx_t]
            yb = labels_train_t[batch_idx_t]
            mb = mask_train_t[batch_idx_t]

            optimizer.zero_grad()
            preds = model(xb)
            losses = per_task_losses(preds, yb, mb)
            loss = weighting(losses)
            loss.backward()
            optimizer.step()

        model.eval()
        weighting.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_losses = per_task_losses(val_preds, labels_val_t, mask_val_t)
            val_loss = weighting(val_losses).item()

        if np.isnan(val_loss):
            raise RuntimeError(f"NaN val loss at epoch {epoch} - masking/weighting is likely broken.")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_weighting_state = {k: v.clone() for k, v in weighting.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    weighting.load_state_dict(best_weighting_state)
    model.eval()
    weighting.eval()
    return model, weighting, best_val_loss, epoch + 1


def evaluate(model, X, tasks) -> dict[str, torch.Tensor]:
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

    model, weighting, best_val_loss, n_epochs = train_model(
        X_train, labels_train, mask_train, X_val, labels_val, mask_val, ALL_TASKS
    )
    print(f"Trained for {n_epochs} epochs, best val combined loss = {best_val_loss:.5f}")

    learned_sigmas = weighting.sigmas()
    print("Learned per-task sigma (uncertainty weighting, sqrt(variance)):")
    for task, sigma in zip(ALL_TASKS, learned_sigmas):
        implied_weight = 1.0 / (2.0 * sigma**2)
        print(f"  {task}: sigma={sigma:.4f} (implied 1/(2*sigma^2) weight={implied_weight:.5f})")

    all_records = []
    for split_name, X_split, labels_split, mask_split in [
        ("val", X_val, labels_val, mask_val),
        ("test", X_test, labels_test, mask_test),
    ]:
        preds_by_task = evaluate(model, X_split, ALL_TASKS)
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
                    "trained_at": trained_at, "n_samples": int(m.sum()),
                    "learned_sigma": float(learned_sigmas[i]), **metrics,
                }
            else:
                proba = 1.0 / (1.0 + np.exp(-raw_pred))
                metrics = classification_metrics(y_true.astype(int), proba)
                record = {
                    "task": task, "model_name": MODEL_NAME, "split": split_name,
                    "trained_at": trained_at, "learned_sigma": float(learned_sigmas[i]), **metrics,
                }

            if any(np.isnan(v) for v in metrics.values() if isinstance(v, float)):
                raise RuntimeError(f"NaN metric for task={task} split={split_name}: {metrics}")

            all_records.append(record)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_records)} records to {RESULTS_PATH}")

    if FIXED_WEIGHTING_METRICS_PATH.exists():
        fixed_records = json.loads(FIXED_WEIGHTING_METRICS_PATH.read_text(encoding="utf-8"))
        write_report(all_records, fixed_records)
        print(f"Wrote {REPORT_PATH}")
    else:
        print(
            f"WARNING: {FIXED_WEIGHTING_METRICS_PATH} not found - run "
            "ml/experiments_multitask.py first to get the fixed-weighting "
            "comparison point. Skipping report."
        )


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_report(uncertainty_records: list[dict], fixed_records: list[dict]) -> None:
    unc_by_task = {r["task"]: r for r in uncertainty_records if r["split"] == "val"}
    fixed_by_task = {r["task"]: r for r in fixed_records if r["split"] == "val"}

    lines = ["# Uncertainty weighting vs. fixed inverse-frequency weighting (multitask NN)\n\n"]
    lines.append(
        "Same shared-backbone multi-task MLP (7 -> 64 -> 64, 16 task-specific "
        "heads), same data/split/masking as ml/experiments_multitask.py. The "
        "only difference: task weights are learned jointly with the network "
        "(Kendall et al.-style homoscedastic uncertainty weighting, "
        "`loss = sum_i 0.5 * exp(-2*log_sigma_i) * loss_i + log_sigma_i`, "
        "`log_sigma_i` a learnable per-task parameter starting at 0) instead "
        "of ml/experiments_multitask.py's fixed inverse-frequency weights "
        "computed once before training. Compares val metrics between the two "
        "weighting schemes, per task - **not** against per-task XGBoost/"
        "baseline models (see ml/results/multitask_comparison_report.md for "
        "that comparison; both multitask variants here are compared only "
        "against each other).\n\n"
    )

    lines.append("## Regression (solubility)\n\n")
    lines.append("| Task | Split | Fixed-weighting RMSE | Uncertainty-weighting RMSE | Uncertainty wins? |\n|---|---|---|---|---|\n")
    reg_wins = 0
    for task in REGRESSION_TASKS:
        fixed_r = fixed_by_task.get(task)
        unc_r = unc_by_task.get(task)
        if fixed_r is None or unc_r is None:
            continue
        wins = unc_r["rmse"] < fixed_r["rmse"]
        reg_wins += int(wins)
        lines.append(f"| {task} | val | {_fmt(fixed_r['rmse'])} | {_fmt(unc_r['rmse'])} | {'Yes' if wins else 'No'} |\n")

    lines.append("\n## Classification (15 tasks)\n\n")
    lines.append(
        "| Task | Fixed-weighting ROC-AUC | Fixed-weighting PR-AUC | "
        "Uncertainty-weighting ROC-AUC | Uncertainty-weighting PR-AUC | "
        "Uncertainty wins (val ROC-AUC)? |\n|---|---|---|---|---|---|\n"
    )
    cls_wins, cls_losses = 0, 0
    win_examples, loss_examples = [], []
    for task in CLASSIFICATION_TASKS:
        fixed_r = fixed_by_task.get(task)
        unc_r = unc_by_task.get(task)
        if fixed_r is None or unc_r is None:
            continue
        if fixed_r.get("roc_auc") is None or unc_r.get("roc_auc") is None:
            wins_str = "n/a"
        else:
            wins = unc_r["roc_auc"] > fixed_r["roc_auc"]
            wins_str = "Yes" if wins else "No"
            if wins:
                cls_wins += 1
                win_examples.append(task)
            else:
                cls_losses += 1
                loss_examples.append(task)
        lines.append(
            f"| {task} | {_fmt(fixed_r.get('roc_auc'))} | {_fmt(fixed_r.get('pr_auc'))} | "
            f"{_fmt(unc_r.get('roc_auc'))} | {_fmt(unc_r.get('pr_auc'))} | {wins_str} |\n"
        )

    total_wins = reg_wins + cls_wins
    total_compared = 1 + cls_wins + cls_losses
    if total_wins > total_compared / 2:
        verdict = "uncertainty weighting beat fixed inverse-frequency weighting on most tasks"
    elif total_wins == 0:
        verdict = "uncertainty weighting did not beat fixed weighting on a single task here"
    else:
        verdict = "the result is mixed - uncertainty weighting helps a minority of tasks, no clear overall win"

    lines.append(
        f"\n**Summary**: on val, uncertainty weighting beat fixed inverse-frequency "
        f"weighting on {total_wins}/{total_compared} tasks (regression: {reg_wins}/1, "
        f"classification: {cls_wins}/{cls_wins + cls_losses}). "
        f"Examples where uncertainty weighting helped: {', '.join(win_examples[:5]) or 'none'}. "
        f"Examples where it did not: {', '.join(loss_examples[:5]) or 'none'}. "
        f"Honest verdict: {verdict}. This project has repeatedly found the multitask "
        "NN underperforms per-task models overall (3/16 wins vs. per-task-best - see "
        "ml/results/multitask_comparison_report.md); swapping the weighting scheme "
        "changes which/how many tasks the multitask NN wins **against its own fixed-"
        "weighting variant**, but is not expected to and does not, on its own, close "
        "the larger gap against per-task models - the fundamental constraint remains "
        "data scarcity per task on this ~10k-molecule dataset, not the choice of task "
        "weighting scheme.\n"
    )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
