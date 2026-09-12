"""Optuna hyperparameter search for the multi-task NN's architecture/training
config.

`ml/experiments_multitask.py` trains one fixed shared-backbone MLP
(7 -> 64 -> 64, lr=1e-3, no dropout, fixed inverse-frequency task weights,
up to 300 epochs with patience 20) - no hyperparameter search at all. This
script asks whether tuning hidden layer size, learning rate, dropout, and
the task-weighting scheme (fixed inverse-frequency vs. uniform) does any
better, with a small Optuna budget (torch training is comparatively
expensive, so this is intentionally much smaller than
ml/experiment_optuna_per_task.py's XGBoost search).

Search space:
  - hidden_dim: {32, 64, 128, 256}
  - learning_rate: log-uniform [1e-4, 1e-2]
  - dropout: [0.0, 0.5] (applied after each backbone ReLU; 0.0 reproduces
    the current no-dropout architecture as a reachable point in the space)
  - task_weighting: {"inverse_freq", "uniform"} - "inverse_freq" is what
    ml/experiments_multitask.py already uses (see its module docstring for
    why Tox21's 12 tasks would otherwise dominate the gradient); "uniform"
    is the naive alternative, included so the search can tell us whether the
    existing choice is actually load-bearing or not.

Model selection discipline matches the rest of the project: each trial
trains on TRAIN only and is scored by combined val loss (never touching
test) with a SHORT epoch budget/patience to keep the search affordable; the
winning trial's config is then refit with the full epoch budget/patience
(matching ml/experiments_multitask.py exactly) and reported once on val (to
compare against the currently-registered fixed-architecture multitask
model) and once on test (report-only). This script does not modify
ml/model_registry.py or promote any tuned model into production - it is a
pure comparison experiment, like ml/experiments_multitask.py itself.

Usage:
    pip install -r ml/requirements-multitask.txt
    python ml/experiment_optuna_multitask.py

Writes ml/results/optuna_multitask_metrics.json and
ml/results/optuna_multitask_report.md.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
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
MULTITASK_METRICS_PATH = ROOT / "ml" / "results" / "multitask_metrics.json"
RESULTS_PATH = ROOT / "ml" / "results" / "optuna_multitask_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "optuna_multitask_report.md"

SEED = 42
MODEL_NAME = "multitask_nn_optuna"
N_TRIALS = 15  # small budget - torch training is expensive per trial
BATCH_SIZE = 256

# Short budget used only inside the search loop, to keep N_TRIALS affordable.
SEARCH_MAX_EPOCHS = 100
SEARCH_PATIENCE = 10

# Full budget for the final refit of the winning config - matches
# ml/experiments_multitask.py exactly, so the comparison is apples-to-apples.
FINAL_MAX_EPOCHS = 300
FINAL_PATIENCE = 20

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MultiTaskNet(nn.Module):
    """Same shared-backbone-plus-per-task-linear-head design as
    ml/experiments_multitask.py's MultiTaskNet, generalized to a
    configurable hidden width and an optional dropout after each backbone
    ReLU (dropout=0.0 reproduces the original architecture exactly)."""

    def __init__(self, n_descriptors: int, tasks: list[str], hidden_dim: int, dropout: float):
        super().__init__()
        self.tasks = tasks
        layers = [
            nn.Linear(n_descriptors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, 1) for task in tasks})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.backbone(x)
        return {task: self.heads[task](shared).squeeze(-1) for task in self.tasks}


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_row = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_row * mask).sum() / denom


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_row = (preds - targets) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (per_row * mask).sum() / denom


def build_label_and_mask_matrices(df: pd.DataFrame, tasks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mask = df[tasks].notna().to_numpy().astype(np.float32)
    labels = df[tasks].fillna(0.0).to_numpy().astype(np.float32)
    return labels, mask


def compute_task_weights(train_mask: np.ndarray, scheme: str) -> np.ndarray:
    """Two task-weighting schemes: "inverse_freq" (1 / non-missing train
    count, renormalized to sum to 1 - what ml/experiments_multitask.py
    already uses) and "uniform" (1 / n_tasks each, i.e. no correction for
    how many labeled rows each task has)."""
    n_tasks = train_mask.shape[1]
    if scheme == "uniform":
        return np.full(n_tasks, 1.0 / n_tasks, dtype=np.float32)
    counts = np.maximum(train_mask.sum(axis=0), 1.0)
    inv = 1.0 / counts
    return (inv / inv.sum()).astype(np.float32)


def train_model(
    X_train, labels_train, mask_train, X_val, labels_val, mask_val, tasks, task_weights,
    hidden_dim, dropout, lr, max_epochs, patience, seed=SEED,
):
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    model = MultiTaskNet(n_features, tasks, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    epoch = 0

    for epoch in range(max_epochs):
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
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss, epoch + 1


def regression_metrics(y_true, y_pred) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def classification_metrics(y_true, y_proba) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def _evaluate_records(model, X, labels, mask, tasks, split_name, trained_at) -> list[dict]:
    with torch.no_grad():
        preds_by_task = model(torch.from_numpy(X))
    records = []
    for i, task in enumerate(tasks):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            continue
        y_true = labels[m, i]
        raw_pred = preds_by_task[task].numpy()[m]
        if task in REGRESSION_TASKS:
            metrics = regression_metrics(y_true, raw_pred)
        else:
            proba = 1.0 / (1.0 + np.exp(-raw_pred))
            metrics = classification_metrics(y_true.astype(int), proba)
        records.append({
            "task": task, "model_name": MODEL_NAME, "split": split_name,
            "trained_at": trained_at, "n_samples": int(m.sum()), **metrics,
        })
    return records


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

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weighting = trial.suggest_categorical("task_weighting", ["inverse_freq", "uniform"])

        task_weights = compute_task_weights(mask_train, weighting)
        _, best_val_loss, _ = train_model(
            X_train, labels_train, mask_train, X_val, labels_val, mask_val, ALL_TASKS, task_weights,
            hidden_dim=hidden_dim, dropout=dropout, lr=lr,
            max_epochs=SEARCH_MAX_EPOCHS, patience=SEARCH_PATIENCE, seed=SEED,
        )
        return best_val_loss

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"Best trial: val combined loss={study.best_value:.5f}, params={best}")

    final_task_weights = compute_task_weights(mask_train, best["task_weighting"])
    model, final_val_loss, n_epochs = train_model(
        X_train, labels_train, mask_train, X_val, labels_val, mask_val, ALL_TASKS, final_task_weights,
        hidden_dim=best["hidden_dim"], dropout=best["dropout"], lr=best["lr"],
        max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, seed=SEED,
    )
    print(f"Final refit: {n_epochs} epochs, best val combined loss={final_val_loss:.5f}")

    all_records = []
    all_records.extend(_evaluate_records(model, X_val, labels_val, mask_val, ALL_TASKS, "val", trained_at))
    all_records.extend(_evaluate_records(model, X_test, labels_test, mask_test, ALL_TASKS, "test", trained_at))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "best_params": best,
        "best_search_val_loss": study.best_value,
        "n_trials": len(study.trials),
        "final_val_loss": final_val_loss,
        "records": all_records,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")

    fixed_records = json.loads(MULTITASK_METRICS_PATH.read_text(encoding="utf-8"))
    write_report(all_records, fixed_records, best, study.best_value)
    print(f"Wrote {REPORT_PATH}")


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_report(tuned_records: list[dict], fixed_records: list[dict], best_params: dict, best_search_val_loss: float) -> None:
    tuned_by_task = {}
    for r in tuned_records:
        tuned_by_task.setdefault(r["task"], {})[r["split"]] = r
    fixed_by_task = {}
    for r in fixed_records:
        fixed_by_task.setdefault(r["task"], {})[r["split"]] = r

    lines = ["# Optuna multi-task NN hyperparameter search vs. fixed architecture\n\n"]
    lines.append(
        f"{N_TRIALS} Optuna trials over hidden_dim ({{'32,64,128,256'}}), "
        "learning rate (log-uniform 1e-4 to 1e-2), dropout (0.0-0.5), and "
        "task-weighting scheme (inverse-frequency vs. uniform). Each trial "
        f"trains on TRAIN only with a short budget ({SEARCH_MAX_EPOCHS} max "
        f"epochs, patience {SEARCH_PATIENCE}) and is scored by combined val "
        "loss - test is never touched during the search. The winning "
        f"config is refit with the full budget ({FINAL_MAX_EPOCHS} max "
        f"epochs, patience {FINAL_PATIENCE}) used by "
        "`ml/experiments_multitask.py`'s fixed architecture, for an "
        "apples-to-apples comparison.\n\n"
        f"**Best config found**: hidden_dim={best_params['hidden_dim']}, "
        f"lr={best_params['lr']:.2e}, dropout={best_params['dropout']:.3f}, "
        f"task_weighting={best_params['task_weighting']} "
        f"(search val combined loss={best_search_val_loss:.5f}, vs. the "
        "fixed architecture's own combined loss reported in "
        "`ml/results/multitask_comparison_report.md`).\n\n"
        "`ml/experiments_multitask.py`'s fixed architecture (7 -> 64 -> 64, "
        "lr=1e-3, no dropout, inverse-frequency task weighting) is the "
        "comparison point (\"fixed\" below). Val decides whether tuning "
        "helped; test is reported for completeness only.\n\n"
    )

    lines.append("## Regression (solubility)\n\n")
    lines.append("| Split | Fixed RMSE | Optuna RMSE | Improved? |\n|---|---|---|---|\n")
    reg_improved = None
    for split in ["val", "test"]:
        fixed = fixed_by_task.get("solubility", {}).get(split)
        tuned = tuned_by_task.get("solubility", {}).get(split)
        if fixed is None or tuned is None:
            continue
        improved = tuned["rmse"] < fixed["rmse"]
        if split == "val":
            reg_improved = improved
        lines.append(f"| {split} | {_fmt(fixed['rmse'])} | {_fmt(tuned['rmse'])} | {'Yes' if improved else 'No'} |\n")

    lines.append("\n## Classification (15 tasks)\n\n")
    lines.append(
        "| Task | Split | Fixed ROC-AUC | Fixed PR-AUC | Optuna ROC-AUC | "
        "Optuna PR-AUC | Improved (val PR-AUC)? |\n|---|---|---|---|---|---|---|\n"
    )
    improved_count, not_improved_count = 0, 0
    for task in CLASSIFICATION_TASKS:
        val_improved = None
        for split in ["val", "test"]:
            fixed = fixed_by_task.get(task, {}).get(split)
            tuned = tuned_by_task.get(task, {}).get(split)
            if fixed is None or tuned is None:
                continue
            if fixed.get("pr_auc") is None or tuned.get("pr_auc") is None:
                improved_str = "n/a"
            else:
                is_improved = tuned["pr_auc"] > fixed["pr_auc"]
                improved_str = "Yes" if is_improved else "No"
                if split == "val":
                    val_improved = is_improved
            lines.append(
                f"| {task} | {split} | {_fmt(fixed.get('roc_auc'))} | {_fmt(fixed.get('pr_auc'))} | "
                f"{_fmt(tuned.get('roc_auc'))} | {_fmt(tuned.get('pr_auc'))} | {improved_str} |\n"
            )
        if val_improved is True:
            improved_count += 1
        elif val_improved is False:
            not_improved_count += 1

    total_improved = improved_count + (1 if reg_improved else 0)
    total_compared = improved_count + not_improved_count + (1 if reg_improved is not None else 0)
    lines.append(
        f"\n**Summary**: the Optuna-tuned multitask NN improved val PR-AUC/RMSE "
        f"over the fixed architecture on {total_improved} of {total_compared} tasks "
        f"with a defined comparison ({improved_count}/{improved_count + not_improved_count} "
        "classification tasks). This is not promoted into "
        "`ml/model_registry.py` regardless of outcome (the multitask model never "
        "has been) - it is an exploratory comparison only, same status as "
        "`ml/experiments_multitask.py` itself.\n"
    )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
