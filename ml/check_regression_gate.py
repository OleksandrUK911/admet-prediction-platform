"""CI regression gate: fails the build if the CURRENT run's per-task val
metrics, or calibration ECE, are worse than a committed floor.

Two checks, both against ml/results/metric_thresholds.json:

  1. Per-task metric regression: for every task registered in
     models/production/metadata.json, compare its CURRENT val metric
     (PR-AUC for classification, RMSE for regression) against the
     committed floor value for that task, with a small tolerance
     (default 0.02 absolute - see "tolerance" in the threshold file).
     Classification fails if current PR-AUC < floor - tolerance.
     Regression fails if current RMSE > floor + tolerance.

  2. Calibration regression: for the 4 representative tasks explored in
     ml/calibration.py (NR-AR, SR-MMP, bbbp_penetration, ct_tox), compare
     the CURRENT Platt-scaled (sigmoid) val ECE against the committed
     floor, with tolerance (default 0.05 absolute). Fails if current ECE
     > floor + tolerance (ECE getting worse = probabilities less
     trustworthy).

This is a real gate, not a decorative one: it reads whatever
models/production/metadata.json and ml/results/calibration_metrics.json
say RIGHT NOW (freshly regenerated earlier in the same CI run, after
"Run model registry" / "Run calibration + applicability domain") and
exits non-zero the moment either has regressed past its committed floor.

ml/results/metric_thresholds.json is a committed file, not regenerated
by this script. To deliberately move the floor after a reviewed,
intentional metric change (e.g. a genuinely better model, or an accepted
trade-off), regenerate it explicitly - do not hand-edit thresholds down
just to make a real regression pass:

    python -c "
    import json
    meta = json.load(open('models/production/metadata.json'))
    calib = json.load(open('ml/results/calibration_metrics.json'))
    thresholds = json.load(open('ml/results/metric_thresholds.json'))
    for task, w in meta['per_task_winners'].items():
        entry = thresholds['per_task'][task]
        entry['value'] = w['val']['rmse'] if entry['metric'] == 'rmse' else w['val']['pr_auc']
    for r in calib:
        task = r['calibration'][0]['task']
        platt = next(c for c in r['calibration'] if c['method'] == 'platt_sigmoid')
        thresholds['calibration_ece'][task]['value'] = platt['ece']
    json.dump(thresholds, open('ml/results/metric_thresholds.json', 'w'), indent=2)
    "

Usage:
    python ml/check_regression_gate.py

Exit code 0 if everything is within tolerance, 1 if anything regressed
(prints a human-readable report either way).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "models" / "production" / "metadata.json"
CALIBRATION_PATH = ROOT / "ml" / "results" / "calibration_metrics.json"
THRESHOLDS_PATH = ROOT / "ml" / "results" / "metric_thresholds.json"


def check_per_task_metrics(metadata: dict, thresholds: dict) -> list[dict]:
    tol_pr_auc = thresholds["tolerance"]["classification_pr_auc"]
    tol_rmse = thresholds["tolerance"]["regression_rmse"]
    winners = metadata.get("per_task_winners", {})

    results = []
    for task, floor in thresholds["per_task"].items():
        winner = winners.get(task)
        if winner is None:
            results.append({
                "task": task, "ok": False,
                "reason": "no registered production model for this task (was registered when thresholds were committed)",
            })
            continue

        current = winner["val"].get(floor["metric"])
        if current is None:
            results.append({"task": task, "ok": False, "reason": f"current val.{floor['metric']} is missing"})
            continue

        if floor["direction"] == "higher_is_better":
            gate = floor["value"] - tol_pr_auc
            ok = current >= gate
            reason = f"val {floor['metric']}={current:.4f} vs floor {floor['value']:.4f} (- tol {tol_pr_auc} = {gate:.4f})"
        else:
            gate = floor["value"] + tol_rmse
            ok = current <= gate
            reason = f"val {floor['metric']}={current:.4f} vs floor {floor['value']:.4f} (+ tol {tol_rmse} = {gate:.4f})"

        results.append({"task": task, "ok": ok, "reason": reason, "current": current, "floor": floor["value"]})
    return results


def check_calibration_ece(calibration_records: list[dict] | None, thresholds: dict) -> list[dict]:
    tol_ece = thresholds["tolerance"]["calibration_ece"]
    results = []

    by_task = {}
    if calibration_records:
        for r in calibration_records:
            task = r["calibration"][0]["task"]
            platt = next((c for c in r["calibration"] if c["method"] == "platt_sigmoid"), None)
            if platt is not None:
                by_task[task] = platt["ece"]

    for task, floor in thresholds["calibration_ece"].items():
        current = by_task.get(task)
        if current is None:
            results.append({
                "task": task, "ok": False,
                "reason": "no current platt_sigmoid ECE found (ml/calibration.py may not have run, or no longer covers this task)",
            })
            continue
        gate = floor["value"] + tol_ece
        ok = current <= gate
        results.append({
            "task": task, "ok": ok,
            "reason": f"val ECE={current:.4f} vs floor {floor['value']:.4f} (+ tol {tol_ece} = {gate:.4f})",
            "current": current, "floor": floor["value"],
        })
    return results


def main() -> int:
    if not THRESHOLDS_PATH.exists():
        print(f"ERROR: {THRESHOLDS_PATH} not found - nothing to check against.")
        return 1
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))

    if not METADATA_PATH.exists():
        print(f"ERROR: {METADATA_PATH} not found - run ml/model_registry.py first.")
        return 1
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    calibration_records = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else None
    if calibration_records is None:
        print(f"WARNING: {CALIBRATION_PATH} not found - skipping calibration ECE gate (run ml/calibration.py first).")

    metric_results = check_per_task_metrics(metadata, thresholds)
    ece_results = check_calibration_ece(calibration_records, thresholds)

    print("=== Per-task metric regression gate (val PR-AUC / val RMSE) ===")
    for r in metric_results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['task']}: {r['reason']}")

    print("\n=== Calibration ECE regression gate (val, Platt/sigmoid) ===")
    for r in ece_results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['task']}: {r['reason']}")

    failures = [r for r in metric_results + ece_results if not r["ok"]]
    print(f"\n{len(metric_results) + len(ece_results) - len(failures)} / {len(metric_results) + len(ece_results)} checks passed.")
    if failures:
        print(f"REGRESSION GATE FAILED: {len(failures)} check(s) regressed past their committed floor.")
        return 1
    print("Regression gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
