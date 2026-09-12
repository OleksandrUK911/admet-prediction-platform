import numpy as np
import pandas as pd
import torch

from ml.experiments_multitask import (
    build_label_and_mask_matrices,
    compute_task_weights,
    masked_bce_loss,
    masked_mse_loss,
)


def test_masked_bce_loss_ignores_masked_out_rows():
    logits = torch.tensor([0.0, 0.0, 100.0])  # row 2 would blow up BCE if unmasked
    targets = torch.tensor([1.0, 0.0, 0.0])  # placeholder target at masked-out row 2
    mask = torch.tensor([1.0, 1.0, 0.0])

    loss_with_bad_row = masked_bce_loss(logits, targets, mask)

    # Recompute with the masked-out row entirely removed - must match exactly,
    # proving the masked-out entry never contributes to the reduction.
    loss_without_bad_row = masked_bce_loss(logits[:2], targets[:2], mask[:2])

    assert torch.isfinite(loss_with_bad_row)
    assert torch.allclose(loss_with_bad_row, loss_without_bad_row)


def test_masked_mse_loss_ignores_masked_out_rows():
    preds = torch.tensor([1.0, 2.0, 1000.0])
    targets = torch.tensor([1.0, 2.0, -1000.0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    loss = masked_mse_loss(preds, targets, mask)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0  # only the two perfectly-matched, non-missing rows count


def test_masked_bce_loss_all_missing_does_not_divide_by_zero():
    logits = torch.tensor([0.5, -0.5])
    targets = torch.tensor([1.0, 0.0])
    mask = torch.tensor([0.0, 0.0])

    loss = masked_bce_loss(logits, targets, mask)

    assert torch.isfinite(loss)


def test_build_label_and_mask_matrices_marks_missing_correctly():
    df = pd.DataFrame({
        "taskA": [1.0, np.nan, 0.0],
        "taskB": [np.nan, 1.0, np.nan],
    })
    labels, mask = build_label_and_mask_matrices(df, ["taskA", "taskB"])

    np.testing.assert_array_equal(mask, [[1, 0], [0, 1], [1, 0]])
    # Missing slots get a numeric placeholder (0.0) but the mask says not to read them.
    np.testing.assert_array_equal(labels[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(labels[:, 1], [0.0, 1.0, 0.0])


def test_compute_task_weights_favors_rare_tasks_and_sums_to_one():
    # task "rare" has far fewer non-missing train rows than "common".
    train_mask = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    weights = compute_task_weights(train_mask, ["rare", "common"])

    assert weights[0] > weights[1]  # rarer task gets a bigger per-example weight
    assert np.isclose(weights.sum(), 1.0)
