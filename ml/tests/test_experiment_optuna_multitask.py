import numpy as np
import pytest

torch = pytest.importorskip("torch")  # optional, heavy dep - see ml/requirements-multitask.txt
optuna = pytest.importorskip("optuna")  # optional - see ml/requirements-multitask.txt

from ml.experiment_optuna_multitask import (
    MultiTaskNet,
    compute_task_weights,
    masked_bce_loss,
    masked_mse_loss,
)


def test_compute_task_weights_inverse_freq_favors_rare_tasks_and_sums_to_one():
    train_mask = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    weights = compute_task_weights(train_mask, "inverse_freq")

    assert weights[0] > weights[1]  # rarer task gets a bigger per-example weight
    assert np.isclose(weights.sum(), 1.0)


def test_compute_task_weights_uniform_ignores_frequency():
    train_mask = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    weights = compute_task_weights(train_mask, "uniform")

    assert np.allclose(weights, [0.5, 0.5])


def test_masked_bce_loss_ignores_masked_out_rows():
    logits = torch.tensor([0.0, 0.0, 100.0])
    targets = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    loss_with_bad_row = masked_bce_loss(logits, targets, mask)
    loss_without_bad_row = masked_bce_loss(logits[:2], targets[:2], mask[:2])

    assert torch.isfinite(loss_with_bad_row)
    assert torch.allclose(loss_with_bad_row, loss_without_bad_row)


def test_masked_mse_loss_ignores_masked_out_rows():
    preds = torch.tensor([1.0, 2.0, 1000.0])
    targets = torch.tensor([1.0, 2.0, -1000.0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    loss = masked_mse_loss(preds, targets, mask)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_multitask_net_dropout_zero_matches_deterministic_forward():
    torch.manual_seed(0)
    model = MultiTaskNet(n_descriptors=7, tasks=["a", "b"], hidden_dim=16, dropout=0.0)
    model.eval()
    x = torch.randn(4, 7)

    out1 = model(x)
    out2 = model(x)

    for task in ["a", "b"]:
        assert torch.allclose(out1[task], out2[task])


def test_multitask_net_forward_shapes():
    model = MultiTaskNet(n_descriptors=7, tasks=["t1", "t2", "t3"], hidden_dim=32, dropout=0.2)
    x = torch.randn(5, 7)

    out = model(x)

    assert set(out.keys()) == {"t1", "t2", "t3"}
    for v in out.values():
        assert v.shape == (5,)
