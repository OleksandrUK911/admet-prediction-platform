import pytest

torch = pytest.importorskip("torch")  # optional, heavy dep - see ml/requirements-multitask.txt

from ml.experiment_multitask_uncertainty_weighting import UncertaintyWeighting


def test_uncertainty_weighting_at_log_sigma_zero_is_half_unweighted_sum():
    # log_sigma initialized at 0 => sigma=1 => precision=exp(0)=1 => each
    # task's term is 0.5 * loss_i + 0 (log(1) == 0). No task is
    # up/down-weighted relative to another at initialization - it starts
    # equivalent to an (evenly split) unweighted sum, then adapts from there.
    weighting = UncertaintyWeighting(n_tasks=2)
    per_task_losses = torch.tensor([2.0, 4.0])

    total = weighting(per_task_losses)

    assert torch.allclose(total, torch.tensor(0.5 * 2.0 + 0.5 * 4.0))


def test_uncertainty_weighting_sigmas_start_at_one():
    weighting = UncertaintyWeighting(n_tasks=3)
    sigmas = weighting.sigmas()

    assert sigmas.shape == (3,)
    assert all(abs(s - 1.0) < 1e-6 for s in sigmas)


def test_uncertainty_weighting_log_sigma_is_learnable():
    weighting = UncertaintyWeighting(n_tasks=2)
    per_task_losses = torch.tensor([2.0, 4.0])

    total = weighting(per_task_losses)
    total.backward()

    assert weighting.log_sigma.grad is not None
    assert torch.all(torch.isfinite(weighting.log_sigma.grad))


def test_uncertainty_weighting_down_weights_the_persistently_harder_task():
    # A task whose loss never goes below 4.0 (task 1) vs. one that goes to
    # 0 (task 0) - after a few gradient steps on log_sigma alone (network
    # weights held fixed), the harder task's implied precision
    # (1/(2*sigma^2)) should shrink relative to the easy task's, since
    # that is exactly what minimizes the combined objective analytically
    # (optimal log_sigma_i = 0.5*log(loss_i), so a bigger loss_i pushes
    # log_sigma_i - and therefore sigma_i - up, which shrinks precision).
    weighting = UncertaintyWeighting(n_tasks=2)
    optimizer = torch.optim.SGD(weighting.parameters(), lr=0.1)
    per_task_losses = torch.tensor([0.01, 4.0])

    for _ in range(200):
        optimizer.zero_grad()
        loss = weighting(per_task_losses)
        loss.backward()
        optimizer.step()

    precisions = torch.exp(-2.0 * weighting.log_sigma.detach())
    assert precisions[0] > precisions[1]  # easier (smaller-loss) task keeps more weight
