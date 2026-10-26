import pytest
import torch
import torch.nn as nn

from RLAlg.alg._common import alpha_loss, mse, soft_update, validate_same_shape, weighted_models


def test_mse_is_half_mean_squared_error_and_checks_shapes():
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([0.0, 1.0])

    assert mse(prediction, target).item() == pytest.approx(0.5 * (1 + 4) / 2)
    with pytest.raises(ValueError, match="same shape"):
        mse(prediction, target.unsqueeze(-1))
    with pytest.raises(ValueError, match="same shape"):
        validate_same_shape("a", prediction, "b", target[:1])


def test_soft_update_averages_parameters_and_copies_buffers():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    target = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    model[1].running_mean.fill_(3.0)
    model[1].num_batches_tracked.fill_(7)
    before = [p.clone() for p in target.parameters()]

    soft_update(model, target, 0.25)

    for old, new, source in zip(before, target.parameters(), model.parameters()):
        torch.testing.assert_close(new, 0.75 * old + 0.25 * source)
    assert torch.all(target[1].running_mean == 3.0)
    assert target[1].num_batches_tracked.item() == 7


def test_soft_update_rejects_mismatched_models():
    with pytest.raises(ValueError, match="same parameters"):
        soft_update(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), 0.1)


def test_weighted_models_pairs_single_and_sequences():
    a, b = nn.Linear(1, 1), nn.Linear(1, 1)

    assert weighted_models(a, None) == [(a, 1.0)]
    assert weighted_models([a, b], [0.3, 0.7]) == [(a, 0.3), (b, 0.7)]
    for models, weights in ((a, [1.0]), ([a, b], None), ([a, b], [1.0]), ([], [])):
        with pytest.raises(ValueError):
            weighted_models(models, weights)


def test_alpha_loss_matches_sac_formula_and_detaches_log_probs():
    log_alpha = torch.tensor(0.5, requires_grad=True)
    log_probs = torch.tensor([-1.0, -3.0], requires_grad=True)

    result = alpha_loss(log_alpha, log_probs, target_entropy=-2.0)
    result["loss"].backward()

    expected = -(torch.exp(torch.tensor(0.5)) * torch.tensor([-3.0, -5.0])).mean()
    torch.testing.assert_close(result["loss"].detach(), expected)
    assert log_probs.grad is None and log_alpha.grad is not None
    torch.testing.assert_close(result["alpha"], torch.exp(torch.tensor(0.5)))
