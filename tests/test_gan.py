import pytest
import torch
import torch.nn as nn

from RLAlg.alg.gan import GAN


class SumDiscriminator(nn.Module):
    def forward(self, x):
        if isinstance(x, dict):
            outputs = []
            for value in x.values():
                batch = value.shape[0]
                outputs.append(value.reshape(batch, -1).sum(dim=1, keepdim=True))
            return sum(outputs)

        batch = x.shape[0]
        return x.reshape(batch, -1).sum(dim=1, keepdim=True)


def _build_dict_batch(batch: int = 4):
    real = {
        "vector": torch.randn(batch, 3),
        "image": torch.randn(batch, 2, 2),
    }
    fake = {
        "vector": torch.randn(batch, 3),
        "image": torch.randn(batch, 2, 2),
    }
    return real, fake


def test_compute_hinge_loss_supports_dict_input_with_detach_and_r1():
    discriminator = SumDiscriminator()
    real_data, fake_data = _build_dict_batch()

    output = GAN.compute_hinge_loss(
        discriminator,
        real_data=real_data,
        fake_data=fake_data,
        detach_fake=True,
        r1_gamma=1.0,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert isinstance(output["gradient_penalty"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["gradient_penalty"])


def test_compute_wasserstein_loss_supports_dict_input_with_mixed_ranks():
    discriminator = SumDiscriminator()
    real_data, fake_data = _build_dict_batch()

    output = GAN.compute_wasserstein_loss(
        discriminator,
        real_data=real_data,
        fake_data=fake_data,
        lambda_gp=10.0,
        detach_fake=True,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert isinstance(output["gradient_penalty"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["gradient_penalty"])


def test_compute_wasserstein_loss_matches_fake_minus_real_when_gp_disabled():
    discriminator = SumDiscriminator()
    real_data = torch.full((4, 3), 2.0)
    fake_data = torch.zeros(4, 3)

    output = GAN.compute_wasserstein_loss(
        discriminator,
        real_data=real_data,
        fake_data=fake_data,
        lambda_gp=0.0,
        detach_fake=True,
    )

    expected = output["loss_fake"] - output["loss_real"]
    assert torch.allclose(output["loss"], expected)


def test_losses_reject_positional_data():
    discriminator = SumDiscriminator()
    real_data, fake_data = torch.randn(4, 3), torch.randn(4, 3)
    for loss in (GAN.compute_bce_loss, GAN.compute_soft_bce_loss, GAN.compute_hinge_loss,
                 GAN.compute_lsgan_loss, GAN.compute_wasserstein_loss):
        with pytest.raises(TypeError):
            loss(discriminator, real_data, fake_data)
        assert torch.isfinite(loss(discriminator, real_data=real_data, fake_data=fake_data)["loss"])
    for loss in (GAN.compute_generator_softplus_loss, GAN.compute_generator_mean_loss, GAN.compute_generator_ls_loss):
        with pytest.raises(TypeError):
            loss(discriminator, fake_data)
        assert torch.isfinite(loss(discriminator, fake_data=fake_data)["loss"])


def test_bce_loss_matches_reference_and_reports_detached_diagnostics():
    discriminator = nn.Linear(3, 1)
    real_data, fake_data = torch.randn(5, 3), torch.randn(5, 3)

    output = GAN.compute_bce_loss(discriminator, real_data=real_data, fake_data=fake_data)

    expected_real = torch.nn.functional.softplus(-discriminator(real_data)).mean()
    expected_fake = torch.nn.functional.softplus(discriminator(fake_data)).mean()
    torch.testing.assert_close(output["loss"], expected_real + expected_fake)
    torch.testing.assert_close(output["loss_real"], expected_real.detach())
    assert output["loss"].requires_grad
    assert not output["loss_real"].requires_grad
    assert not output["loss_fake"].requires_grad
    assert isinstance(output["gradient_penalty"], torch.Tensor)
    assert output["gradient_penalty"].item() == 0.0


def test_r1_penalty_is_added_to_the_loss():
    discriminator = nn.Linear(3, 1)
    real_data, fake_data = torch.randn(5, 3), torch.randn(5, 3)

    plain = GAN.compute_bce_loss(discriminator, real_data=real_data.clone(), fake_data=fake_data)
    penalised = GAN.compute_bce_loss(discriminator, real_data=real_data.clone(), fake_data=fake_data, r1_gamma=2.0)

    expected_gp = 0.5 * 2.0 * discriminator.weight.pow(2).sum()
    torch.testing.assert_close(penalised["gradient_penalty"], expected_gp.detach())
    torch.testing.assert_close(penalised["loss"], plain["loss"] + expected_gp)
