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
        discriminator=discriminator,
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
        discriminator=discriminator,
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
        discriminator=discriminator,
        real_data=real_data,
        fake_data=fake_data,
        lambda_gp=0.0,
        detach_fake=True,
    )

    expected = output["loss_fake"] - output["loss_real"]
    assert torch.allclose(output["loss"], expected)
