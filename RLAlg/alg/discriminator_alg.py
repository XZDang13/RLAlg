import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.steps import ValueStep

NNMODEL = nn.Module

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

# If your model returns a struct with `.value`, this helper extracts the tensor.
def _to_logits(x):
    return getattr(x, "value", x)

class DiscriminatorAlg:
    @staticmethod
    def compute_bce_loss(
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        detach_fake: bool = True,
        label_smoothing: float = 0.0,  # e.g., 0.1 -> real=0.9, fake=0.1
        r1_gamma: float = 0.0,         # set >0 to enable R1 gradient penalty on real
    ) -> Dict[str, Any]:
        """
        Logistic/BCE loss for the discriminator (a.k.a. non-saturating loss).
        Uses a numerically stable softplus form. Optionally applies R1 penalty.
        """
        if detach_fake:
            fake_data = fake_data.detach()

        logits_real = discriminator(real_data).value
        logits_fake = discriminator(fake_data).value

        # Stable logistic loss (equivalent to BCEWithLogits):
        # real:  log(1 + exp(-r))  => F.softplus(-r)
        # fake:  log(1 + exp(f))   => F.softplus(f)
        loss_real = F.softplus(-logits_real).mean()
        loss_fake = F.softplus(logits_fake).mean()

        # Optional label smoothing (simple convex blend toward the opposite label)
        # This keeps the stable form while nudging targets away from 0/1.
        if label_smoothing > 0.0:
            # Smooth real->(1-ε), fake->ε. In stable form, implement by mixing the opposite terms.
            eps = float(label_smoothing)
            # mix: L = (1-ε)*L + ε*alternative_L
            loss_real = (1 - eps) * loss_real + eps * F.softplus(logits_real).mean()
            loss_fake = (1 - eps) * loss_fake + eps * F.softplus(-logits_fake).mean()

        # Optional R1 gradient penalty: (γ/2) * E[||∇_x D(x)||^2]
        loss_r1 = torch.tensor(0.0, device=real_data.device)
        if r1_gamma > 0.0:
            real_data = real_data.detach().requires_grad_(True)
            logits_real_r1 = discriminator(real_data).value
            grad_real = torch.autograd.grad(
                outputs=logits_real_r1.sum(),
                inputs=real_data,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_real = grad_real.view(grad_real.size(0), -1)
            r1 = (grad_real.pow(2).sum(dim=1)).mean()
            loss_r1 = (r1_gamma * 0.5) * r1

        total = loss_real + loss_fake + loss_r1

        return total

    @staticmethod
    def compute_hinge_loss(
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        detach_fake: bool = True,
    ) -> Dict[str, Any]:
        """
        Hinge loss (often used in modern GANs like BigGAN):
          L_D = E[relu(1 - D(x_real))] + E[relu(1 + D(x_fake))]
        """
        if detach_fake:
            fake_data = fake_data.detach()

        logits_real = discriminator(real_data).value
        logits_fake = discriminator(fake_data).value

        loss_real = F.relu(1 - logits_real).mean()
        loss_fake = F.relu(1 + logits_fake).mean()
        total = loss_real + loss_fake

        return total

    @staticmethod
    def compute_wasserstein_gp_loss(
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_gp: float = 10.0,
        detach_fake: bool = True,
        gp_center: float = 1.0,  # center for gradient norm (usually 1.0)
    ) -> Dict[str, Any]:
        """
        WGAN-GP discriminator loss:
          L_D = E[D(fake)] - E[D(real)] + λ * E[(||∇_x̂ D(x̂)||_2 - c)^2]
        where x̂ = ε * real + (1-ε) * fake, ε~U[0,1].
        """
        if detach_fake:
            fake_data = fake_data.detach()

        logits_real = discriminator(real_data).value
        logits_fake = discriminator(fake_data).value

        loss_real = -logits_real.mean()
        loss_fake = logits_fake.mean()
        loss_wgan = loss_real + loss_fake

        # Gradient penalty on interpolated samples
        batch_sz = real_data.size(0)
        device = real_data.device
        # Make epsilon broadcastable across non-batch dimensions:
        eps_shape = [batch_sz] + [1] * (real_data.dim() - 1)
        eps = torch.rand(eps_shape, device=device)

        x_hat = eps * real_data + (1 - eps) * fake_data
        x_hat = x_hat.detach().requires_grad_(True)

        d_hat = discriminator(x_hat).value
        # d_hat should be per-sample scalar; sum to get a scalar for grad
        grad_outputs = torch.ones_like(d_hat, device=device)
        gradients = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_sz, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - gp_center) ** 2).mean()
        loss_gp = lambda_gp * gp

        total = loss_wgan + loss_gp

        return total