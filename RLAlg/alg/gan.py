import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.steps import ValueStep

NNMODEL = nn.Module

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

class GAN:
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

        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
            
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

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
            logits_real_r1 = discriminator(real_data)
            if isinstance(logits_real_r1, ValueStep):
                logits_real_r1 = logits_real_r1.value

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
        r1_gamma: float = 0.0,
    ) -> torch.Tensor:
        """
        Hinge GAN discriminator loss:
        L_D = E[relu(1 - D(x_real))] + E[relu(1 + D(x_fake))]
        Optionally adds R1 gradient penalty on real samples.
        """
        if detach_fake:
            fake_data = fake_data.detach()

        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        from ..nn.steps import ValueStep
        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_real = F.relu(1 - logits_real).mean()
        loss_fake = F.relu(1 + logits_fake).mean()
        loss_r1 = torch.tensor(0.0, device=real_data.device)

        if r1_gamma > 0.0:
            real_data_r1 = real_data.detach().requires_grad_(True)
            d_real_r1 = discriminator(real_data_r1)
            if isinstance(d_real_r1, ValueStep):
                d_real_r1 = d_real_r1.value

            grad_real = torch.autograd.grad(
                outputs=d_real_r1.sum(),
                inputs=real_data_r1,
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
    def compute_lsgan_loss(
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        detach_fake: bool = True,
        real_target: float = 1.0,
        fake_target: float = 0.0,
        r1_gamma: float = 0.0,   # optional R1 on real; leave 0.0 to disable
    ) -> torch.Tensor:
        """
        LSGAN discriminator loss (Mao et al. 2017):
          L_D = 0.5 * E[(D(x_real) - a)^2] + 0.5 * E[(D(x_fake) - b)^2]
        where typical targets are a=1 (real), b=0 (fake).
        D should output an unconstrained scalar (no sigmoid).
        """
        if detach_fake:
            fake_data = fake_data.detach()

        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        # Unwrap ValueStep if your discriminator returns it
        from ..nn.steps import ValueStep  # keep local to avoid circulars if needed
        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        # Mean squared error to targets
        loss_real = 0.5 * (logits_real - real_target).pow(2).mean()
        loss_fake = 0.5 * (logits_fake - fake_target).pow(2).mean()
        loss_r1 = torch.tensor(0.0, device=real_data.device)

        # Optional R1 regularizer on real samples
        if r1_gamma > 0.0:
            real_data_r1 = real_data.detach().requires_grad_(True)
            d_real_r1 = discriminator(real_data_r1)
            if isinstance(d_real_r1, ValueStep):
                d_real_r1 = d_real_r1.value
            grad_real = torch.autograd.grad(
                outputs=d_real_r1.sum(),
                inputs=real_data_r1,
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
    def compute_wasserstein_loss(
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_gp: float = 10.0,
        detach_fake: bool = True,
        gp_center: float = 1.0,
    ) -> torch.Tensor:
        """
        WGAN-GP discriminator loss:
        L_D = E[D(fake)] - E[D(real)] + λ * E[(||∇_x̂ D(x̂)||_2 - c)^2]
        """
        if detach_fake:
            fake_data = fake_data.detach()

        device = real_data.device

        # Forward on real/fake
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)

        from ..nn.steps import ValueStep
        if isinstance(d_real, ValueStep):
            d_real = d_real.value
        if isinstance(d_fake, ValueStep):
            d_fake = d_fake.value

        loss_wgan = d_fake.mean() - d_real.mean()

        # Interpolate
        bsz = real_data.size(0)
        eps_shape = [bsz] + [1] * (real_data.dim() - 1)  # broadcastable
        eps = torch.rand(eps_shape, device=device, dtype=real_data.dtype)
        x_hat = (eps * real_data + (1.0 - eps) * fake_data).detach().requires_grad_(True)

        d_hat = discriminator(x_hat)
        if isinstance(d_hat, ValueStep):
            d_hat = d_hat.value

        # grad_outputs should match d_hat's device/dtype/shape
        grad_outputs = torch.ones_like(d_hat)

        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(bsz, -1)
        grad_norm = grads.norm(2, dim=1)
        gp = (grad_norm - gp_center).pow(2).mean()
        loss_gp = lambda_gp * gp

        return loss_wgan + loss_gp
    
    @staticmethod
    def compute_generator_softplus_loss(
        discriminator,
        fake_data: torch.Tensor,
    ) -> Dict[str, Any]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = F.softplus(-logits_fake).mean()

        return loss_gen
    
    @staticmethod
    def compute_generator_mean_loss(
        discriminator,
        fake_data: torch.Tensor,
    ) -> Dict[str, Any]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = -logits_fake.mean()

        return loss_gen
    
    @staticmethod
    def compute_generator_ls_loss(
        discriminator,
        fake_data: torch.Tensor,
    ) -> Dict[str, Any]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = 0.5 * (logits_fake - 1.0).pow(2).mean()

        return loss_gen
    
