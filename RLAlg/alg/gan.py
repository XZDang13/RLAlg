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
    def compute_gradient(inputs:torch.Tensor|list[torch.Tensor],
                         outputs:torch.Tensor,
                        ) -> torch.Tensor:

        grads = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        return torch.cat([torch.flatten(grad, 1) for grad in grads], dim=1)
    
    @staticmethod
    def compute_r1_gradient_penalty(inputs:torch.Tensor|dict[str, torch.Tensor],
                                    outputs:torch.Tensor,
                                    gamma:float,
                                    grad_penalty_on: list[str]|None = None) -> torch.Tensor:
        selected_inputs = None

        if isinstance(inputs, dict):
            if grad_penalty_on is None:
                selected_inputs = inputs.values()
            else:
                selected_inputs = [inputs[key] for key in grad_penalty_on]
        else:
            selected_inputs = inputs

        grads = GAN.compute_gradient(selected_inputs, outputs)
        
        grad_sq = grads.pow(2).sum(dim=1)
        gp = 0.5 * gamma * grad_sq.mean()

        return gp
    
    @staticmethod
    def compute_wp_gradient_penalty(inputs:torch.Tensor|dict[str, torch.Tensor],
                                    outputs:torch.Tensor,
                                    lambda_gp:float,
                                    gp_center:float,
                                    grad_penalty_on: list[str]|None = None) -> torch.Tensor:
        
        selected_inputs = None

        if isinstance(inputs, dict):
            if grad_penalty_on is None:
                selected_inputs = inputs.values()
            else:
                selected_inputs = [inputs[key] for key in grad_penalty_on]
        else:
            selected_inputs = inputs

        grads = GAN.compute_gradient(selected_inputs, outputs)

        grad_norm = grads.norm(2, dim=1)
        gp = lambda_gp * ((grad_norm - gp_center) ** 2).mean()

        return gp

    @staticmethod
    def compute_bce_loss(
        discriminator,
        real_data: torch.Tensor|dict[str, torch.Tensor],
        fake_data: torch.Tensor|dict[str, torch.Tensor],
        detach_fake: bool = True,
        r1_gamma: float = 0.0,         # set >0 to enable R1 gradient penalty on real
        grad_penalty_on: list[str]|None = None
    ) -> dict[str, torch.Tensor]:
        """
        Logistic/BCE loss for the discriminator (a.k.a. non-saturating loss).
        Uses a numerically stable softplus form. Optionally applies R1 penalty.
        """
        if detach_fake:
            fake_data = fake_data.detach()

        if r1_gamma != 0:
            real_data.requires_grad_(True)

        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
            
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        true_label = torch.ones_like(logits_real, device=logits_real.device)
        fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)

        loss_real = F.binary_cross_entropy_with_logits(logits_real, true_label)
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_label)

        # Optional R1 gradient penalty: (γ/2) * E[||∇_x D(x)||^2]
        loss_r1 = 0
        if r1_gamma > 0.0:
            loss_r1 = GAN.compute_r1_gradient_penalty(real_data, logits_real, r1_gamma, grad_penalty_on)

        total = loss_real + loss_fake + loss_r1

        return {
            "loss": total,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "gradient_penalty": loss_r1,
        }
    
    @staticmethod
    def compute_soft_bce_loss(
        discriminator,
        real_data: torch.Tensor|dict[str, torch.Tensor],
        fake_data: torch.Tensor|dict[str, torch.Tensor],
        detach_fake: bool = True,
        label_smoothing: float = 0.0,  # e.g., 0.1 -> real=0.9, fake=0.1
        random_smoothing: bool = False,
        one_sided_smoothing: bool = False,
        r1_gamma: float = 0.0,         # set >0 to enable R1 gradient penalty on real
        grad_penalty_on: list[str]|None = None
    ) -> dict[str, torch.Tensor]:
        """
        Logistic/BCE loss for the discriminator (a.k.a. non-saturating loss).
        Uses a numerically stable softplus form. Optionally applies R1 penalty.
        """
        if detach_fake:
            fake_data = fake_data.detach()

        if r1_gamma != 0:
            real_data.requires_grad_(True)

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
            if random_smoothing:
                eps = torch.rand(1).item() * label_smoothing
            else:
                eps = float(label_smoothing)
            # mix: L = (1-ε)*L + ε*alternative_L
            loss_real = (1 - eps) * loss_real + eps * F.softplus(logits_real).mean()
            if not one_sided_smoothing:
                loss_fake = (1 - eps) * loss_fake + eps * F.softplus(-logits_fake).mean()

        # Optional R1 gradient penalty: (γ/2) * E[||∇_x D(x)||^2]
        loss_r1 = 0
        if r1_gamma > 0.0:
            loss_r1 = GAN.compute_r1_gradient_penalty(real_data, logits_real, r1_gamma, grad_penalty_on)

        total = loss_real + loss_fake + loss_r1

        return {
            "loss": total,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "gradient_penalty": loss_r1,
        }

    @staticmethod
    def compute_hinge_loss(
        discriminator,
        real_data: torch.Tensor|dict[str, torch.Tensor],
        fake_data: torch.Tensor|dict[str, torch.Tensor],
        detach_fake: bool = True,
        r1_gamma: float = 0.0,
        grad_penalty_on: list[str]|None = None
    ) -> dict[str, torch.Tensor]:
        """
        Hinge GAN discriminator loss:
        L_D = E[relu(1 - D(x_real))] + E[relu(1 + D(x_fake))]
        Optionally adds R1 gradient penalty on real samples.
        """
        if detach_fake:
            fake_data = fake_data.detach()

        if r1_gamma != 0:
            real_data.requires_grad_(True)

        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        from ..nn.steps import ValueStep
        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_real = F.relu(1 - logits_real).mean()
        loss_fake = F.relu(1 + logits_fake).mean()

        loss_r1 = 0
        if r1_gamma > 0.0:
            loss_r1 = GAN.compute_r1_gradient_penalty(real_data, logits_real, r1_gamma, grad_penalty_on)

        total = loss_real + loss_fake + loss_r1

        return {
            "loss": total,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "gradient_penalty": loss_r1,
        }
    
    @staticmethod
    def compute_lsgan_loss(
        discriminator,
        real_data: torch.Tensor|dict[str, torch.Tensor],
        fake_data: torch.Tensor|dict[str, torch.Tensor],
        detach_fake: bool = True,
        real_target: float = 1.0,
        fake_target: float = 0.0,
        r1_gamma: float = 0.0,   # optional R1 on real; leave 0.0 to disable
        grad_penalty_on: list[str]|None = None
    ) -> dict[str, torch.Tensor]:
        """
        LSGAN discriminator loss (Mao et al. 2017):
          L_D = 0.5 * E[(D(x_real) - a)^2] + 0.5 * E[(D(x_fake) - b)^2]
        where typical targets are a=1 (real), b=0 (fake).
        D should output an unconstrained scalar (no sigmoid).
        """
        if detach_fake:
            fake_data = fake_data.detach()

        if r1_gamma != 0:
            real_data.requires_grad_(True)

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

        loss_r1 = 0
        if r1_gamma > 0.0:
            loss_r1 = GAN.compute_r1_gradient_penalty(real_data, logits_real, r1_gamma, grad_penalty_on)

        total = loss_real + loss_fake + loss_r1

        return {
            "loss": total,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "gradient_penalty": loss_r1,
        }

    @staticmethod
    def compute_wasserstein_loss(
        discriminator,
        real_data: torch.Tensor|dict[str, torch.Tensor],
        fake_data: torch.Tensor|dict[str, torch.Tensor],
        lambda_gp: float = 10.0,
        detach_fake: bool = True,
        gp_center: float = 1.0,
        grad_penalty_on: list[str]|None = None
    ) -> dict[str, torch.Tensor]:
        """
        WGAN-GP discriminator loss:
        L_D = E[D(fake)] - E[D(real)] + λ * E[(||∇_x̂ D(x̂)||_2 - c)^2]
        """
        if detach_fake:
            fake_data = fake_data.detach()

        device = real_data.device

        # Forward on real/fake
        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data)

        from ..nn.steps import ValueStep
        if isinstance(logits_real, ValueStep):
            logits_real = logits_real.value
        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_wgan = logits_real.mean() - logits_fake.mean()

        # Interpolate

        if isinstance(real_data, torch.Tensor):
            bsz = real_data.size(0)
            eps_shape = [bsz] + [1] * (real_data.dim() - 1)  # broadcastable
            eps = torch.rand(eps_shape, device=device, dtype=real_data.dtype)
            x_hat = (eps * real_data + (1.0 - eps) * fake_data).detach().requires_grad_(True)
        elif isinstance(real_data, dict):
            bsz = next(iter(real_data.values())).size(0)
            eps_shape = [bsz] + [1] * (next(iter(real_data.values())).dim() - 1)
            eps = torch.rand(eps_shape, device=device, dtype=next(iter(real_data.values())).dtype)
            x_hat = {}
            for key in real_data.keys():
                x_hat[key] = (eps * real_data[key] + (1.0 - eps) * fake_data[key]).detach().requires_grad_(True)

        d_hat = discriminator(x_hat)
        if isinstance(d_hat, ValueStep):
            d_hat = d_hat.value

        loss_gp = GAN.compute_wp_gradient_penalty(x_hat, d_hat, lambda_gp, gp_center, grad_penalty_on)

        total = loss_wgan + loss_gp

        return {
            "loss": total,
            "loss_real": logits_real.mean(),
            "loss_fake": logits_fake.mean(),
            "gradient_penalty": loss_gp,
        }
    
    @staticmethod
    def compute_generator_softplus_loss(
        discriminator,
        fake_data: torch.Tensor|dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = F.softplus(-logits_fake).mean()

        return {
            "loss": loss_gen
        }
    
    @staticmethod
    def compute_generator_mean_loss(
        discriminator,
        fake_data: torch.Tensor|dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = -logits_fake.mean()

        return {
            "loss": loss_gen
        }
    
    @staticmethod
    def compute_generator_ls_loss(
        discriminator,
        fake_data: torch.Tensor|dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits_fake = discriminator(fake_data)

        if isinstance(logits_fake, ValueStep):
            logits_fake = logits_fake.value

        loss_gen = 0.5 * (logits_fake - 1.0).pow(2).mean()

        return {
            "loss": loss_gen
        }
    
