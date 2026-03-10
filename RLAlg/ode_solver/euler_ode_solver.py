import torch
import torch.nn as nn

from RLAlg.nn.steps import ValueStep

NNMODEL = nn.Module

class EulerODESolver:
    @staticmethod
    def _extract_prediction(model_output: ValueStep | torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(model_output, tuple):
            if len(model_output) == 0:
                raise ValueError("Model output tuple must be non-empty.")
            model_output = model_output[0]

        if isinstance(model_output, ValueStep):
            prediction = model_output.value
        elif torch.is_tensor(model_output):
            prediction = model_output
        else:
            raise TypeError(
                "Model output must be ValueStep or Tensor, "
                f"got {type(model_output)}."
            )

        return prediction

    @staticmethod
    def _call_model(
        model: NNMODEL,
        obs: torch.Tensor,
        current_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        call_errors: list[TypeError] = []
        call_attempts = (
            lambda: model(obs, current_action, time=time),
            lambda: model(obs, current_action, time),
            lambda: model(obs, current_action),
            lambda: model(obs),
        )

        for call_fn in call_attempts:
            try:
                return EulerODESolver._extract_prediction(call_fn())
            except TypeError as error:
                call_errors.append(error)

        raise TypeError(
            "Could not call model with supported signatures: "
            "(obs, action, time=...), (obs, action, time), (obs, action), or (obs)."
        ) from call_errors[-1]

    @staticmethod
    def denoise(
        model: NNMODEL,
        sample_steps: int,
        obs: torch.Tensor,
        init_noise: torch.Tensor,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be > 0, got {sample_steps}.")
        if not torch.is_tensor(init_noise):
            raise TypeError(f"init_noise must be a torch.Tensor, got {type(init_noise)}.")

        denoised_x = init_noise
        denoised_path = [denoised_x]
        dt = 1.0 / float(sample_steps)

        batch_size = denoised_x.shape[0] if denoised_x.ndim > 0 else 1
        for step_idx in range(sample_steps):
            t_value = 1.0 - step_idx * dt
            t_tensor = torch.full(
                (batch_size, 1),
                t_value,
                dtype=denoised_x.dtype,
                device=denoised_x.device,
            )

            prediction = EulerODESolver._call_model(model, obs, denoised_x, t_tensor)
            if prediction.shape != denoised_x.shape:
                raise ValueError(
                    "Model prediction shape must match current action shape, "
                    f"got {tuple(prediction.shape)} and {tuple(denoised_x.shape)}."
                )

            denoised_x = denoised_x - prediction * dt
            denoised_path.append(denoised_x)

        x = denoised_x
        if not deterministic:
            x = x + torch.randn_like(x) * dt

        return x, torch.stack(denoised_path, dim=1)
