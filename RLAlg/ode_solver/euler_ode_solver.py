import inspect
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
        obs: torch.Tensor | dict[str, torch.Tensor],
        current_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        signature = inspect.signature(model.forward)
        call_attempts = (
            ((obs, current_action), {"time": time}),
            ((obs, current_action), {"t": time}),
            ((obs, current_action, time), {}),
            ((obs, current_action), {}),
            ((obs,), {}),
        )
        for args, kwargs in call_attempts:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            # A TypeError from inside forward is a model error. Do not run the
            # model again under a different signature or hide its traceback.
            return EulerODESolver._extract_prediction(model(*args, **kwargs))

        raise TypeError(
            "Could not call model with supported signatures: "
            "(obs, action, time=...), (obs, action, t=...), "
            "(obs, action, time), (obs, action), or (obs)."
        )

    @staticmethod
    def denoise(
        model: NNMODEL,
        flow_steps: int | None = None,
        obs: torch.Tensor | dict[str, torch.Tensor] | None = None,
        init_noise: torch.Tensor | None = None,
        deterministic: bool = False,
        sample_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flow_steps is None:
            flow_steps = sample_steps
        elif sample_steps is not None and sample_steps != flow_steps:
            raise ValueError(
                f"flow_steps and sample_steps must match when both are provided, got {flow_steps} and {sample_steps}."
            )
        if flow_steps is None:
            raise ValueError("flow_steps must be provided.")
        if flow_steps <= 0:
            raise ValueError(f"flow_steps must be > 0, got {flow_steps}.")
        if obs is None:
            raise ValueError("obs must be provided.")
        if not torch.is_tensor(init_noise):
            raise TypeError(f"init_noise must be a torch.Tensor, got {type(init_noise)}.")
        if init_noise.ndim < 1 or init_noise.numel() == 0 or not init_noise.is_floating_point():
            raise ValueError("init_noise must be a non-empty floating-point action tensor.")

        denoised_x = init_noise
        denoised_path = [denoised_x]
        dt = 1.0 / float(flow_steps)

        for step_idx in range(flow_steps):
            t_value = 1.0 - step_idx * dt
            t_tensor = torch.full(
                (*denoised_x.shape[:-1], 1),
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

        return x, torch.stack(denoised_path, dim=-2)
