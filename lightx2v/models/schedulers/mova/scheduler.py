import math
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class FlowMatchScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)

        self.num_train_timesteps = config["num_train_timesteps"]
        self.num_inference_steps = config["num_inference_steps"]
        self.shift = config["shift"]
        self.sigma_max = config["sigma_max"]
        self.sigma_min = config["sigma_min"]
        self.inverse_timesteps = config["inverse_timesteps"]
        self.extra_one_step = config["extra_one_step"]
        self.reverse_sigmas = config["reverse_sigmas"]
        self.exponential_shift = config["exponential_shift"]
        self.exponential_shift_mu = config["exponential_shift_mu"]
        self.shift_terminal = config["shift_terminal"]
        self.set_timesteps(self.num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dynamic_shift_len=None, device=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)

        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class FlowMatchPairScheduler(BaseScheduler):

    def __init__(self, config):
        self.config = config
        self._pair_postprocess_fn = None
        self._pair_postprocess_requires_source = False
        self.pair_timesteps: torch.Tensor | None = None
        self.pair_sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        super().__init__(self, config)

    def set_pair_postprocess(self, fn):
        if fn is not None and not callable(fn):
            raise TypeError("pair_postprocess must be a callable object or None")
        self._pair_postprocess_fn = fn
        self._pair_postprocess_requires_source = False if fn is None else bool(getattr(fn, "_requires_source", False))
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        self._refresh_pair_cache()

    def set_pair_postprocess_by_name(self, name: str | None, **kwargs):
        if name is None or str(name).lower() in ("none", "off", "false", "no"):
            self.set_pair_postprocess(None)
            return
        if name == "dual_sigma_shift":
            visual_shift = float(kwargs.get("visual_shift", self.shift))
            audio_shift = float(kwargs.get("audio_shift", self.shift))
            visual_denoising_strength = float(kwargs.get("visual_denoising_strength", 1.0))
            audio_denoising_strength = float(kwargs.get("audio_denoising_strength", 1.0))
            visual_mu = kwargs.get("visual_exponential_shift_mu", self.exponential_shift_mu)
            audio_mu = kwargs.get("audio_exponential_shift_mu", self.exponential_shift_mu)

            def _dual_sigma_shift(pairs: torch.Tensor, *, source: str):
                if not isinstance(pairs, torch.Tensor):
                    raise TypeError("pairs must be a torch.Tensor")
                if pairs.ndim != 2 or pairs.shape[1] != 2:
                    raise ValueError("pairs must be a torch.Tensor with shape [N, 2]")
                if pairs.shape[0] == 0:
                    raise ValueError("pairs length must be greater than 0")
                if source not in ("timesteps", "sigmas"):
                    raise ValueError("source only supports 'timesteps' or 'sigmas'")

                num_steps = pairs.shape[0]
                device = pairs.device
                dtype = pairs.dtype

                def _build_column(shift_value: float, denoising_strength: float, mu_override):
                    if shift_value <= 0:
                        raise ValueError("shift must be a positive number")
                    if denoising_strength <= 0:
                        raise ValueError("denoising_strength must be a positive number")

                    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
                    if self.extra_one_step:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps + 1, device=device, dtype=dtype)[:-1]
                    else:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps, device=device, dtype=dtype)

                    if self.inverse_timesteps:
                        base = torch.flip(base, dims=[0])

                    if self.exponential_shift:
                        mu_value = mu_override
                        if mu_value is None:
                            raise RuntimeError("exponential_shift is enabled but exponential_shift_mu is not provided")
                        exp_mu = math.exp(float(mu_value))
                        base = exp_mu / (exp_mu + (1 / base - 1))
                    else:
                        base = shift_value * base / (1 + (shift_value - 1) * base)

                    if self.shift_terminal is not None:
                        one_minus_z = 1 - base
                        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
                        base = 1 - (one_minus_z / scale_factor)

                    if self.reverse_sigmas:
                        base = 1 - base

                    if source == "timesteps":
                        return base * self.num_train_timesteps
                    return base

                visual_column = _build_column(visual_shift, visual_denoising_strength, visual_mu)
                audio_column = _build_column(audio_shift, audio_denoising_strength, audio_mu)
                return torch.stack([visual_column, audio_column], dim=1)

            _dual_sigma_shift._requires_source = True
            self.set_pair_postprocess(_dual_sigma_shift)
            return
        raise ValueError(f"Unsupported pair postprocessing name: {name}")

    def _make_pairs_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
        if not isinstance(vec, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if vec.ndim != 1:
            raise ValueError("input vector must be a one-dimensional tensor")
        if vec.numel() == 0:
            raise ValueError("input vector length must be greater than 0")

        # Default each row is (t, t), shape [N, 2]
        pairs = torch.stack([vec, vec], dim=1)
        return pairs

    def get_pairs(self, source: str = "timesteps") -> torch.Tensor:
        if source == "timesteps":
            pairs = self.pair_timesteps
        elif source == "sigmas":
            pairs = self.pair_sigmas
        else:
            raise ValueError("source only supports 'timesteps' or 'sigmas'")

        if pairs is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")

        return pairs

    @property
    def visual_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return self.pair_timesteps[:, 0]

    @property
    def audio_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return self.pair_timesteps[:, 1]

    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)
        self._refresh_pair_cache()

    def timestep_to_sigma(self, timestep: torch.Tensor | float) -> torch.Tensor:
        t = timestep
        t_value = float(t)
        t_cpu = torch.tensor(t_value)
        idx = torch.argmin((self.train_timesteps - t_cpu).abs())
        return self.train_sigmas[idx]

    def step_from_to(self, model_output: torch.Tensor, timestep_from: torch.Tensor, timestep_to: torch.Tensor | None, sample: torch.Tensor) -> torch.Tensor:
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            sigma_to = torch.tensor(1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0, device=sigma_from.device, dtype=sigma_from.dtype)
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        prev_sample = sample + model_output * (sigma_to - sigma_from)
        return prev_sample

    def _refresh_pair_cache(self) -> None:
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")

        def _apply_postprocess(pairs: torch.Tensor, source: str) -> torch.Tensor:
            if self._pair_postprocess_fn is None:
                return pairs
            if self._pair_postprocess_requires_source:
                modified = self._pair_postprocess_fn(pairs, source=source)
            else:
                modified = self._pair_postprocess_fn(pairs)
            if not isinstance(modified, torch.Tensor):
                raise TypeError("pair_postprocess return value must be a torch.Tensor")
            if modified.shape != pairs.shape:
                raise ValueError("pair_postprocess return tensor shape must be the same as the input")
            return modified

        base_pairs_timesteps = self._make_pairs_from_vector(self.timesteps)
        base_pairs_sigmas = self._make_pairs_from_vector(self.sigmas)

        self.pair_timesteps = _apply_postprocess(base_pairs_timesteps, "timesteps")
        self.pair_sigmas = _apply_postprocess(base_pairs_sigmas, "sigmas")
