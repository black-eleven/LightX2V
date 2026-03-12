import gc
import json
import os
from typing import Optional, Union, List, Tuple

import numpy as np
import torch

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, build_wan_model_with_lora
from lightx2v.models.schedulers.mova.scheduler import MovaPairScheduler
# from lightx2v.models.video_encoders.hf.mova.audio_vae.audio_vae import DacVAE
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("mova")
class MovaRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.scheduler = None
        self.video_vae = None
        self.audio_vae = None
        self.vae_cls = WanVAE
        # self.audio_vae_cls = DacVAE

    def init_scheduler(self):
        self.scheduler = MovaPairScheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer() # 包含三个模型 video_dit & video_dit_2 & audio_dit
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()

        # TODO 改成 load dual_tower_bridge
        # if self.config.get("use_upsampler", False):
        #    self.upsampler = self.load_upsampler()

    def load_transformer(self):
        # TODO 需要适配成 MOVA 模型, video_dit & video_dit_2 & audio_dit
        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            lora_configs = self.config.get("lora_configs")
            high_model_kwargs = {
                "model_path": self.high_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_high_noise",
            }
            low_model_kwargs = {
                "model_path": self.low_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_low_noise",
            }
            if not lora_configs:
                high_noise_model = WanModel(**high_model_kwargs)
                low_noise_model = WanModel(**low_model_kwargs)
            else:
                high_noise_model = build_wan_model_with_lora(WanModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                low_noise_model = build_wan_model_with_lora(WanModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")

            return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"])
        else:
            model_struct = MultiModelStruct([None, None], self.config, self.config["boundary"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct


    def load_text_encoder(self):
        pass

    def load_image_encoder(self):
        pass

    def get_vae_parallel(self):
        if isinstance(self.config.get("parallel", False), bool):
            return self.config.get("parallel", False)
        if isinstance(self.config.get("parallel", False), dict):
            return self.config.get("parallel", {}).get("vae_parallel", True)
        return False

    def load_vae(self):
        # load audio_vae(DACVae) & load video vae(AutoencoderKLWan)
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)

        vae_path = os.path.join(self.config["model_path"], "video_vae.pth")

        vae_config = {
            "vae_path": vae_path,
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "use_lightvae": self.config.get("use_lightvae", False),
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
        }

        video_vae = self.vae_cls(**vae_config)

        vae_path = os.path.join(self.config["model_path"], "audio_vae")
        # audio_vae_config
        # audio_vae = self.audio_vae_cls()
        audio_vae = None

        return video_vae, audio_vae

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]

        # TODO 需要对齐当前WanVAE的实现
        video_latent_shape = [
            self.config.video_vae.config.get("z_dim", 16),
            (self.config["target_video_length"] - 1) // self.config.video_vae.config["scale_factor_temporal"] + 1,
            int(target_height) // self.config.video_vae.config["scale_factor_spatial"],
            int(target_width) // self.config.video_vae.config["scale_factor_spatial"],
        ]

        audio_sample_rate = self.audio_vae.config["sample_rate"]
        audio_num_samples = int(audio_sample_rate * self.config["target_video_length"] / self.config["fps"])

        audio_vae_scale_factor = int(np.prod(self.audio_vae.config["encoder_rates"]))
        latent_t = (audio_num_samples - 1) // audio_vae_scale_factor + 1
        audio_latent_shape = (self.config.get("audio_vae_latent_dim"), latent_t)

        return video_latent_shape, audio_latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2av(self):
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)
        self.video_denoise_mask, self.initial_video_latent = self.run_vae_encoder()
        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
        }

    def prepare_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.video_vae.dtype)

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.video_vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.video_vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = self.normalize_video_latents(latent_condition)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    def prepare_audio_latents(
        self,
        audio: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
        num_samples: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        shape = (batch_size, num_channels, latent_t)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents
