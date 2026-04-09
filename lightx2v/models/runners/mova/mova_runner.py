import gc
import json
import os
import wave
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace


import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.mova.bridge import DualTowerConditionalBridge
from lightx2v.models.networks.mova.model import MovaAudioModel, MovaVideoModel
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, build_wan_model_with_lora, WanRunner
from lightx2v.models.schedulers.mova.scheduler import MovaPairScheduler, MovaAudioPairScheduler
from lightx2v.models.video_encoders.hf.mova.audio_vae.audio_vae import DacVAE
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.ltx2_media_io import encode_video as save_video
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("mova")
class MovaRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_cls = WanVAE
        self.audio_vae_cls = DacVAE
        self.high_noise_model_path = os.path.join(config["model_path"], "video_dit.safetensors")
        self.low_noise_model_path = os.path.join(config["model_path"], "video_dit_2.safetensors")

    def init_scheduler(self):
        self.scheduler = MovaPairScheduler(self.config)
        self.audio_scheduler = MovaAudioPairScheduler(self.scheduler)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model, self.audio_model = self.load_transformer() # 包含三个模型 video_dit & video_dit_2 & audio_dit
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.audio_model.set_scheduler(self.audio_scheduler)

        self.text_encoders = self.load_text_encoder()
        self.vae_encoder, self.audio_vae = self.load_vae()

        self.dual_tower_bridge = self.load_dual_tower_bridge()

    def load_dual_tower_bridge(self):
        bridge_path = os.path.join(self.config["model_path"], "dual_tower_bridge")
        if not os.path.exists(bridge_path):
            logger.warning(f"dual_tower_bridge not found at {bridge_path}, skip bridge loading.")
            return None
        with open(os.path.join(bridge_path, "config.json"), "r") as f:
            bridge_config = json.load(f)
        dual_tower_bridge = DualTowerConditionalBridge(**bridge_config)
        self._load_dual_tower_bridge_weights(dual_tower_bridge, bridge_path)
        dual_tower_bridge.eval()
        return dual_tower_bridge

    def _load_dual_tower_bridge_weights(self, dual_tower_bridge, bridge_path):
        state_dict = None
        safetensors_path = None
        pt_path = None

        for filename in ["model.safetensors", "diffusion_pytorch_model.safetensors"]:
            candidate = os.path.join(bridge_path, filename)
            if os.path.exists(candidate):
                safetensors_path = candidate
                break

        if safetensors_path is not None:
            from safetensors.torch import load_file

            logger.info(f"Loading MOVA dual_tower_bridge weights from {safetensors_path}")
            state_dict = load_file(safetensors_path, device="cpu")
        else:
            for filename in ["pytorch_model.bin", "model.pt"]:
                candidate = os.path.join(bridge_path, filename)
                if os.path.exists(candidate):
                    pt_path = candidate
                    break
            if pt_path is not None:
                logger.info(f"Loading MOVA dual_tower_bridge weights from {pt_path}")
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        if state_dict is None:
            raise FileNotFoundError(
                f"Cannot find dual_tower_bridge weight file under {bridge_path}. "
                "Expected one of: model.safetensors, diffusion_pytorch_model.safetensors, "
                "pytorch_model.bin, model.pt"
            )

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict = self._normalize_bridge_state_dict_keys(state_dict, dual_tower_bridge)
        model_state_keys = set(dual_tower_bridge.state_dict().keys())
        loaded_key_count = sum(1 for k in state_dict.keys() if k in model_state_keys)
        if loaded_key_count == 0:
            raise RuntimeError(
                "MOVA dual_tower_bridge weights were found but no parameter names matched. "
                "Please check checkpoint compatibility (e.g., key prefixes)."
            )

        missing, unexpected = dual_tower_bridge.load_state_dict(state_dict, strict=False)
        logger.info(
            f"MOVA dual_tower_bridge matched {loaded_key_count}/{len(model_state_keys)} params before strict=False loading."
        )
        if missing:
            logger.warning(f"MOVA dual_tower_bridge missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"MOVA dual_tower_bridge unexpected keys: {len(unexpected)}")

    def _normalize_bridge_state_dict_keys(self, state_dict, bridge_module):
        if not isinstance(state_dict, dict):
            return state_dict

        model_state_keys = set(bridge_module.state_dict().keys())
        if any(k in model_state_keys for k in state_dict.keys()):
            return state_dict

        candidate_prefixes = [
            "dual_tower_bridge.",
            "module.dual_tower_bridge.",
            "model.dual_tower_bridge.",
            "module.",
            "model.",
        ]
        for prefix in candidate_prefixes:
            normalized = {}
            changed = False
            for k, v in state_dict.items():
                if isinstance(k, str) and k.startswith(prefix):
                    normalized[k[len(prefix):]] = v
                    changed = True
                else:
                    normalized[k] = v
            if changed and any(k in model_state_keys for k in normalized.keys()):
                logger.info(f"Normalized dual_tower_bridge state_dict keys by stripping prefix: {prefix}")
                return normalized

        return state_dict

    def load_transformer(self):
        audio_config = deepcopy(self.config)
        audio_config["rope_type"] = audio_config.get("audio_rope_type", audio_config.get("rope_type", "flashinfer"))
        audio_config["num_layers"] = 30
        # MOVA audio tower uses a smaller hidden size than the video tower.
        # Do not reuse video num_heads/dim directly, otherwise reshape in self-attn breaks.
        audio_hidden_dim = int(audio_config.get("audio_hidden_dim", 1536))
        audio_head_dim = int(audio_config.get("audio_head_dim", 128))
        audio_num_heads = int(audio_config.get("audio_num_heads", max(1, audio_hidden_dim // audio_head_dim)))
        audio_config["dim"] = audio_hidden_dim
        audio_config["num_heads"] = audio_num_heads
        audio_model_kwargs = {"model_path": os.path.join(self.config["model_path"], "audio_dit.safetensors"), "config": audio_config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            audio_model = MovaAudioModel(**audio_model_kwargs)
        else:
            audio_model = build_wan_model_with_lora(MovaAudioModel, self.config, audio_model_kwargs, lora_configs, model_type="mova_audio")

        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            lora_configs = self.config.get("lora_configs")
            high_model_kwargs = {
                "model_path": self.high_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "mova_video",
            }
            low_model_kwargs = {
                "model_path": self.low_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "mova_video",
            }
            if not lora_configs:
                high_noise_model = MovaVideoModel(**high_model_kwargs)
                low_noise_model = MovaVideoModel(**low_model_kwargs)
            else:
                high_noise_model = build_wan_model_with_lora(MovaVideoModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                low_noise_model = build_wan_model_with_lora(MovaVideoModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")

            return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"]), audio_model
        else:
            model_struct = MultiModelStruct([None, None], self.config, self.config["boundary"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct, audio_model

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

        audio_vae_path = os.path.join(self.config["model_path"], "audio_vae")
        with open(os.path.join(audio_vae_path, "config.json"), "r") as f:
            audio_vae_config = json.load(f)
            self.audio_vae_config = audio_vae_config
        audio_vae = self.audio_vae_cls(**audio_vae_config)
        self._load_audio_vae_weights(audio_vae, audio_vae_path)

        return video_vae, audio_vae

    def _load_audio_vae_weights(self, audio_vae, audio_vae_path):
        state_dict = None
        safetensors_path = None
        pt_path = None

        for filename in ["model.safetensors", "diffusion_pytorch_model.safetensors"]:
            candidate = os.path.join(audio_vae_path, filename)
            if os.path.exists(candidate):
                safetensors_path = candidate
                break

        if safetensors_path is not None:
            from safetensors.torch import load_file

            logger.info(f"Loading MOVA audio_vae weights from {safetensors_path}")
            state_dict = load_file(safetensors_path, device="cpu")
        else:
            for filename in ["pytorch_model.bin", "model.pt"]:
                candidate = os.path.join(audio_vae_path, filename)
                if os.path.exists(candidate):
                    pt_path = candidate
                    break
            if pt_path is not None:
                logger.info(f"Loading MOVA audio_vae weights from {pt_path}")
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        if state_dict is None:
            raise FileNotFoundError(
                f"Cannot find audio_vae weight file under {audio_vae_path}. "
                "Expected one of: model.safetensors, diffusion_pytorch_model.safetensors, "
                "pytorch_model.bin, model.pt"
            )

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict = self._normalize_audio_vae_state_dict_keys(state_dict, audio_vae)
        model_state_keys = set(audio_vae.state_dict().keys())
        loaded_key_count = sum(1 for k in state_dict.keys() if k in model_state_keys)
        if loaded_key_count == 0:
            raise RuntimeError(
                "MOVA audio_vae weights were found but no parameter names matched. "
                "Please check checkpoint compatibility (e.g., key prefixes)."
            )

        missing, unexpected = audio_vae.load_state_dict(state_dict, strict=False)
        logger.info(
            f"MOVA audio_vae matched {loaded_key_count}/{len(model_state_keys)} params before strict=False loading."
        )
        if missing:
            logger.warning(f"MOVA audio_vae missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"MOVA audio_vae unexpected keys: {len(unexpected)}")
        audio_vae.eval()

    def _normalize_audio_vae_state_dict_keys(self, state_dict, audio_vae_module):
        if not isinstance(state_dict, dict):
            return state_dict

        model_state_keys = set(audio_vae_module.state_dict().keys())
        if any(k in model_state_keys for k in state_dict.keys()):
            return state_dict

        candidate_prefixes = [
            "audio_vae.",
            "module.audio_vae.",
            "model.audio_vae.",
            "module.",
            "model.",
        ]
        for prefix in candidate_prefixes:
            normalized = {}
            changed = False
            for k, v in state_dict.items():
                if isinstance(k, str) and k.startswith(prefix):
                    normalized[k[len(prefix):]] = v
                    changed = True
                else:
                    normalized[k] = v
            if changed and any(k in model_state_keys for k in normalized.keys()):
                logger.info(f"Normalized audio_vae state_dict keys by stripping prefix: {prefix}")
                return normalized

        return state_dict

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]

        video_latent_shape = [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            int(target_height) // self.config["vae_stride"][1],
            int(target_width) // self.config["vae_stride"][2],
        ]

        audio_sample_rate = self.audio_vae_config["sample_rate"]
        audio_num_samples = int(audio_sample_rate * self.config["target_video_length"] / self.config["fps"])

        audio_vae_scale_factor = int(np.prod(self.audio_vae_config["encoder_rates"]))
        latent_t = (audio_num_samples - 1) // audio_vae_scale_factor + 1
        audio_latent_shape = (self.audio_vae_config["latent_dim"], latent_t)

        return video_latent_shape, audio_latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2av(self):
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        self.input_info.latent_shape = self.input_info.video_latent_shape
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2av(self):
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        self.input_info.latent_shape = self.input_info.video_latent_shape
        text_encoder_output = self.run_text_encoder(self.input_info)

        first_frame, _ = self.read_image_input(self.input_info.image_path)
        if self.input_info.last_frame_path:
            last_frame, _ = self.read_image_input(self.input_info.last_frame_path)
        else:
            last_frame = None

        vae_encode_out, latent_shape = self.run_vae_encoder(first_frame, last_frame)
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info

        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "vae_encoder_out": vae_encode_out,
            }
        }

    @ProfilingContext4DebugL1("Run VAE Decoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_decode_duration, metrics_labels=["MovaRunner"])
    def run_vae_decoder(self, v_latent, a_latent):
        video = self.vae_encoder.decode(v_latent.to(GET_DTYPE()))
        audio_device = self.audio_vae.device if hasattr(self.audio_vae, "device") else a_latent.device
        # Align with official MOVA: decode audio under fp32 autocast on CUDA.
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if getattr(audio_device, "type", "") == "cuda" else nullcontext()
        with amp_ctx:
            audio = self.audio_vae.decode(a_latent.unsqueeze(0).to(device=audio_device, dtype=torch.float32))
        return video, audio

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(prompt))
        neg_prompt = input_info.negative_prompt

        if self.config.get("enable_cfg", False) and self.config["cfg_parallel"]:
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            cfg_p_rank = dist.get_rank(cfg_p_group)
            if cfg_p_rank == 0:
                context = self.text_encoders[0].infer([prompt])
                context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
                text_encoder_output = {"context": context}
            else:
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
                text_encoder_output = {"context_null": context_null}
        else:
            context = self.text_encoders[0].infer([prompt])
            context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
            if self.config.get("enable_cfg", False):
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
            else:
                context_null = None
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return text_encoder_output

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()
        self._bridge_prev_audio_layers = {}
        self._bridge_video_grid_size = None

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model, self.audio_model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
            self.audio_model.set_scheduler(self.audio_scheduler)

        # Image conditioning (if any) is already prepared in run_input_encoder
        # and stored in self.video_denoise_mask and self.initial_video_latent
        self.scheduler.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape,
                               audio_latent_shape=self.input_info.audio_latent_shape)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)
        for segment_idx in range(self.video_segment_num):
            logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                    f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["DefaultRunner"],
            ):
                self.check_stop()
                # 1. default do nothing
                self.init_run_segment(segment_idx)
                # 2. main inference loop
                v_latent, a_latent = self.run_segment(segment_idx)

                # 3. vae decoder
                self.gen_video, self.gen_audio = self.run_vae_decoder(v_latent, a_latent)

                # 4. default do nothing
                self.end_run_segment(segment_idx)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def run_segment(self, segment_idx=0, stage_name=None):
        infer_steps = self.model.scheduler.infer_steps
        self._set_bridge_callbacks(None, None)

        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            with ProfilingContext4DebugL1(
                f"Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                if self.video_segment_num == 1:
                    self.check_stop()

                # Use stage_name for logging if provided, otherwise use default
                if stage_name:
                    logger.info(f"==> {stage_name} step_index: {step_index + 1} / {infer_steps}")
                else:
                    logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4DebugL1("🚀 infer_dual_sync"):
                    self._infer_dual_tower(self.inputs)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()

                # Progress callback only for regular segments (not upsampler)
                if self.progress_callback and segment_idx is not None:
                    current_step = segment_idx * infer_steps + step_index + 1
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)

        self._set_bridge_callbacks(None, None)
        return self.model.scheduler.latents, self.model.scheduler.audio_latents

    def _get_active_video_model(self):
        if isinstance(self.model, MultiModelStruct):
            self.model.get_current_model_index()
            cur = self.model.model[self.model.cur_model_index]
            if cur is None:
                return None
            return cur
        return self.model

    def _infer_dual_tower(self, inputs):
        # Resolve active video model once per step to avoid duplicate
        # get_current_model_index() calls in CFG positive/negative branches.
        video_model = self._get_active_video_model()
        if self.config.get("enable_cfg", False):
            v_pos, a_pos = self._infer_dual_tower_cond(inputs, infer_condition=True, video_model=video_model)
            v_neg, a_neg = self._infer_dual_tower_cond(inputs, infer_condition=False, video_model=video_model)
            cfg_scale = self.model.scheduler.sample_guide_scale
            v_diff = (v_pos - v_neg)
            self.model.scheduler.noise_pred = v_neg + cfg_scale * v_diff
            self.audio_model.scheduler.noise_pred = a_neg + cfg_scale * (a_pos - a_neg)
        else:
            v_pred, a_pred = self._infer_dual_tower_cond(inputs, infer_condition=True, video_model=video_model)
            self.model.scheduler.noise_pred = v_pred
            self.audio_model.scheduler.noise_pred = a_pred

    def _infer_dual_tower_cond(self, inputs, infer_condition=True, video_model=None):
        if video_model is None:
            video_model = self._get_active_video_model()
        if video_model is None:
            # Safety fallback when model struct is not materialized.
            self.model.infer(inputs)
            self.audio_model.infer(inputs)
            return self.model.scheduler.noise_pred, self.audio_model.scheduler.noise_pred

        self.model.scheduler.infer_condition = infer_condition
        self.audio_model.scheduler.infer_condition = infer_condition

        pre_v = video_model.pre_infer.infer(video_model.pre_weight, inputs)
        pre_a = self.audio_model.pre_infer.infer(self.audio_model.pre_weight, inputs)

        t_v = video_model.transformer_infer
        t_a = self.audio_model.transformer_infer
        t_v.cos_sin = pre_v.cos_sin
        t_a.cos_sin = pre_a.cos_sin
        t_v.reset_infer_states()
        t_a.reset_infer_states()

        x_v = pre_v.x
        x_a = pre_a.x

        v_blocks = video_model.transformer_weights.blocks
        a_blocks = self.audio_model.transformer_weights.blocks
        min_layers = min(len(v_blocks), len(a_blocks))

        x_freqs = None
        y_freqs = None
        bridge_dtype = None
        if self.dual_tower_bridge is not None:
            self._ensure_bridge_device(x_v)
            params = list(self.dual_tower_bridge.parameters())
            bridge_dtype = params[0].dtype if params else x_v.dtype
        if self.dual_tower_bridge is not None and getattr(self.dual_tower_bridge, "apply_cross_rope", False):
            x_freqs, y_freqs = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=float(self.config.get("fps", 24)),
                grid_size=pre_v.grid_sizes.tuple,
                audio_steps=x_a.shape[0],
                device=x_v.device,
                dtype=bridge_dtype if bridge_dtype is not None else x_v.dtype,
            )
            if bridge_dtype is not None:
                x_freqs = tuple(t.to(dtype=bridge_dtype) for t in x_freqs)
                y_freqs = tuple(t.to(dtype=bridge_dtype) for t in y_freqs)

        for layer_idx in range(min_layers):
            if self.dual_tower_bridge is not None and self.dual_tower_bridge.should_interact(layer_idx, "a2v"):
                x_v_dtype = x_v.dtype
                x_a_dtype = x_a.dtype
                bridge_scale = float(self.config.get("mova_bridge_scale", 1.0))
                x_v_in = x_v.unsqueeze(0).to(dtype=bridge_dtype)
                x_a_in = x_a.unsqueeze(0).to(dtype=bridge_dtype)
                x_v_out, x_a_out = self.dual_tower_bridge(
                    layer_idx=layer_idx,
                    visual_hidden_states=x_v_in,
                    audio_hidden_states=x_a_in,
                    x_freqs=x_freqs,
                    y_freqs=y_freqs,
                    condition_scale=bridge_scale,
                    video_grid_size=pre_v.grid_sizes.tuple,
                )
                x_v = x_v_out.squeeze(0).to(dtype=x_v_dtype)
                x_a = x_a_out.squeeze(0).to(dtype=x_a_dtype)

            t_v.block_idx = layer_idx
            x_v = t_v.infer_block(v_blocks[layer_idx], x_v, pre_v)
            t_a.block_idx = layer_idx
            x_a = t_a.infer_block(a_blocks[layer_idx], x_a, pre_a)

        for layer_idx in range(min_layers, len(v_blocks)):
            t_v.block_idx = layer_idx
            x_v = t_v.infer_block(v_blocks[layer_idx], x_v, pre_v)

        x_v = t_v.infer_non_blocks(video_model.transformer_weights, x_v, pre_v.embed)
        x_a = t_a.infer_non_blocks(self.audio_model.transformer_weights, x_a, pre_a.embed)

        noise_v = video_model.post_infer.infer(x_v, pre_v)[0]
        noise_a = self.audio_model.post_infer.infer(x_a, pre_a)[0]
        return noise_v, noise_a

    def _iter_video_models_for_bridge(self):
        if isinstance(self.model, MultiModelStruct):
            return [m for m in self.model.model if m is not None]
        return [self.model]

    def _set_bridge_callbacks(self, video_callback, audio_callback):
        for video_model in self._iter_video_models_for_bridge():
            if hasattr(video_model, "transformer_infer"):
                video_model.transformer_infer.layer_hidden_state_callback = video_callback
        if hasattr(self.audio_model, "transformer_infer"):
            self.audio_model.transformer_infer.layer_hidden_state_callback = audio_callback

    def _ensure_bridge_device(self, ref_tensor):
        if self.dual_tower_bridge is None:
            return
        params = list(self.dual_tower_bridge.parameters())
        if not params:
            return
        if params[0].device != ref_tensor.device:
            self.dual_tower_bridge.to(ref_tensor.device)

    def _video_bridge_callback(self, layer_idx, hidden_states, pre_infer_out):
        if self.dual_tower_bridge is None or hidden_states.ndim != 2:
            self._bridge_cur_video_layers[layer_idx] = hidden_states.detach()
            return hidden_states

        hidden_dtype = hidden_states.dtype
        self._bridge_video_grid_size = getattr(pre_infer_out.grid_sizes, "tuple", self._bridge_video_grid_size)
        audio_cond = self._bridge_prev_audio_layers.get(layer_idx, None)
        if audio_cond is None:
            self._bridge_cur_video_layers[layer_idx] = hidden_states.detach()
            return hidden_states

        self._ensure_bridge_device(hidden_states)
        bridge_dtype = next(self.dual_tower_bridge.parameters()).dtype
        bridge_scale = float(self.config.get("mova_bridge_scale", 1.0))
        video_in = hidden_states.unsqueeze(0).to(bridge_dtype)
        audio_in = audio_cond.unsqueeze(0).to(bridge_dtype)
        video_out, _ = self.dual_tower_bridge(
            layer_idx=layer_idx,
            visual_hidden_states=video_in,
            audio_hidden_states=audio_in,
            condition_scale=bridge_scale,
            video_grid_size=self._bridge_video_grid_size,
        )
        hidden_states = video_out.squeeze(0).to(dtype=hidden_dtype)
        self._bridge_cur_video_layers[layer_idx] = hidden_states.detach()
        return hidden_states

    def _audio_bridge_callback(self, layer_idx, hidden_states, pre_infer_out):
        if self.dual_tower_bridge is None or hidden_states.ndim != 2:
            self._bridge_cur_audio_layers[layer_idx] = hidden_states.detach()
            return hidden_states

        hidden_dtype = hidden_states.dtype
        video_cond = self._bridge_cur_video_layers.get(layer_idx, None)
        if video_cond is None:
            self._bridge_cur_audio_layers[layer_idx] = hidden_states.detach()
            return hidden_states

        self._ensure_bridge_device(hidden_states)
        bridge_dtype = next(self.dual_tower_bridge.parameters()).dtype
        bridge_scale = float(self.config.get("mova_bridge_scale", 1.0))
        video_in = video_cond.unsqueeze(0).to(bridge_dtype)
        audio_in = hidden_states.unsqueeze(0).to(bridge_dtype)
        _, audio_out = self.dual_tower_bridge(
            layer_idx=layer_idx,
            visual_hidden_states=video_in,
            audio_hidden_states=audio_in,
            condition_scale=bridge_scale,
            video_grid_size=self._bridge_video_grid_size,
        )
        hidden_states = audio_out.squeeze(0).to(dtype=hidden_dtype)
        self._bridge_cur_audio_layers[layer_idx] = hidden_states.detach()
        return hidden_states

    def _prepare_audio_for_export(self, audio):
        if audio is None:
            return None
        if audio.ndim == 3:
            # [B, C, T] -> [C, T]
            audio = audio[0]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError(f"Unsupported audio shape for export: {audio.shape}")
        # [C, T] -> [T, C]
        audio = audio.transpose(0, 1).contiguous()
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        return audio.float().cpu()

    def _save_video(self, video, fps, audio, output_path, video_chunks_number=1):
        audio_for_save = None
        if audio is not None:
            if hasattr(audio, "waveform") and hasattr(audio, "sampling_rate"):
                audio_for_save = audio
            else:
                audio_for_save = SimpleNamespace(
                    waveform=audio,
                    sampling_rate=int(self.audio_vae_config.get("sample_rate", self.config.get("audio_fps", 24000))),
                )
        save_video(
            video=video,
            fps=fps,
            audio=audio_for_save,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_video
        self.gen_audio_final = self.gen_audio

    def process_images_after_vae_decoder(self):
        self.gen_video_final = self.gen_video_final.float()
        self.gen_video_final = wan_vae_to_comfy(self.gen_video_final)
        self.gen_audio_final = self._prepare_audio_for_export(self.gen_audio_final)
        video_for_save = (self.gen_video_final.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "audio": self.gen_audio_final}
        elif self.input_info.save_result_path is not None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("🎬 Start to save video+audio 🎬")
                self._save_video(
                    video=video_for_save,
                    fps=self.config.get("fps", 24),
                    audio=self.gen_audio_final,
                    output_path=self.input_info.save_result_path,
                    video_chunks_number=1,
                )
                logger.info(f"✅ Video saved successfully to: {self.input_info.save_result_path} ✅")
            return {"video": None}
