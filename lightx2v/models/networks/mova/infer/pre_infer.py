import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d, guidance_scale_embedding


class MovaPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        # Align MOVA audio RoPE with official implementation:
        # precompute over full head dim, then split into 3 chunks.
        audio_rope_max_seq_len = int(config.get("audio_rope_max_seq_len", 16384))
        self.freqs = self._build_mova_audio_freqs(self.head_size, audio_rope_max_seq_len)

    def _build_mova_audio_freqs(self, head_size: int, max_seq_len: int, theta: float = 10000.0):
        if head_size % 2 != 0:
            raise ValueError(f"head_size must be even for RoPE, got {head_size}")
        device = self.freqs.device
        base = torch.arange(0, head_size, 2, dtype=torch.float64, device=device)
        inv = torch.pow(torch.tensor(theta, dtype=torch.float64, device=device), -(base / float(head_size)))
        pos = torch.arange(max_seq_len, dtype=torch.float64, device=device)
        freqs = torch.outer(pos, inv)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        return freqs_cis.chunk(3, dim=1)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        # Diffusion state may be fp32; align to model inference dtype before convs.
        x = x.to(self.infer_dtype)
        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))

        grid_sizes_t = x.shape[2]
        x = x.flatten(2).transpose(1, 2).contiguous()
        # seq_lens = torch.tensor(x.size(1), dtype=torch.int32, device=x.device).unsqueeze(0)

        # NOTE: MMWeight.apply requires input dtype == weight dtype.
        # In this codebase, time_embedding weights are typically loaded as infer_dtype (bf16/fp16),
        # so we keep embedding compute in infer_dtype to avoid addmm dtype mismatch.
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.infer_dtype)
        if self.enable_dynamic_cfg:
            s = torch.tensor([self.cfg_scale], dtype=torch.float32, device=x.device)
            cfg_embed = guidance_scale_embedding(
                s, embedding_dim=256, cfg_range=(1.0, 6.0), target_range=1000.0, dtype=torch.float32
            ).to(self.infer_dtype)
            cfg_embed = weights.cfg_cond_proj_1.apply(cfg_embed)
            cfg_embed = torch.nn.functional.silu(cfg_embed)
            cfg_embed = weights.cfg_cond_proj_2.apply(cfg_embed)
            embed = embed + cfg_embed
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))
        embed = embed.to(self.infer_dtype)
        embed0 = embed0.to(self.infer_dtype)

        # text embeddings
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        grid_sizes = GridOutput(
            tensor=torch.tensor([[grid_sizes_t]], dtype=torch.int32, device=x.device),
            tuple=grid_sizes_t,
        )

        if self.cos_sin is None or self.grid_sizes != grid_sizes.tuple:
            if isinstance(self.freqs, (tuple, list)):
                freqs = tuple(f.clone() for f in self.freqs)
            else:
                freqs = self.freqs.clone()  # self.freqs init param can not be changed
            self.grid_sizes = grid_sizes.tuple
            self.cos_sin = self.prepare_cos_sin(grid_sizes.tuple, freqs)

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=self.cos_sin,
        )

    def prepare_cos_sin(self, grid_sizes, freqs):
        c = self.head_size // 2
        if isinstance(freqs, (tuple, list)):
            freqs = tuple(freqs)
            if len(freqs) != 3:
                raise ValueError(f"Expected 3 RoPE chunks, got {len(freqs)}")
        else:
            freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f = grid_sizes
        cos_sin = torch.cat(
            [
                freqs[0][:f].view(f, -1).expand(f, -1),
                freqs[1][:f].view(f, -1).expand(f, -1),
                freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1,
        )
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            cos_sin = cos_sin.reshape(f, -1)
            cos_half = cos_sin.real.contiguous().to(torch.float32)
            sin_half = cos_sin.imag.contiguous().to(torch.float32)
            cos_sin = torch.cat([cos_half, sin_half], dim=-1)
        else:
            cos_sin = cos_sin.reshape(f, 1, -1)
            if self.seq_p_group is not None:
                world_size = dist.get_world_size(self.seq_p_group)
                cur_rank = dist.get_rank(self.seq_p_group)
                seqlen = cos_sin.shape[0]
                multiple = world_size * f
                padding_size = (multiple - (seqlen % multiple)) % multiple
                if padding_size > 0:
                    cos_sin = F.pad(cos_sin, (0, 0, 0, 0, 0, padding_size))
                cos_sin = torch.chunk(cos_sin, world_size, dim=0)[cur_rank]
        return cos_sin