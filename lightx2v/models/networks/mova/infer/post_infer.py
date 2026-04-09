import torch
from einops import rearrange


class MovaAudioPostInfer:
    def __init__(self, config):
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        patch_size = config.get("patch_size", 1)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = (int(patch_size),)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, "b f (p c) -> b c (f p)",
            f=grid_size[0],
            p=self.patch_size[0],
        )

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Unexpected MOVA audio post input shape: {tuple(x.shape)}")

        grid_size = pre_infer_out.grid_sizes.tuple
        if isinstance(grid_size, int):
            grid_size = (grid_size,)
        noise_pred = self.unpatchify(x, grid_size).squeeze(0).contiguous()

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()
        return [noise_pred.float()]
