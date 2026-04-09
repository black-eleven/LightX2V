import torch

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class MovaTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.layer_hidden_state_callback = None

    def _apply_layer_hidden_state_callback(self, x, block_idx, pre_infer_out):
        callback = getattr(self, "layer_hidden_state_callback", None)
        if callback is None:
            return x
        return callback(block_idx, x, pre_infer_out)

    def infer_without_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out)
            x = self._apply_layer_hidden_state_callback(x, block_idx, pre_infer_out)
        return x


class MovaOffloadTransformerInfer(WanOffloadTransformerInfer, MovaTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.layer_hidden_state_callback = None

    def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            self.offload_manager.prefetch_weights((block_idx + 1) % len(blocks), blocks)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)
                x = self._apply_layer_hidden_state_callback(x, block_idx, pre_infer_out)

            self.offload_manager.swap_blocks()

        if self.clean_cuda_cache:
            del (pre_infer_out.embed0, pre_infer_out.context)
            torch_device_module.empty_cache()

        return x

    def infer_with_phases_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(blocks)
                self.offload_manager.start_prefetch_block(next_prefetch)

            x = self.infer_phases(block_idx, blocks, x, pre_infer_out)
            x = self._apply_layer_hidden_state_callback(x, block_idx, pre_infer_out)

            if self.clean_cuda_cache:
                del (
                    self.phase_params["attn_out"],
                    self.phase_params["y_out"],
                    self.phase_params["y"],
                )
                torch_device_module.empty_cache()

        if self.clean_cuda_cache:
            self.clear_offload_params(pre_infer_out)

        return x
