from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV1D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class MovaPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        patch_size = config.get("patch_size", 1)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.config = config

        self.add_module(
            "patch_embedding",
            CONV1D_WEIGHT_REGISTER["Default"](
                "patch_embedding.weight",
                "patch_embedding.bias",
                stride=self.patch_size,
                lora_prefix="diffusion_model.patch_embedding",
            ),
        )

        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.0.weight",
                "text_embedding.0.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.2.weight",
                "text_embedding.2.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.0.weight",
                "time_embedding.0.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.2.weight",
                "time_embedding.2.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_projection_1",
            MM_WEIGHT_REGISTER["Default"](
                "time_projection.1.weight",
                "time_projection.1.bias",
                lora_prefix="diffusion_model.time_projection",
            ),
        )
