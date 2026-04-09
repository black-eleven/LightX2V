from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.mova.infer.post_infer import MovaAudioPostInfer
from lightx2v.models.networks.mova.infer.pre_infer import MovaPreInfer
from lightx2v.models.networks.mova.infer.transformer_infer import MovaOffloadTransformerInfer, MovaTransformerInfer
from lightx2v.models.networks.mova.weights.pre_weights import MovaPreWeights
from lightx2v.models.networks.wan.model import WanModel


class MovaAudioModel(WanModel):
    pre_weight_class = MovaPreWeights

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = MovaPreInfer
        self.post_infer_class = MovaAudioPostInfer
        if self.config.get("feature_caching", "NoCaching") == "NoCaching":
            self.transformer_infer_class = MovaTransformerInfer if not self.cpu_offload else MovaOffloadTransformerInfer


class MovaVideoModel(WanModel):
    def _init_infer_class(self):
        super()._init_infer_class()
        if self.config.get("feature_caching", "NoCaching") == "NoCaching":
            self.transformer_infer_class = MovaTransformerInfer if not self.cpu_offload else MovaOffloadTransformerInfer

