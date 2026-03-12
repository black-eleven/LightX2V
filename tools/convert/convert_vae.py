import argparse
import re
from typing import Dict, List, Optional, Tuple

import torch


DIRECT_KEY_MAPPING = {
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
}

MIDDLE_RESIDUAL_SUFFIX_MAP = {
    "norm1.gamma": "residual.0.gamma",
    "conv1.weight": "residual.2.weight",
    "conv1.bias": "residual.2.bias",
    "norm2.gamma": "residual.3.gamma",
    "conv2.weight": "residual.6.weight",
    "conv2.bias": "residual.6.bias",
}

MIDDLE_ATTN_SUFFIX_MAP = {
    "norm.gamma": "norm.gamma",
    "to_qkv.weight": "to_qkv.weight",
    "to_qkv.bias": "to_qkv.bias",
    "proj.weight": "proj.weight",
    "proj.bias": "proj.bias",
}


def _convert_residual_suffix(suffix: str) -> str:
    if suffix in MIDDLE_RESIDUAL_SUFFIX_MAP:
        return MIDDLE_RESIDUAL_SUFFIX_MAP[suffix]
    if suffix == "conv_shortcut.weight":
        return "shortcut.weight"
    if suffix == "conv_shortcut.bias":
        return "shortcut.bias"
    return suffix


def _convert_encoder_middle_key(key: str) -> Optional[str]:
    m = re.match(r"^encoder\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        resnet_idx = int(m.group(1))
        if resnet_idx not in (0, 1):
            return None
        middle_idx = 0 if resnet_idx == 0 else 2
        suffix = _convert_residual_suffix(m.group(2))
        return f"encoder.middle.{middle_idx}.{suffix}"

    m = re.match(r"^encoder\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        suffix = m.group(1)
        if suffix not in MIDDLE_ATTN_SUFFIX_MAP:
            return None
        return f"encoder.middle.1.{MIDDLE_ATTN_SUFFIX_MAP[suffix]}"
    return None


def _convert_decoder_middle_key(key: str) -> Optional[str]:
    m = re.match(r"^decoder\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        resnet_idx = int(m.group(1))
        if resnet_idx not in (0, 1):
            return None
        middle_idx = 0 if resnet_idx == 0 else 2
        suffix = _convert_residual_suffix(m.group(2))
        return f"decoder.middle.{middle_idx}.{suffix}"

    m = re.match(r"^decoder\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        suffix = m.group(1)
        if suffix not in MIDDLE_ATTN_SUFFIX_MAP:
            return None
        return f"decoder.middle.1.{MIDDLE_ATTN_SUFFIX_MAP[suffix]}"
    return None


def _convert_encoder_down_key(key: str) -> Optional[str]:
    if not key.startswith("encoder.down_blocks."):
        return None
    suffix_converted = (
        key.replace(".norm1.gamma", ".residual.0.gamma")
        .replace(".conv1.weight", ".residual.2.weight")
        .replace(".conv1.bias", ".residual.2.bias")
        .replace(".norm2.gamma", ".residual.3.gamma")
        .replace(".conv2.weight", ".residual.6.weight")
        .replace(".conv2.bias", ".residual.6.bias")
        .replace(".conv_shortcut.weight", ".shortcut.weight")
        .replace(".conv_shortcut.bias", ".shortcut.bias")
    )
    return suffix_converted.replace("encoder.down_blocks.", "encoder.downsamples.")


def _convert_decoder_up_key(key: str) -> Optional[str]:
    if not key.startswith("decoder.up_blocks."):
        return None

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", key)
    if m:
        up_block_idx = int(m.group(1))
        native_idx = up_block_idx * 4 + 3
        tail = m.group(2)
        return f"decoder.upsamples.{native_idx}.{tail}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        up_block_idx = int(m.group(1))
        resnet_idx = int(m.group(2))
        native_idx = up_block_idx * 4 + resnet_idx
        suffix = _convert_residual_suffix(m.group(3))
        return f"decoder.upsamples.{native_idx}.{suffix}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.conv_shortcut\.(weight|bias)$", key)
    if m:
        up_block_idx = int(m.group(1))
        native_idx = up_block_idx * 4
        return f"decoder.upsamples.{native_idx}.shortcut.{m.group(2)}"

    return key.replace("decoder.up_blocks.", "decoder.upsamples.")


def convert_diffusers_to_native_key(key: str) -> str:
    if key in DIRECT_KEY_MAPPING:
        return DIRECT_KEY_MAPPING[key]

    converted = _convert_encoder_middle_key(key)
    if converted is not None:
        return converted

    converted = _convert_decoder_middle_key(key)
    if converted is not None:
        return converted

    converted = _convert_encoder_down_key(key)
    if converted is not None:
        return converted

    converted = _convert_decoder_up_key(key)
    if converted is not None:
        return converted

    return key


def convert_state_dict(diffusers_state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    native_state_dict: Dict[str, torch.Tensor] = {}
    passthrough_keys: List[str] = []
    for key, value in diffusers_state_dict.items():
        new_key = convert_diffusers_to_native_key(key)
        native_state_dict[new_key] = value
        if new_key == key:
            passthrough_keys.append(key)
    return native_state_dict, passthrough_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Diffusers-format Wan VAE weights to native Wan checkpoint format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Diffusers model directory or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .pth file path for native Wan VAE state_dict.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="vae",
        help="Subfolder containing the VAE in a Diffusers pipeline repo.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional checkpoint variant (e.g. fp16, bf16).",
    )
    parser.add_argument(
        "--wrap_with_state_dict",
        action="store_true",
        help="Save as {'state_dict': ...} instead of raw state_dict.",
    )
    parser.add_argument(
        "--allow_passthrough",
        action="store_true",
        help="Allow keys that are not remapped (recommended only for debugging).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from diffusers.models.autoencoders import AutoencoderKLWan
    except ImportError as e:
        raise ImportError(
            "diffusers is required to run this script. Please install it in your current environment."
        ) from e

    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
    )
    diffusers_state_dict = vae.state_dict()
    native_state_dict, passthrough_keys = convert_state_dict(diffusers_state_dict)

    if passthrough_keys and not args.allow_passthrough:
        sample = "\n".join(passthrough_keys[:20])
        raise RuntimeError(
            f"Detected {len(passthrough_keys)} unmapped keys. "
            "Use --allow_passthrough to keep them as-is for debugging.\n"
            f"Sample keys:\n{sample}"
        )

    save_obj = {"state_dict": native_state_dict} if args.wrap_with_state_dict else native_state_dict
    torch.save(save_obj, args.output_path)
    print(f"Saved native Wan VAE checkpoint to: {args.output_path}")
    if passthrough_keys:
        print(f"Warning: {len(passthrough_keys)} keys were passed through unchanged.")


if __name__ == "__main__":
    main()
