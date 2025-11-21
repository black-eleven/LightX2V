from typing import Dict, Optional, Tuple, Union

import gguf
import numpy as np
import torch
from loguru import logger

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


class GGMLTensor(torch.Tensor):
    def __new__(
        cls,
        data: Union[torch.Tensor, np.ndarray, None] = None,
        shape: Tuple[int, ...] = None,
        orig_shape: Tuple[int, ...] = None,
        dtype: torch.dtype = None,
        gguf_type: gguf.GGMLQuantizationType = None,
        requires_grad: bool = False,
        aligned: bool = True,
        pin_memory: bool = False,
        preallocated: bool = False,
    ):
        # 预分配模式处理
        if preallocated and shape is not None:
            if pin_memory:
                torch_data = torch.empty(shape, dtype=dtype, requires_grad=requires_grad, pin_memory=True)
            else:
                torch_data = torch.empty(shape, dtype=dtype, requires_grad=requires_grad)
        # 正常数据模式
        elif data is not None:
            if isinstance(data, np.ndarray):
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                    torch_data = torch.from_numpy(data)
            else:
                torch_data = data

            # 类型转换
            if dtype is not None and torch_data.dtype != dtype:
                torch_data = torch_data.to(dtype)
        else:
            raise ValueError("Either data or shape must be provided for preallocated tensors")

        result = super().__new__(cls, torch_data)
        return result

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, None] = None,
        shape: Tuple[int, ...] = None,
        orig_shape: Tuple[int, ...] = None,
        dtype: torch.dtype = None,
        gguf_type: gguf.GGMLQuantizationType = None,
        requires_grad: bool = False,
        aligned: bool = True,
        pin_memory: bool = False,
        preallocated: bool = False,
    ):
        super().__init__()

        assert orig_shape is not None
        assert gguf_type is not None

        self.gguf_type = gguf_type
        self._orig_shape = orig_shape
        self._aligned = aligned
        self._pinned_memory = pin_memory
        self._requires_grad = requires_grad
        self._preallocated = preallocated

        # 量化相关属性
        self._quantized = self._is_quantized_type(gguf_type)
        self._q_type = self._get_quant_type_str(gguf_type)

        # 重塑张量形状
        if not self._quantized:
            self.data = self.reshape(*self._orig_shape).data

        self.requires_grad_(requires_grad)

        # 内存优化
        if aligned:
            self._make_aligned()
        if pin_memory:
            self._pin_memory()

    def _is_quantized_type(self, gguf_type: gguf.GGMLQuantizationType) -> bool:
        return gguf_type not in TORCH_COMPATIBLE_QTYPES

    def _get_quant_type_str(self, gguf_type: gguf.GGMLQuantizationType) -> str:
        type_mapping = {
            gguf.GGMLQuantizationType.F32: "ggml_f32",
            gguf.GGMLQuantizationType.F16: "ggml_f16",
            gguf.GGMLQuantizationType.Q4_0: "ggml_q4_0",
            gguf.GGMLQuantizationType.Q4_1: "ggml_q4_1",
            gguf.GGMLQuantizationType.Q5_0: "ggml_q5_0",
            gguf.GGMLQuantizationType.Q5_1: "ggml_q5_1",
            gguf.GGMLQuantizationType.Q8_0: "ggml_q8_0",
            gguf.GGMLQuantizationType.Q8_1: "ggml_q8_1",
            gguf.GGMLQuantizationType.Q2_K: "ggml_q2_k",
            gguf.GGMLQuantizationType.Q3_K: "ggml_q3_k",
            gguf.GGMLQuantizationType.Q4_K: "ggml_q4_k",
            gguf.GGMLQuantizationType.Q5_K: "ggml_q5_k",
            gguf.GGMLQuantizationType.Q6_K: "ggml_q6_k",
            gguf.GGMLQuantizationType.Q8_K: "ggml_q8_k",
        }
        return type_mapping.get(gguf_type, "unknown")

    @classmethod
    def empty_pinned(
        cls, shape: Tuple[int, ...], orig_shape: Tuple[int, ...] = None, dtype: torch.dtype = torch.float32, gguf_type: gguf.GGMLQuantizationType = None, aligned: bool = True
    ) -> "GGMLTensor":
        return cls(shape=shape, dtype=dtype, orig_shape=orig_shape, gguf_type=gguf_type, pin_memory=True, aligned=aligned, preallocated=True)

    @classmethod
    def empty_aligned(
        cls, shape: Tuple[int, ...], orig_shape: Tuple[int, ...] = None, dtype: torch.dtype = torch.float32, gguf_type: gguf.GGMLQuantizationType = None, pin_memory: bool = False
    ) -> "GGMLTensor":
        return cls(shape=shape, dtype=dtype, orig_shape=orig_shape, gguf_type=gguf_type, pin_memory=pin_memory, aligned=True, preallocated=True)

    def copy_from(self, source: Union[torch.Tensor, "GGMLTensor"], transpose: bool = False, non_blocking: bool = False) -> "GGMLTensor":
        if not self._preallocated:
            raise RuntimeError("copy_from can only be used with preallocated tensors")

        # 获取源数据
        if transpose:
            source_data = source.t().contiguous()
        else:
            source_data = source.contiguous()

        # 检查形状是否匹配
        if self.shape != source_data.shape:
            raise ValueError(f"Shape mismatch: target {self.shape} vs source {source_data.shape}")

        # 执行复制
        self.copy_(source_data)

        return self

    def copy_from_dict(self, weight_dict: Dict[str, torch.Tensor], weight_name: str, transpose: bool = False, non_blocking: bool = False) -> "GGMLTensor":
        if weight_name not in weight_dict:
            raise KeyError(f"Weight '{weight_name}' not found in weight dictionary")

        source_weight = weight_dict[weight_name]
        return self.copy_from(source_weight, transpose=transpose, non_blocking=non_blocking)

    def copy_to(self, target: Union[torch.Tensor, "GGMLTensor"], transpose: bool = False, non_blocking: bool = False) -> "GGMLTensor":
        source_data = self
        if transpose:
            source_data = self.t().contiguous()

        if isinstance(target, GGMLTensor):
            target.copy_from(source_data, non_blocking=non_blocking)
        else:
            target.copy_(source_data)

        return self

    def _make_aligned(self, alignment: int = 32):
        """确保张量数据内存对齐"""
        if not self.is_contiguous():
            self.data = self.contiguous().data

        ptr = self.data_ptr()
        if ptr % alignment == 0:
            return

        if self._pinned_memory:
            aligned_data = torch.empty(self.shape, dtype=self.dtype, device=self.device, pin_memory=True)
        else:
            aligned_data = torch.empty(self.shape, dtype=self.dtype, device=self.device)

        aligned_data.copy_(self)
        self.data = aligned_data.data

    def _pin_memory(self) -> "GGMLTensor":
        if self._pinned_memory or self.device.type != "cpu":
            return self

        pinned_data = self.pin_memory()
        self.data = pinned_data.data
        self._pinned_memory = True
        return self

    @classmethod
    def from_torch(cls, tensor: torch.Tensor, tensor_type: gguf.GGMLQuantizationType, tensor_shape: Tuple[int, ...], aligned: bool = True, pin_memory: bool = False) -> "GGMLTensor":
        return cls(tensor, gguf_type=tensor_type, shape=tensor_shape, dtype=tensor.dtype, aligned=aligned, pin_memory=pin_memory)

    def to_torch(self) -> torch.Tensor:
        return torch.as_tensor(self)

    # 属性访问方法
    @property
    def tensor_type(self) -> gguf.GGMLQuantizationType:
        """获取GGUF张量类型"""
        return self.gguf_type

    @property
    def quant_type(self) -> str:
        """获取量化类型字符串"""
        return self._q_type

    @property
    def is_quantized(self) -> bool:
        """是否已量化"""
        return self._quantized

    @property
    def orig_shape(self) -> Tuple[int, ...]:
        """获取原始形状"""
        return self._orig_shape

    @property
    def blocksize(self) -> Optional[int]:
        _blocksize, _ = gguf.GGML_QUANT_SIZES[self.qtype]
        return _blocksize

    @property
    def is_pinned(self) -> bool:
        """是否固定内存"""
        return self._pinned_memory

    def memory_footprint(self) -> int:
        """计算内存占用（字节）"""
        if self._quantized:
            # 量化张量的实际内存占用
            return self.numel() * self.element_size()
        else:
            return self.numel() * self.element_size()

    def __repr__(self) -> str:
        return f"GGMLTensor(shape={self.shape}, orig_shape={self.orig_shape}, dtype={self.dtype}, quantized={self.is_quantized}, quant_type='{self.quant_type}', pinned={self.is_pinned})"
        # f"data={self.data}")

    def cuda(self, device: Optional[Union[int, torch.device]] = None, non_blocking: bool = False) -> "GGMLTensor":
        # 使用父类的cuda方法移动数据
        if device is None:
            cuda_tensor = super().cuda(non_blocking=non_blocking)
        else:
            cuda_tensor = super().cuda(device=device, non_blocking=non_blocking)

        # 创建新的GGMLTensor，保持所有属性
        result = GGMLTensor.from_torch(
            cuda_tensor,
            self.gguf_type,
            self._orig_shape,
            aligned=False,  # CUDA张量不需要内存对齐
            pin_memory=False,  # CUDA张量不能固定内存
        )

        # 手动复制所有属性
        result._quantized = self._quantized
        result._q_type = self._q_type
        result._requires_grad = self._requires_grad

        return result

    def cpu(self, pin_memory: bool = False) -> "GGMLTensor":
        # 使用父类的cpu方法移动数据
        cpu_tensor = super().cpu()

        # 创建新的GGMLTensor，保持所有属性
        result = GGMLTensor.from_torch(cpu_tensor, self.gguf_type, self._orig_shape, aligned=self._aligned, pin_memory=pin_memory)

        # 手动复制所有属性
        result._quantized = self._quantized
        result._q_type = self._q_type
        result._requires_grad = self._requires_grad

        return result

    def to(self, *args, **kwargs) -> "GGMLTensor":
        # 调用父类的to方法
        result_tensor = super().to(*args, **kwargs)

        # 如果设备或类型发生变化，创建新的GGMLTensor
        if result_tensor.device != self.device or result_tensor.dtype != self.dtype or not isinstance(result_tensor, GGMLTensor):
            # 确定是否固定内存（仅对CPU张量有效）
            pin_memory = kwargs.get("pin_memory", False)
            if result_tensor.device.type != "cpu":
                pin_memory = False

            result = GGMLTensor.from_torch(result_tensor, self.gguf_type, self._orig_shape, aligned=self._aligned if result_tensor.device.type == "cpu" else False, pin_memory=pin_memory)

            # 复制属性
            result._quantized = self._quantized
            result._q_type = self._q_type
            result._requires_grad = self._requires_grad

            return result

        return self


def load_gguf_clip_ckpt(gguf_path):
    pass


# 修改后的加载函数
def load_gguf_sd_ckpt(gguf_path, return_arch=False, to_device: Optional[Union[int, torch.device]] = None):
    import warnings

    logger.info(f"Loading gguf-quant dit model from {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        tensor_name = tensor.name

        # 处理NumPy数组（避免mmap警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        # 获取原始形状
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            # TODO 由于测试用的t5模型都是bf16的，这里先强转成bf16
            state_dict[tensor.name] = torch_tensor.to(torch.bfloat16).to(to_device)
        else:
            # 创建GGMLTensor并添加到state_dict
            state_dict[tensor.name] = GGMLTensor(
                data=torch_tensor,
                gguf_type=tensor.tensor_type,
                orig_shape=shape,
                aligned=True,  # 启用内存对齐
                pin_memory=False,  # 根据需求调整
            ).to(to_device)

    if return_arch:
        # 提取模型架构信息
        arch = get_model_architecture(reader)
        return state_dict, arch

    return state_dict


# 辅助函数
def get_orig_shape(reader, tensor_name: str) -> Optional[Tuple[int, ...]]:
    """从GGUF读取器获取原始张量形状"""
    # 实现根据GGUF格式获取原始形状的逻辑
    # TODO 这里正式上线的时候，需要更换
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")


def get_model_architecture(reader) -> str:
    """从GGUF读取器获取模型架构信息"""
    # 实现获取模型架构的逻辑
    arch_str = get_field(reader, "general.architecture", str)
    return arch_str


def dequantize_tensor(tensor, dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        return dequantize(tensor.data, qtype, oshape, dtype=dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {qtype}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)


def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}


if __name__ == "__main__":
    sd, arch = load_gguf_sd_ckpt("/home/SENSETIME/yihuiwen/yihuiwen/workspace/models/city96/Wan2.1-I2V-14B-720P-gguf/wan2.1-i2v-14b-720p-Q4_K_S.gguf", return_arch=True)

    for k, s in sd.items():
        print(k)
        print(s.dtype)
        print(getattr(s, "gtype", s.dtype))
