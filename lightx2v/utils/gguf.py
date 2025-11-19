from loguru import logger
import torch
import numpy as np
from typing import Optional, Union, List, Tuple, Dict
import gguf

class GGMLTensor(torch.Tensor):
    """
    专门用于GGUF文件加载的GGMLTensor类
    支持各种量化格式和GGUF特性
    """
    
    @staticmethod
    def __new__(cls, 
                data: Union[torch.Tensor, np.ndarray],
                tensor_type: gguf.GGMLQuantizationType,
                tensor_shape: Tuple[int, ...],
                dtype: torch.dtype = None,
                requires_grad: bool = False,
                aligned: bool = True,
                pin_memory: bool = False,
                preallocated: bool = False,
                ):
        """
        专门为GGUF加载设计的构造函数
        
        Args:
            data: 原始张量数据（可能是量化的）
            tensor_type: GGUF量化类型
            tensor_shape: 原始张量形状
            dtype: 目标数据类型（自动推断）
            requires_grad: 是否需要梯度
            aligned: 是否内存对齐
            pin_memory: 是否固定内存
            preallocated: 是否预分配内存
        """
        # 处理NumPy数组
        if isinstance(data, np.ndarray):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                torch_data = torch.from_numpy(data)
        else:
            torch_data = data
        
        # 根据量化类型处理数据
        result = super().__new__(cls, torch_data)
        return result
    
    def __init__(self, 
                 data: Union[torch.Tensor, np.ndarray],
                 tensor_type: gguf.GGMLQuantizationType,
                 tensor_shape: Tuple[int, ...],
                 dtype: torch.dtype = None,
                 requires_grad: bool = False,
                 aligned: bool = True,
                 pin_memory: bool = False):
        super().__init__()
        
        self._tensor_type = tensor_type
        self._orig_shape = tensor_shape
        self._aligned = aligned
        self._pinned_memory = pin_memory
        self._requires_grad = requires_grad
        
        # 量化相关属性
        self._quantized = self._is_quantized_type(tensor_type)
        self._q_type = self._get_quant_type_str(tensor_type)
        self._scale = None
        self._zero_point = None
        self._blocksize = None
        
        # 根据量化类型初始化
        self._init_from_gguf_type()
        
        # 重塑张量形状
        if not self._quantized:
            self.data = self.reshape(*self._orig_shape).data
        
        # 设置梯度
        self.requires_grad_(requires_grad)
        
        # 内存优化
        if aligned:
            self._make_aligned()
        if pin_memory:
            self._pin_memory()
    
    def _is_quantized_type(self, tensor_type: gguf.GGMLQuantizationType) -> bool:
        """检查是否为量化类型"""
        quant_types = {
            gguf.GGMLQuantizationType.F32: False,
            gguf.GGMLQuantizationType.F16: False,
            gguf.GGMLQuantizationType.Q4_0: True,
            gguf.GGMLQuantizationType.Q4_1: True,
            gguf.GGMLQuantizationType.Q5_0: True,
            gguf.GGMLQuantizationType.Q5_1: True,
            gguf.GGMLQuantizationType.Q8_0: True,
            gguf.GGMLQuantizationType.Q8_1: True,
            gguf.GGMLQuantizationType.Q2_K: True,
            gguf.GGMLQuantizationType.Q3_K: True,
            gguf.GGMLQuantizationType.Q4_K: True,
            gguf.GGMLQuantizationType.Q5_K: True,
            gguf.GGMLQuantizationType.Q6_K: True,
            gguf.GGMLQuantizationType.Q8_K: True,
        }
        return quant_types.get(tensor_type, False)
    
    def _get_quant_type_str(self, tensor_type: gguf.GGMLQuantizationType) -> str:
        """获取量化类型字符串"""
        type_mapping = {
            gguf.GGMLQuantizationType.F32: "f32",
            gguf.GGMLQuantizationType.F16: "f16",
            gguf.GGMLQuantizationType.Q4_0: "q4_0",
            gguf.GGMLQuantizationType.Q4_1: "q4_1",
            gguf.GGMLQuantizationType.Q5_0: "q5_0",
            gguf.GGMLQuantizationType.Q5_1: "q5_1",
            gguf.GGMLQuantizationType.Q8_0: "q8_0",
            gguf.GGMLQuantizationType.Q8_1: "q8_1",
            gguf.GGMLQuantizationType.Q2_K: "q2_k",
            gguf.GGMLQuantizationType.Q3_K: "q3_k",
            gguf.GGMLQuantizationType.Q4_K: "q4_k",
            gguf.GGMLQuantizationType.Q5_K: "q5_k",
            gguf.GGMLQuantizationType.Q6_K: "q6_k",
            gguf.GGMLQuantizationType.Q8_K: "q8_k",
        }
        return type_mapping.get(tensor_type, "unknown")
    
    def _init_from_gguf_type(self):
        """根据GGUF类型初始化张量"""
        if not self._quantized:
            # 非量化类型直接使用
            if self._tensor_type == gguf.GGMLQuantizationType.F16:
                # F16转换为FP32以便PyTorch处理
                if self.dtype != torch.float32:
                    self.data = self.float().data
            return
        
        # 量化类型处理
        if self._tensor_type in [gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q4_1]:
            self._blocksize = 32
        elif self._tensor_type in [gguf.GGMLQuantizationType.Q5_0, gguf.GGMLQuantizationType.Q5_1]:
            self._blocksize = 32
        elif self._tensor_type in [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q8_1]:
            self._blocksize = 32
        elif self._tensor_type in [gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q3_K,
                                 gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K,
                                 gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_K]:
            self._blocksize = 256  # K-quants通常使用256块大小
    
    def dequantize(self, target_dtype: torch.dtype = torch.float32) -> 'GGMLTensor':
        """
        反量化张量为目标数据类型
        
        Args:
            target_dtype: 目标数据类型
        """
        if not self._quantized:
            # 非量化张量直接转换类型
            if self.dtype != target_dtype:
                converted = self.to(target_dtype)
                return GGMLTensor.from_torch(converted, self._tensor_type, self._orig_shape)
            return self
        
        # 这里实现具体的反量化逻辑
        # 注意：实际的反量化实现需要根据GGUF格式详细实现
        if self._tensor_type == gguf.GGMLQuantizationType.Q4_0:
            dequantized_data = self._dequantize_q4_0()
        elif self._tensor_type == gguf.GGMLQuantizationType.Q8_0:
            dequantized_data = self._dequantize_q8_0()
        # 其他量化类型的处理...
        else:
            # 默认简单处理（实际使用时需要完整实现）
            dequantized_data = self.float()
        
        # 重塑形状并创建新的GGMLTensor
        dequantized_data = dequantized_data.reshape(*self._orig_shape)
        return GGMLTensor.from_torch(dequantized_data.to(target_dtype), 
                                   gguf.GGMLQuantizationType.F32, 
                                   self._orig_shape)
    
    def _dequantize_q4_0(self) -> torch.Tensor:
        """Q4_0反量化实现（简化版）"""
        # 实际实现需要根据GGUF的Q4_0格式解析
        # 这里返回一个简单实现
        return torch.randn(self._orig_shape, dtype=torch.float32)
    
    def _dequantize_q8_0(self) -> torch.Tensor:
        """Q8_0反量化实现（简化版）"""
        # 实际实现需要根据GGUF的Q8_0格式解析
        return torch.randn(self._orig_shape, dtype=torch.float32)
    
    @classmethod
    def empty_pinned(cls, 
                    shape: Tuple[int, ...], 
                    dtype: torch.dtype = torch.float32,
                    aligned: bool = True) -> 'GGMLTensor':
        """
        创建预分配的固定内存张量
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            aligned: 是否内存对齐
        """
        return cls(shape=shape, dtype=dtype, pin_memory=True, aligned=aligned, preallocated=True)
    
    @classmethod
    def empty_aligned(cls,
                     shape: Tuple[int, ...],
                     dtype: torch.dtype = torch.float32,
                     pin_memory: bool = False) -> 'GGMLTensor':
        """
        创建预分配的对齐内存张量
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            pin_memory: 是否固定内存
        """
        return cls(shape=shape, dtype=dtype, pin_memory=pin_memory, aligned=True, preallocated=True)
    
    def copy_from(self, 
                 source: Union[torch.Tensor, 'GGMLTensor'],
                 transpose: bool = False,
                 non_blocking: bool = False) -> 'GGMLTensor':
        """
        从源张量复制数据到当前张量
        
        Args:
            source: 源张量
            transpose: 是否转置源数据
            non_blocking: 是否非阻塞复制
        """
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
    
    def copy_from_dict(self,
                      weight_dict: Dict[str, torch.Tensor],
                      weight_name: str,
                      transpose: bool = False,
                      non_blocking: bool = False) -> 'GGMLTensor':
        """
        从权重字典中复制指定名称的权重
        
        Args:
            weight_dict: 权重字典
            weight_name: 权重名称
            transpose: 是否转置权重
            non_blocking: 是否非阻塞复制
        """
        if weight_name not in weight_dict:
            raise KeyError(f"Weight '{weight_name}' not found in weight dictionary")
        
        source_weight = weight_dict[weight_name]
        return self.copy_from(source_weight, transpose=transpose, non_blocking=non_blocking)
    
    def copy_to(self, 
               target: Union[torch.Tensor, 'GGMLTensor'],
               transpose: bool = False,
               non_blocking: bool = False) -> 'GGMLTensor':
        """
        复制当前张量数据到目标张量
        
        Args:
            target: 目标张量
            transpose: 是否转置数据
            non_blocking: 是否非阻塞复制
        """
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
    
    def _pin_memory(self) -> 'GGMLTensor':
        """固定张量内存"""
        if self._pinned_memory or self.device.type != 'cpu':
            return self
        
        pinned_data = self.pin_memory()
        self.data = pinned_data.data
        self._pinned_memory = True
        return self
    
    @classmethod
    def from_torch(cls, 
                  tensor: torch.Tensor,
                  tensor_type: gguf.GGMLQuantizationType,
                  tensor_shape: Tuple[int, ...],
                  aligned: bool = True,
                  pin_memory: bool = False) -> 'GGMLTensor':
        """从PyTorch张量创建GGMLTensor"""
        return cls(tensor, tensor_type, tensor_shape, 
                  dtype=tensor.dtype, aligned=aligned, pin_memory=pin_memory)
    
    def to_torch(self) -> torch.Tensor:
        """转换为普通PyTorch张量"""
        return torch.as_tensor(self)
    
    # 属性访问方法
    @property
    def tensor_type(self) -> gguf.GGMLQuantizationType:
        """获取GGUF张量类型"""
        return self._tensor_type
    
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
        """获取量化块大小"""
        return self._blocksize
    
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
        return (f"GGMLTensor(shape={self.shape}, orig_shape={self.orig_shape}, "
                f"dtype={self.dtype}, quantized={self.is_quantized}, "
                f"quant_type='{self.quant_type}', pinned={self.is_pinned})")

    def cuda(self, device: Optional[Union[int, torch.device]] = None, non_blocking: bool = False) -> 'GGMLTensor':
        """
        移动到CUDA设备，保持GGMLTensor类型和属性

        Args:
            device: CUDA设备
            non_blocking: 是否非阻塞传输
        """
        # 使用父类的cuda方法移动数据
        if device is None:
            cuda_tensor = super().cuda(non_blocking=non_blocking)
        else:
            cuda_tensor = super().cuda(device=device, non_blocking=non_blocking)

        # 创建新的GGMLTensor，保持所有属性
        result = GGMLTensor.from_torch(
            cuda_tensor,
            self._tensor_type,
            self._orig_shape,
            aligned=False,  # CUDA张量不需要内存对齐
            pin_memory=False  # CUDA张量不能固定内存
        )

        # 手动复制所有属性
        result._quantized = self._quantized
        result._q_type = self._q_type
        result._scale = self._scale
        result._zero_point = self._zero_point
        result._blocksize = self._blocksize
        result._requires_grad = self._requires_grad

        return result

    def cpu(self, pin_memory: bool = False) -> 'GGMLTensor':
        """
        移动到CPU设备，保持GGMLTensor类型和属性

        Args:
            pin_memory: 是否固定内存
        """
        # 使用父类的cpu方法移动数据
        cpu_tensor = super().cpu()

        # 创建新的GGMLTensor，保持所有属性
        result = GGMLTensor.from_torch(
            cpu_tensor,
            self._tensor_type,
            self._orig_shape,
            aligned=self._aligned,
            pin_memory=pin_memory
        )

        # 手动复制所有属性
        result._quantized = self._quantized
        result._q_type = self._q_type
        result._scale = self._scale
        result._zero_point = self._zero_point
        result._blocksize = self._blocksize
        result._requires_grad = self._requires_grad

        return result

    def to(self, *args, **kwargs) -> 'GGMLTensor':
        """
        重写to方法，保持GGMLTensor类型

        支持各种用法:
        - tensor.to(device)
        - tensor.to(dtype)
        - tensor.to(device, dtype)
        - tensor.to(other_tensor)
        """
        # 调用父类的to方法
        result_tensor = super().to(*args, **kwargs)

        # 如果设备或类型发生变化，创建新的GGMLTensor
        if (result_tensor.device != self.device or
            result_tensor.dtype != self.dtype or
            not isinstance(result_tensor, GGMLTensor)):

            # 确定是否固定内存（仅对CPU张量有效）
            pin_memory = kwargs.get('pin_memory', False)
            if result_tensor.device.type != 'cpu':
                pin_memory = False

            result = GGMLTensor.from_torch(
                result_tensor,
                self._tensor_type,
                self._orig_shape,
                aligned=self._aligned if result_tensor.device.type == 'cpu' else False,
                pin_memory=pin_memory
            )

            # 复制属性
            result._quantized = self._quantized
            result._q_type = self._q_type
            result._scale = self._scale
            result._zero_point = self._zero_point
            result._blocksize = self._blocksize
            result._requires_grad = self._requires_grad

            return result

        return self



# 修改后的加载函数
def load_gguf_sd_ckpt(gguf_path, return_arch=False):
    import warnings
    import gguf
    
    logger.info(f"Loading gguf-quant dit model from {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}
    qtype_dict = {}
    
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

        # 创建GGMLTensor并添加到state_dict
        state_dict[tensor.name] = GGMLTensor(
            data=torch_tensor,
            tensor_type=tensor.tensor_type,
            tensor_shape=shape,
            aligned=True,  # 启用内存对齐
            pin_memory=False  # 根据需求调整
        )

        # 统计加载的张量类型
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
    
    # 打印加载统计信息
    logger.info("Loaded tensor types:")
    for qtype, count in qtype_dict.items():
        logger.info(f"  {qtype}: {count}")
    
    if return_arch:
        # 提取模型架构信息
        arch = get_model_architecture(reader)
        return state_dict, arch
    
    return state_dict


# 辅助函数
def get_orig_shape(reader, tensor_name: str) -> Optional[Tuple[int, ...]]:
    """从GGUF读取器获取原始张量形状"""
    # 实现根据GGUF格式获取原始形状的逻辑
    return None

def get_model_architecture(reader) -> str:
    """从GGUF读取器获取模型架构信息"""
    # 实现获取模型架构的逻辑
    return "unknown"

# for remapping llama.cpp -> original key names
# TODO 转模型的时候就把这些key对应好
T5_SD_MAP = {
    "enc.blk": "blocks",
    "token_embd": "token_embedding",
    "enc.output_norm": "norm",
    "attn_norm": "norm1",
    "attn_q": "attn.q",
    "attn_k": "attn.k",
    "attn_v": "attn.v",
    "attn_o": "attn.o",
    "attn_rel_b": "pos_embedding.embedding",
    "ffn_up": "ffn.fc1",
    "ffn_down": "ffn.fc2",
    "ffn_gate": "ffn.gate.0",
    "ffn_norm": "norm2",

    # "attn_q": "layer.0.SelfAttention.q",
    # "attn_k": "layer.0.SelfAttention.k",
    # "attn_v": "layer.0.SelfAttention.v",
    # "attn_o": "layer.0.SelfAttention.o",
    # "attn_norm": "layer.0.norm1",
    # "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    # "ffn_up": "layer.1.DenseReluDense.wi_1",
    # "ffn_down": "layer.1.DenseReluDense.wo",
    # "ffn_gate": "layer.1.DenseReluDense.wi_0",
    # "ffn_norm": "layer.1.norm2",
}


def load_gguf_clip_ckpt(path):
    sd, arch = load_gguf_sd_ckpt(path, return_arch=True)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        sd = sd_map_replace(sd, T5_SD_MAP)
    else:
        pass
    return sd
