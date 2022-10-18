from onnx2torchscript import onnx_op
from typing import Callable, Dict, List, Optional, Union, Tuple
from torch import Tensor
import torch


binary_ops: Dict[str, Callable] = {
    "Add-1": torch.add,
    "And-1": torch.logical_and,
    "BitwiseAnd-18": torch.bitwise_and,
    "BitwiseOr-18": torch.bitwise_xor,
    "Div-1": torch.true_divide,
    "Equal-1": torch.eq,
    "Greater-1": torch.greater,
    "Less-1": torch.less,
    "MatMul-1": torch.matmul,
    "Mul-1": torch.mul,
    "Or-7": torch.logical_or,
    "Sub-1": torch.sub,
    "Xor-1": torch.logical_xor,
}

for k, o in binary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: Tensor, y: Tensor) -> Tensor:
        return o(x, y)

unary_ops: Dict[str, Callable] = {
    "Abs-1": torch.abs,
    "Acos-7": torch.acos,
    "Acosh-9": torch.acosh,
    "Asin-7": torch.asin,
    "Asinh-9": torch.asinh,
    "Atan-7": torch.atan,
    "Atanh-9": torch.atanh,
    "Ceil-1": torch.ceil,
    "Cos-7": torch.cos,
    "Cosh-9": torch.cosh,
    "Det-11": torch.det,
    "Exp-1": torch.exp,
    "Floor-1": torch.floor,
    "IsNaN-9": torch.isnan,
    "Log-1": torch.log,
    "Neg-1": torch.neg,
    "Not-1": torch.logical_not,
    "Reciprocal-1": torch.reciprocal,
    "Relu-1": torch.relu,
    "Round-11": torch.round,
    "Sign-9": torch.sign,
    "Sin-7": torch.sin,
    "Sinh-9": torch.sinh,
    "Sqrt-1": torch.sqrt,
    "Tan-7": torch.tan,
    "Tanh-1": torch.tanh,
}

for k, o in unary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: Tensor) -> Tensor:
        return o(x)


@onnx_op("Gemm", 1)
def op_Gemm(
    a: Tensor, b: Tensor, c: Optional[Tensor] = None,
    # *,  # Commenting out due to kwargs unsupported in trace mode
    alpha: float = 1.0, beta: float = 1.0, transA: int = 0, transB: int = 0
) -> Tensor:
    if transA:
        a = a.swapaxes(-1, -2)
    if transB:
        b = b.swapaxes(-1, -2)
    if c is not None:
        return torch.addmm(c, a, b, beta=beta, alpha=alpha)
    else:
        return torch.mm(a, b) * alpha


@onnx_op("Constant", 1)
def op_Constant(
    # *,
    value: Tensor
) -> Tensor:
    return value


@onnx_op("Conv", 1)
def op_Conv(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None,
    # *,
    dilations: Optional[List[int]] = None,  group: int = 1, kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None, strides: Optional[List[int]] = None,
) -> Tensor:
    if dilations is None:
        dilations = [1]
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    elif all([p == pads[0] for p in pads]):
        pads = [pads[0]]
    return torch.convolution(
        x, w, b,
        stride=strides, padding=pads, dilation=dilations, groups=group, 
        transposed=False, output_padding=[0])


@onnx_op("BatchNormalization", 1)
def op_BatchNorm(
    x: Tensor, scale: Tensor, b: Tensor,
    input_mean: Tensor, input_var: Tensor,
    # *,
    epsilon: float = 1e-05, momentum: float = 0.9, training_mode: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.native_batch_norm(
        x, scale, b, input_mean, input_var, 
        training=training_mode != 0, momentum=momentum, eps=epsilon)


@onnx_op("Softmax", 1)
def op_Softmax(
    x: Tensor,
    # *,
    axis: int = -1
) -> Tensor:
    return torch.softmax(x, dim=axis)


@onnx_op("LogSoftmax", 1)
def op_LogSoftmax(
    x: Tensor,
    # *,
    axis: int = -1
) -> Tensor:
    return torch.log_softmax(x, dim=axis)


@onnx_op("Trilu", 14)
def op_Trilu(
    input: Tensor,
    k: Optional[Tensor] = None,
    # *,
    upper: int = 1
) -> Tensor:
    if k is None:
        k = torch.scalar_tensor(0)
    if upper:
        return torch.triu(input, diagonal=k.item())
    else:
        return torch.tril(input, diagonal=k.item())


@onnx_op("Where", 9)
def op_Where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return torch.where(cond != 0, x, y)


@onnx_op("TopK", 1)
def op_TopK(
    x: Tensor, k: Tensor,
    # *,
    axis: int = -1, largest: int = 1, sorted: int = 1,
) -> Tuple[Tensor, Tensor]:
    return torch.topk(x, k.item(), dim=axis, largest=largest != 0, sorted=sorted != 0)


OnnxAny = Union[Tensor, List[Tensor], Optional[Tensor]]

@onnx_op("Identity", 1)
def op_Identity(input: OnnxAny) -> OnnxAny:
    return input


@onnx_op("Reshape", 1)
def op_Reshape(
    data: Tensor, shape: Tensor,
    # *,
    allowzero: int = 0
) -> Tensor:
    return torch.reshape(data, torch.jit.annotate(List[int], shape.tolist()))


@onnx_op("BitShift", 11)
def op_BitShift(
    x: Tensor, y: Tensor,
    # *,
    direction: str,
) -> Tensor:
    if direction == "LEFT":
        return torch.bitwise_left_shift(x, y)
    else:
        assert direction == "RIGHT"
        return torch.bitwise_right_shift(x, y)


@onnx_op("Shape", 1)
def op_Shape(
    data: Tensor,
    # *,
    end: Optional[int] = None, start: int = 0,
) -> Tensor:
    s = data.shape
    if end is None:
        end = len(s)
    return torch.tensor(s[start:end])


@onnx_op("Transpose", 1)
def op_Transpose(
    data: Tensor,
    # *,
    perm: Optional[List[int]] = None,
) -> Tensor:
    if perm is None:
        l = list(range(data.dim()))
        l.reverse()
        return data.permute(l)
    return torch.permute(data, perm)


@onnx_op("Tile", 1)
def op_Tile(input: Tensor, repeats: Tensor) -> Tensor:
    return torch.tile(input, torch.jit.annotate(List[int], repeats.tolist()))


@onnx_op("Pow", 1)
def op_Pow(x: Tensor, y: Tensor) -> Tensor:
    return torch.pow(x, y).to(x.dtype)


@onnx_op("ArgMax", 1)
def op_ArgMax(
    data: Tensor,
    # *,
    axis: int = 0, keepdims: int = 1, select_last_index: int = 0,
) -> Tensor:
    return torch.argmax(data, dim=axis, keepdim=keepdims != 0)


@onnx_op("ArgMin", 1)
def op_ArgMin(
    data: Tensor,
    # *,
    axis: int = 0, keepdims: int = 1, select_last_index: int = 0,
) -> Tensor:
    return torch.argmin(data, dim=axis, keepdim=keepdims != 0)
