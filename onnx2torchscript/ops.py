from math import ceil
from opcode import hasjabs
from optparse import Option
from typing import Callable, Dict, List, Optional, Union, Tuple
from torch import Tensor
from torch.jit import annotate
from onnx2torchscript import onnx_op
import torch
import onnx
import pytorch_pfn_extras as ppe


binary_ops: Dict[str, Callable] = {
    "Add-1": torch.add,
    "And-1": torch.logical_and,
    "BitwiseAnd-18": torch.bitwise_and,
    "BitwiseOr-18": torch.bitwise_xor,
    "Div-1": torch.true_divide,
    "Equal-1": torch.eq,
    "Greater-1": torch.greater,
    "GreaterOrEqual-12": torch.greater_equal,
    "Less-1": torch.less,
    "LessOrEqual-12": torch.less_equal,
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
    op.__nmae__ = f"op_{op_type}_{ver}"

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
    "Erf-9": torch.erf,
    "Exp-1": torch.exp,
    "Floor-1": torch.floor,
    "IsNaN-9": torch.isnan,
    "Log-1": torch.log,
    "Neg-1": torch.neg,
    "Not-1": torch.logical_not,
    "Reciprocal-1": torch.reciprocal,
    "Relu-1": torch.relu,
    "Round-11": torch.round,
    "Sigmoid-1": torch.sigmoid,
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
    op.__nmae__ = f"op_{op_type}_{ver}"


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
    value: Optional[Tensor] = None,
    value_int: Optional[int] = None,
) -> Tensor:
    if value_int is not None:
        return torch.scalar_tensor(value_int, dtype=torch.int64)
    assert value is not None
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


@onnx_op("ConvTranspose", 1)
def op_ConvTranspose(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None,
    # *,
    dilations: Optional[List[int]] = None,  group: int = 1, kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None, strides: Optional[List[int]] = None,
    output_padding: Optional[List[int]] = None,
) -> Tensor:
    if dilations is None:
        dilations = [1]
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    elif all([p == pads[0] for p in pads]):
        pads = [pads[0]]
    if output_padding is None:
        output_padding = [0]
    return torch.convolution(
        x, w, b,
        stride=strides, padding=pads, dilation=dilations, groups=group, 
        transposed=True, output_padding=output_padding)


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


OnnxAny = Union[Tensor, List[Tensor], Optional[Tensor], Optional[List[Tensor]]]

@onnx_op("Identity", 1)
def op_Identity(input: Tensor) -> Tensor:
    return input


@onnx_op("Reshape", 1)
def op_Reshape(
    data: Tensor, shape: Tensor,
    # *,
    allowzero: int = 0
) -> Tensor:
    shape = annotate(List[int], shape.tolist())
    if allowzero != 0:
        return torch.reshape(data, shape)

    new_shape: List[int] = []    
    for idx, d in enumerate(shape):
        new_shape.append(data.shape[idx] if d == 0 else d)
    return torch.reshape(data, new_shape)


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
    return torch.tensor(s[start:end], device=data.device)


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
    return torch.tile(input, annotate(List[int], repeats.tolist()))


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


@onnx_op("Size", 1)
def op_Size(data: Tensor) -> Tensor:
    return torch.scalar_tensor(data.numel(), dtype=torch.int64)


@onnx_op("Einsum", 12)
def op_Einsum(
    inputs: List[Tensor],
    # *,
    equation: str,
) -> Tensor:
    return torch.einsum(equation, inputs)


@onnx_op("Max", 1)
def op_Max(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = torch.maximum(ret, i)
    return ret


@onnx_op("Min", 1)
def op_Min(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = torch.minimum(ret, i)
    return ret


@onnx_op("Mean", 1)
def op_Mean(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = ret + i
    return ret / len(inputs)


@onnx_op("Sum", 1)
def op_Sum(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = ret + i
    return ret


@onnx_op("Mod", 1)
def op_Mod(
    a: Tensor, b: Tensor,
    # *,
    fmod: int = 0,
) -> Tensor:
    if fmod:
        return torch.fmod(a, b)
    else:
        return a % b


@onnx_op("PRelu", 1)
def op_PRelu(x: Tensor, slope: Tensor) -> Tensor:
    return torch.prelu(x, slope)


@onnx_op("SequenceLength", 11)
def op_SequenceLength(x: List[Tensor]) -> Tensor:
    return torch.scalar_tensor(len(x), dtype=torch.int64)


@onnx_op("ReduceMax", 1)
def op_ReduceMax(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.max(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.amax(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceMin", 1)
def op_ReduceMin(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.min(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.amin(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceL1", 1)
def op_ReduceL1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = torch.abs(data)
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceL2", 1)
def op_ReduceL2(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = data * data
    if axes is None:
        ret = torch.sqrt(torch.sum(data))
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sqrt(torch.sum(data, dim=axes, keepdim=keepdims != 0))


@onnx_op("ReduceSum", 1)
def op_ReduceSum_1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tensor:
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceSum", 13)
def op_ReduceSum_13(
    data: Tensor,
    axes: Optional[Tensor] = None,
    # *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tensor:
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        if axes.numel() == 0 and noop_with_empty_axes != 0:
            return data
        return torch.sum(data, dim=annotate(List[int], axes.tolist()), keepdim=keepdims != 0)


@onnx_op("ReduceLogSum", 1)
def op_ReduceLogSum_1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    return torch.log(op_ReduceSum_1(data, axes=axes, keepdims=keepdims))


@onnx_op("ReduceLogSumExp", 1)
def op_ReduceLogSumExp_1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    return op_ReduceLogSum_1(torch.exp(data), axes=axes, keepdims=keepdims)


@onnx_op("ReduceSumSquare", 1)
def op_ReduceSumSquare(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = data * data
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceMean", 1)
def op_ReduceMean(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.mean(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.mean(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceProd", 1)
def op_ReduceProd(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.prod(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        assert len(axes) == 1
        return torch.prod(data, dim=axes[0], keepdim=keepdims != 0)


@onnx_op("Clip", 1)
def op_Clip_1(
    input: Tensor,
    # *,
    min: float = float("+inf"), max: float = float("-inf"),
) -> Tensor:
    return torch.clamp(input, min, max)


@onnx_op("Clip", 11)
def op_Clip_11(input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
    if min is not None and max is not None:
        return torch.clamp(input, min, max)
    elif min is not None:
        return torch.clamp_min(input, min)
    elif max is not None:
        return torch.clamp_max(input, max)
    return input


@onnx_op("Concat", 1)
def op_Concat(
    inputs: List[Tensor],
    # *,
    axis: int,
) -> Tensor:
    return torch.concat(inputs, dim=axis)


@onnx_op("Squeeze", 13)
def op_Squeeze_13(data: Tensor, axes: Optional[Tensor] = None) -> Tensor:
    if axes is None:
        return torch.squeeze(data)
    axes_list = annotate(List[int], axes.tolist())
    axes_list = [a if a >= 0 else a + data.dim() for a in axes_list]
    s = data.shape
    for a in sorted(axes_list):
        assert s[a] == 1
        del s[a]
    return torch.reshape(data, s)


@onnx_op("Squeeze", 1)
def op_Squeeze_1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None
) -> Tensor:
    if axes is None:
        return torch.squeeze(data)
    axes_list = axes
    axes_list = [a if a >= 0 else a + data.dim() for a in axes_list]
    s = data.shape
    for a in sorted(axes_list):
        assert s[a] == 1
        del s[a]
    return torch.reshape(data, s)


@onnx_op("Unsqueeze", 13)
def op_Unsqueeze_13(data: Tensor, axes: Tensor) -> Tensor:
    axes_list = annotate(List[int], axes.tolist())
    axes_list = [a if a >= 0 else a + data.dim() for a in axes_list]
    ret = data
    for a in sorted(axes_list):
        ret = torch.unsqueeze(ret, dim=a)
    return ret


@onnx_op("Unsqueeze", 1)
def op_Unsqueeze_1(
    data: Tensor,
    # *,
    axes: List[int],
) -> Tensor:
    axes_list = [a if a >= 0 else a + data.dim() for a in axes]
    ret = data
    for a in sorted(axes_list):
        ret = torch.unsqueeze(ret, dim=a)
    return ret


@torch.jit.script
def _to_torch_dtype(d: int) -> torch.dtype:
    table: Dict[int, torch.dtype] = {
        1: torch.float,
        2: torch.uint8,
        3: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        11: torch.double,
        9: torch.bool,
    }

    return table[d]


@onnx_op("Cast", 1)
def op_Cast(
    input: Tensor,
    # *,
    to: int,
) -> Tensor:
    return  input.to(_to_torch_dtype(to))


@onnx_op("Compress", 9)
def op_Compress(
    input: Tensor, condition: Tensor,
    # *,
    axis: Optional[int] = None
) -> Tensor:
    if axis is None:
        input = torch.flatten(input)
        axis = 0
    return torch.index_select(input, dim=axis, index=torch.nonzero(condition).squeeze())


@onnx_op("GlobalMaxPool", 1)
def op_GlobalMaxPool(x: Tensor) -> Tensor:
    dims = annotate(List[int], [])
    for i in range(x.dim() - 2):
        dims.append(i + 2)
    return torch.amax(x, dim=dims, keepdim=True)


@onnx_op("GlobalAveragePool", 1)
def op_GlobalAveragePool(x: Tensor) -> Tensor:
    dims = annotate(List[int], [])
    for i in range(x.dim() - 2):
        dims.append(i + 2)
    return torch.mean(x, dim=dims, keepdim=True)


@onnx_op("IsInf", 10)
def op_IsInf(
    x: Tensor,
    # *,
    detect_negative: int = 1, detect_positive: int = 1,
) -> Tensor:
    ret = torch.isinf(x)
    if detect_positive != 0 and detect_negative == 0:
        return torch.logical_and((x > 0), ret)
    if detect_positive == 0 and detect_negative != 0:
        return torch.logical_and((x < 0), ret)
    return ret


@onnx_op("GridSample", 16)
def op_GridSample(
    x: Tensor, grid: Tensor,
    # *,
    align_corners: int = 0, mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    # ref: aten/src/ATen/native/GridSamplerUtils.h
    i = 0
    if mode == "bilinear":
        i = 0
    elif mode == "nearest":
        i = 1
    else:
        assert mode == "bicubic"
        i = 2
    p = 0
    if padding_mode == "zeros":
        p = 0
    elif padding_mode == "border":
        p = 1
    else:
        assert padding_mode == "reflection"
        p = 2
    return torch.grid_sampler(
        x, grid,
        interpolation_mode=i, padding_mode=p,
        align_corners=align_corners != 0)


@onnx_op("AveragePool", 1)
def op_AveragePool(
    x: Tensor,
    # *,
    ceil_mode: int = 0, count_include_pad: int = 0,
    kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Tensor:
    assert kernel_shape is not None
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    else:
        for i in range(len(kernel_shape)):
            assert pads[i] ==  pads[i + len(kernel_shape)]
        pads = pads[0:len(kernel_shape)]

    if len(kernel_shape) == 1:
        return torch.nn.functional.avg_pool1d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)
    elif len(kernel_shape) == 2:
        return torch.nn.functional.avg_pool2d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)
    else:
        assert len(kernel_shape) == 3
        return torch.nn.functional.avg_pool3d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)


@onnx_op("MaxPool", 1)
def op_MaxPool(
    x: Tensor,
    # *,
    ceil_mode: int = 0, count_include_pad: int = 0,
    dilations: Optional[List[int]] = None,
    kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    assert kernel_shape is not None
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    else:
        for i in range(len(kernel_shape)):
            assert pads[i] ==  pads[i + len(kernel_shape)]
        pads = pads[0:len(kernel_shape)]
    if dilations is None:
        dilations = [1]

    if len(kernel_shape) == 1:
        return torch.nn.functional.max_pool1d(
            x, kernel_shape, stride=strides, padding=pads,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)
    elif len(kernel_shape) == 2:
        return torch.nn.functional.max_pool2d(
            x, kernel_shape, stride=strides, padding=pads,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)
    else:
        assert len(kernel_shape) == 3
        return torch.nn.functional.max_pool3d(
            x, kernel_shape, stride=strides, padding=0,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)


@onnx_op("SequenceEmpty", 11)
def op_SequneceEmpty(
    # *,
) -> List[Tensor]:
    return annotate(List[Tensor], [])


@onnx_op("LeakyRelu", 1)
def op_LeakyRelu(
    x: Tensor,
    # *,
    alpha: float = 0.01
) -> Tensor:
    return torch.where(x >= 0, x, x * alpha)


@onnx_op("LayerNormalization", 17)
def op_LayerNormalization(
    x: Tensor, scale: Tensor, b: Optional[Tensor] = None,
    # *,
    axis: int = -1, epsilon: float = 1e-5, stash_type: int = 1
) -> Tuple[Tensor, Tensor, Tensor]:
    s: List[int] = []
    if axis < 0:
        axis = x.dim() + axis
    for i in range(axis, x.dim()):
        s.append(x.shape[i])

    return torch.native_layer_norm(x, normalized_shape=s, weight=scale, bias=b, eps=epsilon)


@onnx_op("ConstantOfShape", 9)
def op_ConstantOfShape(
    input: Tensor,
    # *,
    value: Optional[Tensor] = None
) -> Tensor:
    if value is None:
        value = torch.scalar_tensor(0.0)
    return value.expand(annotate(List[int], input.tolist()))


@onnx_op("CumSum", 11)
def op_CumSum(
    x: Tensor, axis: Tensor,
    # *,
    exclusive: int = 0, reverse: int = 0,
) -> Tensor:
    assert exclusive == 0
    assert reverse == 0
    return torch.cumsum(x, axis.item())


@onnx_op("InstanceNormalization", 1)
def op_InstanceNormalization(
    input: Tensor, scale: Tensor, b: Tensor,
    # *,
    epsilon: float = 1e-5,
) -> Tensor:
    return torch.instance_norm(
        input, weight=scale, bias=b,
        running_mean=None, running_var=None, use_input_stats=True,
        momentum=0.0, cudnn_enabled=True,
        eps=epsilon)


@onnx_op("Softplus", 1)
def op_Softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) + 1)


@onnx_op("CastLike", 15)
def op_CastLike(input: Tensor, target_type: Tensor) -> Tensor:
    return input.to(dtype=target_type.dtype)


@onnx_op("Selu", 1)
def op_Selu(
    x: Tensor,
    # *,
    alpha: float = 1.67326, gamma: float = 1.0507,
) -> Tensor:
    return torch.where(
        x <= 0,
        gamma * (alpha * torch.exp(x) - alpha),
        gamma * x)


@onnx_op("ThresholdedRelu", 10)
def op_ThresholdedRelu(
    x: Tensor,
    # *,
    alpha: float = 1.0
) -> Tensor:
    return torch.where(x > alpha, x, torch.zeros_like(x))


@onnx_op("HardSigmoid", 6)
def op_HardSigmoid(
    x: Tensor,
    # *,
    alpha: float = 0.2, beta: float = 0.4
) -> Tensor:
    return torch.clamp(alpha * x + beta, 0, 1)


@onnx_op("HardSwish", 14)
def op_HardSwish(x: Tensor) -> Tensor:
    return x * op_HardSigmoid(x, alpha=1.0 / 6, beta=0.5)


@onnx_op("Expand", 1)
def op_Expannd(input: Tensor, shape: Tensor) -> Tensor:
    t = annotate(List[int], shape.tolist())
    f = input.shape

    if len(t) > len(f):
        t, f = f, t
    offset = len(f) - len(t)
    ex: List[int] = []
    for idx in range(len(f)):
        ex.append(max(f[idx], t[idx - offset] if idx >= offset else 1))

    return input.expand(ex)


@onnx_op("Split", 13)
def op_Split_13(
    input: Tensor, split: Optional[Tensor] = None,
    # *,
    axis: int = 0, num_outputs: Optional[int] = None,
    _num_outputs: int = 0,  # Special argument name
) -> List[Tensor]:
    if split is None:
        if num_outputs is None:
            num_outputs = _num_outputs
        return torch.split(input, int(ceil(input.size(axis) / num_outputs)), dim=axis)
    else:
        return torch.split(input, annotate(List[int], split.tolist()), dim=axis)


@onnx_op("Split", 1)
def op_Split_1(
    input: Tensor,
    # *,
    axis: int = 0, split: Optional[List[int]] = None,
    _num_outputs: int = 0,  # Special argument name
) -> List[Tensor]:
    if split is None:
        num_outputs = _num_outputs
        return torch.split(input, int(ceil(input.size(axis) / num_outputs)), dim=axis)
    else:
        return torch.split(input, split, dim=axis)


@onnx_op("Flatten", 1)
def op_Flatten(
    input: Tensor,
    # *,
    axis: int = 1,
) -> Tensor:
    if axis < 0:
        axis += input.dim()
    s = [1, 1]
    for i in range(axis):
        s[0] *= input.size(i)
    for i in range(axis, input.dim()):
        s[1] *= input.size(i)
    return torch.reshape(input, s)


@onnx_op("Shrink", 9)
def op_Shrink(
    x: Tensor,
    # *,
    bias: float = 0.0, lambd: float = 0.5
) -> Tensor:
    return torch.where(
        x < -lambd,
        x + bias,
        torch.where(
            x > lambd,
            x - bias,
            torch.zeros_like(x)))


@onnx_op("Softsign", 1)
def op_Softsign(x: Tensor) -> Tensor:
    return x / (1 + torch.abs(x))


@onnx_op("NonZero", 9)
def op_NonZero(x: Tensor) -> Tensor:
    return torch.nonzero(x).transpose(0, 1)


@onnx_op("SplitToSequence", 11)
def op_SplitToSequence(
    input: Tensor, split: Optional[Tensor] = None,
    # *,
    axis: int = 0, keepdims: int = 1,
) -> List[Tensor]:
    if split is None:
        ret = list(torch.split(input, 1, dim=axis))
        if keepdims == 0:
            ret = [op_Squeeze_1(t, [axis]) for t in ret]
        return ret
    if split.numel() == 1:
        return list(torch.split(input, split.item(), dim=axis))
    else:
        return list(torch.split(
            input, annotate(List[int], split.tolist()), dim=axis))


@onnx_op("SequenceInsert", 11)
def op_SequenceInsert(
    s: List[Tensor], t: Tensor, p: Optional[Tensor] = None,
) -> List[Tensor]:
    s = s.copy()
    if p is None:
        s.append(t)
        return s
    s.insert(p.item(), t)
    return s


@onnx_op("Slice", 10)
def op_Slice_10(
    data: Tensor, starts: Tensor, ends: Tensor,
    axes: Optional[Tensor] = None, steps: Optional[Tensor] = None,
) -> Tensor:
    axes_list: List[int] = []
    assert starts.numel() == ends.numel()
    if axes is None:
        axes_list = list(range(starts.numel()))
    else:
        axes_list = annotate(List[int], axes.tolist())
    
    steps_list: List[int] = []
    if steps is None:
        steps_list = [1] * starts.numel()
    else:
        assert steps.numel() == starts.numel()
        steps_list = annotate(List[int], steps.tolist())

    ret = data
    for a, s, e, step in zip(
        axes_list,
        annotate(List[int], starts.tolist()),
        annotate(List[int], ends.tolist()),
        steps_list
    ):
        if s < 0:
            s += data.size(a)
        if e < 0:
            e += data.size(a)
        e = min(data.size(a), e)
        s = min(data.size(a), s)
        if s > e:
            s, e = e, s
            step = -step
        idx = torch.arange(s, e, step, device=data.device)
        ret = torch.index_select(ret, dim=a, index=idx)
    return ret


@onnx_op("Gather", 1)
def op_Gather(
    data: Tensor, indices: Tensor,
    # *,
    axis: int = 0,
) -> Tensor:
    out_shape = data.shape[:axis] + indices.shape + data.shape[axis + 1:]
    indices = torch.where(indices < 0, indices + data.size(axis), indices).flatten()
    return torch.index_select(data, dim=axis, index=indices).reshape(out_shape)


@onnx_op("GatherND", 1)
def op_GatherND(
    data: Tensor, indices: Tensor,
    # *,
    batch_dims: int = 0,
) -> Tensor:
    assert indices.shape[-1] <= len(data.shape)

    batch_dims_shape = indices.shape[0:batch_dims]
    batch_dims_numel = int(torch.prod(torch.tensor(batch_dims_shape)).item())

    reshaped_indices = indices.reshape([batch_dims_numel, -1, indices.shape[-1]])
    if indices.shape[-1] == len(data.shape):
        output_shape = batch_dims_shape + indices.shape[batch_dims:-1]
        reshaped_data = data.reshape([batch_dims_numel, -1])
        shapes = data.shape[-indices.shape[-1] + 1:] + [1]
    else:
        rest_dims = data.shape[batch_dims + indices.shape[-1]:]
        output_shape = batch_dims_shape + indices.shape[batch_dims:-1] + rest_dims
        reshaped_data = data.reshape([batch_dims_numel, -1] + rest_dims)
        shapes = data.shape[batch_dims + 1:-len(rest_dims)] + [1]

    shapes.reverse()
    idx_prod = torch.tensor(shapes).cumprod(0).flip(0).reshape(-1, 1)
    assert len(shapes) == indices.shape[-1]

    out_buf: List[Tensor] = []
    for batch_dim in range(batch_dims_numel):
        idx = reshaped_indices[batch_dim]
        idx = torch.mm(idx.cpu(), idx_prod.cpu()).to(data.device).reshape(-1)
        out_buf.append(reshaped_data[batch_dim].index_select(dim=0, index=idx))
    out_buf = torch.stack(out_buf)
    return out_buf.reshape(output_shape)



@onnx_op("GatherElements", 11)
def op_GatherElements(
    data: Tensor, indices: Tensor,
    # *,
    axis: int = 0,
) -> Tensor:
    indices = torch.where(indices < 0, indices + data.size(axis), indices)
    return torch.gather(data, dim=axis, index=indices)


if ppe.requires("1.12"):
    @onnx_op("ScatterElements", 11)
    def op_ScatterElements(
        data: Tensor, indices: Tensor, updates: Tensor,
        # *,
        axis: int = 0, reduction: Optional[str] = None,
    ) -> Tensor:
        indices = torch.where(indices < 0, indices + data.size(axis), indices)
        if reduction is None or reduction == "none":
            return torch.scatter(data, dim=axis, index=indices, src=updates)

        if reduction == "add":
            reduction = "sum"
        elif reduction == "mul":
            reduction = "prod"
        elif reduction == "mean":
            reduction = "mean"
        elif reduction == "max":
            reduction = "amax"
        else:
            assert reduction == "min"
            reduction = "amin"
        return torch.scatter_reduce(
            data, dim=axis, index=indices, src=updates, reduce=reduction)
else:
    @onnx_op("ScatterElements", 11)
    def op_ScatterElements(
        data: Tensor, indices: Tensor, updates: Tensor,
        # *,
        axis: int = 0, reduction: Optional[str] = None,
    ) -> Tensor:
        indices = torch.where(indices < 0, indices + data.size(axis), indices)
        if reduction is None or reduction == "none":
            return torch.scatter(data, dim=axis, index=indices, src=updates)
        elif reduction == "add":
            reduction = "add"
        else:
            assert reduction == "mul"
            reduction = "multiply"
        return torch.scatter(
            data, dim=axis, index=indices, src=updates,
            reduce=reduction)


@onnx_op("Scatter", 9)
def op_Scatter(
    data: Tensor, indices: Tensor, updates: Tensor,
    # *,
    axis: int = 0, reduction: Optional[str] = None,
) -> Tensor:
    return op_ScatterElements(
        data, indices, updates, axis=axis, reduction=reduction)


@onnx_op("SequenceAt", 11)
def op_SequenceAt(input_sequence: List[Tensor], position: Tensor) -> Tensor:
    return input_sequence[position.item()]


@onnx_op("SequenceConstruct", 11)
def op_SequenceConstruct(inputs: List[Tensor]) -> List[Tensor]:
    return inputs


@onnx_op("SequenceErase", 11)
def op_SequenceErase(inputs: List[Tensor], position: Optional[Tensor] = None) -> List[Tensor]:
    inputs = inputs.copy()
    if position is None:
        del inputs[-1]
        return inputs
    del inputs[int(position.item())]
    return inputs


@onnx_op("ConcatFromSequence", 11)
def op_ConcatFromSequence(
    inputs: List[Tensor],
    # *,
    axis: Optional[int] = None, new_axis: int = 0, 
) -> Tensor:
    assert axis is not None
    if new_axis != 0:
        return torch.stack(inputs, dim=axis)
    else:
        return torch.concat(inputs, dim=axis)


if hasattr(torch, "index_reduce"):
    @onnx_op("ScatterND", 11)
    def op_ScatterND(
        data: Tensor, indices: Tensor, updates: Tensor,
        # *,
        reduction: str = "none",
    ) -> Tensor:
        assert data.dim() >= 1
        assert indices.dim() > 1
        assert (data.dim() + indices.dim()) - (indices.shape[-1] + 1)
        out = data.clone()
        idx = indices.reshape(-1, indices.shape[-1])
        nd = torch.tensor([1] + data.shape[0:data.dim() - indices.dim() - 1], device=indices.device).reshape(1, -1)
        idx = torch.mm(idx, nd).reshape(-1)
        assert idx.shape[0] == torch.prod(torch.tensor(indices.shape[:-1]))
        updates = updates.reshape([-1] + updates.shape[indices.dim() - 1:])
        out = out.reshape([-1] + data.shape[data.dim() - indices.dim():])
        if reduction == "none":
            out.index_copy_(dim=0, index=idx, source=updates)
        elif reduction == "add":
            out.index_add_(dim=0, index=idx, source=updates)
        else:
            if reduction == "mul":
                reduction = "prod"
            elif reduction == "min":
                reduction = "amin"
            elif reduction == "max":
                reduction = "amax"
            else:
                assert reduction == "mean"
                reduction = "mean"
            out = out.index_reduce_(dim=0, index=idx, source=updates, reduce=reduction)
        return out.reshape(data.shape)
else:
    @onnx_op("ScatterND", 11)
    def op_ScatterND(
        data: Tensor, indices: Tensor, updates: Tensor,
        # *,
        reduction: str = "none",
    ) -> Tensor:
        assert data.dim() >= 1
        assert indices.dim() > 1
        assert (data.dim() + indices.dim()) - (indices.shape[-1] + 1)
        out = data.clone()
        idx = indices.reshape(-1, indices.shape[-1])
        nd = torch.tensor([1] + data.shape[0:data.dim() - indices.dim() - 1], device=indices.device).reshape(1, -1)
        idx = torch.mm(idx, nd).reshape(-1)
        assert idx.shape[0] == torch.prod(torch.tensor(indices.shape[:-1]))
        updates = updates.reshape([-1] + updates.shape[indices.dim() - 1:])
        out = out.reshape([-1] + data.shape[data.dim() - indices.dim():])
        if reduction == "none":
            out.index_copy_(dim=0, index=idx, source=updates)
        else:
            assert reduction == "add"
            out.index_add_(dim=0, index=idx, source=updates)
        return out.reshape(data.shape)



@onnx_op("Elu", 1)
def op_Elu(
    x: Tensor,
    # *,
    alpha: float = 1.0,
) -> Tensor:
    return torch.where(x < 0, alpha * (torch.exp(x) - 1.0), x)


@onnx_op("Celu", 12)
def op_Celu(
    x: Tensor,
    # *,
    alpha: float = 1.0,
) -> Tensor:
    return torch.clamp_min(x, 0) + torch.clamp_max(alpha * torch.exp(x / alpha) - 1, 0)


def resolve_negative_indices(data: Tensor, indices: Tensor, axis: int) -> Tensor:
    indices = indices.to(torch.long)
    if indices.numel() == 0:
        return indices
    return torch.where(indices < 0, indices + data.shape[axis], indices)


@onnx_op("Gather", 13)
def op_Gather(
    data: Tensor, indices: Tensor,
    # *,
    axis: int = 0,
) -> Tensor:
    positive_indices = resolve_negative_indices(data, indices, axis)
    indices_collapsed = positive_indices.reshape(indices.numel())
    shape = data.shape[0:axis] + indices.shape + data.shape[axis + 1:]
    return data.index_select(dim=axis, index=indices_collapsed).reshape(shape)


@onnx_op("OptionalGetElement", 15)
def op_OptionalGetElement(input: Optional[OnnxAny]) -> OnnxAny:
    assert input is not None
    return input

@onnx_op("Bernoulli", 15)
def op_Bernoulli(
    input: Tensor,
    # *,
    dtype: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    dtype = _to_torch_dtype(dtype) if dtype is not None else input.dtype
    out = torch.empty_like(input, dtype=dtype)
    return torch.bernoulli(input, out=out)


@onnx_op("NegativeLogLikelihoodLoss", 12)
def op_nll_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None,
    # *,
    ignore_index: Optional[int] = None,
    reduction: str = "none",
) -> Tensor:
    if reduction == '':
        reduction = 'none'
    return torch.nn.functional.nll_loss(
        input, target, weight, 
        ignore_index=(ignore_index if ignore_index is not None else -100),
        reduction=reduction)


@onnx_op("SoftmaxCrossEntropyLoss", 12)
def op_sce_loss(
    scores: Tensor, labels: Tensor, weights: Optional[Tensor] = None,
    # *,
    ignore_index: Optional[int] = None,
    reduction: str = "none",
) -> Tensor:
    if reduction == '':
        reduction = 'none'
    return torch.nn.functional.cross_entropy(
        torch.log_softmax(scores, dim=-1), labels, weights, 
        ignore_index=(ignore_index if ignore_index is not None else -100),
        reduction=reduction)
