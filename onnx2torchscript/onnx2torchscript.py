from ast import Call
from typing import Any, Callable, Dict, Optional, Tuple

import onnx
import onnx.numpy_helper
import torch


_op_table: Dict[int, Dict[str, torch._C.ScriptFunction]] = {}
_version_cache: Dict[Tuple[str, int], torch._C.ScriptFunction] = {}


def onnx_op(op_type: str, opset_version: int = 0, domain: str = "") -> Callable:
    def fn(f: Callable) -> Callable:
        ret = torch.jit.script(f)
        _op_table.setdefault(opset_version, {})
        assert op_type not in _op_table[opset_version]
        _op_table[opset_version][op_type] = ret
        _version_cache = {}  # Clear cache
        return ret
    return fn


binary_ops = {
    "Add-1": torch.add,
    "And-1": torch.logical_and,
    "BitwiseAnd-18": torch.bitwise_and,
    "BitwiseOr-18": torch.bitwise_xor,
    "Greater-1": torch.greater,
    "Less-1": torch.less,
    "Mul-1": torch.mul,
    # "Pow-1": torch.pow,
    "Sub-1": torch.sub,
}

for k, o in binary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return o(x, y)

unary_ops = {
    "Abs-1": torch.abs,
    "Acos-7": torch.acos,
    "Acosh-9": torch.acosh,
    "Ceil-1": torch.ceil,
    "Log-1": torch.log,
    "Neg-1": torch.neg,
    "Not-1": torch.logical_not,
    "Reciprocal-1": torch.reciprocal,
    "Sign-9": torch.sign,
    "Sin-7": torch.sin,
    "Sinh-9": torch.sinh,
    "Tan-7": torch.tan,
    "Tanh-9": torch.tanh,
}

for k, o in unary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: torch.Tensor) -> torch.Tensor:
        return o(x)


@onnx_op("Gemm", 11)
def op_Gemm(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None,
    # *,  # Commenting out due to kwargs unsupported in trace mode
    alpha: float = 1.0, beta: float = 1.0, transA: int = 0, transB: int = 0
) -> torch.Tensor:
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
    value: torch.Tensor
) -> torch.Tensor:
    return value


def get_onnx_ts(op_type: str, opset_version: int = 0, domain: str = "") -> Optional[torch._C.ScriptFunction]:
    k = (op_type, opset_version)
    if k in _version_cache:
        return _version_cache[k]

    if opset_version <= 0:
        opset_version = onnx.defs.onnx_opset_version()
    while not(opset_version in _op_table and \
            op_type in _op_table[opset_version]):
        opset_version -= 1
        if opset_version == 0:
            return None
        continue

    ret = _version_cache[k] = _op_table[opset_version][op_type]
    return ret


def onnx2ts(model: onnx.ModelProto, args: Any, verbose: bool = False) -> torch._C.ScriptModule:
    def p(*a):
        if verbose:
            print(*a)

    domain2opset: Dict[str, int] = {}
    for o in model.opset_import:
        domain2opset[o.domain] = o.version

    class OnnxModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for i in model.graph.initializer:
                p = torch.from_numpy(onnx.numpy_helper.to_array(i).copy())
                setattr(self, i.name, torch.nn.parameter.Parameter(p))

        def forward(self, *args):
            values: Dict[str, Any] = {}
            for a, i in zip(args, model.graph.input):
                values[i.name] = a
            for i in model.graph.initializer:
                values[i.name] = getattr(self, i.name)
            for o_n in model.graph.node:
                o_sch = onnx.defs.get_schema(o_n.op_type, domain2opset[o_n.domain], o_n.domain)

                o_attrs: Dict[str, onnx.AttributeProto] = {}
                for name, o_sch_a in o_sch.attributes.items():
                    if o_sch_a.default_value is not None:
                        o_attrs[name] = o_sch_a.default_value
                for o_a in o_n.attribute:
                    o_attrs[o_a.name] = o_a

                o_attr_vals: Dict[str, Any] = {}
                for name, o_a in o_attrs.items():
                    if o_a.type == onnx.AttributeProto.AttributeType.UNDEFINED:
                        continue

                    attr_value = onnx.helper.get_attribute_value(o_a)
                    if o_a.type == onnx.AttributeProto.AttributeType.TENSOR:
                        attr_value = torch.from_numpy(onnx.numpy_helper.to_array(attr_value).copy())
                    o_attr_vals[name] = attr_value

                t_s = get_onnx_ts(o_n.op_type, domain2opset[o_n.domain], o_n.domain)
                if t_s is None:
                    raise NotImplementedError(f"{o_n.domain}::{o_n.op_type}-{domain2opset[o_n.domain]} not found")
                t_sch = t_s.schema
                ins = [values[n] for n in o_n.input]
                for idx in range(len(ins), len(t_sch.arguments)):
                    arg = t_sch.arguments[idx]
                    if arg.name in o_attr_vals:
                        ins.append(o_attr_vals[arg.name])
                    elif arg.has_default_value():
                        ins.append(arg.default_value)
                    else:
                        raise RuntimeError(f"{arg} not provided")
                outs = t_s(*ins)
                if not isinstance(outs, (tuple, list)):
                    outs = (outs,)

                for n, o in zip(o_n.output, outs):
                    assert n not in values
                    values[n] = o

            ret = tuple([values[i.name] for i in model.graph.output])
            if len(ret) == 1:
                return ret[0]
            return ret

    m = OnnxModule()
    t = torch.jit.trace(m, args, check_trace=False)

    return t
