from ast import Call
from fnmatch import fnmatch
from typing import Callable, Dict, Optional, Tuple


import onnx
import torch


_op_table: Dict[int, Dict[str, torch._C.Graph]] = {}
_version_cache: Dict[Tuple[str, int], torch._C.Graph] = {}


def onnx_op(op_type: str, opset_version: int = 0, domain = "") -> Callable:
    def fn(f: Callable) -> Callable:
        ret = torch.jit.script(f)
        _op_table.setdefault(opset_version, {})
        assert op_type not in _op_table[opset_version]
        _op_table[opset_version][op_type] = ret.graph
        _version_cache = {}  # Clear cache
        print(ret)
        # print(ret.graph)
        return ret
    return fn


@onnx_op("Add", 1)
def op_Add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


@onnx_op("Gemm", 11)
def op_Gemm(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None,
    alpha = 1.0, beta = 1.0, transA = 0, transB = 0
) -> torch.Tensor:
    if transA:
        a = a.swapaxes(-1, -2)
    if transB:
        b = b.swapaxes(-1, -2)
    if c is not None:
        return torch.addbmm(c, a, b, beta=beta, alpha=alpha)
    else:
        return torch.mm(a, b) * alpha


def graph_of_onnx_op(op_type: str, opset_version: int = 0) -> torch._C.Graph:
    k = (op_type, opset_version)
    if k in _version_cache:
        return _version_cache[k]

    if opset_version <= 0:
        opset_version = onnx.defs.onnx_opset_version()
    while opset_version not in _op_table or \
            op_type not in _op_table[opset_version]:
        opset_version -= 1
        if opset_version == 0:
            raise NotImplementedError(f"${op_type}-${opset_version} not found")
        continue

    ret = _version_cache[k] = _op_table[opset_version][op_type]
    return ret


def onnx2ts(model: onnx.ModelProto, verbose: bool = False) -> torch._C.Graph:
    def p(*a):
        if verbose:
            print(*a)

    ret: torch._C.Graph = torch._C.Graph()

    values: Dict[str, torch._C.Value] = {}
    for v in model.graph.input:
        values[v.name] = ret.addInput()
        values[v.name].setDebugName(v.name)

    domain2opset: Dict[str, int] = {}
    for o in model.opset_import:
        domain2opset[o.domain] = o.version

    model_output_names = [v.name for v in model.graph.output]

    for o_n in model.graph.node:
        t_g = graph_of_onnx_op(o_n.op_type, domain2opset[o_n.domain])

        t_values: Dict[str, str] = {}
        for o_i, t_i in zip(o_n.input, t_g.inputs()):
            t_i.setDebugName(f"{o_n.name}_{t_i.debugName()}")
            t_values[t_i.debugName()] = o_i

        for t_n in t_g.nodes():
            ins = [values[t_values[t_i.debugName()]] for t_i in t_n.inputs()]
            op_name = t_n.kind()
            c = ret.create(op_name, ins, len(list(t_n.outputs())))
            c.copyAttributes(t_n)
            for c_o, t_o in zip(c.outputs(), t_n.outputs()):
                c_o.setDebugName(f"{o_n.name}_{t_n.kind().split('::')[-1]}_{t_o.debugName()}")
                values[c_o.debugName()] = c_o
                t_values[t_o.debugName()] = c_o.debugName()
                c_o.setType(t_o.type())
            ret.insertNode(c)

        for o_o, t_o in zip(o_n.output, t_g.outputs()):
            if o_o in model_output_names:
                values[o_o] = values[t_values[t_o.debugName()]]

    for v in model.graph.output:
        ret.registerOutput(values[v.name])

    return ret
