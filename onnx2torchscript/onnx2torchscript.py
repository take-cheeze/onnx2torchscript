from ast import Call
from curses.ascii import isalnum
from fnmatch import fnmatch
from pyclbr import Function
from typing import Any, Callable, Dict, Optional, Tuple

import onnx
import torch


_op_table: Dict[int, Dict[str, torch._C.ScriptFunction]] = {}
_version_cache: Dict[Tuple[str, int], torch._C.ScriptFunction] = {}


def onnx_op(op_type: str, opset_version: int = 0, domain = "") -> Callable:
    def fn(f: Callable) -> Callable:
        ret = torch.jit.script(f)
        _op_table.setdefault(opset_version, {})
        assert op_type not in _op_table[opset_version]
        _op_table[opset_version][op_type] = ret
        _version_cache = {}  # Clear cache
        return ret
    return fn


@onnx_op("Add", 1)
def op_Add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

@onnx_op("Gemm", 11)
def op_Gemm(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None,
    *,
    alpha: float, beta: float, transA: int, transB: int
) -> torch.Tensor:
    if transA:
        a = a.swapaxes(-1, -2)
    if transB:
        b = b.swapaxes(-1, -2)
    if c is not None:
        return torch.addmm(c, a, b, beta=beta, alpha=alpha)
    else:
        return torch.mm(a, b) * alpha


def get_onnx_ts(op_type: str, opset_version: int = 0, domain: str = "") -> torch._C.ScriptFunction:
    k = (op_type, opset_version)
    if k in _version_cache:
        return _version_cache[k]

    if opset_version <= 0:
        opset_version = onnx.defs.onnx_opset_version()
    while opset_version not in _op_table or \
            op_type not in _op_table[opset_version]:
        opset_version -= 1
        if opset_version == 0:
            raise NotImplementedError(f"{domain}::{op_type}-{opset_version} not found")
        continue

    ret = _version_cache[k] = _op_table[opset_version][op_type]
    return ret


def _copy_node(dst: torch._C.Node, src: torch._C.Node, t_values: Dict[str, str], values: Dict[str, torch._C.Value]) -> None:
    dst.copyAttributes(src)
    for b in src.blocks():
        new_b = dst.addBlock()

        for n in b.nodes():
            new_inputs = []
            for i in n.inputs():
                new_i = values[t_values[i.debugName()]]
                new_inputs.append(new_i)
            new_n = new_b.addNode(n.kind(), new_inputs)
            for o in n.outputs():
                new_o = new_n.addOutput()
                new_o_name = o.debugName()
                if new_o_name.isnumeric():
                    new_o_name = f"{n.kind().split('::')[-1]}_{new_o_name}"
                new_o.setDebugName(new_o_name)
                t_values[o.debugName()] = new_o.debugName()
                values[new_o.debugName()] = new_o
            _copy_node(new_n, n, t_values, values)

        for o in b.outputs():
            new_b.registerOutput(values[t_values[o.debugName()]])


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
        t_s = get_onnx_ts(o_n.op_type, domain2opset[o_n.domain], o_n.domain)
        t_sch = t_s.schema
        t_g = t_s.graph
        o_sch = onnx.defs.get_schema(o_n.op_type, domain2opset[o_n.domain], o_n.domain)

        t_values: Dict[str, str] = {}
        for o_i, t_i in zip(o_n.input, t_g.inputs()):
            t_i.setDebugName(f"{o_n.name}_{t_i.debugName()}")
            t_values[t_i.debugName()] = o_i

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

            v = None
            if o_a.type == onnx.AttributeProto.AttributeType.INT:
                v = o_a.i
            elif o_a.type == onnx.AttributeProto.AttributeType.FLOAT:
                v = o_a.f
            assert v is not None
            o_attr_vals[name] = v

        for t_a_idx, (t_a, t_i) in enumerate(zip(t_sch.arguments, t_g.inputs())):
            if t_a.kwarg_only:
                t_v = ret.insertConstant(o_attr_vals[t_a.name])
                t_v.setDebugName(f"{o_n.name}_{t_i.debugName()}")
                t_values[t_i.debugName()] = t_v.debugName()
                values[t_v.debugName()] = t_v
            elif t_a_idx >= len(o_n.input):
                t_v = ret.insertConstant(t_a.default_value)
                t_v.setDebugName(f"{o_n.name}_{t_a.name}")
                t_values[t_i.debugName()] = t_v.debugName()
                values[t_v.debugName()] = t_v

        for t_n in t_g.nodes():
            ins = [values[t_values[t_i.debugName()]] for t_i in t_n.inputs()]
            op_name = t_n.kind()
            c = ret.create(op_name, ins, len(list(t_n.outputs())))
            _copy_node(c, t_n, t_values, values)
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
