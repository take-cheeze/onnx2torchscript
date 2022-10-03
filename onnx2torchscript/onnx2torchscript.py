from ast import Call
from fnmatch import fnmatch
from typing import Callable, Dict, Tuple

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
def op_Add(a, b):
    return a + b


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
    # for v in model.graph.output:
    #     values[v.name] = ret.registerOutput()

    domain2opset: Dict[str, int] = {}
    for o in model.opset_import:
        domain2opset[o.domain] = o.version

    for n in model.graph.node:
        g = graph_of_onnx_op(n.op_type, domain2opset[n.domain])
        p(g)
        for n_idx, t_n in enumerate(g.nodes()):
            p(t_n)
            for s_idx, s_n in enumerate(g.nodes()):
                ins = []
                c = ret.create(s_n.kind(), ins, len(list(s_n.outputs())))
                p(c)

    return ret
