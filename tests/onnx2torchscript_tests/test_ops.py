from typing import Any, Callable, Dict, List, Tuple
import onnx2torchscript as o2t

import torch
import numpy as np
import onnx
import onnx.backend.test
import tempfile
import os
from onnx.backend.base import Backend, BackendRep


def run_op_test(
    f: Callable, *args, opset_version: int = 11
) -> onnx.ModelProto:
    if isinstance(f, torch.nn.Module):
        mod = f
    else:
        class M(torch.nn.Module):
            def forward(self, *args):
                return f(*args)
        mod = M()

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.close()
        torch.onnx.export(mod, args, tmp.name, opset_version=opset_version)
        m: onnx.ModelProto = onnx.load_model(tmp.name)
        ts = o2t.onnx2ts(m, args)
        call_func_count = 0
        for n in ts.graph.nodes():
            if n.kind() == "prim::CallFunction":
                call_func_count += 1
        assert call_func_count == len(m.graph.node)
        assert torch.allclose(mod(*args), ts(*args))

        with tempfile.NamedTemporaryFile() as tmp_mod:
            tmp_mod.close()
            torch.jit.save(ts, tmp_mod.name)
            ts_reload = torch.jit.load(tmp_mod.name)
            assert torch.allclose(mod(*args), ts_reload(*args))

        return m


def test_add():
    run_op_test(lambda a, b: a + b, torch.randn(10), torch.randn(10))


def test_gemm():
    run_op_test(
        lambda a, b: torch.mm(a, b),
        torch.randn(10, 10), torch.randn(10, 10))
    run_op_test(
        lambda c, a, b: torch.addmm(c, a, b),
        torch.randn(10, 10), torch.randn(10, 10),
        torch.randn(10, 10))


def test_initializer():
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = torch.nn.parameter.Parameter(torch.rand(10))

        def forward(self, a):
            return a * self.v

    m = run_op_test(M(), torch.rand(10))
    assert len(m.graph.initializer) == 1


def test_dir():
    d = os.path.join(os.path.dirname(onnx.__file__), "backend/test/data/node/test_abs")
    ts, datas = o2t.onnx_testdir_to_torchscript(d)
    for inputs, outputs in datas:
        actual_outs = ts(*inputs)
        if not isinstance(actual_outs, (list, tuple)):
            actual_outs = (actual_outs,)
        assert len(actual_outs) == len(outputs)
        for e_o, a_o in zip(outputs, actual_outs):
            assert torch.allclose(e_o, a_o)
