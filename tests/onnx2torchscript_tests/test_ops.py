from turtle import forward
from typing import Callable
import onnx2torchscript as o2t

import torch
import onnx
import tempfile

def run_op_test(f: Callable, *args, opset_version: int = 11) -> onnx.ModelProto:
    if isinstance(f, torch.nn.Module):
        mod = f
    else:
        class M(torch.nn.Module):
            def forward(self, *args):
                return f(*args)
        mod = M()

    tmp = tempfile.NamedTemporaryFile()
    torch.onnx.export(mod, args, tmp.name, opset_version=opset_version)
    m: onnx.ModelProto = onnx.load_model(tmp.name)
    ts = o2t.onnx2ts(m, args)
    call_func_count = 0
    for n in ts.graph.nodes():
        if n.kind() == "prim::CallFunction":
            call_func_count += 1
    assert call_func_count == len(m.graph.node)
    assert torch.allclose(mod(*args), ts(*args))

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
