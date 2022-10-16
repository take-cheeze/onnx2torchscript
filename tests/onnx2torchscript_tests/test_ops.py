from turtle import forward
from typing import Callable
import onnx2torchscript as o2t

import torch
import onnx
import tempfile

def run_op_test(f: Callable, *args, opset_version: int = 11):
    class M(torch.nn.Module):
        def forward(self, *args):
            return f(*args)

    tmp = tempfile.NamedTemporaryFile()
    torch.onnx.export(M(), args, tmp.name, opset_version=opset_version)
    m: onnx.ModelProto = onnx.load_model(tmp.name)
    ts = o2t.onnx2ts(m, args)
    call_func_count = 0
    for n in ts.graph.nodes():
        if n.kind() == "prim::CallFunction":
            call_func_count += 1
    assert call_func_count == len(m.graph.node)
    assert torch.allclose(M()(*args), ts(*args))

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
