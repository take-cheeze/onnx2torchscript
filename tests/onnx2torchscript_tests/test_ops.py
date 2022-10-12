from turtle import forward
from typing import Callable
import onnx2torchscript as o2t

import torch
import onnx
import tempfile

def run_op_test(f: Callable, *args):
    class M(torch.nn.Module):
        def forward(self, *args):
            return f(*args)

    tmp = tempfile.NamedTemporaryFile()
    torch.onnx.export(M(), args, tmp.name)
    m: onnx.ModelProto = onnx.load_model(tmp.name)
    ts = o2t.onnx2ts(m)
    torch._C._jit_pass_lint(ts)
    ts_f = torch._C._create_function_from_graph(m.graph.name, ts)
    assert torch.allclose(M()(*args), ts_f(*args))

def test_add():
    run_op_test(lambda a, b: a + b, torch.randn(10), torch.randn(10))

def test_gemm():
    # o2t.onnx2torchscript.get_onnx_op("Gemm", 11)(
    #     torch.randn(10, 10), torch.randn(10, 10),
    #     alpha=1.0, beta=1.0, transA=0, transB=0)
    run_op_test(
        lambda a, b: torch.mm(a, b),
        torch.randn(10, 10), torch.randn(10, 10))
    run_op_test(
        lambda c, a, b: torch.addmm(c, a, b),
        torch.randn(10, 10), torch.randn(10, 10),
        torch.randn(10, 10))
