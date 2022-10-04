from turtle import forward
import onnx2torchscript as o2t

import torch
import onnx
import tempfile

def test_add():
    class M(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    a, b = torch.randn(10), torch.randn(10)
    f = tempfile.NamedTemporaryFile()
    torch.onnx.export(M(), (a, b), f.name)
    m: onnx.ModelProto = onnx.load_model(f.name)
    ts = o2t.onnx2ts(m)
    f = torch._C._create_function_from_graph(m.graph.name, ts)
    assert torch.allclose(M()(a, b), f(a, b))
