import onnx2torchscript as o2ts

import sys
from typing import List


def main(argv: List[str]) -> None:
    assert len(argv) == 2

    test_dir = argv[1]

    ts, datas = o2ts.onnx_testdir_to_torchscript(test_dir)
    for inputs, outputs in datas:
        actual_outs = ts(*inputs)
        if not isinstance(actual_outs, (list, tuple)):
            actual_outs = (actual_outs,)
        assert len(actual_outs) == len(outputs)
        for e_o, a_o in zip(outputs, actual_outs):
            assert torch.allclose(e_o, a_o)


if __name__ == "__main__":
    main(sys.argv)
