from typing import Any, Dict, List, Tuple
import onnx.backend.test
from onnx.backend.base import Backend, BackendRep
import numpy as np
import onnx
import torch
import pytorch_pfn_extras as ppe

import onnx2torchscript as o2t


pytest_plugins = 'onnx.backend.test.report',


_has_mps: bool = hasattr(torch.backends, "mps") and \
    torch.backends.mps.is_available()


def _move_to(ins: List[Any], device: torch.device) -> List[Any]:
    ret = []
    for i in ins:
        if isinstance(i, list):
            ret.append(_move_to(i, device))
        else:
            assert isinstance(i, torch.Tensor)
            ret.append(i.to(device))
    return ret


class TorchScriptBackendRep(BackendRep):
    def __init__(self, model: onnx.ModelProto, device: str):
        super().__init__()
        self.model = model
        if device == "CUDA":
            if _has_mps:
                self.device = "mps"
            else:
                assert torch.backends.cuda.is_avaiable()
                self.device = "cuda"
        else:
            assert device == "CPU"
            self.device = "cpu"

        self.device = torch.device(self.device)

    def run(self, inputs: Any, **kwargs) -> Tuple[Any, ...]:
        ins = []
        for i in inputs:
            if isinstance(i, np.ndarray):
                ins.append(torch.from_numpy(i.copy()))
            elif isinstance(i, list):
                ins.append([torch.from_numpy(j.copy()) for j in i])
            else:
                raise f"Unsupported input: {i}"
        self.ts = o2t.onnx2ts(self.model, ins)
        self.ts.to(device=self.device)
        ins =  _move_to(ins, self.device)
        ret = self.ts(*ins)
        if not isinstance(ret, (list, tuple)):
            ret = (ret,)
        return tuple([t.detach().cpu().numpy() for t in ret])


class TorchScriptBackend(Backend):
    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str, **kwargs):
        return TorchScriptBackendRep(model, device)

    @classmethod
    def is_compatible(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> bool:
        domain2opset: Dict[str, int] = {}
        for o in model.opset_import:
            domain2opset[o.domain] = o.version

        for n in model.graph.node:
            s = o2t.get_onnx_ts(n.op_type, domain2opset[n.domain], n.domain)
            if s is None:
                print(n.op_type, domain2opset[n.domain])
                return False

        return True

    @classmethod
    def supports_device(cls, device: str) -> bool:
        if device == "CPU":
            return True
        elif _has_mps and device == "CUDA":
            return True
        elif torch.cuda.is_available() and device =="CUDA":
            return True
        return False


backend_test = onnx.backend.test.runner.Runner(TorchScriptBackend, __name__)

xfails = [
    "uint16",
    "uint32",
    "uint64",
    "BFLOAT16",
    "STRING",
    "test_div_uint8",
    "test_reshape_zero",
    "test_identity_opt",
    "test_identity_sequence",
    "test_averagepool_2d_pads_count_include_pad_cpu",
    "test_averagepool_2d_pads_cpu",
    "test_averagepool_2d_precomputed_same_upper_cpu",
    "test_averagepool_2d_same_lower_cpu",
    "test_averagepool_2d_same_upper_cpu",
    "test_maxpool_2d_uint8_cpu",
    "test_maxpool_2d_precomputed_same_upper",
    "test_maxpool_2d_pads",
    "test_maxpool_2d_same_lower",
    "test_maxpool_2d_same_upper",
    "test_maxpool_with_argmax_2d_precomputed_strides",
    "test_inception_v2",
    "test_MaxPool3d_stride_padding_cpu",
    "test_convtranspose_autopad_same",
    "test_convtranspose_output_shape",
    "test_convtranspose_pads_",
    "test_cumsum_1d_exclusive",
    "test_cumsum_1d_reverse_exclusive",
    "test_cumsum_1d_reverse",
    "test_slice_neg_steps",
]

if not ppe.requires("1.12"):
    xfails += ["test_scatternd_multiply_"]

excludes = [
    "test_arg.*_select_last_index",
    "test_BatchNorm",
    "test_batchnorm_.*training_mode",
    "conv_with_autopad_same",
    "conv_with_strides_and_asymmetric_padding",
    "test_prelu_broadcast",
    "test_prelu_example",
    "FLOAT16",
    "test_strnorm",
    "test_tfidf",
    "test_resnet50_",
    "test_densenet121_",
    "test_sequence_insert_at_",
    "test_sequence_model[12345678]_",
]

if _has_mps:
    excludes += [
        "test_and.*_cuda",
        "test_arg.*_cuda",
        "test_det.*_cuda",
        "test_not.*_cuda",
        "test_or.*_cuda",
        "test_greater_equal.*_cuda",
        "test_less_equal.*_cuda",
        "test_gemm_.*_bias_cuda",
        "test_pow_.*_cuda",
        "test_round.*_cuda",
        "test_xor.*_cuda",
        "test_tri[lu]_zero_cuda",
        "test_Conv1d_dilated_cuda",
        "test_Conv1d_stride_cuda",
        "test_Conv3d.*_cuda",
        "test_PoissonNLLLLoss_no_reduce_cuda",
        "test_operator_add_broadcast_cuda",
        "test_operator_add_size1_broadcast_cuda",
        "test_operator_add_size1_right_broadcast_cuda",
        "test_operator_add_size1_singleton_broadcast_cuda",
        "test_operator_addconstant_cuda",
        "test_operator_mm_cuda",
        "test_einsum.*_cuda",
        "float64",
        "test_.*int8_cuda",
        "test_mod.*_cuda",
        "test_prelu.*_cuda",
        "test_PReLU.*_cuda",
        "test_reduce_max.*_cuda",
        "test_reduce_min.*_cuda",
        "test_logsoftmax.*expanded_cuda",
        "test_softmax.*expanded_cuda",
        "test_clip.*_cuda",
        "test_operator_clip_cuda",
        "DOUBLE",
        "test_globalmaxpool.*_cuda",
        "test_isinf.*_cuda",
        "test_averagepool.*_cuda",
        "test_AvgPool3d.*_cuda",
        "test_MaxPool3d.*_cuda",
        "test_maxpool_3d_default_cuda",
        "test_gridsample.*_cuda",
        "test_convtranspose_3d_cuda",
        "test_ConvTranspose2d.*_cuda",
        "test_convtranspose_pad_cuda",
        "test_convtranspose_kernel_shape_cuda",
        "test_convtranspose_with_kernel_cuda",
        "test_cumsum.*_cuda",
        "test_unique.*_cuda",
        "test_slice_cuda",
        "test_slice_start_out_of_bounds_cuda",
        "test_nllloss_.*_expanded_cuda",
        "test_scatter_elements_with_duplicate_indices_cuda",
        "test_scatternd_cuda",
        "test_scatternd_add_cuda",
        "test_scatternd_multiply_cuda",
        "test_reduce_log_sum_exp.*_cuda",
        "test_celu_expanded_cuda",
        "test_layer_normalization.*_expanded_cuda",
    ]

for x in xfails:
    backend_test.xfail(x)
for x in excludes:
    backend_test.exclude(x)

globals().update(backend_test.enable_report().test_cases)
