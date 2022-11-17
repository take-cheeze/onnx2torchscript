import warnings
import os
import glob
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import onnx
import onnx.numpy_helper
import torch


_op_table: Dict[int, Dict[str, torch._C.ScriptFunction]] = {}
_version_cache: Dict[Tuple[str, int], torch._C.ScriptFunction] = {}


class MetaWarning(Warning):
    pass


def onnx_op(
    op_type: str, opset_version: int = 0, domain: str = ""
) -> Callable:
    def fn(f: Callable) -> Callable:
        ret = torch.jit.script(f)
        _op_table.setdefault(opset_version, {})
        assert op_type not in _op_table[opset_version]
        _op_table[opset_version][op_type] = ret
        global _version_cache
        _version_cache = {}  # Clear cache
        return ret
    return fn


def get_onnx_ts(
    op_type: str, opset_version: int = 0, domain: str = ""
) -> Optional[torch._C.ScriptFunction]:
    k = (op_type, opset_version)
    if k in _version_cache:
        return _version_cache[k]

    if opset_version <= 0:
        opset_version = onnx.defs.onnx_opset_version()
    while not (
        opset_version in _op_table and
        op_type in _op_table[opset_version]
    ):
        opset_version -= 1
        if opset_version == 0:
            return None
        continue

    ret = _version_cache[k] = _op_table[opset_version][op_type]
    return ret


class OnnxModule(torch.nn.Module):
    def __init__(self, model: onnx.ModelProto) -> None:
        super().__init__()
        self.model = model

        self.meta_mode: bool = False

        # Construct domain opset mapping
        self.domain2opset: Dict[str, int] = {}
        for o in self.model.opset_import:
            self.domain2opset[o.domain] = o.version

        # Register constants as buffers to use as meta tensor
        for o_n in self.model.graph.node:
            o_sch = self.node_schema(o_n)
            o_attrs = self.attribute_dict(o_sch, o_n)
            for name, o_a in o_attrs.items():
                if o_a.type != onnx.AttributeProto.AttributeType.TENSOR:
                    continue
                attr_value = onnx.helper.get_attribute_value(o_a)
                attr_value = torch.from_numpy(
                    onnx.numpy_helper.to_array(attr_value).copy())
                t_name = self.attribute_state_name(o_n, name)
                assert not hasattr(self, t_name)
                self.register_buffer(t_name, attr_value)

        # Register initializers as buffers
        for i in self.model.graph.initializer:
            p = torch.from_numpy(onnx.numpy_helper.to_array(i).copy())
            self.register_buffer(self.escape_buffer_name(i.name), p)
            # if p.dtype.is_floating_point:
            #     setattr(self, i.name, torch.nn.parameter.Parameter(p))
            # else:
            #     self.register_buffer(i.name, p)
        self.original_params = self.state_dict()

    def enable_meta_mode(self, mode: bool = True) -> None:
        if mode:
            for k, v in self.original_params.items():
                self.register_buffer(k, v.to("meta"))
        else:
            self.load_state_dict(self.original_params)
            for k, v in self.original_params.items():
                setattr(self, k, v)
        self.meta_mode = mode

    def attribute_dict(self, o_sch: Optional[onnx.defs.OpSchema], o_n: onnx.NodeProto) -> Dict[str, onnx.AttributeProto]:
        o_attrs: Dict[str, onnx.AttributeProto] = {}
        if o_sch is not None:
            for name, o_sch_a in o_sch.attributes.items():
                if o_sch_a.default_value is not None:
                    o_attrs[name] = o_sch_a.default_value
        for o_a in o_n.attribute:
            o_attrs[o_a.name] = o_a
        return o_attrs

    def escape_buffer_name(self, name: str) ->  str:
        return name.replace('.', '_')

    def attribute_state_name(self, o_n: onnx.NodeProto, name: str) -> str:
        return self.escape_buffer_name(f"{o_n.name}_{o_n.output[0]}_{name}")

    def attribute_values(self, o_sch: Optional[onnx.defs.OpSchema], o_n: onnx.NodeProto) -> Dict[str, Any]:
        o_attrs: Dict[str, onnx.AttributeProto] = self.attribute_dict(o_sch, o_n)
        o_attr_vals: Dict[str, Any] = {}
        for name, o_a in o_attrs.items():
            if o_a.type == onnx.AttributeProto.AttributeType.UNDEFINED:
                continue

            if o_a.type == onnx.AttributeProto.AttributeType.TENSOR:
                o_attr_vals[name] = getattr(self, self.attribute_state_name(o_n, name))
            else:
                o_attr_vals[name] = onnx.helper.get_attribute_value(o_a)

        return o_attr_vals

    def opset(self, domain: str) -> int:
        return self.domain2opset.get(domain, 1)

    def node_schema(self, node: onnx.NodeProto) -> Optional[onnx.defs.OpSchema]:
        ver = self.opset(node.domain)
        if onnx.defs.has(node.op_type, node.domain):
            return onnx.defs.get_schema(
                node.op_type, ver, node.domain)
        return None

    def forward(self, *args: Any) -> Any:
        values: Dict[str, Any] = {}
        initializer_names: Set[str] = set()
        for i in self.model.graph.initializer:
            n = self.escape_buffer_name(i.name)
            values[n] = getattr(self, n)
            initializer_names.add(n)
        input_names: List[str] = []
        for i in self.model.graph.input:
            n = self.escape_buffer_name(i.name)
            if n not in initializer_names:
                input_names.append(n)
        assert len(input_names) == len(args)
        for a, n in zip(args, input_names):
            values[n] = a

        Variadic = onnx.defs.OpSchema.FormalParameterOption.Variadic
        for o_n in self.model.graph.node:
            o_sch = self.node_schema(o_n)
            o_attr_vals: Dict[str, Any] = self.attribute_values(o_sch, o_n)

            t_s = get_onnx_ts(
                o_n.op_type, self.opset(o_n.domain), o_n.domain)
            if t_s is None:
                msg = f"{o_n.domain}::{o_n.op_type}-{self.opset(o_n.domain)} not found"
                raise NotImplementedError(msg)
            t_sch = t_s.schema
            ins = [None if n == '' else values[self.escape_buffer_name(n)] for n in o_n.input]
            if o_sch is not None and len(o_sch.inputs) == 1 and o_sch.inputs[0].option == Variadic:
                ins = [ins]
            for idx in range(len(ins), len(t_sch.arguments)):
                arg = t_sch.arguments[idx]
                if arg.name in o_attr_vals:
                    ins.append(o_attr_vals[arg.name])
                elif arg.has_default_value():
                    if arg.name == "_num_outputs":
                        ins.append(len(o_n.output))
                        continue
                    ins.append(arg.default_value)
                else:
                    raise RuntimeError(f"{arg.name} not provided")
            outs = t_s(*ins)
            if not isinstance(outs, (tuple, list)):
                outs = (outs,)

            if len(outs) >= len(o_n.output):
                for n, o in zip(o_n.output, outs):
                    n = self.escape_buffer_name(n)
                    assert n not in values
                    values[n] = o
            else:
                if o_sch is not None and len(o_sch.outputs) == 1 and o_sch.outputs[0].option == Variadic:
                    assert len(o_n.output) == len(outs)
                    for n, o in zip(o_n.output, outs):
                        n = self.escape_buffer_name(n)
                        assert n not in values
                        values[n] = o
                else:
                    raise RuntimeError(f"Cannot supply outputs: {o_n.output[len(outs):]}")

        ret = tuple([values[i.name] for i in self.model.graph.output])
        if len(ret) == 1:
            return ret[0]
        return ret


def onnx2ts(
    model: onnx.ModelProto, args: Any, verbose: bool = False
) -> torch._C.ScriptModule:
    m = OnnxModule(model)
    try:
        meta_args = tuple(a.to('meta') for a in args)
        m.enable_meta_mode(True)
        t = torch.jit.trace(m, meta_args, check_trace=False)
        t.load_state_dict(m.original_params)
        for k, v in m.original_params.items():
            setattr(t, k, v)
    except NotImplementedError as e:
        warnings.warn(f"Failed meta tracing mode, fallbacking: {e}", MetaWarning)
        m.enable_meta_mode(False)
        t = torch.jit.trace(m, args, check_trace=False)

    return t


def onnx_testdir_to_torchscript(test_dir: str) -> Tuple[torch._C.ScriptModule, List[Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
    model_path = os.path.join(test_dir, "model.onnx")
    assert os.path.exists(model_path)

    cases = glob.glob(os.path.join(test_dir, "test_data_set_*"))
    ret: List[Tuple[List[torch.Tensor], List[torch.Tensor]]] = []
    m = onnx.load_model(model_path)
    for c in cases:
        input_files = glob.glob(os.path.join(c, "input_*.pb"))
        in_dict: Dict[str, torch.Tensor] = {}
        for i in input_files:
            with open(i, 'rb') as f:
                p = onnx.TensorProto()
                p.ParseFromString(f.read())
                in_dict[p.name] = torch.from_numpy(onnx.numpy_helper.to_array(p).copy())
        ins: List[torch.Tensor] = []
        for i in m.graph.input:
            ins.append(in_dict[i.name])
        output_files = glob.glob(os.path.join(c, "output_*.pb"))
        out_dict: Dict[str, torch.Tensor] = {}
        for i in output_files:
            with open(i, 'rb') as f:
                p = onnx.TensorProto()
                p.ParseFromString(f.read())
                out_dict[p.name] = torch.from_numpy(onnx.numpy_helper.to_array(p).copy())
        outs: List[torch.Tensor] = []
        for i in m.graph.output:
            outs.append(out_dict[i.name])
        ret.append((ins, outs))
    assert len(ret) >= 1

    ts = onnx2ts(m, ret[0][0])

    return ts, ret
