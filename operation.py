#shamelessly taken from forge

import torch
import contextlib
from . import stream, utils
from .operations_bnb import ForgeLoader4Bit, functional_linear_4bits, functional_dequantize_4bit
from comfy.ops import manual_cast


_stash = {}

def get_weight_and_bias(layer, weight_args=None, bias_args=None, weight_fn=None, bias_fn=None):
    patches = getattr(layer, 'forge_online_loras', None)
    weight_patches, bias_patches = None, None

    if patches is not None:
        weight_patches = patches.get('weight', None)

    if patches is not None:
        bias_patches = patches.get('bias', None)

    weight = None
    if layer.weight is not None:
        weight = layer.weight
        if weight_fn is not None:
            if weight_args is not None:
                fn_device = weight_args.get('device', None)
                if fn_device is not None:
                    weight = weight.to(device=fn_device)
            weight = weight_fn(weight)
        if weight_args is not None:
            weight = weight.to(**weight_args)
        if weight_patches is not None:
            weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online weight lora", computation_dtype=weight.dtype)

    bias = None
    if layer.bias is not None:
        bias = layer.bias
        if bias_fn is not None:
            if bias_args is not None:
                fn_device = bias_args.get('device', None)
                if fn_device is not None:
                    bias = bias.to(device=fn_device)
            bias = bias_fn(bias)
        if bias_args is not None:
            bias = bias.to(**bias_args)
        if bias_patches is not None:
            bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online bias lora", computation_dtype=bias.dtype)
    return weight, bias

def weights_manual_cast(layer, x, skip_weight_dtype=False, skip_bias_dtype=False, weight_fn=None, bias_fn=None):
    weight, bias, signal = None, None, None
    non_blocking = True

    if getattr(x.device, 'type', None) == 'mps':
        non_blocking = False

    target_dtype = x.dtype
    target_device = x.device

    if skip_weight_dtype:
        weight_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if skip_bias_dtype:
        bias_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)
            signal = stream.mover_stream.record_event()
    else:
        weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)

    return weight, bias, signal

@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if signal is None or not stream.should_use_stream():
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        _stash[id(finished_signal)] = (weight, bias, finished_signal)

    garbage = []
    for k, (w, b, s) in _stash.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del _stash[k]
    return

def cleanup_cache():
    if not stream.should_use_stream():
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    _stash.clear()
    return


current_device = None
current_dtype = None
current_manual_cast_enabled = True
current_bnb_dtype = None

class OPS(manual_cast):
    class Linear(ForgeLoader4Bit):
        def __init__(self, *args, **kwargs):
            super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
            self.parameters_manual_cast = current_manual_cast_enabled

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                # Maybe this can also be set to all non-bnb ops since the cost is very low.
                # And it only invokes one time, and most linear does not have bias
                self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

            if hasattr(self, 'forge_online_loras'):
                weight, bias, signal = weights_manual_cast(self, x, weight_fn=functional_dequantize_4bit, bias_fn=None, skip_bias_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)

            if not self.parameters_manual_cast:
                return functional_linear_4bits(x, self.weight, self.bias)
            elif not self.weight.bnb_quantized:
                assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                layer_original_device = self.weight.device
                self.weight = self.weight._quantize(x.device)
                bias = self.bias.to(x.device) if self.bias is not None else None
                out = functional_linear_4bits(x, self.weight, bias)
                self.weight = self.weight.to(layer_original_device)
                return out
            else:
                weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return functional_linear_4bits(x, weight, bias)

