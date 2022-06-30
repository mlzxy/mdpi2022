# fmt: off
import gc
import os.path as osp
import math
import warnings
from collections import deque
from typing import Tuple
import warnings
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn
from scipy import optimize

from qsparse.common import (QuantizeCallback, QuantizeOptimizer, TensorOrFloat,
                            TensorOrInt, ensure_tensor)
from qsparse.imitation import imitate
from qsparse.util import get_option, logging, nd_slice
from qsparse.sparse import id2mask

# fmt: on

def compress_arr(v):
    shape = v.shape
    data = [round(a, 10) for a in v.flatten().tolist()]
    return {'arr': data, 'shape': shape}


class LinearQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator.

    Please look for detailed description on arguments in [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
        channel_index: int = 1,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
    ):
        """quantize the input tensor and prepare for backward computation."""
        ctx.backward_passthrough = backward_passthrough
        ctx.notch = 1 if flip_axis else 0
        limit = 2.0 ** (bits - 1)
        tof = 2.0**-decimal
        toi = 2.0**decimal
        shape = [1 for _ in input.shape]
        if isinstance(decimal, torch.Tensor) and sum(decimal.shape) > 1:
            assert (
                len(decimal) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            tof, toi = tof.view(*shape), toi.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(tof))
        q = (input * toi).int()
        if use_uint:
            q.clamp_(0, 2 * limit - 1)
        else:
            q.clamp_(
                -limit + ctx.notch,
                limit - 1 + ctx.notch,
            )
        return q.float() * tof

    @staticmethod
    def backward(ctx, grad_output):
        """gradient computation for quantization operation."""
        limit, tof = ctx.saved_tensors
        if ctx.backward_passthrough:
            v = grad_output
        else:
            v = grad_output.clamp_(
                (-limit + ctx.notch) * tof,
                (limit - 1 + ctx.notch) * tof,
            )
            v[v != grad_output] = 0  # reset the clampped values to 0
        return (v,) + (None,) * 6


class ScalerQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator (with scaler).

    Please look for detailed description on arguments in [scaler\_quantize\_callback][qsparse.quantize.scaler_quantize_callback].
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        scaler: TensorOrFloat = 0.1,
        channel_index: int = 1,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
    ):
        """quantize the input tensor and prepare for backward computation."""
        ctx.backward_passthrough = backward_passthrough
        ctx.notch = 1 if flip_axis else 0
        limit = 2.0 ** (bits - 1)
        shape = [1 for _ in input.shape]
        if isinstance(scaler, torch.Tensor) and math.prod(scaler.shape) > 1:
            assert (
                len(scaler) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            scaler = scaler.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(scaler))
        q = (input / scaler).round().int()
        if use_uint:
            q.clamp_(0, 2 * limit - 1)
        else:
            q.clamp_(
                -limit + ctx.notch,
                limit - 1 + ctx.notch,
            )
        return q.float() * scaler

    @staticmethod
    def backward(ctx, grad_output):
        """gradient computation for quantization operation."""
        limit, scaler = ctx.saved_tensors
        if ctx.backward_passthrough:
            v = grad_output
        else:
            v = grad_output.clamp_(
                (-limit + ctx.notch) * scaler,
                (limit - 1 + ctx.notch) * scaler,
            )
            v[v != grad_output] = 0  # reset the clampped values to 0
        return (v,) + (None,) * 6


class BaseQuantizer(nn.Module):
    weight_size = 1

    def optimize(self, tensor, bits, weight=None, batched=False, channel_index=-1, return_new_weight=False):
        raise NotImplementedError

    def forward(self, tensor, bits, weight=None, batched=False, channel_index=-1):
        raise NotImplementedError

    def get_weight_shape(self, x, channelwise):
        return (1 if channelwise < 0 else x.shape[channelwise], self.weight_size)


class DecimalQuantizer(BaseQuantizer):
    weight_size = 1

    def __init__(
        self,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
        group_num=-1
    ):
        super().__init__()
        self.use_uint = use_uint
        self.backward_passthrough = backward_passthrough
        self.flip_axis = flip_axis
        self.function = LinearQuantization.apply
        self.t = 0
        self._groups = None
        self._group_trees = None
        self.group_num = group_num
        if group_num > 0:
            logging.danger(f"alq: group_num = {group_num}")

    def quantize(self, tensor, bits, decimal, channel_index=-1):
        return self.function(
            tensor,
            bits,
            decimal,
            channel_index,
            self.use_uint,
            self.backward_passthrough,
            self.flip_axis,
        )

    def optimize(self, x, bits, weight=None, channel_index=-1, return_new_weight=False, **kwargs):
        # if weight is None or return_new_weight:
        #     with torch.no_grad():
        #         new_weight = weight if weight is not None else torch.zeros(*self.get_weight_shape(x, channel_index)).to(x.device)
        #         index =  [slice(0, s) for s in x.shape]
        #         for i in range(x.shape[channel_index] if channel_index >= 0 else 1):
        #             if channel_index >=0:
        #                 index[channel_index] = i
        #             err = float("inf")
        #             tensor = x[index]
        #             best_n = None
        #             for n in range(0, 20):
        #                 tensor_q = self.quantize(tensor, bits, weight)
        #                 err_ = torch.sum((tensor - tensor_q) ** 2).item()
        #                 if err_ < err:
        #                     best_n = n
        #                     err = err_
        #             new_weight.data[i, :] = best_n
        #         return new_weight
        # return None
        with torch.no_grad():
            x = x.abs()
            if channel_index == -1:
                x = x.view(1, -1)
            elif channel_index != 0:
                num_channel = x.shape[channel_index]
                x = x.transpose(0, channel_index)
                x = x.view(num_channel, -1)
            else:
                x = x.view(x.shape[0], -1)

            new_weight = x.max(dim=1) / (2 ** (bits - 1))
            new_weight = new_weight.view(
                self.get_weight_shape(x, channel_index))

        if self.t == 0:
            weight = new_weight.view(self.get_weight_shape(x, channel_index))
        else:
            weight.data[:] = (self.t * weight + new_weight) / (self.t + 1)
        self.t = t + 1
        return weight

    def forward(self, tensor, bits, decimal, channel_index=-1, **kwargs):
        # ! important, group quantization
        if self.t >= 512 and self.group_num > 0 and decimal.numel() > self.group_num:
            if self._groups is None:
                logging.danger(
                    f"(W) start to clustering channels into {self.group_num} groups")
                clustering = AgglomerativeClustering(
                    n_clusters=self.group_num)
                clustering.fit(decimal.detach().cpu().numpy())
                self._groups = nn.Parameter(torch.from_numpy(
                    clustering.labels_).to(decimal.device), requires_grad=False)
                # self._group_trees = nn.Parameter(torch.from_numpy(
                #     clustering.children_).to(lines.device), requires_grad=False)

            gdecimal = torch.clone(decimal)
            for ci in range(self.group_num):
                ind = self._groups == ci
                avg = gdecimal[ind].mean(dim=0)
                gdecimal[ind] = avg
            decimal = gdecimal
        return self.quantize(tensor, bits, decimal, channel_index)


class ScalerQuantizer(DecimalQuantizer):
    weight_size = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = ScalerQuantization.apply

    def optimize(self, x, bits, weight=None, channel_index=-1, return_new_weight=False, **kwargs):
        # if weight is None or return_new_weight:
        #     logging.warn("apply scaler quantization")
        #     with torch.no_grad():
        #         new_weight = weight if weight is not None else torch.zeros(*self.get_weight_shape(x, channel_index)).to(x.device)
        #         index =  [slice(0, s) for s in x.shape]
        #         for i in range(x.shape[channel_index] if channel_index >= 0 else 1):
        #             if channel_index >=0:
        #                 index[channel_index] = i
        #             tensor = x[index]
        #             init = (tensor.abs().mean() if weight is None else weight[i]).item()
        #             x0 = np.array(init)

        #             def func(x):
        #                 tensor_q = self.quantize(tensor, bits, float(x))
        #                 return torch.mean((tensor - tensor_q) ** 2).item()

        #             result = optimize.minimize(func, x0, method="Nelder-Mead")
        #             best = abs(float(result.x))
        #             new_weight.data[i, :] = best
        #         return new_weight
        # return None
        if weight is None:
            logging.warning("triggering scaler quantization!")
        weight_shape = self.get_weight_shape(x, channel_index)

        with torch.no_grad():
            x = x.abs()
            if channel_index == -1:
                x = x.view(1, -1)
            elif channel_index != 0:
                num_channel = x.shape[channel_index]
                x = x.transpose(0, channel_index)
                x = x.reshape(num_channel, -1)
            else:
                x = x.view(x.shape[0], -1)
            new_weight = x.max(dim=1).values / (2 ** (bits - 1))
            new_weight = new_weight.view(weight_shape)

            if self.t == 0:
                weight = new_weight.view(weight_shape)
            else:
                weight.data[:] = (self.t * weight + new_weight) / (self.t + 1)
        # import ipdb; ipdb.set_trace()
        self.t = self.t + 1
        return weight


Z_INT = True

class LineQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int = 8, lines=(-0.1, 0.9), channel_index=-1, inplace=False, training=True):
        with torch.no_grad():
            N = 2**bits
            shape = [1] * len(x.shape)
            if channel_index >= 0:
                shape[channel_index] = -1
                assert x.shape[channel_index] == lines.shape[0]
            assert lines.shape[1] == 2
            start, end = lines[:, 0].view(shape), lines[:, 1].view(shape)
            x = torch.clamp(x, start, end)
            step = (end - start) / N
            step[step == 0] = 0.0001
            # qa = torch.clamp(((x - start) / step).round(), 0, N - 1) * step + start

            if  not training:
                qa = (x / step).round()
                qstart = (start / step).round()
                qa = (qa - qstart).clamp(0, N-1)
                qa = (qa + qstart) * step
                return qa
            else:
                if inplace:
                    x = x - start
                    x /= step
                    x = x.round_().clamp_(0, N - 1)
                    x = x * step
                    x += start
                    return x
                else:
                    qa = x - start
                    qa /= step
                    qa = qa.round_().clamp_(0, N - 1)
                    qa = qa * step
                    qa += start
                    return qa

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,) + (None,) * 5


# today just test this
class AdaptiveLineQuantizer(nn.Module):
    weight_size = 2

    def __init__(self, alpha="mean", outlier_ratio=0, always_running_average=False, spa=False, group_num=-1):
        super().__init__()
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio
        self._lines = None
        self._groups = None
        self._group_trees = None

        if always_running_average:
            logging.danger("always use running average")
        self.always_running_average = always_running_average
        self.spa = spa
        self.group_num = group_num
        if group_num > 0:
            logging.danger(f"alq: group_num = {group_num}")

    def estimate_bound(self, bound, lower_bound):
        dist = (bound - lower_bound).abs()
        flag = ((dist / bound) < 1.1) | (dist < 0.2)
        v = torch.cat([bound.view(-1, 1), lower_bound.view(-1, 1)], dim=1)
        return v[torch.cat([~flag.view(-1, 1), flag.view(-1, 1)], dim=1)]

    def optimize(self, x, bits, weight=None, channel_index=-1, batched=False, **kwargs):
        batch_size = x.shape[0]
        with torch.no_grad():
            if self.spa:
                assert batched
                if hasattr(x, "qsparse_mask_id"):
                    mask = id2mask[x.qsparse_mask_id].repeat(
                        len(x), *([1] * (len(x.shape) - 1)))
                else:
                    warnings.warn(
                        "spa disable for certain tensors without a fixed mask")
                    mask = None
            else:
                mask = None
                # if channel_index >= 0:
                #     assert channel_index == 1
                #     for ci in range(x.shape[channel_index]):
                #         x[:, ci, ~mask[0, ci]] =  x[:, ci, mask[0, ci]].median(dim=1).values.view(-1, 1)
                # else:
                #     x[:, ~mask[0]] =  x[:,  mask[0]].median(dim=1).values.view(-1, 1) # ensure masked values won't affect min/max

            if channel_index >= 0:
                if batched:
                    if channel_index != 1:
                        x = x.transpose(1, channel_index)
                        if mask is not None:
                            mask = mask.transpose(1, channel_index)
                    shape = tuple(x.shape)
                    x = x.view(-1, math.prod(shape[2:]))
                    if mask is not None:
                        mask = mask.view(-1, math.prod(shape[2:]))
                else:
                    if channel_index != 0:
                        x = x.transpose(0, channel_index)
                    shape = tuple(x.shape)
                    # contiguous()
                    x = x.contiguous().view(-1, math.prod(shape[1:]))
            else:
                if batched:
                    x = x.view(len(x), -1)
                    if mask is not None:
                        mask = mask.view(len(x), -1)
                else:
                    x = x.view(1, -1)

            if self.outlier_ratio > 0:
                # lb = self.estimate_bound(
                #     x.quantile(self.outlier_ratio, dim=1), x.min(dim=1).values
                # )
                # ub = self.estimate_bound(
                #     x.quantile(1 - self.outlier_ratio, dim=1), x.max(dim=1).values
                # )
                assert False, "havin't use this branch yet!"
                lb = x.quantile(self.outlier_ratio, dim=1)
                ub = x.quantile(1 - self.outlier_ratio, dim=1)
            else:

                if mask is None:
                    lb = x.min(dim=1).values
                    ub = x.max(dim=1).values
                else:
                    mask = ~mask
                    # olb = x.min(dim=1).values
                    # oub = x.max(dim=1).values
                    ub = (x - mask * 10000).max(dim=1).values
                    ub[ub == -10000] = 0

                    lb = (x + mask * 10000).min(dim=1).values
                    lb[lb == 10000] = 0
                    lb[lb == ub] = 0

            if weight is None:
                logging.warning("triggering adaptive line quantization!")

            _buf = torch.cat([lb.view(-1, 1), ub.view(-1, 1)], dim=1)
            # if weight is None or len(lb) != len(self._buf):
            #     self._buf = torch.cat([lb.view(-1, 1), ub.view(-1, 1)], dim=1)
            # else:
            #     self._buf[:len(lb), 0], self._buf[:len(ub), 1] = lb, ub  # avoid too many `cat`
            _lines = _buf[:len(lb), :]
            # _lines = torch.cat([lb.view(-1, 1), ub.view(-1, 1)], dim=1)

            if batched:
                _lines = _lines.view(batch_size, -1, 2)
                # avg_lines = torch.cat([_lines[:, :, 0].quantile(0.02, dim=0).view(-1, 1), _lines[:, :, 1].quantile(0.98, dim=0).view(-1, 1)], dim=1)
                avg_lines = torch.cat([_lines[:, :, 0].min(
                    dim=0).values.view(-1, 1), _lines[:, :, 1].max(dim=0).values.view(-1, 1)], dim=1)
                if not self.always_running_average:
                    self._lines = _lines.view(-1, 2)
            else:
                avg_lines = _lines
                if not self.always_running_average:
                    self._lines = _lines

            if self.always_running_average:
                self._lines = None

            if weight is None:
                self.t = nn.Parameter(torch.zeros(1).to(
                    x.device), requires_grad=False)
                self.t += 1
                return avg_lines
            else:
                assert avg_lines.shape == weight.shape
                # how to maintain the running min/max [to be customized]
                if self.alpha == "mean":
                    self.t += 1
                    return (weight * (self.t - 1) + avg_lines) / self.t
                else:
                    return weight * (1 - self.alpha) + avg_lines * self.alpha

    def forward(self, tensor, bits, lines, channel_index=-1, inplace=False):
        origin_shape = tuple(tensor.shape)
        if self.training and (not self.always_running_average):
            assert self._lines is not None, "statistics of current batch is not calculated yet."
            if channel_index >= 0:  # channelwise
                if self._lines.shape[0] == origin_shape[channel_index]:  # only channel-wise
                    result = LineQuantization.apply(
                        tensor, bits, self._lines, channel_index, inplace, self.training)
                else:
                    assert channel_index != 0
                    # channel-wise and sample-wise
                    if channel_index != 1:
                        tensor = tensor.transpose(1, channel_index)
                    tensor = tensor.view(-1, *origin_shape[2:])
                    assert self._lines.shape[0] == tensor.shape[0]
                    result = LineQuantization.apply(
                        tensor, bits, self._lines, 0, inplace, self.training)
                    if channel_index != 1:
                        result = result.transpose(1, channel_index)
            else:
                if self._lines.shape[0] != 1:
                    # sample-wise along the batch dimension
                    result = LineQuantization.apply(
                        tensor, bits, self._lines, 0, inplace, self.training)
                else:
                    result = LineQuantization.apply(
                        tensor, bits, self._lines, -1, inplace, self.training)
            self._lines = None
            # import gc; gc.collect()
            # import torch; torch.cuda.empty_cache()
        else:
            # TODO: we can possibly expand the lines here
            if self.t >= 512 and self.group_num > 0:
                if self._groups is None:
                    logging.danger(
                        f"start to clustering channels into {self.group_num} groups") # self.training == True (because self.always_running_average == True!); OR self.training = False
                    clustering = AgglomerativeClustering(
                        n_clusters=self.group_num)
                    clustering.fit(lines.detach().cpu().numpy())
                    self._groups = nn.Parameter(torch.from_numpy(
                        clustering.labels_).to(lines.device), requires_grad=False)
                    self._group_trees = nn.Parameter(torch.from_numpy(
                        clustering.children_).to(lines.device), requires_grad=False)

                glines = torch.clone(lines)
                for ci in range(self.group_num):
                    ind = self._groups == ci
                    avg = glines[ind].mean(dim=0)
                    glines[ind] = avg
                lines = glines
            result = LineQuantization.apply(
                tensor, bits, lines, channel_index, inplace, self.training)
        result = result.view(origin_shape)
        if self.spa and hasattr(tensor, "qsparse_mask_id"):
            mask = id2mask[tensor.qsparse_mask_id]
            return result * mask
        else:
            return result


class QuantizeLayer(nn.Module):
    """Applies quantization over input tensor.

    Please look for detailed description in [quantize][qsparse.quantize.quantize]
    """

    def __str__(self):
        return f"QuantizeLayer(bits={self.bits})"

    def __repr__(self):
        return str(self)

    def __init__(
        self,
        bits: int = 8,
        channelwise: int = 1,
        # for step-wise training
        timeout: int = 1000,
        # for customization
        callback: BaseQuantizer = None,
        interval=-1,
        # for debug purpose
        collapse: int = 0,
        collect_q_stats="",
        name: str = "",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Quantize{name if name == '' else f' @ {name}'}] bits={bits} channelwise={channelwise} timeout={timeout}"
            )
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self.bits = bits
        self.callback = callback  # type: BaseQuantizer
        self._batch_dim = collapse
        self.interval = interval
        self._quantized = False
        self.collect_q_stats = collect_q_stats
        self.db = None
        self.prev_w = None
        self.hook = None

    @property
    def initted(self) -> bool:
        """whether the parameters of the quantize layer are initialized."""
        return hasattr(self, '_n_updates')

    def forward(self, x):
        """Quantize input tensor according to given configuration.

        Args:
            x (torch.Tensor): tensor to be quantized

        Returns:
            torch.Tensor: quantized tensor
        """
        # if self.hook is not None:
        #     self.hook.remove()
        #     self.hook = None

        if not self.initted:
            self.weight = nn.Parameter(
                torch.zeros(
                    1 if self.channelwise < 0 else x.shape[self.channelwise],
                    self.callback.weight_size,
                ).to(x.device),
                requires_grad=False,
            )
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )


        t = self._n_updates.item()
        if self.timeout > 0:
            if t >= self.timeout:
                if self.training:
                    new_weight = self.callback.optimize(x, self.bits, None if t == self.timeout else self.weight, batched=self._batch_dim == 0,
                                                        channel_index=self.channelwise, return_new_weight=False if self.interval <= 0 or ((t - self.timeout) % self.interval != 0) else True)
                    if new_weight is not None:
                        self.weight.data[:] = new_weight
                    self._quantized = True
                if self._quantized:
                    out = self.callback(
                        x, self.bits, self.weight, channel_index=self.channelwise, inplace=self._batch_dim == 0)
                else:
                    out = x
            else:
                out = x

            if self.training:
                self._n_updates += 1
        else:
            out = x
        return out


def quantize(
    inp: nn.Module = None,
    bits: int = 8,
    channelwise: int = 1,
    # for step-wise training
    timeout: int = 1000,
    # for customization
    callback: BaseQuantizer = None,
    interval=-1,
    # for bias quantization, default to -1 is to not quantize bias
    bias_bits: int = -1,
    # for debug purpose
    collect_q_stats="",
    name: str = "",
) -> nn.Module:
    """Creates a [QuantizeLayer][qsparse.quantize.QuantizeLayer] which is
    usually used for feature quantization if no input module is provided, or
    creates a weight-quantized version of the input module.

    Args:
        inp (nn.Module, optional): input module whose weight is to be quantized. Defaults to None.
        bits (int, optional): bitwidth for weight. Defaults to 8.
        channelwise (int, optional): dimension index for channel. Defaults to 1. When channelwise >= 0, channel-wise quantization is enabled. When set to -1, channel-wise quantization is disabled.
        timeout (int, optional): the steps to compute the best decimal bits. Defaults to 1000.
        interval (int, optional): interval of steps before each time to compute the best decimal bits. Defaults to -1, means only calculating the decimal bits once.
        window_size (int, optional): number of tensors used for computing the decimal bits. Defaults to 1.
        on_device_window (bool, optional): whether keep the tensor window on gpu device being used, or move to cpu. Default to False, means moving to cpu.
        optimizer (QuantizeOptimizer, optional): optimizer used to compute the best quantization weight. Defaults to `DecimalOptimizer()`.
        callback (QuantizeCallback, optional):  callback for actual operation of quantizing tensor, used for customization. Defaults to [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].
        bias_bits (int, optional): bitwidth for bias. Defaults to -1, means not quantizing bias.
        name (str, optional): name of the quantize layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight quantized or a instance of [QuantizeLayer][qsparse.quantize.QuantizeLayer] for feature quantization
    """
    callback = callback or DecimalQuantizer()

    kwargs = dict(
        bits=bits,
        channelwise=channelwise,
        timeout=timeout,
        callback=callback,
        bias_bits=bias_bits,
        name=name,
        interval=interval,
        collect_q_stats=collect_q_stats
    )

    def get_quantize_layer(feature_collapse=0, is_bias=False):
        if bias_bits == -1 and is_bias:
            return lambda a: a
        else:
            return QuantizeLayer(
                bits=bias_bits if is_bias else bits,
                channelwise=(0 if channelwise >= 0 else
                             - 1) if is_bias else channelwise,
                timeout=int(timeout),
                callback=callback,
                name=name,
                collapse=feature_collapse,
                collect_q_stats=collect_q_stats,
                interval=interval
            )

    if inp is None:
        layer = get_quantize_layer()
        setattr(layer, "_kwargs", kwargs)
        return layer
    elif isinstance(inp, nn.Module):
        return imitate(
            inp,
            "quantize",
            get_quantize_layer(-1),
            get_quantize_layer(-1, is_bias=True),
        )
    else:
        raise ValueError(f"{inp} is not a valid argument for quantize")


if __name__ == "__main__":
    layer = quantize(timeout=0)
    print(layer)
    print(quantize(torch.nn.Conv2d(10, 30, 3)))

    data = torch.rand(10, 10)
    print(data)
    print(layer(data))
