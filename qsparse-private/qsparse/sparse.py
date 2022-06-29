import cvxpy as cp
import math
import warnings
from argparse import ArgumentError
import os.path as osp
from collections import deque
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import use

from qsparse.common import PruneCallback
from qsparse.imitation import imitate
from qsparse.util import align_tensor_to_shape, get_option, logging, nd_slice
import numba


id2mask = {}


@numba.jit(nopython=True, parallel=True)
def my_conv2d(inp_up, out_shape, weight, padding=1, stride=1, groups=1, **kwargs):
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    in_shape = inp_up.shape
    inp = np.zeros((1, in_shape[1], in_shape[2]
                   + 2 * padding[0], in_shape[3] + 2 * padding[1]))
    if padding[0] == 0:
        inp = inp_up
    else:
        inp[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] = inp_up

    out = np.zeros(out_shape)

    group_size = inp.shape[1] // groups
    out_group_size = out.shape[1] // groups

    ks_h = weight.shape[2]
    ks_w = weight.shape[3]

    for co in numba.prange(out_group_size):
        for gi in range(groups):
            for ci in range(group_size):
                for i in range(out.shape[2]):
                    for j in range(out.shape[3]):
                        center_h = i * stride[0]
                        center_w = j * stride[1]
                        tmp = (inp[0, gi * group_size + ci, center_h:center_h + ks_h,
                               center_w:center_w + ks_w] * weight[gi * out_group_size + co, ci, :, :]).sum()
                        out[0, gi * out_group_size + co, i, j] += tmp
    return out


@numba.jit(nopython=True)
def my_deconv2d(inp, out_shape, weight, padding=0, stride=1, output_padding=0, **kwargs):
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    assert padding == output_padding

    in_shape = inp.shape
    unpadded_output = np.zeros(
        (out_shape[0], out_shape[1], out_shape[2] + padding[0] * 2, out_shape[3] + padding[1] * 2))
    # print(unpadded_output.shape)
    ks_h = weight.shape[2]
    ks_w = weight.shape[3]

    for co in range(out_shape[1]):
        for ci in range(inp.shape[1]):
            for i in range(inp.shape[2]):
                for j in range(inp.shape[3]):
                    center_h = i * stride[0]
                    center_w = j * stride[1]
                    unpadded_output[0, co, center_h:center_h + ks_h, center_w:center_w + ks_w] += (
                        inp[0, ci, i, j] * weight[ci, co, :, :])

    if padding[0] > 0 and padding[1] > 0:
        return unpadded_output[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return unpadded_output[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return unpadded_output[:, :, :, padding[1]:-padding[1]]
    else:
        return unpadded_output


@numba.jit(nopython=True)
def my_deconv2d_trace(inp_m, out_m, weight_shape, padding=0, stride=2, output_padding=0, **kwargs):
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    # assert padding == output_padding
    out_shape = out_m.shape
    unpadded_out_m = np.zeros(
        (out_shape[0], out_shape[1], out_shape[2] + padding[0] * 2, out_shape[3] + padding[1] * 2), dtype='bool')
    if padding[0] > 0 and padding[1] > 0:
        unpadded_out_m[:, :, padding[0]:-padding[0],
                       padding[1]:-padding[1]] = out_m
    elif padding[0] > 0:
        unpadded_out_m[:, :, padding[0]:-padding[0], :] = out_m
    elif padding[1] > 0:
        unpadded_out_m[:, :, :, padding[1]:-padding[1]] = out_m
    else:
        unpadded_out_m = out_m

    weight = np.zeros((weight_shape[0], weight_shape[1]))  # ci, co
    in_shape = inp_m.shape
    ks_h = weight_shape[2]
    ks_w = weight_shape[3]

    tasks = []
    for co in range(out_shape[1]):
        for ci in range(in_shape[1]):
            tasks.append((co, ci))

    for ind in range(len(tasks)):
        co, ci = tasks[ind]
        tmp = 0
        for i in range(in_shape[2]):
            for j in range(in_shape[3]):
                if inp_m[0, ci, i, j] == 0:
                    continue
                else:
                    center_h = i * stride[0]
                    center_w = j * stride[1]
                    tmp += unpadded_out_m[0, co, center_h:center_h
                                          + ks_h, center_w:center_w + ks_w].sum()
        weight[ci, co] += tmp
    return weight


@numba.jit(nopython=True, parallel=True)
def my_conv2d_trace(inp_m, out_m, weight_shape, padding=(1, 1), stride=(1, 1), groups=1, **kwargs):

    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    in_shape = inp_m.shape
    inp = np.zeros((1, in_shape[1], in_shape[2]
                   + 2 * padding[0], in_shape[3] + 2 * padding[1]))
    if padding[0] == 0:
        inp[:] = inp_m
    else:
        inp[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] = inp_m
    # inp[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] = inp_m
    out = out_m
    weight = np.zeros((weight_shape[0], weight_shape[1]))

    group_size = inp.shape[1] // groups
    out_group_size = out.shape[1] // groups

    ks_h = weight_shape[2]
    ks_w = weight_shape[3]

    tasks = []
    for co in range(out_group_size):
        for gi in range(groups):
            for ci in range(group_size):  # group_size
                tasks.append((co, gi, ci))

    for ind in numba.prange(len(tasks)):
        co, gi, ci = tasks[ind]
        tmp = 0
        o_c_ind = gi * out_group_size + co
        i_c_ind = gi * group_size + ci
        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                center_h = i * stride[0]
                center_w = j * stride[1]
                if out[0, o_c_ind, i, j] == 0:
                    continue
                tmp += inp[0, i_c_ind, center_h:center_h
                           + ks_h, center_w:center_w + ks_w].sum()
        weight[o_c_ind, ci] += tmp
    return weight


def calculate_weight_mag(input_mask, output_mask, weight_shape, op_name, op_kwargs):
    if op_name == 'Conv2d':
        return torch.from_numpy(my_conv2d_trace(input_mask.detach().cpu().numpy(), output_mask.detach().cpu().numpy(), tuple(weight_shape),
                                                padding=op_kwargs['padding'], stride=op_kwargs['stride'], groups=op_kwargs['groups']))
    elif op_name == 'ConvTranspose2d':
        return torch.from_numpy(my_deconv2d_trace(input_mask.detach().cpu().numpy(), output_mask.detach().cpu().numpy(), tuple(weight_shape),
                                                  padding=op_kwargs['padding'], stride=op_kwargs['stride'], output_padding=op_kwargs['output_padding']))
    else:
        raise RuntimeError()


def find_prune_threshold(importance, sparsity):
    values = importance.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    return threshold


def calculate_act_filter_mask(importance, sparsity, method="max", kernel=(3, 3), padding=(1, 1), stride=(1, 1)):
    with torch.no_grad():
        if kernel == (1, 1):
            return importance >= find_prune_threshold(importance, sparsity)
        else:
            if method == "max":
                pool_m = F.max_pool2d
            else:
                pool_m = F.avg_pool2d

            signal = pool_m(importance, kernel, stride, padding)
            if signal.shape == importance.shape:
                return signal >= find_prune_threshold(signal, sparsity)
            else:
                assert stride != (1, 1)
                mask_sm = signal >= find_prune_threshold(signal, sparsity)
                _weight = torch.ones(
                    mask_sm.shape[1], 1, *stride).to(importance.device)
                mask = F.conv_transpose2d(mask_sm.float(
                ), _weight, stride=stride, groups=mask_sm.shape[1], padding=0).bool()
                # assert mask.shape == importance.shape
                return mask[:, :, :importance.shape[2], :importance.shape[3]]


def optimize_mask(importance, sparsity, min_ratio=0.25):
    dev = importance.device
    origin_shape = tuple(importance.shape)

    if np.prod(origin_shape) >= (128 * 128 * 256):
        print("subsample")
        importance = F.avg_pool2d(importance, (2, 2), stride=(2, 2))
        shrink_shape = tuple(importance.shape)

    A = importance.cpu().numpy().reshape(importance.shape[1], -1)
    O = cp.Variable(A.shape, boolean=True)
    O.value = (A >= np.quantile(A, sparsity)).astype('int32')  # initialization
    objective = cp.Maximize(cp.sum(cp.multiply(A, O)))
    constraints = [
        cp.sum(O) == int(np.prod(A.shape) * (1 - sparsity)),
        cp.sum(O, axis=0) >= int(importance.shape[1] * min_ratio)
    ]
    prob = cp.Problem(objective, constraints)
    # print(int(np.prod(A.shape) * (1 - sparsity)), int(importance.shape[1] * min_ratio))
    print(
        f"solving MIP optimization problem, sparsity = {sparsity}, min active neurons per location = {int(importance.shape[1] * min_ratio)}")
    prob.solve(solver=cp.CPLEX)
    print(f"solved, value = {prob.value}")

    if abs(prob.value) == np.inf:
        return None
    mask = torch.from_numpy(O.value)

    if np.prod(origin_shape) >= (128 * 128 * 256):
        _weight = torch.ones(shrink_shape[1], 1, 2, 2)
        mask = F.conv_transpose2d(mask.view(shrink_shape).float(
        ), _weight, stride=(2, 2), groups=shrink_shape[1], padding=0)
    return mask.view(origin_shape).bool().to(dev)


class MagnitudePruningCallback(nn.Module):
    def __init__(
        self,
        mask_refresh_interval: int = -1,
        stop_mask_refresh: int = float("inf"),
        use_gradient: bool = False,
        running_average: bool = True,
        filter_based=False,
        act_filter_p=None,
        structure=False,
        bernoulli=False,
        preserve_existing_mask=False,
        sp_balance=False
    ):
        """
        Magnitude-based pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

        Args:
            mask_refresh_interval (int, optional): number of steps to refresh mask. Defaults to 1.
            stop_mask_refresh (int, optional): when to stop refreshing mask. Defaults to float('inf').
            use_gradient (bool, optional): whether use the magnitude of gradients
            running_average (bool, optional): whether use the running average of magnitude. Defaults to True.
        """
        super().__init__()
        self.mask_refresh_interval = mask_refresh_interval
        self.stop_mask_refresh = stop_mask_refresh
        self.use_gradient = use_gradient
        self.t = nn.Parameter(torch.full((1,), -1), requires_grad=False)
        self.prev_hook = None
        if use_gradient and not running_average:
            raise ArgumentError(
                "the combination of `use_gradient=True` and `running_average=False` is not supported"
            )
        self.running_average = running_average

        self.forward_mag = None
        self.backward_mag = None
        self.name = None
        self.joint_weight_act = False
        self.hook_handles = []
        self.parent_op_name = ''
        self.filter_based = filter_based
        self.structure = structure
        self.bernoulli = bernoulli
        self.parent_op_kwargs = {}
        self.prev_bn = None
        if self.bernoulli:
            logging.danger("using bernoulli magnitude")

        self.act_filter_p = act_filter_p
        if self.act_filter_p is not None:
            logging.danger(f"using act_filter_p = {self.act_filter_p}")
        self.act_filter_kernel = None
        self.act_filter_stride = None
        self.act_filter_padding = None
        self.preserve_existing_mask = preserve_existing_mask
        self.existing_mask = None
        self.first_time_inference = True

        self.sp_balance = sp_balance
        self.use_existing_stat = False

    @property
    def initted(self) -> bool:
        return self.t.item() != -1

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.running_average:
            importance = self.magnitude
        else:
            importance = align_tensor_to_shape(x.abs(), mask.shape)

        if not self.filter_based:
            if not self.structure:  # second cond checks for linear layers
                if self.act_filter_p is not None:
                    # grid-like pruning
                    mask.data[:] = calculate_act_filter_mask(
                        importance, sparsity, self.act_filter_p, self.act_filter_kernel, self.act_filter_padding, self.act_filter_stride)
                else:
                    # unstructure pruning
                    if self.sp_balance:
                        mask.data[:] = optimize_mask(importance, sparsity)
                    else:
                        if self.preserve_existing_mask:
                            importance = importance * self.existing_mask
                        values = importance.flatten().sort()[0]
                        n = len(values)
                        idx = max(int(sparsity * n - 1), 0)
                        threshold = values[idx]
                        mask.data[:] = importance >= threshold
                        if self.preserve_existing_mask:
                            mask.data[:] = mask * self.existing_mask
                return self.broadcast_mul(x, mask)
            else:
                # structure pruning
                if mask.shape[0] == 1:  # activation
                    num_c = mask.shape[1]
                    importance = importance.view(num_c, -1).mean(-1)

                    values = importance.flatten().sort()[0]
                    n = len(values)
                    idx = max(int(sparsity * n - 1), 0)
                    threshold = values[idx + 1]
                    _mask = importance >= threshold
                    _mask = _mask.view(1, num_c, 1, 1)
                    # logging.danger(f"actual sparsity = {1 - _mask.sum() / _mask.numel()}")
                    mask.data[:] = _mask
                else:
                    # weight pruning
                    from exp_helper import layer_types
                    if "transpose" in layer_types[self.name]["name"]:
                        importance = importance.transpose(0, 1)
                    num_c = importance.shape[0]
                    importance = importance.reshape(num_c, -1).mean(-1)
                    values = importance.flatten().sort()[0]
                    n = len(values)
                    idx = max(int(sparsity * n - 1), 0)
                    threshold = values[idx + 1]
                    _mask = importance >= threshold
                    if "linear" in layer_types[self.name]["name"]:
                        _mask = _mask.view(num_c, 1)
                    else:
                        if "transpose" in layer_types[self.name]["name"]:
                            _mask = _mask.view(1, num_c, 1, 1)
                        else:
                            _mask = _mask.view(num_c, 1, 1, 1)
                    mask.data[:] = _mask
                    # logging.danger(
                    #     f"{self.name}: actual sparsity = {1 - _mask.sum() / _mask.numel()}")
                return self.broadcast_mul(x, mask)
        else:
            logging.warn("run filter based pruning (shall be on weight)")
            target_shape = tuple(importance.shape[:2])
            importance = importance.view(*target_shape, -1).mean(-1)

            values = importance.flatten().sort()[0]
            n = len(values)
            idx = max(int(sparsity * n - 1), 0)
            threshold = values[idx]
            _mask = importance >= threshold
            _mask = _mask.view(*target_shape, 1, 1)
            mask.data[:] = _mask
            return self.broadcast_mul(x, mask)

    def broadcast_mul(self, x: torch.Tensor, mask: torch.Tensor, inplace=False):
        if inplace:
            x *= mask
            return x
        else:
            return x * mask

    def receive_input(self, x: torch.Tensor):
        if self.use_gradient:
            if self.prev_hook is not None:
                self.prev_hook.remove()
            if x.requires_grad:
                self.prev_hook = x.register_hook(
                    lambda grad: self.update_magnitude(grad))
            else:
                logging.error("meeting no-grad tensor")
                self.prev_hook = None
        else:
            self.update_magnitude(x)

    def update_magnitude(self, x):
        if self.running_average:
            with torch.no_grad():
                if self.bernoulli:
                    if x.min().item() == 0:
                        x = (x != 0).float()
                    # else:
                    #     logging.warn("non relu layer encounter!")
                x = align_tensor_to_shape(x.abs(), self.magnitude.shape)
                t = self.t.item()
                self.magnitude.data[:] = (t * self.magnitude + x) / (t + 1)

    def initialize(self, mask: torch.Tensor):
        if self.running_average:
            self.magnitude = nn.Parameter(
                torch.zeros(*mask.shape, device=mask.device,
                            dtype=torch.float),
                requires_grad=False,
            )

    def register_bn(self, bn):
        self.prev_bn = bn

    def forward(self, x: torch.Tensor, sparsity: float, mask: torch.Tensor, inplace=False, name=""):
        self.name = name
        if self.training:
            if not self.initted:
                self.initialize(mask)
                self.t.data[:] = 0
                if self.mask_refresh_interval <= 0:
                    self.mask_refresh_interval = 1

            t_item = self.t.item()
            if not self.use_existing_stat:
                if t_item < self.stop_mask_refresh:
                    self.receive_input(x)

            if (
                sparsity >= 0
                and (t_item % self.mask_refresh_interval == 0 and t_item <= self.stop_mask_refresh ) and (t_item != 0 or name.endswith(".prune")) # if name endswith `.prune`, then it's weight pruning, can prune directly
            ):
                logging.danger(f"prune_and_update_mask {name} target sparsity {sparsity}")
                if x.min().item() != 0:
                    logging.danger(f"non-relu layer {name} found!")
                out = self.prune_and_update_mask(x, sparsity, mask)
            else:
                out = self.broadcast_mul(x, mask, inplace=inplace)

            if self.use_existing_stat:
                if t_item < self.stop_mask_refresh:
                    self.receive_input(x)
            self.t += 1
            if self.name.endswith(".prune") and self.structure and self.t >= self.stop_mask_refresh:
                # print('setting bn to zero')
                from exp_helper import layer_types
                # also setting bias and bn parameters to zero
                assert inplace == False
                if layer_types[self.name]["link"] is None:
                    layer_types[self.name]["link"] = self.name

                if self.name == layer_types[self.name]["link"]:
                    name = layer_types[self.name]["name"]
                    op = layer_types[self.name]["op"]
                    bn = layer_types[self.name]["bn"]

                    if "transpose" in name:
                        _mask = mask[0, :, 0, 0]
                    elif "linear" in name:
                        _mask = mask[:, 0]
                    else:
                        _mask = mask[:, 0, 0, 0]

                    if bn is not None:
                        if bn.weight is not None:
                            if bn.weight.numel() != _mask.numel():
                                start = bn.weight.numel() - _mask.numel()
                                if bn.weight.data[start:].numel() == _mask.numel():
                                    bn.weight.data[start:] = bn.weight.data[start:] * _mask
                                    bn.bias.data[start:] = bn.bias.data[start:] * _mask
                                    # return x # prune at bn level, instead of here
                            else:
                                bn.weight.data[:] = bn.weight * _mask
                                bn.bias.data[:] = bn.bias * _mask
                                # return x # prune at bn level, instead of here

                    if op.bias is not None:
                        op.bias.data[:] = op.bias * _mask

            return out
        else:
            return self.broadcast_mul(x, mask, inplace=inplace)


class UniformPruningCallback(MagnitudePruningCallback):
    """unstructured uniform pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].
    This function will prune uniformly without considering magnitude of the input tensors. If a init mask is provided,
    this function will not reactivate those already pruned locations in init mask.
    """

    def initialize(self, mask: torch.Tensor):
        pass

    def receive_input(self, x: torch.Tensor):
        pass

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        cur_sparsity = (~mask).sum().item() / mask.numel()
        if cur_sparsity > sparsity:
            logging.warning("sparsity is decreasing, which shall not happen")
        budget = int((sparsity - cur_sparsity) * np.prod(mask.shape))
        slots = mask.nonzero(as_tuple=True)
        selected_indexes = np.random.choice(
            range(len(slots[0])), size=budget, replace=False
        )
        mask.data[[slot[selected_indexes] for slot in slots]] = False
        return self.broadcast_mul(x, mask)


def get_lower_conf_costs(count, cumsum, cumsum_square, t):
    safe_count = count + 0.0001
    mean = cumsum / safe_count
    variance = (cumsum_square / safe_count) - mean**2
    if t.item() == 0:
        T = t + 1
    else:
        T = t + 0.0001
    variance += torch.sqrt(2.0 * torch.log(T) / safe_count)
    lower_conf_costs = mean - torch.sqrt(torch.log(T) * variance / safe_count)
    lower_conf_costs[count < 1] = -float("inf")
    return lower_conf_costs


class BanditPruningFunction(torch.autograd.Function):
    """Pruning method based on multi-arm bandits"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        sparsity: float,
        mask_shape: Tuple[int],
        cumsum: torch.Tensor,  # initialized as all zeros
        cumsum_square: torch.Tensor,  # initialized as all zeros
        count: torch.Tensor,  # initialize as all zeros
        t: torch.Tensor,  # total number of experiments, t/0.8 => real T
        normalizer: torch.Tensor,  # use to normalize gradient distribution
        deterministic: bool,
        mask_out: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for bandit-based pruning algorithm

        Args:
            ctx: pytorch context
            inp (torch.Tensor): input tensor to be pruned
            sparsity (float): target sparsity ratio
            mask_shape (Tuple[int]): shape of the output mask
            cumsum (torch.Tensor): cumulative sum of the cost for each arm / neuron
            deterministic (bool): whether run in a deterministic mode, True will disable bandit parameters updates
            mask_out (torch.Tensor): output binary mask

        Returns:
            torch.Tensor: pruned input
        """

        ctx.sparsity = sparsity
        ctx.deterministic = deterministic

        dim = cumsum.numel()
        m = int(sparsity * dim)

        # UCBVTune Iteration Equation
        # costs -> importance
        lower_conf_costs = get_lower_conf_costs(
            count, cumsum, cumsum_square, t)

        # select the topk
        indexes = torch.topk(lower_conf_costs, m, largest=False).indices
        mask = torch.ones(dim, device=inp.device, dtype=torch.bool)
        mask[indexes] = 0  # top m -> pruned
        mask = mask.view(mask_shape)
        if deterministic:
            ctx.save_for_backward(mask)
        else:
            ctx.save_for_backward(
                mask, indexes, cumsum, cumsum_square, count, t, normalizer
            )
        mask_out.data[:] = mask
        return inp * mask

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.deterministic:
            mask = ctx.saved_tensors[0]
        else:
            (
                mask,
                indexes,
                cumsum,
                cumsum_square,
                count,
                t,
                normalizer,
            ) = ctx.saved_tensors

            grad = align_tensor_to_shape(grad_output.abs(), mask.shape)
            costs = grad.view(-1)[indexes]

            if normalizer.item() <= 0:  # set one time
                normalizer.data[:] = costs.quantile(0.95)

            costs /= normalizer

            # update bandit parameters
            count[indexes] += 1
            t.data += 1
            cumsum[indexes] += costs
            cumsum_square[indexes] += costs**2
        result = grad_output * mask.expand(grad_output.shape)
        return (result,) + (None,) * 10


class BanditPruningCallback(MagnitudePruningCallback):
    # through properly setup the bandit parameters for each layer
    # it is possible to implement the layer-wise from bottom pruning scheme
    # just use the `stop_mask_refresh` parameter
    # retrieve the list of prune layers
    # then through the order of it, set parameters one by one
    def __init__(self, **kwargs):
        """Callback to prune the network based on multi-arm bandits algorithms (UCBVTuned is used here)

        Args:
            exploration_steps (int): How many steps used for bandit learning
            collapse_batch_dim (bool, optional): whether treat the first dimension as batch dimension. Defaults to True.
        """
        if "mask_refresh_interval" not in kwargs:
            kwargs["mask_refresh_interval"] = 1
        super().__init__(**kwargs)

    def initialize(self, mask: torch.Tensor):
        device = mask.device
        self.normalizer = nn.Parameter(
            torch.zeros(1).to(device), requires_grad=False)
        self.ucbT = nn.Parameter(torch.zeros(
            1).to(device), requires_grad=False)
        self.count = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )
        self.cumsum = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )
        self.cumsum_square = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )

    def receive_input(self, x: torch.Tensor):
        pass

    @property
    def magnitude(self):
        return self.cumsum / (self.count + 0.00001)

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        if sparsity > 0:
            deterministic = self.t.item() >= self.stop_mask_refresh
            out = BanditPruningFunction.apply(
                x,
                sparsity,
                tuple(self.cumsum.shape),
                self.cumsum.view(-1),
                self.cumsum_square.view(-1),
                self.count.view(-1),
                self.ucbT,
                self.normalizer,
                deterministic,
                mask,
            )
            return out
        else:
            return x


class PruneLayer(nn.Module):
    """Applies pruning over input tensor.

    Please look for detailed description in [prune][qsparse.sparse.prune]
    """

    def __str__(self):
        return f"PruneLayer(sparsity={self.sparsity})"

    def __repr__(self):
        return str(self)

    def __init__(
        self,
        sparsity: float = 0.5,
        # for step-wise training
        start: int = 1000,
        interval: int = 1000,
        repetition: int = 4,
        strict: bool = True,
        # for customization
        callback: PruneCallback = MagnitudePruningCallback(),
        collapse: Union[int, List[int]] = 0,
        rampup: bool = False,
        # for debug purpose
        continue_pruning=False,
        name="",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.warning(
                f"[Prune{name if name == '' else f' @ {name}'}] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} collapse dimension = {collapse}"
            )

        self.schedules = [
            start + interval * ((1 if rampup else 0) + i) for i in range(repetition)
        ]
        self.start = start
        self.interval = interval
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self.rampup_interval = 0 if rampup else interval
        self._collapse = (
            collapse
            if isinstance(collapse, list)
            else [
                collapse,
            ]
        )
        self.strict = strict
        self.continue_pruning = continue_pruning

        for k in [
            "mask",
            "_n_updates",
            "_cur_sparsity",
        ]:
            self.register_parameter(
                k,
                nn.Parameter(
                    torch.tensor(-1, dtype=torch.int), requires_grad=False
                ),  # placeholder
            )

    @property
    def initted(self) -> bool:
        """whether the parameters of the prune layer are initialized."""
        return self._n_updates.item() != -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prunes input tensor according to given sparsification schedule.

        Args:
            x (torch.Tensor): tensor to be pruned

        Raises:
            RuntimeError: when the shape of input tensors mismatch with the shape of binary mask

        Returns:
            torch.Tensor: pruned tensor
        """
        

        if not self.initted:
            assert len(x.shape) > 1
            with torch.no_grad():
                # ! change !
                mask_shape = [
                            1 if i in self._collapse else s
                            for i, s in enumerate(list(x.shape))
                        ]
                # if len(self._collapse) > 0:
                #     # activation
                #     mask_shape = [
                #         1 if i in self._collapse else (
                #             s if i == 1 else 1)  # channelwise mask
                #         for i, s in enumerate(list(x.shape))
                #     ]
                # else:
                #     # weight
                #     mask_shape = [
                #         s for i, s in enumerate(list(x.shape))
                #     ]

                self.mask = nn.Parameter(
                    torch.ones(
                        *mask_shape,
                        dtype=torch.bool,
                    ).to(x.device),
                    requires_grad=False,
                )
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )
            self._cur_sparsity = nn.Parameter(
                torch.zeros(1).to(x.device), requires_grad=False
            )

        if (self._n_updates.item() in self.schedules) and self.training:
            ratio = (
                1.0
                - (self._n_updates.item() - self.start + self.rampup_interval)
                / (self.interval * self.repetition)
            ) ** 3
            self._cur_sparsity[0] = self.sparsity * (1 - ratio)
            logging.warning(
                f"[Prune{self.name if self.name == '' else f' @ {self.name}'}] [Step {self._n_updates.item()}] pruned {self._cur_sparsity.item():.02f}"
            )

        
        inplace = False
        if not self.training:
            if self.strict:
                if inplace:
                    # x *= self.mask.expand(x.shape)
                    x *= self.mask
                    out = x
                else:
                    out = x * self.mask
                    # out = x * self.mask.expand(x.shape)
            else:
                raise RuntimeError("")
                # mask = self.mask
                # if len(self.mask.shape) != len(x.shape):
                #     if len(self.mask.shape) == (len(x.shape) - 1):
                #         mask = mask.view(1, *mask.shape)
                #     else:
                #         raise RuntimeError(
                #             f"mask shape not matched: mask {mask.shape} vs input {x.shape}"
                #         )
                # target_shape = x.shape[1:]
                # final_mask = torch.ones(
                #     (1,) + target_shape, device=x.device, dtype=mask.dtype
                # )
                # repeats = [x.shape[i] // mask.shape[i] for i in range(1, len(x.shape))]
                # mask = mask.repeat(1, *repeats)
                # slices = [0] + [
                #     slice(
                #         (x.shape[i] - mask.shape[i]) // 2,
                #         (x.shape[i] - mask.shape[i]) // 2 + mask.shape[i],
                #     )
                #     for i in range(1, len(x.shape))
                # ]
                # final_mask[slices] = mask[0, :]
                # out = x * final_mask
        else:
            n_updates = self._n_updates.item()
            
            if n_updates >= self.start:
                if n_updates == self.start:
                    logging.warning(f"Start pruning at {self.name} @ {n_updates}")
                # if self._cur_sparsity.item() > 0:
                #     print(1)
                out = self.callback(x, self._cur_sparsity.item(
                ), mask=self.mask, inplace=inplace, name=self.name)
                if self.name:
                    id2mask[self.name] = self.mask
                    setattr(out, "qsparse_mask_id", self.name)
            else:
                out = x
            self._n_updates += 1
        return out


def prune(
    inp: nn.Module = None,
    sparsity: float = 0.5,
    # for step-wise training
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    strict: bool = True,
    collapse: Union[str, int, List[int]] = "auto",
    # for customization
    callback: PruneCallback = None,
    rampup: bool = False,
    continue_pruning: bool = False,
    # for debug purpose
    name="",
) -> nn.Module:
    """Creates a [PruneLayer][qsparse.sparse.PruneLayer] which is usually used
    for feature pruning if no input module is provided, or creates a weight-
    pruned version of the input module.

    Args:
        inp (nn.Module, optional): input module whose weight is to be pruned. Defaults to None.
        sparsity (float, optional): target sparsity. Defaults to 0.5.
        start (int, optional): starting step to apply pruning. Defaults to 1000.
        interval (int, optional): interval of iterations between each sparsity increasing steps. Defaults to 1000.
        repetition (int, optional): number of sparsity increasing steps. Defaults to 4.
        strict (bool, optional): whether enforcing the shape of the binary mask to be equal to the input tensor. Defaults to True.
                                 When strict=False, it will try to expand the binary mask to matched the input tensor shape during evaluation, useful for tasks whose test images are larger, like super resolution.
        collapse (Union[str, int, List[int]]): which dimension to ignore when creating binary mask. It is usually set to 0 for the batch dimension during pruning activations, and -1 when pruning weights. Default to "auto", means setting `collapse` automatically based on `inp` parameter.
        callback (PruneCallback, optional): callback for actual operation of calculating pruning mask (mask refreshing), used for customization. Defaults to [unstructured\_prune\_callback][qsparse.sparse.unstructured_prune_callback].
        rampup (bool, optional): whether to wait another interval before starting to prune. Defaults to False.
        name (str, optional): name of the prune layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight pruned or a instance of [PruneLayer][qsparse.sparse.PruneLayer] for feature pruning
    """

    if hasattr(callback, "mask_refresh_interval"):
        if callback.mask_refresh_interval <= 0:
            callback.mask_refresh_interval = interval

    callback = callback or MagnitudePruningCallback()

    kwargs = dict(
        sparsity=sparsity,
        start=start,
        interval=interval,
        repetition=repetition,
        strict=strict,
        collapse=collapse,
        callback=callback,
        name=name,
        rampup=rampup,
        continue_pruning=continue_pruning,
    )

    def get_prune_layer(
        feature_collapse=[
            0,
        ]
    ):
        if collapse != "auto":
            feature_collapse = collapse
        return PruneLayer(
            start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
            name=name,
            strict=strict,
            callback=callback,
            collapse=feature_collapse,
            rampup=rampup,
            continue_pruning=continue_pruning
        )

    if inp is None:
        layer = get_prune_layer()
        setattr(layer, "_kwargs", kwargs)
        return layer
    elif isinstance(inp, nn.Module):
        return imitate(inp, "prune", get_prune_layer([]))
    else:
        raise ValueError(f"{inp} is not a valid argument for prune")


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
