import torch
from collections import defaultdict, OrderedDict
from qsparse.util import logging
from tqdm import tqdm
import os.path as osp
import jstyleson as json
from require import require
import torch.nn as nn

layer_types = {}


class Config:
    qsparse_params = []
    epoch_size = 0
    pruning_finished_epoch = -1

def partial(func, **kwargs):
    def f(*args, **kw):
        for k,v in kwargs.items():
            kw[k] = v
        return func(*args, **kw)
    return f

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def is_true(dct, key):
    return key in dct and dct[key]

def init(args=None):
    import os, sys, time
    from pprint import pprint
    from datetime import datetime, timedelta
    now = datetime.now() + timedelta(hours=8)
    assert "config_file" in os.environ, "config_file not set in envrionment variables"
    config_file_path = os.environ["config_file"]
    conf = require(config_file_path, default=True)
    if is_true(os.environ, "debug"):
        repo = osp.expanduser("~/debug/mdpi")
    else:
        repo = conf.get("repo", os.environ.get("output_dir", default=osp.expanduser("~/output/mdpi")))
    name = conf["name"]
    folder = f"{repo}/{name}/{now.strftime('%Y-%m-%d')}"
    if not os.path.exists(folder): os.makedirs(folder)
    logfile_path = f'{folder}/log.txt'
    if os.path.exists(logfile_path): os.rename(logfile_path, logfile_path + f".bak.{int(time.time())}")
    logfile = open(logfile_path, 'w')
    with open(f'{folder}/config.json', 'w') as f: f.write(json.dumps(conf, indent=4))
    class Unbuffered:
        def __init__(self, stream): self.stream = stream
        def write(self, data): self.stream.write(data); logfile.write(data); logfile.flush(); self.stream.flush()
        def flush(self): self.stream.flush(); logfile.flush()
    sys.stdout = Unbuffered(sys.stdout)
    sys.stderr = Unbuffered(sys.stderr)
    Config.qsparse_params = conf.get("qsparse_parameters", []) # convert configurations
    if args is not None:
        for k, v in conf.get("task_parameters", {}).items():
            print(f"PARAM {k} <- {v}")
            for a in args:
                setattr(a, k, v)
    pprint(conf)
    for a in args:
        setattr(a, conf['$checkpoint_path'], folder)

def set_epoch_size(loader): Config.epoch_size = loader if isinstance(loader, int) else len(loader)


def convert(net):
    import qsparse, time, math
    from qsparse import quantize, prune
    from qsparse.sparse import PruneLayer
    from qsparse.quantize import QuantizeLayer
    import torch.nn as nn
    import random
    from qsparse.util import logging

    def to_step(x): return int(round(Config.epoch_size * x)) if x > 0 else (1 if x == 0 else -1)
    def layer_names_to_modules(names): return [getattr(nn, layer_name) for layer_name in names]
    def layer_names_to_modules_2rd(indexes): return [(getattr(nn, layer_name), indexes) for layer_name, indexes in indexes]

    if isinstance(Config.qsparse_params, list):
        conversion_params = Config.qsparse_params
        qsparse_params = {}
    else:
        qsparse_params = Config.qsparse_params
        conversion_params = qsparse_params["conversions"]

    if len(conversion_params) > 0:
        for param in conversion_params:
            print(f'***** Convert net with {param} *****')
            callback_kwargs = {}
            if "outlier_ratio" in param:
                callback_kwargs["outlier_ratio"] = param["outlier_ratio"]
            if "line" in param["callback"].lower(): # quantization, use running average
                callback_kwargs["always_running_average"] = True

            if "spa" in param:
                callback_kwargs["spa"] = True

            if "group_num" in param:
                callback_kwargs["group_num"] = param["group_num"]


            net = qsparse.convert(net, quantize(
                bits=param["bits"], channelwise=param["channelwise"], timeout=to_step(param["timeout"]), 
                callback=getattr(qsparse, param["callback"])(**callback_kwargs)
            ) if param["op"] == "quantize" else prune(
                sparsity=param["sparsity"],
                start=to_step(param.get("start", 0)),
                interval=to_step(param.get("interval", 0)),
                rampup=param.get("rampup", False),
                repetition=param.get("repetition", 1),
                callback=qsparse.MagnitudePruningCallback(structure=True, bernoulli=param.get("bernoulli", False),  mask_refresh_interval=to_step(param.get("mask_refresh_interval", 0)), use_gradient=param.get("use_gradient", False), running_average=param.get("running_average", True), stop_mask_refresh=to_step(param.get("stop_mask_refresh", -1)))
                            if param["callback"] == 'MagnitudePruningCallback' else (
                                qsparse.UniformPruningCallback()
                                if param["callback"] == "UniformPruningCallback" else
                                qsparse.BanditPruningCallback(stop_mask_refresh=to_step(param.get("stop_mask_refresh", -1)), mask_refresh_interval=-1))
                ),
                weight_layers=layer_names_to_modules(param.get("weight_layers", [])),
                activation_layers=layer_names_to_modules(param.get("activation_layers", [])),
                excluded_activation_layer_indexes=layer_names_to_modules_2rd(param.get("excluded_activation_layer_indexes", [])),
                excluded_weight_layer_indexes=layer_names_to_modules_2rd(param.get("excluded_weight_layer_indexes", [])), filter=param.get("filter", []), exclude=param.get("exclude", None), input=param.get("input", False), order=param.get('order', 'post'))
    print(str(net))
    all_layers = OrderedDict(net.named_modules())
    all_layer_names = list(all_layers.keys())


    def find_prev_op_name(layer_name):
        ind = all_layer_names.index(layer_name) - 1
        while ind >= 0:
            ln = all_layer_names[ind]
            layer = all_layers[ln]
            layer_type = layer.__class__.__name__.lower()
            if 'conv2d' == layer_type or 'convtranspose2d' == layer_type or 'linear' ==layer_type:
                return ln
            ind -= 1
        raise RuntimeError()

    def find_next_bn(layer_name, parent_op=None):
        ind = all_layer_names.index(layer_name)
        for i in range(ind+1, len(all_layer_names)):
            ln = all_layer_names[i]
            layer = all_layers[ln]
            layer_type = layer.__class__.__name__.lower()
            if (('conv2d' == layer_type or 'convtranspose2d' == layer_type) and 'shortcut' not in ln) or ('linear' == layer_type):
                return None 
            elif layer_type.endswith('norm2d'):
                return layer
        logging.danger(f"no next normalization layer is found for {layer_name}")
        return None

    just_w_players = [mod for mod in net.modules() if isinstance(mod, PruneLayer) and mod.name.endswith("prune")]
    just_a_players = [mod for mod in net.modules() if isinstance(mod, PruneLayer) and not mod.name.endswith("prune")]
    a_w_players = [mod for mod in net.modules() if isinstance(mod, PruneLayer)]
    q_players = [mod for mod in net.modules() if isinstance(mod, QuantizeLayer)]
    players = a_w_players
    if "gradual_pruning" in qsparse_params:
        param = qsparse_params["gradual_pruning"]
        is_weight_prune = all([p.name.endswith('.prune') for p in players])

        non_layerwise = param.get("no_layerwise", False)

        mask_refresh_interval = to_step(param["mask_refresh_interval"])
        print(f"============ Gradual Pruning (Mask Refresh Every {mask_refresh_interval}) ============")
        if param.get("from", "top") == "bottom":
            players = list(reversed(players))

        if param.get("from", "top") == "shuffle":
            random.shuffle(players)

        start = to_step(param["start"]) * param.get("interval_multiplier", 1) #// 10 # TODO
        interval = to_step(param["interval"]) * param.get("interval_multiplier", 1) #// 4 # TODO
        logging.danger(f"start: {start}, interval: {interval}")



        for l in players:
            l.start = start
            l.repetition = 1
            if "rampup" in param:
                l.schedules = [start + to_step(param["rampup"]), ]
                l.rampup_interval = 0
                l.interval = to_step(param["rampup"])
            else:
                l.schedules = [start, ]
            l.callback.mask_refresh_interval = mask_refresh_interval 
            l.callback.stop_mask_refresh = interval
            if is_weight_prune:
                l.callback.running_average = False
            start += (interval + 1)

        if non_layerwise:
            for l in players:
                l.start = start - interval
                l.repetition = 1
                l.schedules = [start, ]
                l.rampup_interval = 0
                l.interval = interval

                l.callback.mask_refresh_interval = interval
                l.callback.stop_mask_refresh = interval
                l.callback.use_existing_stat = True
                if is_weight_prune:
                    l.callback.running_average = False

        Config.pruning_finished_epoch = activation_stop = int(math.ceil( start / Config.epoch_size ))
        logging.danger(f"Pruning stops at epoch - {activation_stop}")

    just_w_players = [mod for mod in net.modules() if isinstance(mod, PruneLayer) and mod.name.endswith("prune")]

    # for weight pruning, we also need to set the corresponding channels of batch norm layers to zero
    for l in just_w_players:
        layer_name = l.name.split('.prune')[0]
        parent_op = all_layers[layer_name]
        assert parent_op is not None
        _ = {
            "name": parent_op.__class__.__name__.lower(),
            "op": parent_op,
            "bn": find_next_bn(layer_name),
            "link": None
        }
        layer_types[l.name] =  _
        logging.warning(f"{l.name} = {_}")

    return net




def epoch_callback(net, epoch):
    net.train()
    return net
