from require import require
base = require('../../base')
to_group_num = base.to_group_num
calculate_exp_id = base.calculate_exp_id
create_conversions = base.create_conversions
cifar100_pruning_and_task_parameters = require(
    '../densenet121/base').cifar100_pruning_and_task_parameters


def setup(pa=0, pw=0, qa=0, qw=0, apoz='', sparse_aware=False, naive_p=False, quantize_mode='g', zg=False, zgf=False,  structure=False, mask_batchnorm=False, bernoulli=False, filter_p=None,  unstructure_2nd=False, sp_balance=False, collect_q_stats="", no_save=False, group_both=False, no_layerwise=False, q_then_p=False, layer_pq=False):
    exp_id, p_exp_id, q_exp_id = calculate_exp_id(**locals())
    resume_after_convert = False
    if (q_exp_id != '' and p_exp_id != '' and (not q_then_p) and (not layer_pq)) or 'ust2' in p_exp_id:
        resume_after_convert = True
        if structure:
            resume_from = {
                50: './checkpoints/cifar100/preactresnet18/a50_mag_top_loc_st',
                75: './checkpoints/cifar100/preactresnet18/a75_mag_top_loc_st',
            }[pa]
    else:
        resume_from = "./checkpoints/cifar100/preactresnet18/baseline"
    assert resume_from is not None
    gradual_pruning, task_parameters = cifar100_pruning_and_task_parameters(
        p_exp_id, q_exp_id, locals())

    if naive_p:
        gradual_pruning = {}

    q_weight_conversions = [
        {
            "op": "quantize",
            "bits": qw,
            "channelwise": 0,
            "timeout": 0.5 if q_exp_id != '' else -1,
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "callback": "ScalerQuantizer",
            "weight_layers": ["Conv2d"]
        },
        {
            "op": "quantize",
            "bits": qw,
            "channelwise": -1,
            "timeout":  0.5 if q_exp_id != '' else -1,
            "callback": "ScalerQuantizer",
            "weight_layers": ["Linear"]
        }
    ]
    q_activation_conversions = [
        {
            "op": "quantize",
            "bits": qa,
            "channelwise": -1 if quantize_mode == 't' else 1,
            "group_num": to_group_num(quantize_mode),
            "timeout": 0.05 if q_exp_id != '' else -1,
            "callback": "AdaptiveLineQuantizer",
            "activation_layers": ["ReLU"],
            "spa": sparse_aware
        }
    ]


    if q_then_p:
        gradual_pruning["gradual_pruning"]["start"] += q_weight_conversions[0]["timeout"] * 4


    if zg:
        gradual_pruning = {}
        p_weight_conversions = [
            {
                "op": "prune",
                "sparsity": pw / 100,
                "callback": "MagnitudePruningCallback",
                "stop_mask_refresh": 2.5,
                "start": 0.5,
                # "rampup": 0.5,
                "interval": 0.5,
                "mask_refresh_interval": 0.5,
                "running_average": False,
                "repetition": 5,
                "weight_layers": ["Conv2d"],
                "structure": True
            }
        ]
    else:
        if naive_p:
            p_weight_conversions = [
                {
                    "op": "prune",
                    "sparsity": pw / 100,
                    "stop_mask_refresh": 12,
                    "start": 0.5,
                    "rampup": 0.5,
                    "interval": 2.5,
                    "mask_refresh_interval": 2.5,
                    "running_average": False,
                    "repetition": 4,
                    "callback": "MagnitudePruningCallback",
                    "weight_layers": ["Conv2d"],
                    "exclude": "shortcut"
                    # "excluded_weight_layer_indexes": [["Conv2d", [0]]]
                }
                # {
                #     "op": "prune",
                #     "sparsity": pw / 100,
                #     "stop_mask_refresh": 12,
                #     "start": 0.5,
                #     "rampup": 0.5,
                #     "interval": 2.5,
                #     "mask_refresh_interval": 2.5,
                #     "running_average": False,
                #     "repetition": 4,
                #     "callback": "MagnitudePruningCallback",
                #     "weight_layers": ["Conv2d"],
                #     "filter": ["stage4", "shortcut"]
                #     # "excluded_weight_layer_indexes": [["Conv2d", [0]]]
                # }
            ]   
        else:
            p_weight_conversions = [
                {
                    "op": "prune",
                    "sparsity": pw / 100,
                    "callback": "MagnitudePruningCallback",
                    "weight_layers": ["Conv2d"],
                    # "excluded_weight_layer_indexes": [["Conv2d", [0]]]
                }
            ]   
        # if pw > 0:
        #     gradual_pruning["gradual_pruning"]["joint_weight_act"] = True
        

    if naive_p:
        p_activation_conversions = [
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "stop_mask_refresh": 12,
                "start": 0.5,
                "rampup": 0.5,
                "interval": 2.5,
                "mask_refresh_interval": 2.5,
                "running_average": False,
                "repetition": 4,
                "activation_layers": ["ReLU"]
            }
        ]
    else:
        p_activation_conversions = [
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "activation_layers": ["ReLU"],
                "bernoulli": bernoulli
            }
        ]

    if mask_batchnorm:
        gradual_pruning["gradual_pruning"]["mask_batchnorm"] = True

    obj = {
        "name": f"cifar100/preactresnet18/{exp_id}",
        "task_parameters": {
            "gpu": True,
            "net": "preactresnet18",
            "warm": 0,
            "resume": True,
            "resume_from": resume_from,
            "resume_after_convert": resume_after_convert,
            **task_parameters
        },
        "$checkpoint_path": "CHECKPOINT_PATH",
        "qsparse_parameters": {
            **gradual_pruning,
            "conversions": create_conversions(**locals()),
            "apoz": "/workspace/code/experiments/MDPI/analysis/apoz/json/cifar100_preactresnet.json" if apoz else ''
        }
    }
    return obj


if __name__ == "__main__":
    pass
