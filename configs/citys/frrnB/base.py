from require import require
base = require('../../base')
to_group_num = base.to_group_num
calculate_exp_id = base.calculate_exp_id
create_conversions = base.create_conversions


def citys_pruning_and_task_parameters(p_exp_id, q_exp_id, resume_after_convert, resume_from, kwargs):
    if kwargs.get("q_then_p", False):
        q_exp_id = ""

    gradual_pruning = {}
    if p_exp_id != '':
        gradual_pruning = {
            "gradual_pruning": {
                "batch": 1,
                "from": "top",
                "interval": 175,
                "mask_refresh_interval": 10,
                "start": 10,
                "freeze_during_pruning": False,
                "adaptive_per_layer": False,
                "normalize_statistics": True,
                "time_between_w_a": 4000
            }
        }

    task_parameters = {
        "training": {
            "loss": {
                "name": "bootstrapped_cross_entropy",
                "K": 32768
            },
            "n_workers": 10,
            "train_iters":  base.switch_quantize_prune(q_exp_id, p_exp_id, 7000 if '_cqs' not in q_exp_id else 2000, 35000),
            "batch_size": 3,
            "val_interval":  base.switch_quantize_prune(q_exp_id, p_exp_id, 500, 1000),
            "print_interval": 25,
            "optimizer": {
                "name": "adam",
                "lr": 0.001
            },
            "lr_schedule": {
                "name": "multi_step",
                "gamma": 0.1,
                "milestones": [
                    base.switch_quantize_prune(q_exp_id, p_exp_id, 2000 if '_cqs' not in q_exp_id else 1000, 20000)
                ]
            },
            "augmentations": {
                "translate": [
                    64,
                    128
                ],
                "gamma": 0.75
            },
            "resume": resume_from,
            "resume_after_convert": resume_after_convert
        },
        "record_after": base.switch_quantize_prune(q_exp_id, p_exp_id, 800, 15000)
    }

    if kwargs.get("no_save", False):
        task_parameters["no_save"] = True

    if kwargs.get("no_layerwise", False):
        gradual_pruning["gradual_pruning"]["no_layerwise"] = True

    if kwargs.get("layer_pq", False):
        gradual_pruning["gradual_pruning"]["layer_pq"] = True

    return gradual_pruning, task_parameters


def setup(pa=0, pw=0, qa=0, qw=0, apoz='', sparse_aware=False, naive_p=False, quantize_mode='g', zg=False, zgf=False,  structure=False, mask_batchnorm=False, bernoulli=False, filter_p=None,  unstructure_2nd=False, sp_balance=False, collect_q_stats="", no_save=False, group_both=False, no_layerwise=False, q_then_p=False, layer_pq=False):
    exp_id, p_exp_id, q_exp_id = calculate_exp_id(**locals())
    resume_after_convert = False
    if q_exp_id != '' and p_exp_id != '' and (not q_then_p) and (not layer_pq):
        resume_after_convert = True
        if structure:
            resume_from = {
                50: './checkpoints/citys/frrnB/a50_mag_top_loc_st/frrnB_cityscapes_final_model.pkl',
                75: './checkpoints/citys/frrnB/a75_mag_top_loc_st/frrnB_cityscapes_final_model.pkl'
            }[pa]
        assert resume_from
    else:
        resume_from = "./checkpoints/citys/frrnB/baseline/frrnB_cityscapes_best_model.pkl"
    assert resume_from is not None

    net_name = "frrnB"
    gradual_pruning_params, task_parameters = citys_pruning_and_task_parameters(
        p_exp_id, q_exp_id, resume_after_convert, resume_from, locals())
    
    if naive_p:
        gradual_pruning_params = {}

    q_weight_conversions = [
        {
            "op": "quantize",
            "bits": qw,
            "channelwise": 0,
            "timeout": 800 if q_exp_id != '' else -1,
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "callback": "ScalerQuantizer",
            "weight_layers": ["Conv2d"]
        }
    ]
    q_activation_conversions = [
        {
            "op": "quantize",
            "bits": qa,
            "channelwise": -1 if quantize_mode == 't' else 1,
            "group_num": to_group_num(quantize_mode),
            "timeout": 100 if q_exp_id != '' else -1,
            "callback": "AdaptiveLineQuantizer",
            "activation_layers": ["Conv2d"],
            "order": "pre",
            "spa": sparse_aware,
            "excluded_activation_layer_indexes": [["Conv2d", [0]]]
        }
    ]

    if q_then_p:
        gradual_pruning_params["gradual_pruning"]["start"] += q_weight_conversions[0]["timeout"] * 2

    if zg: 
        gradual_pruning_params = {}
        p_weight_conversions = [
        {
            "op": "prune",
            "sparsity": pw / 100,
            "callback": "MagnitudePruningCallback",
            "stop_mask_refresh": 14000,
            "start": 12000,
            # "rampup": 300,
            "interval": 500,
            "mask_refresh_interval": 500,
            "running_average": False,
            "repetition": 5,
            "weight_layers": ["Conv2d"],
            "excluded_weight_layer_indexes": [["Conv2d", [-1]]],
            "structure": True
            # "filter_based": zgf
        }
    ]
    else:
        if naive_p:
            p_weight_conversions = [
                {
                    "op": "prune",
                    "sparsity": pw / 100,
                    "callback": "MagnitudePruningCallback",
                    "weight_layers": ["Conv2d"],
                    "stop_mask_refresh": 13100,
                    "start": 10,
                    "interval": 3000,
                    "rampup": 1000,
                    "mask_refresh_interval": 3000,
                    "running_average": False,
                    "repetition": 4,
                    "exclude": "split",
                    "excluded_weight_layer_indexes": [["Conv2d", [-1]]]
                }
            ]   
        else:
            p_weight_conversions = [
            {
                "op": "prune",
                "sparsity": pw / 100,
                "callback": "MagnitudePruningCallback",
                "weight_layers": ["Conv2d"],
                "exclude": "split",
                "excluded_weight_layer_indexes": [["Conv2d", [-1]]]
            }
        ]   
        # if pw > 0:
        #     gradual_pruning_params["gradual_pruning"]["joint_weight_act"] = True

    if naive_p:
        p_activation_conversions = [
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",

                "stop_mask_refresh": 5100,
                "start": 10,
                "interval": 1000,
                "rampup": 1000,
                "mask_refresh_interval": 1000,
                "running_average": False,
                "repetition": 4,

                "activation_layers": ["ReLU"]
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",

                "stop_mask_refresh": 5100,
                "start": 10,
                "interval": 1000,
                "rampup": 1000,
                "mask_refresh_interval": 1000,
                "running_average": False,
                "repetition": 4,

                "activation_layers": ["Conv2d"],
                "filter": "merge_conv"
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",

                "stop_mask_refresh": 5100,
                "start": 10,
                "interval": 1000,
                "rampup": 1000,
                "mask_refresh_interval": 1000,
                "running_average": False,
                "repetition": 4,

                "activation_layers": ["BatchNorm2d"],
                "filter": "cb_unit"
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",

                "stop_mask_refresh": 5100,
                "start": 10,
                "interval": 1000,
                "rampup": 1000,
                "mask_refresh_interval": 1000,
                "running_average": False,
                "repetition": 4,

                "activation_layers": ["Conv2d"],
                "filter": "conv_res"
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
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "activation_layers": ["Conv2d"],
                "filter": "merge_conv"
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "activation_layers": ["BatchNorm2d"],
                "filter": "cb_unit"
            },
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "activation_layers": ["Conv2d"],
                "filter": "conv_res"
            }
        ]

    if mask_batchnorm:
        gradual_pruning_params["gradual_pruning"]["mask_batchnorm"] = True

    obj = {
        "name": f"citys/{net_name}/{exp_id}",
        "task_parameters": {
            "model": {
                "arch": net_name,
                "size": 1
            },
            "data": {
                "dataset": "cityscapes",
                "train_split": "train",
                "val_split": "val",
                "img_rows": 256,
                "img_cols": 512,
                "path": "/home/jovyan/Cityscapes"
            },
            **task_parameters
        },
        "$checkpoint_path": "checkpoint",
        "qsparse_parameters": {
            **gradual_pruning_params,
            "conversions": create_conversions(**locals()),
            "apoz": "/workspace/code/experiments/MDPI/analysis/apoz/json/citys_frrn.json" if apoz else ""
        }
    }

    return obj


if __name__ == "__main__":
    pass
