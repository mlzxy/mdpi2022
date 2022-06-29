from require import require
base = require('../../base')
to_group_num = base.to_group_num
calculate_exp_id = base.calculate_exp_id
create_conversions = base.create_conversions
citys_pruning_and_task_parameters = require(
    '../frrnB/base').citys_pruning_and_task_parameters


def setup(pa=0, pw=0, qa=0, qw=0, apoz='', sparse_aware=False, naive_p=False, quantize_mode='g', zg=False, zgf=False, structure=False, mask_batchnorm=False, bernoulli=False, filter_p=None,  unstructure_2nd=False, sp_balance=False, collect_q_stats="", no_save=False, group_both=False, no_layerwise=False, q_then_p=False, layer_pq=False):
    exp_id, p_exp_id, q_exp_id = calculate_exp_id(**locals())
    resume_after_convert = False
    if q_exp_id != '' and p_exp_id != '' and (not q_then_p) and (not layer_pq):
        resume_after_convert = True
        if structure:
            resume_from = {
                50: './checkpoints/citys/unet/a50_mag_top_loc_st/unet_cityscapes_final_model.pkl',
                75: './checkpoints/citys/unet/a75_mag_top_loc_st/unet_cityscapes_final_model.pkl'
            }[pa]
        assert resume_from
    else:
        resume_from = "./checkpoints/citys/unet/baseline/unet_cityscapes_best_model.pkl"
    assert resume_from is not None

    net_name = "unet"
    gradual_pruning_params, task_parameters = citys_pruning_and_task_parameters(
        p_exp_id, q_exp_id, resume_after_convert, resume_from, locals())
    task_parameters["training"]["batch_size"] = 7
    # if pa >= 75:
    #     task_parameters["training"]["batch_size"] = 16
    
    if naive_p:
        gradual_pruning_params = {}

    q_weight_conversions = [
        {
            "op": "quantize",
            "bits": qw,
            "channelwise": 0,
            "timeout": 800 if q_exp_id != '' else -1,
            "callback": "ScalerQuantizer",
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "weight_layers": ["Conv2d"]
        },
        {
            "op": "quantize",
            "bits": qw,
            "channelwise": 1,
            "timeout": 800 if q_exp_id != '' else -1,
            "callback": "ScalerQuantizer",
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "weight_layers": ["ConvTranspose2d"]
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
            "activation_layers": ["ReLU"],
            "spa": sparse_aware,
        },
        {
            "op": "quantize",
            "bits": qa,
            "channelwise": -1 if quantize_mode == 't' else 1,
            "group_num": to_group_num(quantize_mode),
            "timeout": 100 if q_exp_id != '' else -1,
            "callback": "AdaptiveLineQuantizer",
            "activation_layers": ["ConvTranspose2d"],
            "spa": sparse_aware,
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
            "interval": 500,
            "mask_refresh_interval": 500,
            "running_average": False,
            "repetition": 5,
            "weight_layers": ["Conv2d", "ConvTranspose2d"],
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
                    "weight_layers": ["Conv2d", "ConvTranspose2d"],
                    "stop_mask_refresh": 9100,
                    "start": 10,
                    "interval": 2000,
                    "rampup": 1000,
                    "mask_refresh_interval": 2000,
                    "running_average": False,
                    "repetition": 4,
                    "excluded_weight_layer_indexes": [["Conv2d", [-1]]]
                }
            ]   
        else:
            p_weight_conversions = [
                {
                    "op": "prune",
                    "sparsity": pw / 100,
                    "callback": "MagnitudePruningCallback",
                    "weight_layers": ["Conv2d", "ConvTranspose2d"],
                    "excluded_weight_layer_indexes": [["Conv2d", [-1]]]
                    # "excluded_weight_layer_indexes": [["Conv2d", [0, -1]]]
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

                "activation_layers": ["ReLU", "ConvTranspose2d"]
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
                "activation_layers": ["ConvTranspose2d"]
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
            "apoz": "/workspace/code/experiments/MDPI/analysis/apoz/json/citys_unet.json" if apoz else ''
        }
    }

    return obj


if __name__ == "__main__":
    pass