from require import require
base = require('../../base')
to_group_num = base.to_group_num
calculate_exp_id = base.calculate_exp_id
create_conversions = base.create_conversions



def citys_pruning_and_task_parameters(p_exp_id, q_exp_id, resume_after_convert, resume_from, kwargs):
    gradual_pruning = {}
    if p_exp_id != '':
        gradual_pruning = {
            "gradual_pruning": {
                "batch": 1,
                "from": "top",
                "interval": 0.5,
                "mask_refresh_interval": 0.1,
                "start": 0.1,
                "freeze_during_pruning": False
            }
        }

    task_parameters = {
        "size": 1.0,
        "direction": "BtoA",
        "netG": "unet_256", 
        "dataroot": "/home/jovyan/Cityscapes_pix2pix/cityscapes",
        "batch_size": 3,
        "name": "cityscapes_pix2pix",
        "pool_size": 0,
        "gan_mode": "vanilla",
        "dataset_mode": "aligned",
        "model": "pix2pix",
        "no_dropout": True,
        "n_epochs": base.switch_quantize_prune(q_exp_id, p_exp_id, 20, 20),
        "n_epochs_decay": base.switch_quantize_prune(q_exp_id, p_exp_id, 40, 80),
        "lr": base.switch_quantize_prune(q_exp_id, p_exp_id, 0.0001, 0.0002), 
        "lambda_L1": 100,
        "load_size": 286,
        "crop_size": 256,
        "norm": "batch",
        "save_latest_freq": 15000,
        "save_epoch_freq": 10,
        "display_freq": 4000,
        "fid_cache": "/home/jovyan/Cityscapes_pix2pix/fid_stat.A.256.npz",
        "record_after": base.switch_quantize_prune(q_exp_id, p_exp_id, 5, 12),

        "resume_after_convert": resume_after_convert,
        "resume_from": resume_from
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
                50: 'checkpoints/citysgan/unet/a50_mag_top_loc_st/cityscapes_pix2pix',
                75: 'checkpoints/citysgan/unet/a75_mag_top_loc_st/cityscapes_pix2pix'
            }[pa]
        assert resume_from
    else:
        resume_from = "checkpoints/citysgan/unet/baseline/cityscapes_pix2pix"
    assert resume_from is not None

    gradual_pruning_params, task_parameters = citys_pruning_and_task_parameters(
        p_exp_id, q_exp_id, resume_after_convert, resume_from, locals())


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
            "channelwise": 1,
            "timeout": 0.5 if q_exp_id != '' else -1,
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "callback": "ScalerQuantizer",
            "weight_layers": ["ConvTranspose2d"],
            "excluded_weight_layer_indexes": [["ConvTranspose2d", [-1]]]
        },
        {
            "op": "quantize",
            "bits": 8,
            "channelwise": 1,
            "timeout": 0.5 if q_exp_id != '' else -1,
            "group_num": to_group_num(quantize_mode) if group_both else -1,
            "callback": "ScalerQuantizer",
            "weight_layers": ["ConvTranspose2d"],
            "filter": "model.model.4"
        }
    ]

    q_activation_conversions = [
        {
            "op": "quantize",
            "bits": qa,
            "channelwise": -1 if quantize_mode == 't' else 1,
            "group_num": to_group_num(quantize_mode),
            "timeout": 0.1 if q_exp_id != '' else -1,
            "callback": "AdaptiveLineQuantizer",
            "activation_layers": ["Conv2d", "ConvTranspose2d"],
            "order": "pre",
            "spa": sparse_aware,
            "excluded_activation_layer_indexes": [["Conv2d", [0]], ["ConvTranspose2d", [-1]]]
        },
        {
            "op": "quantize",
            "bits": 8,
            "channelwise": -1 if quantize_mode == 't' else 1,
            "group_num": to_group_num(quantize_mode),
            "timeout": 0.1 if q_exp_id != '' else -1,
            "callback": "AdaptiveLineQuantizer",
            "activation_layers": ["ConvTranspose2d"],
            "order": "pre",
            "spa": sparse_aware,
            "filter": "model.model.4"
        }
    ]

    if q_then_p:
        gradual_pruning_params["gradual_pruning"]["start"] += q_weight_conversions[0]["timeout"] * 4

    if zg:
        gradual_pruning_params = {}
        p_weight_conversions = [
        {
            "op": "prune",
            "sparsity": pw / 100,
            "callback": "MagnitudePruningCallback",
            "stop_mask_refresh": 2,
            "start": 0,
            "interval": 0.5,
            "mask_refresh_interval": 0.5,
            "running_average": False,
            "repetition": 5,
            "weight_layers": ["Conv2d", "ConvTranspose2d"],
            "excluded_weight_layer_indexes": [["Conv2d", [-1]]],
            "structure": True
            # "filter_based": zgf
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
            }
        ]   

    p_activation_conversions = [
            {
                "op": "prune",
                "sparsity": pa / 100,
                "callback": "MagnitudePruningCallback",
                "order": "pre",
                "activation_layers": ["Conv2d", "ConvTranspose2d"],
                "excluded_activation_layer_indexes": [["Conv2d", [0]]]
            },
            # {
            #     "op": "prune",
            #     "sparsity": pa / 100,
            #     "callback": "MagnitudePruningCallback",
            #     "activation_layers": ["ReLU", "LeakyReLU"]
            # }
        ]

    obj = {
        "name": f"citysgan/unet/{exp_id}",
        "task_parameters": {
            **task_parameters
        },
        "$checkpoint_path": "checkpoints_dir",
        "qsparse_parameters": {
            **gradual_pruning_params,
            "conversions": create_conversions(**locals()),
            "apoz": "/workspace/code/experiments/MDPI/analysis/apoz/json/citysgan_p2p.json" if apoz else ""
        }
    }

    return obj
