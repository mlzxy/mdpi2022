# MDPI 2022

This repo contains source code for the experiments reported in the MDPI publication ["Learning Low Precision Structured Subnetworks using Joint Layerwise Channel Pruning and Uniform Quantization"](https://www.mdpi.com/2076-3417/12/15/7829).

- Image Classification on Cifar100: folder [classification](classification/), based on [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100).
- Semantic Segmentation on Cityscape: folder [segmentation](segmentation/), based on [meetps/pytorch-semseg](https://github.com/meetps/pytorch-semseg) and [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- Image Style Transfer on Cityscape: folder [style_transfer](style_transfer/), based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

To keep the code changes on each task minimal, we centralize the network conversion logic in a utility file called [exp_helper.py](exp_helper.py), which is imported by all the experiments. An example of using `exp_helper.py` can be found in [classification/train.py#L232](classification/train.py#L232). There is a customized version of [qsparse](https://github.com/mlzxy/qsparse) residing in the folder [qsparse-private](qsparse-private/), in which we build the core functionalities of pruning and quantization modules and network traversal. The implementation of pruning and quantization module can be found in [qsparse-private/qsparse/sparse.py](qsparse-private/qsparse/sparse.py) and [qsparse-private/qsparse/quantize.py](qsparse-private/qsparse/quantize.py), respectively. We also provide a refactored implementation in the original repo [qsparse](https://github.com/mlzxy/qsparse), which is submoduled at [qsparse-public](qsparse-public/). This version can be installed from [pypi](https://pypi.org/project/qsparse/) and has a much cleaner codebase. Further instructions on how to run experiments with [qsparse-public](qsparse-public/) are provided in the latter of this README. 


Each experiment is defined by a configuration JSON file. Examples can be found in the [configs](configs/) folder. The workflow of running an experiment is:

1. Prerequisite: Install the required environment, download pretrain checkpoints and datasets. We will cover this in the later section.
2. Provide the path to the configuration JSON file, e.g., [configs/cifar100/densenet121/a75_mag_top_loc_st.json](configs/cifar100/densenet121/a75_mag_top_loc_st.json), and training entry, e.g., [classification/train.py](classification/train.py).
3. `train.py` imports `exp_helper.py` and provides the dense network instance. `exp_helper.py`  will (1) parse the configuration JSON file (2) set hyper-parameters based on the given configuration, e.g., training epochs, learning rate, etc. (3) inject pruning / quantization modules into the network.
4. `train.py` starts the training.

## Prerequisite

1. Build or download the docker image from [gitlab.nrp-nautilus.io/zxy/mydocker](https://gitlab.nrp-nautilus.io/zxy/mydocker), which installs all the required dependencies that the codebase relies on.
2. Download the pretrained checkpoints from [https://drive.google.com/file/d/1UsfIXqBhtrGad3NzGzSVwQwUoWawCRks/view?usp=sharing](https://drive.google.com/file/d/1UsfIXqBhtrGad3NzGzSVwQwUoWawCRks/view?usp=sharing). Extract it into the project root, you will get a `checkpoints` folder.  (More checkpoints can be found here [https://drive.google.com/file/d/1hgHUMxVyAaNbn_gbi6rBZq_7dKo2n1_b/view?usp=sharing](https://drive.google.com/file/d/1hgHUMxVyAaNbn_gbi6rBZq_7dKo2n1_b/view?usp=sharing))
3. Download the dataset file from https://drive.google.com/file/d/1Y7PduznYNic-OfbMbIrc48VIB6sDnZ8N/view?usp=sharing, which contains a copy of cifar100 and cityscape datasets. Extract it into your `${HOME}`, for Cityscape dataset used for segmentation training, run `decompress.sh` script located in `~/Cityscapes` after extraction.

## Command to run experiments

Take the DenseNet121 on Cifar100 as an example, to start 75% pruning training, run the following commands:

```bash
./qr.sh cifar100/densenet121/baseline_quarter        # from scratch with 25% channel
./qr.sh cifar100/densenet121/a75_mag_top_loc_st      # layerwise 
./qr.sh cifar100/densenet121/a75_mag_top_loc_st_nly  # one-shot 
./qr.sh cifar100/densenet121/w75_mag_top_loc_zg_st   # stepwise
```

To start 50% pruning training, run the following commands:

```
./qr.sh cifar100/densenet121/baseline_half        # from scratch with 50% channel
./qr.sh cifar100/densenet121/a50_mag_top_loc_st      # layerwise 
./qr.sh cifar100/densenet121/a50_mag_top_loc_st_nly  # one-shot 
./qr.sh cifar100/densenet121/w50_mag_top_loc_zg_st   # stepwise
```

To run joint pruning and quantization:

```bash
./qr.sh cifar100/densenet121/a75_wa8_t_st  # W8A8
./qr.sh cifar100/densenet121/a75_wa4_t_st  # W4A4
```

`qr.sh` is a utility script that auto-completes training command and enables to run experiments with only a partial path of the configuration JSON file.

You can find your output at `~/output/mdpi` folder, the output location can be configured with `output_dir` environment variable as indicated at [exp_helper.py#L45](exp_helper.py#L45).

Each task has its own prefix:

* Classification at Cifar100
  * DenseNet121: `cifar100/densenet121`
  * MobileNetV2: `cifar100/mobilenetv2`
* Segmentation at Cityscape
  * UNet: `citys/unet`
  * FRRNet: `citys/frrnB`
* Style Transfer at Cityscape
  * CycleGAN: `citysgan/res`


### Run experiments with public version of qsparse

You can install qsparse by `pip install qsparse`, or pull it as a submodule through:

```bash
git submodule init
git submodule update
```

Then, prepend the `USE_PUBLIC_QSPARSE=1` to the above commands. For example,

```bash
USE_PUBLIC_QSPARSE=1 ./qr.sh cifar100/mobilenetv2/a75_mag_top_loc_st
```

Note that the public qsparse library is not compatible with the one-shot pruning configuration, e.g.,`75_mag_top_loc_st_nly`, and will yield different results on weight pruning experiments because [qsparse-private](qsparse-private/) supports resetting bias and batchnorm parameters of pruned channels to zero ([code](qsparse-private/qsparse/sparse.py#L515)), in order to align the behavior of weight and activation pruning, while the public version does not. The results in our publication are based on the private version.

## Citing

If you find this open source release useful, please reference in your paper:

> Zhang, X.; Colbert, I.; Das, S. Learning Low-Precision Structured Subnetworks Using Joint Layerwise Channel Pruning and Uniform Quantization. Appl. Sci. 2022, 12, 7829. https://doi.org/10.3390/app12157829

```bibtex
@Article{app12157829,
	AUTHOR = {Zhang, Xinyu and Colbert, Ian and Das, Srinjoy},
	TITLE = {Learning Low-Precision Structured Subnetworks Using Joint Layerwise Channel Pruning and Uniform Quantization},
	JOURNAL = {Applied Sciences},
	VOLUME = {12},
	YEAR = {2022},
	NUMBER = {15},
	ARTICLE-NUMBER = {7829},
	URL = {https://www.mdpi.com/2076-3417/12/15/7829},
	ISSN = {2076-3417}
}
```

