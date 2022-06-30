# MDPI 2022

This repo contains source code for the experiments reported in the MDPI submission "Learning Low Precision Structured Subnetworks using Joint Layerwise Channel Pruning and Uniform Quantization". 

- Image Classification on Cifar100: folder [classification](classification/), based on [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100).
- Semantic Segmentation on Cityscape: folder [segmentation](segmentation/), based on [meetps/pytorch-semseg](https://github.com/meetps/pytorch-semseg) and [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- Image Style Transfer on Cityscape: folder [style_transfer](style_transfer/), based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


To keep the code changes on each task minimal, we centralize the network conversion logic in a utility file called [exp_helper.py](exp_helper.py), which is imported by all the experiments. An example of using `exp_helper.py` can be found in [classification/train.py#L232](classification/train.py#L232). There is a customized version of [qsparse](https://github.com/mlzxy/qsparse) residing in the folder [qsparse-private](qsparse-private/), in which we build the core functionalities of pruning and quantization modules and network traversal. The implementation of pruning and quantization module can be found in [qsparse-private/qsparse/sparse.py](qsparse-private/qsparse/sparse.py) and [qsparse-private/qsparse/quantize.py](qsparse-private/qsparse/quantize.py), respectively.


Each experiment is defined by a configuration JSON file. Examples can be found in the [configs](configs/) folder. The workflow of running an experiment is:

1. Prerequisite: Install the required environment, download pretrain checkpoints and datasets. We will cover this in the later section.
2. Provide the path to the configuration JSON file, e.g., [configs/cifar100/densenet121/a75_mag_top_loc_st.json](configs/cifar100/densenet121/a75_mag_top_loc_st.json), and training entry, e.g., [classification/train.py](classification/train.py).
3. `train.py` imports `exp_helper.py` and provides the dense network instance. `exp_helper.py`  will (1) parse the configuration JSON file (2) set hyper-parameters based on the given configuration, e.g., training epochs, learning rate, etc. (3) inject pruning / quantization modules into the network.
4. `train.py` starts the training.



## Prerequisite



1. Build or download the docker image from [gitlab.nrp-nautilus.io/zxy/mydocker](https://gitlab.nrp-nautilus.io/zxy/mydocker), which installs all the required dependencies that the codebase relies on.
2. Download the pretrained checkpoints from https://drive.google.com/file/d/1UsfIXqBhtrGad3NzGzSVwQwUoWawCRks/view?usp=sharing. Extract it into the project root, you will get a `checkpoints` folder.
3. Download the dataset file from https://drive.google.com/file/d/1UsfIXqBhtrGad3NzGzSVwQwUoWawCRks/view?usp=sharing, which contains a copy of cifar100 and cityscape datasets. Extract it into your `${HOME}`, for Cityscape dataset used for segmentation training, run `decompress.sh` script located in `~/Cityscapes` after extraction.




## Command to run experiments

Take the DenseNet121 on Cifar100 as an example (others are similar and
self-explanatory from the names of configuration files), to start 75% pruning
training, run the following commands:

```bash
./qr.sh cifar100/densenet121/a75_mag_top_loc_st # layerwise 
./qr.sh cifar100/densenet121/a75_mag_top_loc_st_nly # one-shot 
./qr.sh cifar100/densenet121/w75_mag_top_loc_zg_st # stepwise 
```

To run joint pruning and quantization with W8A8: 

```bash
./qr.sh cifar100/densenet121/a75_wa8_t_st
```

`qr.sh` is a utility script that auto-completes training command and enables to run experiments with only a partial path of the configuration JSON file. 

You can find your output at `~/output/mdpi` folder.