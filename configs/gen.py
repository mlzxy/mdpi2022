#!/usr/bin/python3.8
import os.path as osp
from require import require, dump_json

if __name__ == "__main__":
    root = osp.dirname(__file__)

    module_paths = [
        "./citysgan/res",
        "./citysgan/unet",

        "./citys/unet",
        "./citys/frrnB",
        "./cifar100/densenet121",
        "./cifar100/mobilenetv2",
        "./cifar100/preactresnet18"
    ]

    def write(data):
        dump_json(osp.join(root, data['name'] + '.json'), data)
    

    def gen_all(path, setup):

        for p in [50, 75]:
            # A L0 One-Shot
            write(setup(pa=p, structure=True, no_layerwise=True, bernoulli=True)) 

            # A L0 Layerwise
            write(setup(pa=p, structure=True, bernoulli=True))

            # W L1 One-Shot
            write(setup(pw=p, structure=True, no_layerwise=True)) 

            # W L1 Layerwise
            write(setup(pw=p, structure=True)) 

            # A L1 One-Shot
            write(setup(pa=p, structure=True, no_layerwise=True)) 

            # A L1 Layerwise 
            write(setup(pa=p, structure=True)) # layerwise + A

            # W L1 Stepwise
            write(setup(pw=p, zg=True, structure=True))

            # joint prune and quantize
            for qw, qa in [(4, 8), (4, 4), (8, 8)]:
                write(setup(qw=qw, qa=qa, pa=p, quantize_mode='t', structure=True))
    
    for mp in module_paths:
        setup = require(mp).setup
        gen_all(mp, setup)

        
    


