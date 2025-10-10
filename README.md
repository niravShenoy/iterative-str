# Efficient Sparse Training: Combining Continuous Sparsification with Learning Rate Rewinding and Rewiring

<p>The following code contains the implementation of my Master Thesis for the implementation of a novel pruning technique called iterative Soft Threshold Reparameterization (iSTR) and study its stability and performance against baseline continuous sparsification and dynamic sparse training methods in the dense-to-sparse and sparse-to-sparse regimes. We then explore rewinding techniques and neuroregeneration and investigate whether it improves upon the performance of iSTR.</p>

## Requirements
- Python 3.6
- PyTorch v1.5.1
- CUDA v10.2

## Experiments

### iSTR Implementation
Using LRR:
**resnet50-imagenet1k-iter-str-3-LRR.yaml**
```yaml
arch: ResNet50

# ===== Dataset ===== #
name: imagenet_resnet50_balanced_STR_Iter_LRR
set: imagenet
resnet_type: res50

# ===== Learning Rate Policy ======== #
optimizer: sgd
lr: 0.1
lr_policy: multistep_lr_imagenet
lr_adjust: 30
warmup_length: 10

# ===== Network training config ===== #
str_iterations: 3
epochs: 90
weight_decay: 0.00002
weight_decay_multiplier: 1.6
momentum: 0.9
batch_size: 256
label_smoothing: 0.1
lr_rewind: True
weights_sparsity_plot: True

# ===== Sparsity =========== #
er_sparse_init: 1.0
er_sparse_method: balanced
conv_type: STRConvER
bn_type: LearnedBatchNorm
init: kaiming_normal
mode: fan_in
nonlinearity: relu
sparse_function: sigmoid
sInit_value: -200


# ===== Hardware setup ===== #
workers: 8
multigpu: "0,1,2,3"
```

Command:
```
python ~/sparse_to_sparse/STR/main.py --config ~/sparse_to_sparse/STR/configs/largescale/resnet50-imagenet1k-iter-str-3-LRR.yaml --seed 3 
```

### iSTR with neuroregeneration
Additional arguments (compared to above YAML):
With neuroregeneration (append the following to the above YAML file)
**resnet50-imagenet1k-iter-str-3-DST.yaml**
```yaml
# ====== GraNet ============ #
dst_method: None
dst_init_prune_epoch: 90
dst_final_prune_epoch: 130
dst_prune_const: True
dst_const_prune_rate: 0.1
update_frequency: 5
is_reinitialze_grown_weights: True
is_update_pruned_weights: False
```

Command:
```
python ~/sparse_to_sparse/STR/main.py --config ~/sparse_to_sparse/STR/configs/largescale/resnet50-imagenet1k-iter-str-3-LRR.yaml --seed 3 
```

## Acknowledgements
We would like to acknowledge the following code repositories on which our code is based:
- [STR](https://github.com/RAIVNLab/STR)
- [GraNet](https://github.com/VITA-Group/GraNet)
- [Random Pruning](https://github.com/AdvaitGadhikar/sparse\_to\_sparse)
- [FFCV Dataloader](https://github.com/libffcv/ffcv-imagenet)
