#!/bin/bash -l

conda activate tta

# Choose setting from: continual, mixed_domains, correlated, mixed_domains_correlated
setting=continual
options=()

cifar10=("cfgs/cifar10_c/roid.yaml")
cifar100=("cfgs/cifar100_c/roid.yaml")
imagenet_c=("cfgs/imagenet_c/roid.yaml")
imagenet_others=("cfgs/imagenet_others/roid.yaml")

# ---continual---
if [[ $setting = "continual" ]]
then
  for var in "${cifar10[@]}"; do
    python test_time.py --cfg $var SETTING $setting $options
  done
  for var in "${cifar100[@]}"; do
    python test_time.py --cfg $var SETTING $setting $options
  done
  architectures=(resnet50 vit_b_16 swin_b)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch $options
    done
  done
  dataset=(imagenet_r imagenet_k imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds $options
      done
    done
  done
fi

# ---mixed-domains---
if [[ $setting = "mixed_domains" ]]
then
  for var in "${cifar10[@]}"; do
    python test_time.py --cfg $var SETTING $setting $options
  done
  for var in "${cifar100[@]}"; do
    python test_time.py --cfg $var SETTING $setting $options
  done
  architectures=(resnet50 vit_b_16 swin_b)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.NUM_EX 75000 $options
    done
  done
  dataset=(imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds $options
      done
    done
  done
fi

# ---correlated---
if [[ $setting = "correlated" ]]
then
  for var in "${cifar10[@]}"; do
    python test_time.py --cfg $var SETTING $setting MODEL.ARCH resnet26_gn CKPT_PATH ./ckpt/resnet26_gn.pth $options
  done
  architectures=(vit_b_16 swin_b)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch $options
    done
  done
  dataset=(imagenet_r imagenet_k imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds $options
      done
    done
  done
fi

# ---mixed_domains_correlated---
if [[ $setting = "mixed_domains_correlated" ]]
then
  architectures=(vit_b_16 swin_b)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch TEST.ALPHA_DIRICHLET 0.01 $options
    done
  done
  dataset=(imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds TEST.ALPHA_DIRICHLET 0.1 $options
      done
    done
  done
fi
