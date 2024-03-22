#!/bin/bash -l

conda activate tta

method=roid       # choose from: source, norm_test, tent, rpl, eta, eata, rdump, sar, cotta, rotta, adacontrast, lame, deyo, cmf, rmt, gtta, roid
setting=continual # choose from: continual, mixed_domains, correlated, mixed_domains_correlated, ccc
seeds=(1)         # to reproduce the benchmark results, use: (1 2 3 4 5)
options=()        # for RMT source-free without warm-up use: options=("RMT.LAMBDA_CE_SRC 0.0 RMT.NUM_SAMPLES_WARM_UP 0")

ccc=("cfgs/ccc/${method}.yaml")
cifar10=("cfgs/cifar10_c/${method}.yaml")
cifar100=("cfgs/cifar100_c/${method}.yaml")
imagenet_c=("cfgs/imagenet_c/${method}.yaml")
imagenet_others=("cfgs/imagenet_others/${method}.yaml")

# ---continual---
if [[ $setting = "continual" ]]
then
  for var in "${cifar10[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  for var in "${cifar100[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  architectures=(resnet50 swin_b vit_b_16)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      for seed in ${seeds[*]}; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch RNG_SEED $seed $options
      done
    done
  done
  dataset=(imagenet_r imagenet_k imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi

# ---mixed-domains---
if [[ $setting = "mixed_domains" ]]
then
  for var in "${cifar10[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  for var in "${cifar100[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  architectures=(resnet50 swin_b vit_b_16)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      for seed in ${seeds[*]}; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch RNG_SEED $seed $options
      done
    done
  done
  dataset=(imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi

# ---correlated---
if [[ $setting = "correlated" ]]
then
  for var in "${cifar10[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting MODEL.ARCH resnet26_gn MODEL.CKPT_PATH ./ckpt/resnet26_gn.pth RNG_SEED $seed $options
    done
  done
  architectures=(swin_b vit_b_16)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch RNG_SEED $seed $options
      done
    done
  done
  dataset=(imagenet_r imagenet_k imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi

# ---mixed_domains_correlated---
if [[ $setting = "mixed_domains_correlated" ]]
then
  architectures=(swin_b vit_b_16)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      for seed in ${seeds[*]}; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch RNG_SEED $seed TEST.DELTA_DIRICHLET 0.01 $options
      done
    done
  done
  dataset=(imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed TEST.DELTA_DIRICHLET 0.1 $options
        done
      done
    done
  done
fi

# ---ccc benchmark---
if [[ $setting = "ccc" ]]
then
  architectures=(resnet50)
  baseline_accs=(0 20 40)
  speeds=(1000 2000 5000)
  dataset_seeds=(43 44 45)
  for arch in ${architectures[*]}; do
    for base_acc in ${baseline_accs[*]}; do
      for speed in "${speeds[@]}"; do
        for var in "${ccc[@]}"; do
          for ds_seed in ${dataset_seeds[*]}; do
            domain="['baseline_${base_acc}_transition+speed_${speed}_seed_${ds_seed}']"
            python test_time.py --cfg $var CORRUPTION.TYPE $domain MODEL.ARCH $arch PRINT_EVERY 1000 $options
          done
        done
      done
    done
  done
fi
