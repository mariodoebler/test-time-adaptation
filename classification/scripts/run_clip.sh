#!/bin/bash -l

conda activate tta

methods=(source)              # choose (one or multiple) from: source, norm_test, tent, rpl, eta, eata, rdump, sar, cotta, rotta, adacontrast, lame, deyo, cmf, tpt, vte, rmt, gtta, roid
seeds=(1)                     # specify something like: (1 2 3 4 5)
mode=generalization_datasets  # choose from: imagenet_variations , generalization_datasets
architectures=(RN50 ViT-B-16 ViT-L-14)
options=("MIXED_PRECISION True TEST.NUM_WORKERS 6 MODEL.USE_CLIP True MODEL.WEIGHTS openai CLIP.PROMPT_MODE ensemble")


if [[ $mode = "imagenet_variations" ]]
then
  for arch in ${architectures[*]}; do
    for mthd in ${methods[*]}; do
      # experiments on ImageNet val
      for seed in ${seeds[*]}; do
        python test_time.py --cfg "cfgs/imagenet_c/${mthd}.yaml" MODEL.ARCH $arch RNG_SEED $seed CORRUPTION.TYPE "['none']" CORRUPTION.NUM_EX 50000 $options
      done
      # experiments on ImageNet-C
      for seed in ${seeds[*]}; do
        python test_time.py --cfg "cfgs/imagenet_c/${mthd}.yaml" MODEL.ARCH $arch RNG_SEED $seed $options
      done
    done
    dataset=(imagenet_a imagenet_v2 imagenet_r imagenet_k imagenet_d109)
    for ds in ${dataset[*]}; do
      for mthd in ${methods[*]}; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg "cfgs/imagenet_others/${mthd}.yaml" MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi


if [[ $mode = "generalization_datasets" ]]
then
  dataset=(flowers102 dtd oxford_pets stanford_cars ucf101 caltech101 food101 sun397 fgvc_aircraft eurosat)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for mthd in ${methods[*]}; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg "cfgs/imagenet_others/${mthd}.yaml" MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi
