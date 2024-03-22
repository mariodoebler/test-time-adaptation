#!/bin/bash -l

conda activate tta

methods=(roid)    # choose from: source, norm_test, tent, rpl, eta, eata, rdump, sar, cotta, rotta, adacontrast, lame, cmf, deyo, rmt, gtta, roid
seeds=(1)         # to reproduce the benchmark results, use: (1 2 3 4 5)
options=()        # for RMT source-free without warm-up use: options=("RMT.LAMBDA_CE_SRC 0.0 RMT.NUM_SAMPLES_WARM_UP 0")
architectures=(resnet50 swin_b vit_b_16)


# ---catastrophic forgetting---
for arch in ${architectures[*]}; do
  for mthd in ${methods[*]}; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg "cfgs/imagenet_c/${mthd}.yaml" MODEL.ARCH $arch RNG_SEED $seed CORRUPTION.NUM_EX 50000 $options CORRUPTION.TYPE "[
        'gaussian_noise', 'none', 'shot_noise', 'none', 'impulse_noise', 'none',
        'defocus_blur', 'none', 'glass_blur', 'none', 'motion_blur', 'none', 'zoom_blur', 'none',
        'snow', 'none', 'frost', 'none', 'fog', 'none', 'brightness', 'none', 'contrast', 'none',
        'elastic_transform', 'none', 'pixelate', 'none', 'jpeg_compression', 'none']"
    done
  done
done
