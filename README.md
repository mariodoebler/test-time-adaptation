# Online Test-time Adaptation
This is an open source online test-time adaptation repository based on PyTorch. It is joint work by Robert A. Marsden and Mario Döbler. It is also the official repository for the work [Introducing Intermediate Domains for Effective Self-Training during Test-Time](https://arxiv.org/abs/2208.07736) and [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081).

<details open>
<summary>Features</summary>

- **Unified Benchmark**

  We provide a unified benchmark for online test-time adaptation (TTA). For single-target TTA and continual TTA, we provide: CIFAR10-to-CIFAR10C, CIFAR100-to-CIFAR100C, ImageNet-to-ImageNet-C, ImageNet-to-ImageNet-R, and DomainNet126. For the setting of gradual TTA, we provide: CIFAR10-to-CIFAR10C, CIFAR100-to-CIFAR100C, ImageNet-to-ImageNet-C and CarlaTTA (segmentation).

- **Modular Design**

  Adding new methods should be rather simple, thanks to the modular design.

- **Support of multiple methods out of box**

  The repository currently supports the following methods: BN-0 (source), BN-alpha, BN-1, TENT, CoTTA, AdaContrast, GTTA, and RMT.

</details>

## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate tta 
```

## Classification

We provide config files for all experiments and methods. Simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[continual/gradual]/[cifar10_c/cifar100_c/imagenet_c/imagenet_r/domainnet126]/[source/norm_test/norm_alpha/tent/cotta/gtta/adacontrast/rmt].yaml
```

E.g., to run RMT for the continual CIFAR100-to-CIFAR100C task using 1 update step, run the following command.
```bash
python test_time.py --cfg cfgs/continual/cifar100_c/rmt.yaml OPTIM.STEP 1
```

### Settings
The default setting is continual test-time adaptation. If you want to evaluate each domain shift separately, namely single-target test-time adaptation, the following argument has to be passed.
```bash
python test_time.py --cfg cfgs/continual/[dataset]/[method].yaml SETTING reset_each_shift
```

### Datasets
The root folder for the datasets can be specified by `_C.DATA_DIR = "./data"` in `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary. In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them. Some datasets have to be manually downloaded, for more information please read the following.

#### ImageNet-to-ImageNet-C
For the continual and gradual ImageNet-to-ImageNet-C benchmark, it is required to first download ImageNet and ImageNet-C.
+ ImageNet [download](https://www.image-net.org/download.php)
+ ImageNet-C [download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

For GTTA, we provide checkpoint files for the style transfer network. The checkpoints are provided on Google-Drive ([download](https://drive.google.com/file/d/1i1IUZ6XJYBa7TfNVM4LovsP3gBCutm9_/view?usp=sharing)); extract the zip-file within the `classification` subdirectory.

#### ImageNet-to-ImageNet-R
For the ImageNet-to-ImageNet-R benchmark, it is required to first download ImageNet and ImageNet-R.
+ ImageNet [download](https://www.image-net.org/download.php)
+ ImageNet-R [download](https://github.com/hendrycks/imagenet-r)

#### DomainNet126
For the continual DomainNet126 benchmark, it is required to first download DomainNet126. We follow AdaContrast and use a subset that contains 126 classes from 4 domains.
+ DomainNet (cleaned) [download](http://ai.bu.edu/M3SDA/)

For the different sequences you have to pass the `CKPT_PATH` argument. When not specifying a `CKPT_PATH`, the sequence using the real domain as the source domain will be used.
```
python test_time.py --cfg cfgs/continual/[dataset]/[method].yaml CKPT_PATH [./ckpt/domainnet126/best_real_2020.pth ./ckpt/domainnet126/best_clipart_2020.pth ./ckpt/domainnet126/best_painting_2020.pth ./ckpt/domainnet126/best_sketch_2020.pth]
```

For more information about the DomainNet126 benchmark, please have a look at our paper [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081). If you find the benchmark useful, feel free to cite our work.
```
@article{dobler2022robust,
  title={Robust Mean Teacher for Continual and Gradual Test-Time Adaptation},
  author={D{\"o}bler, Mario and Marsden, Robert A and Yang, Bin},
  journal={arXiv preprint arXiv:2211.13081},
  year={2022}
}
```

### Acknowledgements
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)

## Segmentation

For running the experiments based on CarlaTTA, you first have to download the dataset splits as provided below. Again, you probably have to change the data directory `_C.DATA_DIR = "./data"` in `conf.py`. Further, you have to download the pre-trained source checkpoints ([download](https://drive.google.com/file/d/1PoeW-GnFr374j-J76H8udblSwrae74LQ/view?usp=sharing)) and extract the zip-file within the `segmentation` subdirectory.

E.g., to run GTTA, use the config file provided in the directory `cfgs` and run:
```
python test_time.py --cfg cfgs/gtta.yaml
```

You can also change the test sequences by setting `LIST_NAME_TEST` to:
+ day2night: `day_night_1200.txt`
+ clear2fog: `clear_fog_1200.txt`
+ clear2rain: `clear_rain_1200.txt`
+ dynamic: `dynamic_1200.txt`
+ highway: `town04_dynamic_1200.txt`

If you choose highway as the test sequence, you have to change the source list and the corresponding checkpoint paths.
```bash
python test_time.py --cfg cfgs/gtta.yaml LIST_NAME_SRC clear_highway_train.txt LIST_NAME_TEST town04_dynamic_1200.txt CKPT_PATH_SEG ./ckpt/clear_highway/ckpt_seg.pth CKPT_PATH_ADAIN_DEC = ./ckpt/clear_highway/ckpt_adain.pth
```

### CarlaTTA
We provide the different datasets of CarlaTTA as individual zip-files on Google-Drive:
+ clear [download](https://drive.google.com/file/d/19HUmZkL5wo4gY7w5cfztgNVga_uNSVUp/view?usp=sharing)
+ day2night [download](https://drive.google.com/file/d/1R3br738UCPGryhWhJE-Uy4sCJW3FaVTr/view?usp=sharing)
+ clear2fog  [download](https://drive.google.com/file/d/1LeNF9PpdJ7lbpsvNwGy9xpC-AYlPiwMI/view?usp=sharing)
+ clear2rain [download](https://drive.google.com/file/d/1TJfQ4CjIOJtrOpUCQ7VyqKBVYQndGNa_/view?usp=sharing)
+ dynamic [download](https://drive.google.com/file/d/1jb1qJMhOSJ48XUQ7eRqT7agnDK9OBwox/view?usp=sharing)
+ dynamic-slow [download](https://drive.google.com/file/d/1RTciKaw2LhlQ4ecKMlarSKyOzsDgaurT/view?usp=sharing)
+ clear-highway [download](https://drive.google.com/file/d/1lZlxwBVBSBAguONX9K6gI2NlWqAxECvB/view?usp=sharing)
+ highway [download](https://drive.google.com/file/d/1Q_3iOuDK4t-W3lvsHwRddDqHTE8GEAIj/view?usp=sharing)

If you find our dataset CarlaTTA useful, please cite our work.
```
@article{marsden2022introducing,
  title={Introducing Intermediate Domains for Effective Self-Training during Test-Time},
  author={Marsden, Robert A and D{\"o}bler, Mario and Yang, Bin},
  journal={arXiv preprint arXiv:2208.07736},
  year={2022}
}
```

### Acknowledgements
+ Segmentation model is from AdaptSegNet [official](https://github.com/wasidennis/AdaptSegNet)
+ CarlaTTA was generated using Carla [official](https://github.com/carla-simulator/carla)
+ ASM [official](https://github.com/RoyalVane/ASM)
+ SM-PPM [official](https://github.com/W-zx-Y/SM-PPM)
+ MEMO [official](https://github.com/zhangmarvin/memo)
