# Online Test-time Adaptation
This is an open source online test-time adaptation repository based on PyTorch. It is joint work by Robert A. Marsden and Mario Döbler. It is also the official repository for the following works:
- [Introducing Intermediate Domains for Effective Self-Training during Test-Time](https://arxiv.org/abs/2208.07736)
- [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081) (CVPR2023)
- [Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction](https://arxiv.org/abs/2306.00650) (WACV2024)
- [A Lost Opportunity for Vision-Language Models: A Comparative Study of Online Test-time Adaptation for Vision-Language Models](https://arxiv.org/abs/2405.14977) (CVPR2024 MAT Workshop Community Track)

<details>
<summary>Cite</summary>

```
@article{marsden2022gradual,
  title={Gradual test-time adaptation by self-training and style transfer},
  author={Marsden, Robert A and D{\"o}bler, Mario and Yang, Bin},
  journal={arXiv preprint arXiv:2208.07736},
  year={2022}
}
```

```
@inproceedings{dobler2023robust,
  title={Robust mean teacher for continual and gradual test-time adaptation},
  author={D{\"o}bler, Mario and Marsden, Robert A and Yang, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7704--7714},
  year={2023}
}
```

```
@inproceedings{marsden2024universal,
  title={Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction},
  author={Marsden, Robert A and D{\"o}bler, Mario and Yang, Bin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2555--2565},
  year={2024}
}
```

```
@article{dobler2024lost,
  title={A Lost Opportunity for Vision-Language Models: A Comparative Study of Online Test-time Adaptation for Vision-Language Models},
  author={D{\"o}bler, Mario and Marsden, Robert A and Raichle, Tobias and Yang, Bin},
  journal={arXiv preprint arXiv:2405.14977},
  year={2024}
}
```

</details>

We encourage contributions! Pull requests to add methods are very welcome and appreciated.

## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate tta 
```

## Classification
<details open>
<summary>Features</summary>

This repository contains an extensive collection of different methods, datasets, models, and settings,
which we evaluate in a comprehensive benchmark (see below). We also provide a tutorial on how to use this 
repository in combination with CLIP-like models [here](classification/tutorials/tutorial_clip.md). 
A brief overview of the repository's main features is provided below:


- **Datasets**
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  - `imagenet_a` [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
  - `imagenet_r` [ImageNet-R](https://github.com/hendrycks/imagenet-r)
  - `imagenet_v2` [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
  - `imagenet_k` [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)
  - `imagenet_d` [ImageNet-D](https://github.com/bethgelab/robustness/tree/main/examples/imagenet_d)
  - `imagenet_d109`
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  - `Continually Changing Corruptions` [CCC](https://github.com/oripress/CCC)

- **Models**
  - For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/stable/models.html) or [timm](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#models) can be used.
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.
  - For the DomainNet-126 benchmark, there is a pre-trained model for each domain.
  - Further models include [ResNet-26 GN](https://github.com/zhangmarvin/memo).
  - It is also possible to use the models provided by [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main).
  
- **Settings**
  - `reset_each_shift` Reset the model state after the adaptation to a domain.
  - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
  - `gradual` Train the model on a sequence of gradually increasing/decreasing domain shifts without knowing when a domain shift occurs.
  - `mixed_domains` Train the model on one long test sequence where consecutive test samples are likely to originate from different domains.
  - `correlated` Same as the continual setting but the samples of each domain are further sorted by class label.
  - `mixed_domains_correlated` Mixed domains and sorted by class label.
  - Combinations like `gradual_correlated` or `reset_each_shift_correlated` are also possible.

- **Methods**
  - The repository currently supports the following methods: BN-0 (source), BN-alpha, BN-1, [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
  [MEMO](https://openreview.net/pdf?id=vn74m_tWu8O), [ETA](https://arxiv.org/abs/2204.02610), [EATA](https://arxiv.org/abs/2204.02610),
  [CoTTA](https://arxiv.org/abs/2203.13591), [AdaContrast](https://arxiv.org/abs/2204.10377), [LAME](https://arxiv.org/abs/2201.05718), 
  [SAR](https://arxiv.org/abs/2302.12400), [RoTTA](https://arxiv.org/abs/2303.13899), [RPL](https://arxiv.org/abs/2104.12928),
  [RDumb](https://arxiv.org/abs/2306.05401), [SANTA](https://openreview.net/pdf?id=V7guVYzvE4), [CMF](https://openreview.net/forum?id=BllUWdpIOA),
  [DeYO](https://openreview.net/forum?id=9w3iw8wDuE),
  [GTTA](https://arxiv.org/abs/2208.07736), [RMT](https://arxiv.org/abs/2211.13081), and [ROID](https://arxiv.org/abs/2306.00650).
  - While (all) aforementioned approaches can be used with [CLIP](https://arxiv.org/abs/2103.00020), we also support 
  [TPT](https://arxiv.org/abs/2209.07511) and [VTE](...)

- **Mixed Precision Training**
  - Almost all of the aforementioned methods (except SAR and GTTA) can be trained with mixed precision. This greatly 
  speeds up your experiments and requires less memory. However, all benchmark results are generated with fp32.

- **Modular Design**
  - Adding new methods should be rather simple, thanks to the modular design.

</details>

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.
- *CIFAR10-to-CIFAR10-C*: the data is automatically downloaded.
- *CIFAR100-to-CIFAR100-C*: the data is automatically downloaded.
- *ImageNet-to-ImageNet-C*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
- *ImageNet-to-ImageNet-A*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-A](https://github.com/hendrycks/natural-adv-examples).
- *ImageNet-to-ImageNet-R*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).
- *ImageNet-to-ImageNet-V2*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main).
- *ImageNet-to-ImageNet-Sketch*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch).
- *ImageNet-to-ImageNet-D*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php). For [ImageNet-D](https://openreview.net/pdf?id=LiC2vmzbpMO), see the download instructions for DomainNet-126 below. ImageNet-D is created by symlinks, which are set up at the first use.
- *ImageNet-to-ImageNet-D109*: see instructions for DomainNet-126 below.
- *DomainNet-126*: download the 6 splits of the [cleaned version](http://ai.bu.edu/M3SDA/). Following [MME](https://arxiv.org/abs/1904.06487), DomainNet-126 only uses a subset that contains 126 classes from 4 domains.
- *ImageNet-to-CCC*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php). CCC is integrated as a webdataset and does not need to be downloaded! Please note that it cannot be combined with settings such as correlated.

After downloading the missing datasets, you may need to adapt the path to the root directory `_C.DATA_DIR = "./data"` 
located in the file `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary (see function `complete_data_dir_path`). 
In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them.


### Run Experiments

We provide config files for all experiments and methods. Simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[ccc/cifar10_c/cifar100_c/imagenet_c/imagenet_others/domainnet126]/[source/norm_test/norm_alpha/tent/memo/rpl/eta/eata/rdumb/sar/cotta/rotta/adacontrast/lame/gtta/rmt/roid/tpt].yaml
```

For imagenet_others, the argument `CORRUPTION.DATASET` has to be passed:
```bash
python test_time.py --cfg cfgs/imagenet_others/[source/norm_test/norm_alpha/tent/memo/rpl/eta/eata/rdumb/sar/cotta/rotta/adacontrast/lame/gtta/rmt/roid/tpt].yaml CORRUPTION.DATASET [imagenet_a/imagenet_r/imagenet_k/imagenet_v2/imagenet_d109]
```

E.g., to run ROID for the ImageNet-to-ImageNet-R benchmark, run the following command.
```bash
python test_time.py --cfg cfgs/imagenet_others/roid.yaml CORRUPTION.DATASET imagenet_r
```

Alternatively, you can reproduce our experiments by running the `run.sh` in the subdirectory `classification/scripts`.
For the different settings, modify `setting` within `run.sh`.

To run the different continual DomainNet-126 sequences, you have to pass the `MODEL.CKPT_PATH` argument. 
When not specifying a `CKPT_PATH`, the sequence using the *real* domain as the source domain will be used.
The checkpoints are provided by [AdaContrast](https://github.com/DianCh/AdaContrast) and can be downloaded [here](https://drive.google.com/drive/folders/1OOSzrl6kzxIlEhNAK168dPXJcHwJ1A2X). 
Structurally, it is best to download them into the directory `./ckpt/domainnet126`.
```bash
python test_time.py --cfg cfgs/domainnet126/rmt.yaml MODEL.CKPT_PATH ./ckpt/domainnet126/best_clipart_2020.pth
```

For GTTA, we provide checkpoint files for the style transfer network. The checkpoints are provided on 
Google-Drive ([download](https://drive.google.com/file/d/1IpkUwyw8i9HEEjjD6pbbe_MCxM7yqKBq/view?usp=sharing)); 
extract the zip-file within the `classification` subdirectory.


### Changing Configurations
Changing the evaluation configuration is extremely easy. For example, to run TENT on ImageNet-to-ImageNet-C 
in the `reset_each_shift` setting with a ResNet-50 and the `IMAGENET1K_V1` initialization, the arguments below have to be passed. 
Further models and initializations can be found [here (torchvision)](https://pytorch.org/vision/stable/models.html) or [here (timm)](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file).
```bash
python test_time.py --cfg cfgs/imagenet_c/tent.yaml MODEL.ARCH resnet50 MODEL.WEIGHTS IMAGENET1K_V1 SETTING reset_each_shift
```

For ImageNet-C, the default image list provided by robustbench considers 5000 samples per domain 
(see [here](robustbench/data/imagenet_test_image_ids.txt)). If you are interested in running experiments on the full
50,000 test samples, simply set `CORRUPTION.NUM_EX 50000`, i.e. 
```bash
python test_time.py --cfg cfgs/imagenet_c/roid.yaml CORRUPTION.NUM_EX 50000 
```

### Mixed Precision
We support for most methods automatic mixed precision updates with loss scaling. 
By default mixed precision is set to false. To activate mixed precision set the argument `MIXED_PRECISION True`.


### Benchmark
We provide detailed results for each method using different models and settings [here](https://docs.google.com/spreadsheets/d/1xR-3df5xMMsEcMHe4Vo495E35RrIPs5abcR7Pztvucw/edit?usp=drive_link),
The benchmark is updated regularly as new methods, datasets or settings are added to the repository.
Further information on the settings or models can also be found in our [paper](https://arxiv.org/abs/2306.00650).

### Acknowledgements
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)
+ SAR [official](https://github.com/mr-eggplant/SAR)
+ RDumb [official](https://github.com/oripress/CCC)
+ CMF [official](https://openreview.net/forum?id=BllUWdpIOA&noteId=FbQwbITFM0)
+ DeYO [official](https://github.com/Jhyun17/DeYO)
+ TPT [official](https://github.com/azshue/TPT)


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


### Acknowledgements
+ Segmentation model is from AdaptSegNet [official](https://github.com/wasidennis/AdaptSegNet)
+ CarlaTTA was generated using Carla [official](https://github.com/carla-simulator/carla)
+ ASM [official](https://github.com/RoyalVane/ASM)
+ SM-PPM [official](https://github.com/W-zx-Y/SM-PPM)
+ MEMO [official](https://github.com/zhangmarvin/memo)
