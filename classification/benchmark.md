# Online Test-Time Adaptation Benchmark
## Evaluation Protocol
All methods are evaluated in the online TTA setting. The presented results are averaged over five runs for all methods, using the seeds `1, 2, 3, 4, 5`. For the corruption benchmarks, only the highest severity level 5 is considered.

## Continual
#### CIFAR10-C
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| WRN-28       | 43.5   | 20.4  | 20.0                  | 17.9 | 20.4                  | 16.5                  | 19.3  | 18.5     | 17.0                  | 64.3 | **16.2**&plusmn;0.05 |

#### CIFAR100-C
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| ResNext-29   | 46.4   | 35.4  | 62.2 | 32.2 | 32.0                  | 32.8                  | 34.8  | 33.5     | 30.2                  | 98.5 | **29.3**&plusmn;0.04 |

#### ImageNet-C
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 82.0   | 68.6  | 62.6                  | 58.0 | 61.9                  | 63.1                  | 67.3  | 65.5     | 59.9                  | 93.5 | **54.5**&plusmn;0.1  |
| Swin-b       | 64.0   | 64.0  | 64.0                  | 52.8 | 63.7                  | 59.3                  | 62.7  | 58.1     | 52.6                  | 84.8 | **47.0**&plusmn;0.26 |
| ViT-b-16     | 60.2   | 60.2  | 54.5                  | 49.8 | 51.7                  | 77.0 | 58.3  | 57.0     | 72.9 | 79.9 | **45.0**&plusmn;0.09 |

#### ImageNet-R
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 63.8   | 60.5  | 57.6                  | 54.2 | 57.5                  | 57.4                  | 60.7  | 58.9     | 56.1                  | 99.3 | **51.2**&plusmn;0.11 |
| Swin-b       | 54.2   | -     | 53.8                  | 49.9 | 53.0                  | 52.9                  | 53.0  | 52.3     | 47.4                  | 92.7 | **45.8**&plusmn;0.12 |
| ViT-b-16     | 56.0   | -     | 53.3                  | 49.0 | 48.6                  | 69.6 | 54.4  | 54.2     | 68.8 | 95.2 | **44.2**&plusmn;0.13 |

#### ImageNet-Sketch
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 75.9   | 73.6  | 69.5                  | 64.5 | 68.4                  | 69.5                  | 70.8  | 73.0     | 68.4                  | 99.8 | **64.3**&plusmn;0.16 |
| Swin-b       | 68.4   | -     | 68.7 | 60.5 | 72.6 | 71.0 | 67.1  | 64.4     | 69.0 | 94.6 | **58.8**&plusmn;0.15 |
| ViT-b-16     | 70.6   | -     | 70.5                  | 59.7 | 70.6                  | 95.5 | 69.0  | 68.3     | 86.8 | 99.5 | **58.6**&plusmn;0.07 |

#### ImageNet-D109
| Architecture | Source | BN-1 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont. | RMT                   | LAME                  | ROID            |
|--------------|--------|-------|-----------------------|------|-----------------------|-----------------------|-------|----------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 58.8   | 55.1  | 52.9                  | 51.6 | 52.2                  | 50.8                  | 52.3  | 50.4     | 49.4                  | 85.0 | **48.0**&plusmn;0.06 |
| Swin-b       | 51.4   | -     | 66.1 | 47.5 | 54.2 | 49.9                  | 48.7  | 47.3     | 47.6                  | 86.3 | **45.1**&plusmn;0.10 |
| ViT-b-16     | 53.6   | -     | 84.0 | 47.4 | 57.4 | 73.4 | 51.2  | 49.7     | 74.2 | 88.0 | **45.0**&plusmn;0.04 |


## Mixed domains
#### CIFAR10-C
| Architecture | Source | BN-1                 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont.              | RMT                   | LAME                  | ROID            |
|--------------|--------|-----------------------|-----------------------|------|-----------------------|-----------------------|-------|-----------------------|-----------------------|-----------------------|------------------------|
| WRN-28       | 43.5   | 33.8                  | 44.1 | 28.6 | 33.8                  | 32.5                  | 33.4  | **26.2**         | 31.0                  | 75.2 | 28.0&plusmn;0.12          |

#### CIFAR100-C
| Architecture | Source | BN-1                 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont.              | RMT                   | LAME                  | ROID            |
|--------------|--------|-----------------------|-----------------------|------|-----------------------|-----------------------|-------|-----------------------|-----------------------|-----------------------|------------------------|
| ResNext-29   | 46.4   | 45.8                  | 82.5 | 36.9 | 45.5                  | 43.1                  | 45.4  | 41.8                  | 38.6                  | 98.4 | **35.0**&plusmn;0.04 |

#### ImageNet-C
| Architecture | Source | BN-1                 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont.              | RMT                   | LAME                  | ROID            |
|--------------|--------|-----------------------|-----------------------|------|-----------------------|-----------------------|-------|-----------------------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 82.0   | 82.5 | 86.4 | 72.3 | 79.4                  | 76.0                  | 78.1  | 90.8 | 75.4                  | 95.1 | **69.5**&plusmn;0.13 |
| Swin-b       | 64.0   | -                     | 62.6                  | 56.3 | 60.6                  | 63.3                  | 62.6  | 66.0 | 55.4                  | 64.6 | **55.0**&plusmn;0.26 |
| ViT-b-16     | 60.2   | -                     | 55.0                  | 51.8 | 52.3                  | 89.3 | 58.2  | 65.5 | 73.4 | 62.6 | **50.7**&plusmn;0.08 |

#### ImageNet-D109
| Architecture | Source | BN-1                 | TENT                  | EATA | SAR                   | CoTTA                 | RoTTA | AdaCont.              | RMT                   | LAME                  | ROID            |
|--------------|--------|-----------------------|-----------------------|------|-----------------------|-----------------------|-------|-----------------------|-----------------------|-----------------------|------------------------|
| ResNet-50    | 58.8   | 56.2                  | 56.1                  | 53.3 | 53.7                  | **50.3**         | 54.0  | 55.4                  | 50.7                  | 99.1 | 50.9&plusmn;0.04          |
| Swin-b       | 51.4   | -                     | 61.5 | 48.9 | 54.0 | 49.4                  | 48.1  | 49.4                  | **46.5**         | 97.3 | 47.2&plusmn;0.07          |
| ViT-b-16     | 53.6   | -                     | 76.7 | 48.6 | 61.4 | 58.0 | 50.5  | 51.4                  | 70.8 | 98.8 | **46.9**&plusmn;0.02 |


## Correlated
#### CIFAR10-C
| Architecture | Source | TENT                  | EATA                  | SAR                   | CoTTA                 | RoTTA                 | AdaCont.              | RMT                   | LAME          | ROID            |
|--------------|--------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|---------------|------------------------|
| RN-26 GN     | 32.7   | 87.6 | 40.8 | 37.1 | 44.5 | 33.7 | 30.5                  | 57.5 | **11.3** | 15.9&plusmn;0.27          |

#### ImageNet-C
| Architecture | Source | TENT                  | EATA                  | SAR                   | CoTTA                 | RoTTA                 | AdaCont.              | RMT                   | LAME          | ROID            |
|--------------|--------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|---------------|------------------------|
| Swin-b       | 64.0   | 86.7 | 74.2 | 59.3                  | 99.5 | 75.5 | 77.6 | 99.6 | 47.0          | **18.5**&plusmn;0.10 |
| ViT-b-16     | 60.2   | 80.6 | 76.2 | 53.9                  | 98.8 | 65.1 | 87.4 | 99.6 | 44.1          | **16.8**&plusmn;0.72 |

#### ImageNet-R
| Architecture | Source | TENT                  | EATA                  | SAR                   | CoTTA                 | RoTTA                 | AdaCont.              | RMT                   | LAME          | ROID            |
|--------------|--------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|---------------|------------------------|
| Swin-b       | 54.2   | 53.6                  | 53.9                  | 53.1                  | 58.9 | 54.1                  | 56.9 | 48.1                  | **13.6** | 25.2&plusmn;0.37          |
| ViT-b-16     | 56.0   | 53.4                  | 53.6                  | 49.9                  | 81.0 | 55.8                  | 62.1 | 85.8 | **13.0** | 25.8&plusmn;0.13          |

#### ImageNet-Sketch
| Architecture | Source | TENT                  | EATA                  | SAR                   | CoTTA                 | RoTTA                 | AdaCont.              | RMT                   | LAME          | ROID            |
|--------------|--------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|---------------|------------------------|
| Swin-b       | 68.4   | 67.4                  | 66.3                  | 72.3 | 95.3 | 68.1                  | 66.9                  | 91.8 | 58.2          | **43.9**&plusmn;0.19 |
| ViT-b-16     | 70.6   | 66.7                  | 63.7                  | 74.6 | 95.5 | 70.1                  | 72.3 | 97.9 | 61.0          | **44.0**&plusmn;0.14 |

#### ImageNet-D109
| Architecture | Source | TENT                  | EATA                  | SAR                   | CoTTA                 | RoTTA                 | AdaCont.              | RMT                   | LAME          | ROID            |
|--------------|--------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|---------------|------------------------|
| Swin-b       | 51.4   | 68.5 | 53.9 | 55.5 | 58.5 | 50.5                  | 52.1 | 51.9 | **30.4** | 30.6&plusmn;0.16          |
| ViT-b-16     | 53.6   | 84.3 | 57.4 | 58.7 | 93.1 | 53.8 | 56.7 | 90.6 | 35.4          | **31.7**&plusmn;0.08 |


## Mixed domains, correlated
#### ImageNet-C (Dirichlet value set to 0.01)
| Architecture | Source | SAR                   | LAME          | ROID            |
|--------------|--------|-----------------------|---------------|-----------------|
| Swin-b       | 64.0   | 64.9&plusmn;0.81 | 37.4&plusmn;0.12 | **28.6**&plusmn;0.16 |
| ViT-b-16     | 60.2   | 54.3&plusmn;0.59 | 36.1&plusmn;0.15 | **23.6**&plusmn;0.05 |

#### ImageNet-D109 (Dirichlet value set to 0.1)
| Architecture | Source | SAR                   | LAME          | ROID            |
|--------------|--------|-----------------------|---------------|-----------------|
| Swin-b       | 51.4   | 53.9&plusmn;0.52 | **28.0**&plusmn;0.39 | 28.3&plusmn;0.19 |
| ViT-b-16     | 53.6   | 60.8&plusmn;0.48 | **29.2**&plusmn;0.55 | 29.4&plusmn;0.13 |
