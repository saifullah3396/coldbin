# ColDBin: Cold Diffusion for Document Image Binarization
This repository contains the datasets and code for the paper [ColDBin: Cold Diffusion for Document Image Binarization](/to/be/added) by Saifullah Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please download the data from the links below.
## Approach:
<img align="center" src="assets/approach.png">

## Qualitative Results:
<img align="center" src="assets/qualitative.png">

# Quantitative Results
| Dataset | FM | p-FM | PSNR | DRD |
| :---: | :---: | :---: | :---: | :---: |
| DIBCO 2009  | 94.19 | 96.52 | 20.65 | 2.58 |
| DIBCO 2010  | 95.29 | 96.67 | 22.06 | 1.36 |
| DIBCO 2011  | 95.23 | 96.93 | 21.53 | 1.44 |
| DIBCO 2012  | 96.37 | 97.41 | 23.40 | 1.28 |
| DIBCO 2013  | 96.62 | 97.15 | 23.98 | 1.20 |
| DIBCO 2014  | 97.89 | 98.10 | 24.38 | 0.66 |
| DIBCO 2016  | 89.50 | 93.73 | 18.71 | 3.84 |
| DIBCO 2017  | 93.04 | 95.12 | 19.32 | 2.29 |
| DIBCO 2018  | 89.71 | 93.00 | 19.53 | 3.82 |

## Prepare dibco datasets
Download the datasets from the [link](https://drive.google.com/file/d/16pIO4c-mA2kHc1I3uqMs7VwD4Jb4F1Vc/view?usp=sharing):
Use the example dataset preparation script provided for DIBCO 2013 dataset:
```
./scripts/prepare_dataset.sh
```

## Train 
Train a diffusion model in cold manner using the example training script for DIBCO 2013 dataset:
```
./scripts/train.sh
```

## Test:
Test the trained model using the example testing script for DIBCO 2013 dataset:
```
./scripts/test.sh
```

<!-- # Citation
If you find this useful in your research, please consider citing:
```
@INPROCEEDINGS{9956167,
  author={Saifullah, Saifullah and Agne, Stefan and Dengel, Andreas and Ahmed, Sheraz},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Are Deep Models Robust against Real Distortions? A Case Study on Document Image Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1628-1635},
  doi={10.1109/ICPR56361.2022.9956167}}
``` -->

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.