# ColDBin: Cold Diffusion for Document Image Binarization
This repository contains the datasets and code for the paper [ColDBin: Cold Diffusion for Document Image Binarization](/to/be/added) by Saifullah Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please download the data from the links below.

## Prepare dibco datasets
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