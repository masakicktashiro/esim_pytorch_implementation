# Enhanced LSTM for Natural Language Inference
pytorch implementation of esim
this code has not achieved the 88.0% accuracy on test set yet.
89.07% on train and 87.95% on dev set and 87.65% on test set.

**"Enhanced LSTM for Natural Language Inference"**
Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. _ACL (2017)_ 

```
@InProceedings{Chen-Qian:2017:ACL,
  author    = {Chen, Qian and Zhu, Xiaodan and Ling, Zhenhua and Wei, Si and Jiang, Hui and Inkpen, Diana},
  title     = {Enhanced LSTM for Natural Language Inference},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver},
  publisher = {ACL}
}
```
this code is inspired by [theano implementation](https://github.com/lukecq1231/nli)

## Dependencies
- python 3
- torch
- numpy
- tqdm

## Running the Script
1. Download and preprocess 
```
bash fetch_and_preprocess.sh
```
2. Train and test model for ESIM
```
python3 train.py
python3 test.py
```
