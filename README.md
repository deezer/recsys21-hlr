# Hierarchical Latent Relation Modeling for Collaborative Metric Learning

This repository provides Python code to reproduce experiments from our paper:

> V-A. Tran, G. Salha-Galvan, R. Hennequin and M. Moussallam. Hierarchical Latent Relation Modeling for Collaborative Metric Learning. In: *Proceedings of the 15th ACM Conference on Recommender Systems (RecSys 2021)*, September 2021.


## Environment
- python 3.6.9
- tensorflow 1.15
- numpy 1.18.1
- scipy 1.6.2
- sklearn 0.22.2
- pandas 1.0.1
- toolz 0.11.1
- implicit 0.4.4

## Datasets
The following datasets are considered in our work that could be easily downloaded from Internet and put in `exp/data` directory 
- Movielens 20M (https://grouplens.org/datasets/movielens/20m/)
- Echonest (http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip)
- Yelp (https://www.kaggle.com/yelp-dataset/yelp-dataset)
- Amazon Book (https://jmcauley.ucsd.edu/data/amazon/)

## Hyperparameters
Best hyperparameters that we found through grid-search for each model on each dataset are reported in the corresponding configuration file in `configs` directory

## Experiments
All experiment scripts for train / evaluation of our models and other baselines described in the paper could be found in `scripts` directory.

You could do the following steps to run experiment:
1. Download data and put it into `exp/data` directory. For example `exp/data/ml-20m` for Movielens 20M.
2. Change data path and interaction file name in configuration file (for example `configs/mvlens/10-core/cml/*.json`).
3. Run experiment script (that contains both train and evaluation commands) in `scripts` directory
