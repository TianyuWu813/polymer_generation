
#Accurately discovering antimicrobial peptide-mimicking polymers via polymer inverse design framework

## Overview
Our method focus on generating new polymers in an inverse design framework, including a predictive network and a generative network.
Main techniques include multi-modal polymer representation, graph grammar distillation and reinforcement learning.

This repository includes the training process of our generative model. Details about the training process predictive model can be seen in another repository of ours.

## Environment
- python 
- pytroch 
- RDKit
- numpy
- pandas
- tqdm
- scipy


## Usage
The default task of our code is to generate specific polymers according to the given scaffold with fine-tuning a pre-trained model in RL settings. 
Users can customize their own tasks by modifying the code. 
Users can run the following py files in sequence with giving corresponding data.


```
python scaffold_1_train_prior_decorator.py and python scaffold_1_train_prior_decorator_RNN.py 

They are used to pre-train a Transformer-based or RNN based scaffold-decorator model from scratch with ChEMBL.
The idea of scaffold-decorator can be found in (https://github.com/undeadpixel/reinvent-scaffold-decorator).
Codes of constructing multi-modal representation can be found in `./chemreprop/features/` and `feature_calculate.py`
```
```
python scaffold_2_train_middle_model.py 

It is used to pre-train a scaffold-decorator model with the data sampled from the distilled grammar.
We use DEG (https://github.com/gmh14/data_efficient_grammar) to learn specific in our graph grammar distillation process.
 ```
 ```
python scaffold_3_train_rl_agent_RNN_multi_poly_NH3_stable.py 

It is a fine-tuning process for the pre-trained model with RL. And it can be customlized according to the demands.
```

## Acknowledgements
We thank the following code and all cited codes in github, which are all quite helpful for realizing our work.

[![DOI](https://zenodo.org/badge/369146587.svg)](https://zenodo.org/badge/latestdoi/369146587) 

## Contact
tianyuwu813@gmail.com