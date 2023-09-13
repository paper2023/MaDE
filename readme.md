# MaDE

## Introduction 
This GIT is the implementation of the paper 《MaDE: Multi-scale Decision Enhancement for Multi-agent Reinforcement Learning》.

In this work, we introduce a novel methodology, termed **M**ulti-sc**a**le **D**ecision **E**nhancement (MaDE), anchored by a dual-wise bisimulation framework for pre-training agent encoders. 

The MaDE framework aims to facilitate decision-making across three pivotal dimensions: macroscale awareness, mesoscale coordination, and microscale insight. 

At the macro level, a pre-trained global encoder captures a situational awareness map to guide overall strategies. 

At the meso level, specialized local encoders generate cluster-based representations to promote inter-agent cooperation. 

At the micro level, individual agents focus on the accurate decision-making process. 


## Environments supported:

- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- [Seeker]()



##  Installation

``` Bash
# create conda environment
conda env create -f environment.yml
```

## Training


1. we use CN as an example, install package:

``` Python
cd CN
pip install -e .
```

2. pre-train the dual-wise bisimulation


``` Python
cd CN/onpolicy/scripts/train/0_offline_train_states
CUDA_VISIBLE_DEVICES=0 python train_state_6agents_cuda.py
```

3. train the MaDE

``` Python
cd CN/onpolicy/scripts/train/2_train
python train_mpe_ours.py

```


## Contact us

If you have any question about the training, feel free to contact us with e-mail: ruanjingqing2019@ia.ac.cn
